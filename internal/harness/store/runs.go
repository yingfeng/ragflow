package store

import (
	"context"
	"fmt"
	"time"

	"gorm.io/gorm"

	"ragflow/internal/entity"
)

// RunStore persists harness runs, steps, feedback and lessons.
type RunStore struct {
	db *gorm.DB
}

// NewRunStore builds a run store.
func NewRunStore(db *gorm.DB) *RunStore { return &RunStore{db: db} }

// CreateRun inserts a run row.
func (s *RunStore) CreateRun(ctx context.Context, run *entity.HarnessAgentRun) error {
	if run == nil || run.ID == "" {
		return fmt.Errorf("store: run id is required")
	}
	if err := s.db.WithContext(ctx).Create(run).Error; err != nil {
		return fmt.Errorf("store: create run %s: %w", run.ID, err)
	}
	return nil
}

// GetRun returns one run row, or nil when it does not exist.
func (s *RunStore) GetRun(ctx context.Context, runID string) (*entity.HarnessAgentRun, error) {
	var record entity.HarnessAgentRun
	if err := s.db.WithContext(ctx).Where("id = ?", runID).Take(&record).Error; err != nil {
		if err == gorm.ErrRecordNotFound {
			return nil, nil
		}
		return nil, fmt.Errorf("store: get run %s: %w", runID, err)
	}
	return &record, nil
}

// ListChildren returns the child runs one parent spawned, oldest first, so a
// delegating run can inspect what it started.
func (s *RunStore) ListChildren(ctx context.Context, parentRunID string) ([]entity.HarnessAgentRun, error) {
	if parentRunID == "" {
		return nil, nil
	}
	var records []entity.HarnessAgentRun
	if err := s.db.WithContext(ctx).
		Where("parent_run_id = ?", parentRunID).
		Order("create_time").
		Limit(200).
		Find(&records).Error; err != nil {
		return nil, fmt.Errorf("store: list children of %s: %w", parentRunID, err)
	}
	return records, nil
}

// AcquireLease takes the run's lease when it is free, expired or already held by
// the same owner.
func (s *RunStore) AcquireLease(ctx context.Context, runID, owner string, until time.Time) (bool, error) {
	result := s.db.WithContext(ctx).
		Model(&entity.HarnessAgentRun{}).
		Where("id = ?", runID).
		Where("lease_owner = '' OR lease_owner = ? OR lease_until IS NULL OR lease_until < ?", owner, time.Now()).
		Updates(map[string]any{"lease_owner": owner, "lease_until": &until})
	if result.Error != nil {
		return false, fmt.Errorf("store: lease %s: %w", runID, result.Error)
	}
	return result.RowsAffected > 0, nil
}

// ReleaseLease clears the lease of a finished run.
func (s *RunStore) ReleaseLease(ctx context.Context, runID string) error {
	result := s.db.WithContext(ctx).
		Model(&entity.HarnessAgentRun{}).
		Where("id = ?", runID).
		Updates(map[string]any{"lease_owner": "", "lease_until": nil})
	if result.Error != nil {
		return fmt.Errorf("store: release lease %s: %w", runID, result.Error)
	}
	return nil
}

// StartRun marks the run as running.
func (s *RunStore) StartRun(ctx context.Context, runID string) error {
	now := time.Now()
	result := s.db.WithContext(ctx).
		Model(&entity.HarnessAgentRun{}).
		Where("id = ?", runID).
		Updates(map[string]any{"status": "running", "started_at": &now})
	if result.Error != nil {
		return fmt.Errorf("store: start run %s: %w", runID, result.Error)
	}
	return nil
}

// FinishRun closes a run with its terminal status and accounting.
func (s *RunStore) FinishRun(ctx context.Context, runID, status, output, errMessage string, tokensIn, tokensOut int, cost float64) error {
	now := time.Now()
	updates := map[string]any{
		"status":      status,
		"output":      output,
		"error":       errMessage,
		"ended_at":    &now,
		"tokens_in":   tokensIn,
		"tokens_out":  tokensOut,
		"cost":        cost,
		"lease_owner": "",
		"lease_until": nil,
	}
	result := s.db.WithContext(ctx).
		Model(&entity.HarnessAgentRun{}).
		Where("id = ?", runID).
		Updates(updates)
	if result.Error != nil {
		return fmt.Errorf("store: finish run %s: %w", runID, result.Error)
	}
	return nil
}

// AppendStep stores one audit record. Sequence numbers come from the caller so
// replay keeps the original ordering.
func (s *RunStore) AppendStep(ctx context.Context, step *entity.HarnessAgentStep) error {
	if step == nil || step.RunID == "" {
		return fmt.Errorf("store: step run id is required")
	}
	if err := s.db.WithContext(ctx).Create(step).Error; err != nil {
		return fmt.Errorf("store: append step %s#%d: %w", step.RunID, step.Seq, err)
	}
	return nil
}

// NextStepSeq returns the next free sequence number of a run.
func (s *RunStore) NextStepSeq(ctx context.Context, runID string) (int, error) {
	var maxSeq *int
	err := s.db.WithContext(ctx).
		Model(&entity.HarnessAgentStep{}).
		Where("run_id = ?", runID).
		Select("MAX(seq)").
		Scan(&maxSeq).Error
	if err != nil {
		return 0, fmt.Errorf("store: step sequence %s: %w", runID, err)
	}
	if maxSeq == nil {
		return 1, nil
	}
	return *maxSeq + 1, nil
}

// ListSteps returns a run's steps in execution order.
func (s *RunStore) ListSteps(ctx context.Context, runID string, limit int) ([]entity.HarnessAgentStep, error) {
	if limit <= 0 {
		limit = 500
	}
	steps := make([]entity.HarnessAgentStep, 0, limit)
	err := s.db.WithContext(ctx).
		Where("run_id = ?", runID).
		Order("seq ASC").
		Limit(limit).
		Find(&steps).Error
	if err != nil {
		return nil, fmt.Errorf("store: list steps %s: %w", runID, err)
	}
	return steps, nil
}

// SaveFeedback stores a human signal about a run.
func (s *RunStore) SaveFeedback(ctx context.Context, feedback *entity.HarnessAgentFeedback) error {
	if feedback == nil || feedback.RunID == "" {
		return fmt.Errorf("store: feedback run id is required")
	}
	if err := s.db.WithContext(ctx).Create(feedback).Error; err != nil {
		return fmt.Errorf("store: save feedback %s: %w", feedback.RunID, err)
	}
	return nil
}

// ListFeedback returns the feedback of a run, newest first.
func (s *RunStore) ListFeedback(ctx context.Context, runID string, limit int) ([]entity.HarnessAgentFeedback, error) {
	if limit <= 0 {
		limit = 100
	}
	out := make([]entity.HarnessAgentFeedback, 0, limit)
	err := s.db.WithContext(ctx).
		Where("run_id = ?", runID).
		Order("create_time DESC").
		Limit(limit).
		Find(&out).Error
	if err != nil {
		return nil, fmt.Errorf("store: list feedback %s: %w", runID, err)
	}
	return out, nil
}

// SaveLesson stores one promoted or proposed experience entry.
func (s *RunStore) SaveLesson(ctx context.Context, lesson *entity.HarnessAgentLesson) error {
	if lesson == nil || lesson.Title == "" {
		return fmt.Errorf("store: lesson title is required")
	}
	if err := s.db.WithContext(ctx).Create(lesson).Error; err != nil {
		return fmt.Errorf("store: save lesson %q: %w", lesson.Title, err)
	}
	return nil
}

// ListLessons returns the active lessons injected into an agent's prompt.
func (s *RunStore) ListLessons(ctx context.Context, tenantID, agentID string, limit int) ([]entity.HarnessAgentLesson, error) {
	if limit <= 0 {
		limit = 20
	}
	query := s.db.WithContext(ctx).
		Where("tenant_id = ? AND status = ?", tenantID, "active").
		Order("evidence_count DESC, update_time DESC").
		Limit(limit)
	if agentID != "" {
		query = query.Where("agent_id = ? OR agent_id = ''", agentID)
	}
	out := make([]entity.HarnessAgentLesson, 0, limit)
	if err := query.Find(&out).Error; err != nil {
		return nil, fmt.Errorf("store: list lessons: %w", err)
	}
	return out, nil
}

// ListRecoverable returns runs a dead worker left behind: still pending or
// running, and holding no live lease. Recovery resumes them from their last
// checkpoint instead of letting them sit in "running" forever.
func (s *RunStore) ListRecoverable(ctx context.Context, limit int) ([]entity.HarnessAgentRun, error) {
	if limit <= 0 {
		limit = 50
	}
	runs := make([]entity.HarnessAgentRun, 0, limit)
	err := s.db.WithContext(ctx).
		Where("status IN ?", []string{"pending", "running"}).
		Where("lease_until IS NULL OR lease_until < ?", time.Now()).
		Order("create_time ASC").
		Limit(limit).
		Find(&runs).Error
	if err != nil {
		return nil, fmt.Errorf("store: list recoverable runs: %w", err)
	}
	return runs, nil
}

// ReclaimLease clears a stale lease so the recovering worker can claim the run.
func (s *RunStore) ReclaimLease(ctx context.Context, runID string) error {
	result := s.db.WithContext(ctx).
		Model(&entity.HarnessAgentRun{}).
		Where("id = ?", runID).
		Updates(map[string]any{"lease_owner": "", "lease_until": nil})
	if result.Error != nil {
		return fmt.Errorf("store: reclaim lease %s: %w", runID, result.Error)
	}
	return nil
}

// MarkFailed closes a run that cannot be resumed (no checkpoint, corrupted
// state). Leaving it "running" hides the failure from every operator view.
func (s *RunStore) MarkFailed(ctx context.Context, runID, message string) error {
	now := time.Now()
	result := s.db.WithContext(ctx).
		Model(&entity.HarnessAgentRun{}).
		Where("id = ?", runID).
		Updates(map[string]any{
			"status":      "failed",
			"error":       message,
			"ended_at":    &now,
			"lease_owner": "",
			"lease_until": nil,
		})
	if result.Error != nil {
		return fmt.Errorf("store: mark run %s failed: %w", runID, result.Error)
	}
	return nil
}

// PromoteLesson flips a proposed lesson to active after the RSI gate passed.
func (s *RunStore) PromoteLesson(ctx context.Context, lessonID string, evidenceCount int) error {
	result := s.db.WithContext(ctx).
		Model(&entity.HarnessAgentLesson{}).
		Where("id = ?", lessonID).
		Updates(map[string]any{"status": "active", "evidence_count": evidenceCount})
	if result.Error != nil {
		return fmt.Errorf("store: promote lesson %s: %w", lessonID, result.Error)
	}
	return nil
}
