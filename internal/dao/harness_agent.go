package dao

import (
	"context"
	"fmt"

	"gorm.io/gorm"

	"ragflow/internal/entity"
)

// HarnessAgentDAO reads and writes agent definitions and their immutable
// versions. A definition is never updated in place: editing produces a new
// version, so a run can always be replayed against the definition it used.
type HarnessAgentDAO struct {
	db *gorm.DB
}

// NewHarnessAgentDAO builds the DAO over the process database.
func NewHarnessAgentDAO() *HarnessAgentDAO {
	return &HarnessAgentDAO{db: GetDB()}
}

// List returns the tenant's active agents.
func (d *HarnessAgentDAO) List(ctx context.Context, tenantID string, limit int) ([]entity.HarnessAgent, error) {
	if limit <= 0 {
		limit = 100
	}
	out := make([]entity.HarnessAgent, 0, limit)
	err := d.db.WithContext(ctx).
		Where("tenant_id = ? AND status = ?", tenantID, "active").
		Order("update_time DESC").
		Limit(limit).
		Find(&out).Error
	if err != nil {
		return nil, fmt.Errorf("dao: list harness agents: %w", err)
	}
	return out, nil
}

// Get returns one agent scoped to its tenant. The tenant condition is part of
// the lookup, not a post-check: an agent id from another tenant must look like a
// missing row.
func (d *HarnessAgentDAO) Get(ctx context.Context, tenantID, agentID string) (*entity.HarnessAgent, error) {
	var agent entity.HarnessAgent
	if err := d.db.WithContext(ctx).
		Where("id = ? AND tenant_id = ?", agentID, tenantID).
		Take(&agent).Error; err != nil {
		return nil, fmt.Errorf("dao: get harness agent %s: %w", agentID, err)
	}
	return &agent, nil
}

// GetVersion returns one version of an agent.
func (d *HarnessAgentDAO) GetVersion(ctx context.Context, agentID, versionID string) (*entity.HarnessAgentVersion, error) {
	var version entity.HarnessAgentVersion
	query := d.db.WithContext(ctx).Where("agent_id = ?", agentID)
	if versionID != "" {
		query = query.Where("id = ?", versionID)
	}
	if err := query.Order("version DESC").Take(&version).Error; err != nil {
		return nil, fmt.Errorf("dao: get harness agent version %s: %w", agentID, err)
	}
	return &version, nil
}

// ExistsByName reports whether the tenant already uses this agent name. The
// unique index is (tenant_id, name); a chat thread names its agent after the
// thread, so the caller needs to know before inserting.
func (d *HarnessAgentDAO) ExistsByName(ctx context.Context, db *gorm.DB, tenantID, name string) (bool, error) {
	var count int64
	if err := db.WithContext(ctx).Model(&entity.HarnessAgent{}).
		Where("tenant_id = ? AND name = ?", tenantID, name).
		Count(&count).Error; err != nil {
		return false, fmt.Errorf("dao: count harness agent %q: %w", name, err)
	}
	return count > 0, nil
}

// CreateWithVersion stores a new agent and its first version atomically. db may
// be a transaction: a chat thread and its agent are created together, and a
// failure in either must leave neither.
func (d *HarnessAgentDAO) CreateWithVersion(ctx context.Context, db *gorm.DB, agent *entity.HarnessAgent, version *entity.HarnessAgentVersion) error {
	if agent == nil || version == nil {
		return fmt.Errorf("dao: agent and version are required")
	}
	if db == nil {
		db = d.db
	}
	return db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		if err := tx.Create(agent).Error; err != nil {
			return fmt.Errorf("dao: create harness agent: %w", err)
		}
		if err := tx.Create(version).Error; err != nil {
			return fmt.Errorf("dao: create harness agent version: %w", err)
		}
		return tx.Model(&entity.HarnessAgent{}).
			Where("id = ?", agent.ID).
			Update("current_version_id", version.ID).Error
	})
}

// AppendVersion stores a new version and points the agent at it. The agent's own
// name moves with the version, in the same transaction: a row whose name disagrees
// with the definition it points at is two answers to one question.
func (d *HarnessAgentDAO) AppendVersion(ctx context.Context, agentID string, version int, definitionBody, createdBy, name string) (*entity.HarnessAgentVersion, error) {
	record := &entity.HarnessAgentVersion{
		AgentID:    agentID,
		Version:    version,
		Definition: definitionBody,
		CreatedBy:  createdBy,
	}
	err := d.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		if err := tx.Create(record).Error; err != nil {
			return fmt.Errorf("dao: append harness agent version: %w", err)
		}
		return tx.Model(&entity.HarnessAgent{}).
			Where("id = ?", agentID).
			Updates(map[string]any{
				"current_version_id": record.ID,
				"name":               name,
				"update_time":        record.CreateTime,
			}).Error
	})
	if err != nil {
		return nil, err
	}
	return record, nil
}
