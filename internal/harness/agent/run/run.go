// Package run is the harness execution service: it owns the run record and its
// lease, drives the built-in loop, streams events and persists the audit trail.
package run

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/google/uuid"

	"ragflow/internal/entity"
	"ragflow/internal/harness/agent/loop"
	"ragflow/internal/harness/caps"
	"ragflow/internal/harness/graph/types"
	"ragflow/internal/harness/store"
)

// DefaultLeaseTTL is how long a worker owns a run before another may take it.
const DefaultLeaseTTL = 60 * time.Second

// Request starts one run.
type Request struct {
	RunID          string
	AgentID        string
	AgentVersionID string
	TenantID       string
	AgentName      string
	UserMessage    string
	Trigger        string
	Scope          caps.Scope
}

// EventType enumerates the streamed events.
const (
	EventRunStarted  = "run_started"
	EventStep        = "step"
	EventRunFinished = "run_finished"
	EventError       = "error"
)

// Event is one streamed update.
type Event struct {
	Type    string           `json:"type"`
	RunID   string           `json:"run_id"`
	Step    *loop.StepRecord `json:"step,omitempty"`
	Outcome *loop.Outcome    `json:"outcome,omitempty"`
	Error   string           `json:"error,omitempty"`
}

// Config wires the service.
type Config struct {
	// Loop is the loop configuration; its Deps.Sink is chained, not replaced.
	Loop loop.Config
	// Store persists runs and steps. Nil disables persistence.
	Store *store.RunStore
	// LeaseTTL bounds run ownership. Zero uses the default.
	LeaseTTL time.Duration
	// Owner identifies this worker in the lease. Zero value uses a random id.
	Owner string
}

// Service executes runs.
type Service struct {
	runner   *loop.Runner
	store    *store.RunStore
	leaseTTL time.Duration
	owner    string

	mu          sync.Mutex
	cancels     map[string]context.CancelFunc
	subscribers map[string][]chan Event
	seqs        map[string]int
}

// New builds the service from a loop configuration.
func New(cfg Config) (*Service, error) {
	leaseTTL := cfg.LeaseTTL
	if leaseTTL <= 0 {
		leaseTTL = DefaultLeaseTTL
	}
	owner := cfg.Owner
	if owner == "" {
		owner = "worker-" + uuid.NewString()
	}

	service := &Service{
		store:       cfg.Store,
		leaseTTL:    leaseTTL,
		owner:       owner,
		cancels:     make(map[string]context.CancelFunc),
		subscribers: make(map[string][]chan Event),
		seqs:        make(map[string]int),
	}

	// Chain the loop's trace sink so every step is both persisted and streamed.
	originalSink := cfg.Loop.Deps.Sink
	cfg.Loop.Deps.Sink = func(record loop.StepRecord) {
		if originalSink != nil {
			originalSink(record)
		}
		service.persistStep(record)
		service.broadcast(record.RunID, Event{Type: EventStep, RunID: record.RunID, Step: &record})
	}

	runner, err := loop.Build(cfg.Loop)
	if err != nil {
		return nil, fmt.Errorf("run: build loop: %w", err)
	}
	service.runner = runner
	return service, nil
}

// Invoke runs one request to completion.
func (s *Service) Invoke(ctx context.Context, req Request) (loop.Outcome, error) {
	if req.RunID == "" {
		req.RunID = uuid.NewString()
	}
	if req.Trigger == "" {
		req.Trigger = "api"
	}

	if err := s.openRun(ctx, req); err != nil {
		return loop.Outcome{}, err
	}
	s.broadcast(req.RunID, Event{Type: EventRunStarted, RunID: req.RunID})

	runCtx, cancel := context.WithCancel(ctx)
	s.registerCancel(req.RunID, cancel)
	defer func() {
		s.unregisterCancel(req.RunID)
		cancel()
	}()

	// Keep the lease alive for long runs; the heartbeat stops with the run.
	stopHeartbeat := s.startHeartbeat(runCtx, req.RunID)
	defer stopHeartbeat()

	outcome, err := s.runner.Invoke(runCtx, &loop.Input{
		RunID:       req.RunID,
		AgentName:   req.AgentName,
		UserMessage: req.UserMessage,
		Scope:       req.Scope,
	}, configFor(req.RunID))

	s.closeRun(ctx, req.RunID, outcome, err)
	return outcome, err
}

// Resume continues an interrupted run from its checkpoint.
func (s *Service) Resume(ctx context.Context, req Request) (loop.Outcome, error) {
	if req.RunID == "" {
		return loop.Outcome{}, fmt.Errorf("run: run id is required to resume")
	}
	if ok, err := s.claim(ctx, req.RunID); err != nil {
		return loop.Outcome{}, err
	} else if !ok {
		return loop.Outcome{}, fmt.Errorf("run: %s is owned by another worker", req.RunID)
	}
	defer s.releaseLease(ctx, req.RunID)

	runCtx, cancel := context.WithCancel(ctx)
	s.registerCancel(req.RunID, cancel)
	defer func() {
		s.unregisterCancel(req.RunID)
		cancel()
	}()

	outcome, err := s.runner.Resume(runCtx, req.RunID)
	s.closeRun(ctx, req.RunID, outcome, err)
	return outcome, err
}

// Cancel stops a running run. It reports whether the run was owned here.
func (s *Service) Cancel(runID string) bool {
	s.mu.Lock()
	cancel, ok := s.cancels[runID]
	s.mu.Unlock()
	if !ok {
		return false
	}
	cancel()
	return true
}

// Subscribe returns a channel receiving this run's events. The caller must call
// Unsubscribe.
func (s *Service) Subscribe(runID string) chan Event {
	channel := make(chan Event, 64)
	s.mu.Lock()
	s.subscribers[runID] = append(s.subscribers[runID], channel)
	s.mu.Unlock()
	return channel
}

// Unsubscribe removes and closes a subscription.
func (s *Service) Unsubscribe(runID string, channel chan Event) {
	s.mu.Lock()
	defer s.mu.Unlock()
	list := s.subscribers[runID]
	for i, candidate := range list {
		if candidate == channel {
			s.subscribers[runID] = append(list[:i], list[i+1:]...)
			close(channel)
			return
		}
	}
}

// Steps returns the persisted audit trail of a run.
func (s *Service) Steps(ctx context.Context, runID string) ([]entity.HarnessAgentStep, error) {
	if s.store == nil {
		return nil, fmt.Errorf("run: persistence is not configured")
	}
	return s.store.ListSteps(ctx, runID, 0)
}

func (s *Service) openRun(ctx context.Context, req Request) error {
	if s.store == nil {
		return nil
	}
	run := &entity.HarnessAgentRun{
		ID:             req.RunID,
		AgentID:        req.AgentID,
		AgentVersionID: req.AgentVersionID,
		TenantID:       req.TenantID,
		Status:         "pending",
		Trigger:        req.Trigger,
		Input:          req.UserMessage,
	}
	if err := s.store.CreateRun(ctx, run); err != nil {
		return err
	}
	if ok, err := s.claim(ctx, req.RunID); err != nil {
		return err
	} else if !ok {
		return fmt.Errorf("run: %s is owned by another worker", req.RunID)
	}
	if err := s.store.StartRun(ctx, req.RunID); err != nil {
		return err
	}
	return nil
}

func (s *Service) closeRun(ctx context.Context, runID string, outcome loop.Outcome, runErr error) {
	status := "done"
	message := ""
	if runErr != nil {
		status = "failed"
		message = runErr.Error()
	} else if outcome.Interrupted {
		status = "paused"
	} else if outcome.Stopped != "" {
		status = "finished"
	}

	if s.store != nil {
		_ = s.store.FinishRun(ctx, runID, status, outcome.Text, message, outcome.TokensIn, outcome.TokensOut)
	}

	event := Event{Type: EventRunFinished, RunID: runID, Outcome: &outcome}
	if runErr != nil {
		event = Event{Type: EventError, RunID: runID, Error: runErr.Error()}
	}
	s.broadcast(runID, event)
}

func (s *Service) claim(ctx context.Context, runID string) (bool, error) {
	if s.store == nil {
		return true, nil
	}
	return s.store.AcquireLease(ctx, runID, s.owner, time.Now().Add(s.leaseTTL))
}

func (s *Service) releaseLease(ctx context.Context, runID string) {
	if s.store == nil {
		return
	}
	_ = s.store.ReleaseLease(ctx, runID)
}

func (s *Service) startHeartbeat(ctx context.Context, runID string) func() {
	if s.store == nil {
		return func() {}
	}
	ticker := time.NewTicker(s.leaseTTL / 3)
	done := make(chan struct{})
	go func() {
		for {
			select {
			case <-ctx.Done():
				ticker.Stop()
				return
			case <-done:
				ticker.Stop()
				return
			case <-ticker.C:
				_, _ = s.store.AcquireLease(ctx, runID, s.owner, time.Now().Add(s.leaseTTL))
			}
		}
	}()
	return func() { close(done) }
}

func (s *Service) registerCancel(runID string, cancel context.CancelFunc) {
	s.mu.Lock()
	s.cancels[runID] = cancel
	s.mu.Unlock()
}

func (s *Service) unregisterCancel(runID string) {
	s.mu.Lock()
	delete(s.cancels, runID)
	s.mu.Unlock()
}

func (s *Service) broadcast(runID string, event Event) {
	s.mu.Lock()
	list := append([]chan Event(nil), s.subscribers[runID]...)
	s.mu.Unlock()
	for _, channel := range list {
		select {
		case channel <- event:
		default:
			// A slow consumer must not stall the run; it can re-read the audit
			// trail from the store.
		}
	}
}

func (s *Service) persistStep(record loop.StepRecord) {
	if s.store == nil || record.RunID == "" {
		return
	}

	// Holding the lock across the insert keeps sequence allocation monotonic for
	// a run without a second round trip per step.
	s.mu.Lock()
	defer s.mu.Unlock()

	seq, ok := s.seqs[record.RunID]
	if !ok {
		next, err := s.store.NextStepSeq(context.Background(), record.RunID)
		if err != nil {
			return
		}
		seq = next
	}
	s.seqs[record.RunID] = seq + 1

	_ = s.store.AppendStep(context.Background(), &entity.HarnessAgentStep{
		ID:         uuid.NewString(),
		RunID:      record.RunID,
		Seq:        seq,
		Phase:      record.Phase,
		Round:      record.Round,
		Tool:       record.Tool,
		ArgsHash:   record.ArgsHash,
		Decision:   record.Decision,
		Reason:     record.Reason,
		DurationMS: record.DurationMS,
		TokensIn:   record.TokensIn,
		TokensOut:  record.TokensOut,
	})
}

func configFor(runID string) *types.RunnableConfig {
	config := types.NewRunnableConfig()
	config.ThreadID = runID
	config.Configurable["thread_id"] = runID
	return config
}
