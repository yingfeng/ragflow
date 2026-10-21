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
	"ragflow/internal/harness/events"
	"ragflow/internal/harness/graph/types"
	"ragflow/internal/harness/store"
	"ragflow/internal/harness/toolkit"
	"ragflow/internal/harness/toolkit/spill"
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
	Scope          toolkit.Scope
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
	// Approvals settles pending tool approvals. Nil means this worker serves no
	// approval requests, and the gate (if any) denies on its own.
	Approvals *toolkit.ApprovalBroker
	// Grants records "allow always" answers so the same call does not ask again.
	// Nil serves approvals without remembering them.
	Grants toolkit.GrantStore
	// Events receives the fine-grained audit log (model rounds, tool calls,
	// approvals, verdicts). Nil leaves only the step records in the store.
	Events events.EventLog
	// EventLayers selects which data classes reach Events. Nil drops
	// diagnostics, which is the production default.
	EventLayers events.LayerFilter
}

// Service executes runs.
type Service struct {
	runner      *loop.Runner
	store       *store.RunStore
	leaseTTL    time.Duration
	owner       string
	approvals   *toolkit.ApprovalBroker
	grants      toolkit.GrantStore
	events      events.EventLog
	eventLayers events.LayerFilter

	mu          sync.Mutex
	cancels     map[string]context.CancelFunc
	inboxes     map[string]*loop.Inbox
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

	eventLayers := cfg.EventLayers
	if eventLayers == nil {
		eventLayers = events.DefaultLayerFilter
	}

	service := &Service{
		store:       cfg.Store,
		leaseTTL:    leaseTTL,
		owner:       owner,
		approvals:   cfg.Approvals,
		grants:      cfg.Grants,
		events:      cfg.Events,
		eventLayers: eventLayers,
		cancels:     make(map[string]context.CancelFunc),
		inboxes:     make(map[string]*loop.Inbox),
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
	inbox := loop.NewInbox()
	s.registerRun(req.RunID, cancel, inbox)
	defer func() {
		s.unregisterRun(req.RunID)
		cancel()
	}()
	// The spill store is per run: a handle from one run must never resolve in
	// another.
	runCtx = spill.WithStore(runCtx, spill.New(32))
	runCtx = loop.WithInbox(runCtx, inbox)
	runCtx = s.withAudit(runCtx, req.RunID)

	// Keep the lease alive for long runs; the heartbeat stops with the run.
	stopHeartbeat := s.startHeartbeat(runCtx, req.RunID)
	defer stopHeartbeat()

	audit := events.RecorderFromContext(runCtx)
	if audit != nil {
		audit.OnRunStart(runCtx, req.AgentName, req.RunID)
	}

	outcome, err := s.runner.Invoke(runCtx, &loop.Input{
		RunID:       req.RunID,
		AgentName:   req.AgentName,
		UserMessage: req.UserMessage,
		Scope:       req.Scope,
	}, configFor(req.RunID))

	s.closeRun(ctx, req.RunID, outcome, err)
	if audit != nil {
		audit.OnRunEnd(runCtx, req.AgentName, req.RunID, err)
	}
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
	inbox := loop.NewInbox()
	s.registerRun(req.RunID, cancel, inbox)
	defer func() {
		s.unregisterRun(req.RunID)
		cancel()
	}()
	runCtx = spill.WithStore(runCtx, spill.New(32))
	runCtx = loop.WithInbox(runCtx, inbox)
	runCtx = s.withAudit(runCtx, req.RunID)

	audit := events.RecorderFromContext(runCtx)
	if audit != nil {
		audit.OnRunStart(runCtx, req.AgentName, req.RunID)
	}

	outcome, err := s.runner.Resume(runCtx, req.RunID)
	s.closeRun(ctx, req.RunID, outcome, err)
	if audit != nil {
		audit.OnRunEnd(runCtx, req.AgentName, req.RunID, err)
	}
	return outcome, err
}

// withAudit attaches a per-run recorder to the context. The run id doubles as
// the trace id: one run is one auditable unit, and a child run inherits the
// parent's recorder through the context it is spawned with.
func (s *Service) withAudit(ctx context.Context, runID string) context.Context {
	if s.events == nil {
		return ctx
	}
	recorder := events.NewEventRecorder(s.events,
		events.WithTraceID(runID),
		events.WithThreadID(runID),
		events.WithLayers(s.eventLayers),
	)
	return events.ContextWithRecorder(ctx, recorder)
}

// Recover resumes the runs a dead worker left behind and reports how many were
// picked up. A run whose checkpoint cannot be resumed is closed as failed:
// leaving it "running" hides the crash from every operator view and keeps the
// lease row alive forever.
//
// Runs created within the last two lease periods are skipped — they may belong
// to a worker that is still starting, and stealing them would double-execute a
// healthy run.
func (s *Service) Recover(ctx context.Context) (int, error) {
	if s.store == nil {
		return 0, fmt.Errorf("run: persistence is not configured")
	}
	runs, err := s.store.ListRecoverable(ctx, 50)
	if err != nil {
		return 0, err
	}
	cutoff := time.Now().Add(-2 * s.leaseTTL)
	recovered := 0
	for _, item := range runs {
		if item.CreateTime.After(cutoff) {
			continue
		}
		if err := s.store.ReclaimLease(ctx, item.ID); err != nil {
			continue
		}
		if _, err := s.Resume(ctx, Request{
			RunID:    item.ID,
			TenantID: item.TenantID,
			AgentID:  item.AgentID,
		}); err != nil {
			_ = s.store.MarkFailed(ctx, item.ID, "recovery failed: "+err.Error())
			continue
		}
		recovered++
	}
	return recovered, nil
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

// ResolveApproval settles one pending tool approval. It reports whether the
// request was still waiting on this worker. When remember is set and the call is
// allowed, the exact tool-and-arguments pair becomes a standing grant, so the
// next identical call runs without a human.
func (s *Service) ResolveApproval(runID, requestID string, allow bool, reason string, remember bool) bool {
	if s.approvals == nil {
		return false
	}
	pending, hasPending := s.approvals.Pending(runID, requestID)
	if !s.approvals.Resolve(runID, requestID, allow, reason) {
		return false
	}
	if remember && allow && s.grants != nil && hasPending && pending.TenantID != "" {
		_ = s.grants.Remember(context.Background(), toolkit.GrantFor(pending.TenantID, pending.Call))
	}
	return true
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
		_ = s.store.FinishRun(ctx, runID, status, outcome.Text, message, outcome.TokensIn, outcome.TokensOut, outcome.CostUSD)
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

// Steer injects a message into a running run. The run keeps its state and
// budget; the message joins the transcript at the next round.
func (s *Service) Steer(runID, message string) bool {
	s.mu.Lock()
	inbox := s.inboxes[runID]
	s.mu.Unlock()
	return inbox.Push(message)
}

func (s *Service) registerRun(runID string, cancel context.CancelFunc, inbox *loop.Inbox) {
	s.mu.Lock()
	s.cancels[runID] = cancel
	s.inboxes[runID] = inbox
	s.mu.Unlock()
}

func (s *Service) unregisterRun(runID string) {
	s.mu.Lock()
	delete(s.cancels, runID)
	delete(s.inboxes, runID)
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
