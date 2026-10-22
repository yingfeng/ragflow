// Package run is the harness execution service: it owns the run record and its
// lease, drives the built-in loop, streams events and persists the audit trail.
package run

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/google/uuid"

	"ragflow/internal/entity"
	"ragflow/internal/harness/agent/loop"
	"ragflow/internal/harness/agent/model"
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
	// UserID is the actor the run acts for. Model access resolves per user, so a
	// run without it may select a provider the caller does not own.
	UserID    string
	AgentName string
	// SessionID is the conversation this turn belongs to, so a chat thread can
	// list its runs and its audit trail.
	SessionID string
	// ModelRef selects the model this run builds its runtime with. Empty means
	// the host's default (the tenant's configured model), so a caller that names
	// no model still runs, and no process-wide default decides for everyone.
	ModelRef string
	// History is the conversation before this turn (user/assistant text only).
	History     []model.Message
	UserMessage string
	Trigger     string
	Scope       toolkit.Scope
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
	// generations holds the lease generation each owned run was entered under.
	// A terminal write with a stale number is a fence violation: the run moved
	// on with another worker.
	generations map[string]int
	// leaseLost marks runs whose lease was taken while they were executing.
	// Another worker owns the record now, so this worker stops and writes no
	// terminal state: its write would be fenced, and the tokens it would spend
	// finish an answer nobody will read.
	leaseLost map[string]bool
	// userCancels marks runs the user stopped, so closeRun can tell a user
	// cancellation from a timeout or an internal failure.
	userCancels map[string]bool
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
		generations: make(map[string]int),
		leaseLost:   make(map[string]bool),
		userCancels: make(map[string]bool),
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

	generation, err := s.openRun(ctx, req)
	if err != nil {
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
	stopHeartbeat := s.startHeartbeat(runCtx, req.RunID, generation, cancel)
	defer stopHeartbeat()

	audit := events.RecorderFromContext(runCtx)
	if audit != nil {
		audit.OnRunStart(runCtx, req.AgentName, req.RunID)
	}

	outcome, err := s.runner.Invoke(runCtx, &loop.Input{
		RunID:       req.RunID,
		AgentName:   req.AgentName,
		Messages:    req.History,
		UserMessage: req.UserMessage,
		Scope:       req.Scope,
	}, configFor(req.RunID))

	s.closeRun(ctx, req.RunID, outcome, err)
	if audit != nil {
		audit.OnRunEnd(runCtx, req.AgentName, req.RunID, err)
	}
	return outcome, err
}

// ErrOwnedByAnotherWorker reports that a run could not be entered because a
// live lease holds it. Recovery must not treat this as a broken run: the winner
// is executing it right now.
var ErrOwnedByAnotherWorker = errors.New("run is owned by another worker")

// Resume continues an interrupted run from its checkpoint.
func (s *Service) Resume(ctx context.Context, req Request) (loop.Outcome, error) {
	if req.RunID == "" {
		return loop.Outcome{}, fmt.Errorf("run: run id is required to resume")
	}
	claimed, generation, err := s.claim(ctx, req.RunID)
	if err != nil {
		return loop.Outcome{}, err
	}
	if !claimed {
		return loop.Outcome{}, s.claimFailure(ctx, req.RunID)
	}
	s.rememberGeneration(req.RunID, generation)

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

	// A resumed run can outlive one lease period just like a fresh one: without
	// the heartbeat its lease expires mid-run, the next sweep reads it as
	// crashed and a second worker executes the same run.
	stopHeartbeat := s.startHeartbeat(runCtx, req.RunID, generation, cancel)
	defer stopHeartbeat()

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

// Cancel stops a running run on the user's behalf. It reports whether the run
// was owned here.
func (s *Service) Cancel(runID string) bool {
	s.mu.Lock()
	cancel, ok := s.cancels[runID]
	if ok {
		// Remember who asked: closeRun must record a user stop as cancelled,
		// not as a failure, and the model must not read it as a crash either.
		s.userCancels[runID] = true
	}
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
		if err := s.grants.Remember(context.Background(), toolkit.GrantFor(pending.TenantID, pending.Call)); err == nil {
			// A standing grant is a permission change that outlives the run:
			// the audit trail must show who widened what, or the grant is an
			// unreviewable hole in the approval policy.
			s.recordGrant(runID, pending.TenantID, pending.Call)
		}
	}
	return true
}

// recordGrant writes the grant-persisted audit fact with a recorder scoped to
// the run that created it.
func (s *Service) recordGrant(runID, tenantID string, call toolkit.Call) {
	if s.events == nil {
		return
	}
	recorder := events.NewEventRecorder(s.events,
		events.WithTraceID(runID),
		events.WithThreadID(runID),
		events.WithLayers(s.eventLayers),
	)
	recorder.RecordGrantPersisted(context.Background(), tenantID, call.Tool, argumentsFingerprint(call.Arguments))
}

// argumentsFingerprint identifies the arguments a grant covers without copying
// them into the audit trail: arguments can carry user data, and an audit log
// must not become a second copy of it.
func argumentsFingerprint(arguments string) string {
	sum := sha256.Sum256([]byte(arguments))
	return hex.EncodeToString(sum[:8])
}

// Steps returns the persisted audit trail of a run.
func (s *Service) Steps(ctx context.Context, runID string) ([]entity.HarnessAgentStep, error) {
	if s.store == nil {
		return nil, fmt.Errorf("run: persistence is not configured")
	}
	return s.store.ListSteps(ctx, runID, 0)
}

// openRun creates the run row and enters it, returning the lease generation the
// caller now holds.
func (s *Service) openRun(ctx context.Context, req Request) (int, error) {
	if s.store == nil {
		return 0, nil
	}
	run := &entity.HarnessAgentRun{
		ID:             req.RunID,
		AgentID:        req.AgentID,
		AgentVersionID: req.AgentVersionID,
		SessionID:      req.SessionID,
		TenantID:       req.TenantID,
		UserID:         req.UserID,
		Status:         "pending",
		Trigger:        req.Trigger,
		Input:          req.UserMessage,
		Scope:          scopeJSON(req.Scope),
	}
	if err := s.store.CreateRun(ctx, run); err != nil {
		return 0, err
	}
	ok, generation, err := s.claim(ctx, req.RunID)
	if err != nil {
		return 0, err
	}
	if !ok {
		return 0, fmt.Errorf("run: %s: %w", req.RunID, ErrOwnedByAnotherWorker)
	}
	s.rememberGeneration(req.RunID, generation)
	if err := s.store.StartRun(ctx, req.RunID); err != nil {
		return 0, err
	}
	return generation, nil
}

func (s *Service) closeRun(ctx context.Context, runID string, outcome loop.Outcome, runErr error) {
	status := "done"
	message := ""
	switch {
	case s.takeUserCancel(runID) && (runErr == nil || errors.Is(runErr, context.Canceled)):
		// A stopped run is not a failure: the operator asked for it, and a
		// failure status would page someone for a decision that was made on
		// purpose.
		status = "cancelled"
		message = "cancelled by user"
	case runErr != nil:
		status = "failed"
		message = runErr.Error()
	case outcome.Interrupted:
		status = "paused"
	case outcome.Stopped != "":
		status = "finished"
	}

	if s.store != nil {
		if s.takeLeaseLost(runID) {
			// The run belongs to a successor now. It may already have written
			// the answer, so this worker's terminal state would either be
			// fenced out or, worse, race the successor's own write.
			s.recordLeaseLost(ctx, runID)
		} else {
			// The caller's context may already be cancelled (client hung up).
			// The terminal write is not part of that request: losing it leaves
			// the run "running", and the next sweep re-executes finished work.
			writeCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
			applied, err := s.store.FinishRun(writeCtx, runID, s.generationOf(runID),
				status, outcome.Text, message, outcome.TokensIn, outcome.TokensOut, outcome.CostUSD)
			switch {
			case err != nil:
				// Recording the end must never mask the run's own outcome.
			case !applied:
				// The lease changed hands mid-run (recovery, restart) and a
				// successor owns the run now.
				s.recordLeaseLost(ctx, runID)
			}
			cancel()
		}
	}

	event := Event{Type: EventRunFinished, RunID: runID, Outcome: &outcome}
	if runErr != nil {
		event = Event{Type: EventError, RunID: runID, Error: runErr.Error()}
	}
	s.broadcast(runID, event)
}

// recordLeaseLost writes the fence violation to the audit trail: a run whose
// terminal write was rejected means ownership moved, and an operator reading the
// trace needs that fact next to the work that was thrown away.
func (s *Service) recordLeaseLost(ctx context.Context, runID string) {
	if recorder := events.RecorderFromContext(ctx); recorder != nil {
		recorder.RecordError(ctx, "run "+runID+" was taken over by another worker: its terminal write was dropped")
	}
}

// scopeJSON persists the run's data reach so a recovered run keeps the reach it
// was started with instead of inheriting whatever the recovering worker has.
func scopeJSON(scope toolkit.Scope) string {
	encoded, err := json.Marshal(scope)
	if err != nil {
		return ""
	}
	return string(encoded)
}

// rememberGeneration records the lease generation a run was entered under.
func (s *Service) rememberGeneration(runID string, generation int) {
	s.mu.Lock()
	s.generations[runID] = generation
	s.mu.Unlock()
}

func (s *Service) generationOf(runID string) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.generations[runID]
}

func (s *Service) takeUserCancel(runID string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	cancelled := s.userCancels[runID]
	delete(s.userCancels, runID)
	return cancelled
}

func (s *Service) markLeaseLost(runID string) {
	s.mu.Lock()
	s.leaseLost[runID] = true
	s.mu.Unlock()
}

func (s *Service) takeLeaseLost(runID string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	lost := s.leaseLost[runID]
	delete(s.leaseLost, runID)
	return lost
}

func (s *Service) claim(ctx context.Context, runID string) (bool, int, error) {
	if s.store == nil {
		return true, 0, nil
	}
	return s.store.AcquireLease(ctx, runID, s.owner, time.Now().Add(s.leaseTTL))
}

// claimFailure explains why a lease could not be taken. The distinction is not
// cosmetic: an operator resuming a finished run needs to hear that, and recovery
// must not read "somebody owns this" as "this run is broken".
func (s *Service) claimFailure(ctx context.Context, runID string) error {
	record, err := s.store.GetRun(ctx, runID)
	if err != nil {
		return err
	}
	if record == nil {
		return fmt.Errorf("run %s does not exist", runID)
	}
	switch record.Status {
	case "done", "finished", "failed", "cancelled":
		return fmt.Errorf("run %s already ended (%s) and cannot be resumed", runID, record.Status)
	}
	return fmt.Errorf("run %s: %w", runID, ErrOwnedByAnotherWorker)
}

// startHeartbeat extends the lease until the run returns. It renews rather than
// re-acquires: a renewal cannot bump the generation, so the heartbeat can never
// fence the run it is keeping alive. When the renewal stops matching the row,
// the run has moved on and the loop is cancelled instead of spending more
// tokens on an answer this worker will not be allowed to record.
func (s *Service) startHeartbeat(ctx context.Context, runID string, generation int, cancel context.CancelFunc) func() {
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
				renewed, err := s.store.RenewLease(ctx, runID, s.owner, generation, time.Now().Add(s.leaseTTL))
				if err != nil {
					// A transient store failure is not a lost lease: the next
					// tick retries while the current lease is still valid.
					continue
				}
				if renewed {
					continue
				}
				s.markLeaseLost(runID)
				cancel()
				return
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
	delete(s.generations, runID)
	delete(s.leaseLost, runID)
	delete(s.userCancels, runID)
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
