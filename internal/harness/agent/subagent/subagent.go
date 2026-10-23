// Package subagent runs child agents for the subagent capability.
//
// A child gets a fresh State — its own history, evidence and budget — and shares
// only the immutable dependencies (model, tool registry, builder, prompt). That
// isolation is the point: the exploration noise of a child must not evict the
// parent's context.
package subagent

import (
	"context"
	"fmt"
	"sync"

	"github.com/google/uuid"

	"ragflow/internal/entity"
	"ragflow/internal/harness/agent/definition"
	"ragflow/internal/harness/agent/loop"
	"ragflow/internal/harness/events"
	"ragflow/internal/harness/store"
	"ragflow/internal/harness/toolkit"
	"ragflow/internal/harness/toolkit/subagent"
)

// DefaultMaxRounds bounds a child run. A delegated task is narrower than the
// parent's, so it gets a tighter budget by default.
const DefaultMaxRounds = 12

// Spawner implements subagent.Spawner.
//
// One child graph is compiled per round budget and reused. A child's budget is
// not fixed at construction: it is capped by what the parent has left, the same
// way Astra keeps a child deadline inside the parent's so the parent can still
// answer.
type Spawner struct {
	deps         loop.Deps
	policyConfig definition.PolicyConfig
	maxRounds    int
	store        *store.RunStore
	readOnly     bool
	// agentID is the definition the spawner's children run. A child inherits the
	// parent's agent (same prompt, policy and tool surface), so its run row has
	// to name that agent rather than a name the model chose.
	agentID string
	// audit is the event log child runs write to. Nil leaves the child under the
	// parent's recorder, which is the default only because a host without an
	// event log has no trace to write in the first place.
	audit  events.EventLog
	layers events.LayerFilter

	mu      sync.Mutex
	runners map[int]*loop.Runner
	// children are the child runs executing on this worker, keyed by run id.
	children map[string]*childHandle
}

// WithStore persists child runs. A durable child can be inspected after the
// parent returns and survives a parent crash; without a store it exists only for
// the duration of the call.
func (s *Spawner) WithStore(runStore *store.RunStore) *Spawner {
	s.store = runStore
	return s
}

// WithAgentIdentity names the agent a child run belongs to. Children run the
// agent that spawned them, so their run rows are attributable to it.
func (s *Spawner) WithAgentIdentity(agentID string) *Spawner {
	if s == nil {
		return s
	}
	s.mu.Lock()
	s.agentID = agentID
	s.mu.Unlock()
	return s
}

// WithAudit gives each child its own audit trace.
//
// A child is a run, and the run id is the trace id: GET /runs/{child}/events
// reads the child's trace, so its model calls, tool calls and verdicts have to
// be recorded under the child's id. Inheriting the parent's recorder instead
// wrote every child event into the parent's trace, where a reader looking up the
// child by id finds nothing and the child looks like it never ran.
func (s *Spawner) WithAudit(log events.EventLog, layers events.LayerFilter) *Spawner {
	if s == nil || log == nil {
		return s
	}
	s.mu.Lock()
	s.audit = log
	if layers != nil {
		s.layers = layers
	}
	s.mu.Unlock()
	return s
}

// childAudit scopes the child's context to its own trace. Without an event log
// the parent's recorder (if any) is inherited unchanged.
func (s *Spawner) childAudit(ctx context.Context, runID string) context.Context {
	s.mu.Lock()
	log := s.audit
	layers := s.layers
	s.mu.Unlock()
	if log == nil {
		return ctx
	}
	if layers == nil {
		layers = events.DefaultLayerFilter
	}
	recorder := events.NewEventRecorder(log,
		events.WithTraceID(runID),
		events.WithThreadID(runID),
		events.WithLayers(layers),
	)
	return events.ContextWithRecorder(ctx, recorder)
}

// agentIdentity reads the agent a child run belongs to.
func (s *Spawner) agentIdentity() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.agentID
}

// WithReadOnly restricts children to read-only tools. This is the Astra
// read_only delegation profile: the child can research but not mutate.
func (s *Spawner) WithReadOnly(readOnly bool) *Spawner {
	if s == nil || readOnly == s.readOnly {
		return s
	}
	s.readOnly = readOnly
	if readOnly && s.deps.Registry != nil {
		s.deps.Registry = s.deps.Registry.OnlyReadOnly()
	}
	// The compiled child graph captured the old registry; rebuild it.
	s.mu.Lock()
	s.runners = make(map[int]*loop.Runner, 2)
	s.mu.Unlock()
	if _, err := s.runnerFor(s.maxRounds); err != nil {
		s.readOnly = false
	}
	return s
}

// New builds the spawner. With maxRounds <= 0 the child gets DefaultMaxRounds.
func New(deps loop.Deps, policyConfig definition.PolicyConfig, maxRounds int) (*Spawner, error) {
	if maxRounds <= 0 {
		maxRounds = DefaultMaxRounds
	}
	// Children must not delegate again: unbounded nesting is a fork bomb, and a
	// delegated task should be answerable at one level. The spawner owns that
	// rule, so it strips the delegation tools from the registry the child
	// inherits instead of trusting every host to build a child registry by hand.
	childDeps := deps
	if deps.Registry != nil {
		childDeps.Registry = deps.Registry.Without(
			subagent.ToolName, subagent.SpawnManyName,
			subagent.StartName, subagent.StatusName, subagent.StopName, subagent.MessageName,
		)
	}
	spawner := &Spawner{
		deps:         childDeps,
		policyConfig: policyConfig,
		maxRounds:    maxRounds,
		runners:      make(map[int]*loop.Runner, 2),
		children:     make(map[string]*childHandle),
	}
	// Compile eagerly so a broken dependency fails at wiring time, not on the
	// first delegated task.
	if _, err := spawner.runnerFor(maxRounds); err != nil {
		return nil, err
	}
	return spawner, nil
}

// runnerFor returns the compiled child graph for a round budget.
func (s *Spawner) runnerFor(maxRounds int) (*loop.Runner, error) {
	if maxRounds <= 0 {
		maxRounds = DefaultMaxRounds
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if runner, ok := s.runners[maxRounds]; ok {
		return runner, nil
	}
	runner, err := loop.Build(loop.Config{
		Deps:         s.deps,
		PolicyConfig: s.policyConfig,
		MaxRounds:    maxRounds,
	})
	if err != nil {
		return nil, fmt.Errorf("subagent: build child loop: %w", err)
	}
	s.runners[maxRounds] = runner
	return runner, nil
}

// openChild records the child run before it starts, so the delegation is
// visible and attributable instead of an anonymous orphan.
//
// A child holds no lease: the spawner that started it owns the call and blocks
// on it. Recovery therefore skips child rows (ListRecoverable) — "no live
// lease" says nothing about a child that is still running — and resumes the
// parent instead, which re-enters the delegation from its checkpoint.
func (s *Spawner) openChild(ctx context.Context, runID, parentRunID string, req subagent.Request) {
	if s.store == nil || parentRunID == "" {
		return
	}
	record := &entity.HarnessAgentRun{
		ID:          runID,
		AgentID:     s.agentIdentity(),
		ParentRunID: parentRunID,
		Depth:       req.Depth,
		TenantID:    req.Scope.TenantID,
		Status:      "running",
		Trigger:     "subagent",
		Input:       req.Prompt,
	}
	if err := s.store.CreateRun(ctx, record); err != nil {
		return
	}
	_ = s.store.StartRun(ctx, runID)
}

// closeChild writes the child's terminal state and accounting.
func (s *Spawner) closeChild(ctx context.Context, runID string, outcome loop.Outcome, runErr error) {
	if s.store == nil {
		return
	}
	status := "done"
	message := ""
	if runErr != nil {
		status = "failed"
		message = runErr.Error()
	}
	// A child run holds no lease of its own, so its generation stays zero and
	// the fenced write is unconditional from the spawner that owns it. Recovery
	// never leases a child (see openChild), which is what keeps this zero valid.
	_, _ = s.store.FinishRun(ctx, runID, 0, status, outcome.Text, message, outcome.TokensIn, outcome.TokensOut, outcome.CostUSD)
}

// Spawn runs one child to completion.
func (s *Spawner) Spawn(ctx context.Context, req subagent.Request) (subagent.Outcome, error) {
	if s == nil {
		return subagent.Outcome{}, fmt.Errorf("subagent: spawner is not configured")
	}

	// The child sees the delegation depth it was created at, so a registry that
	// still contains the delegation tools cannot recurse past the limit. Its own
	// run state is detached from the parent's context: cancellation still
	// propagates, but the transcript, spill handles and answer stream do not.
	ctx = loop.FreshRunContext(ctx)
	ctx = subagent.WithDepth(ctx, req.Depth)

	// Half the parent's remaining rounds, rounded up: the child needs enough to
	// converge, and the parent needs enough left to answer after it returns.
	runner, err := s.runnerFor(s.childRounds(ctx))
	if err != nil {
		return subagent.Outcome{}, err
	}

	runID := uuid.NewString()
	parentRunID := toolkit.RunIDFrom(ctx)
	s.openChild(ctx, runID, parentRunID, req)
	ctx = s.childAudit(ctx, runID)

	outcome, err := runner.Invoke(ctx, &loop.Input{
		RunID:       runID,
		AgentName:   s.agentIdentity(),
		UserMessage: req.Prompt,
		Scope:       req.Scope,
	}, nil)
	s.closeChild(ctx, runID, outcome, err)
	if err != nil {
		return subagent.Outcome{}, fmt.Errorf("subagent %s: %w", runID, err)
	}
	return subagent.Outcome{RunID: runID, Output: outcome.Text, TokensIn: outcome.TokensIn, TokensOut: outcome.TokensOut, CostUSD: outcome.CostUSD}, nil
}
