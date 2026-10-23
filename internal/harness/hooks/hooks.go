// Package hooks provides lifecycle extension points around the agent loop.
//
// The loop and the toolkit own the runtime semantics; a hook is how embedding
// code participates without forking either. Thirteen points exist, mirroring
// the reference runtimes (codex hook_runtime, Claude Code hooks):
//
//	PreModelRequest / PostModelRequest    around every model call
//	PreToolUse / PostToolUse             around every tool call
//	PreCompact / PostCompact              around every history fold
//	SessionStart / SessionEnd             around the run
//	UserPromptSubmit                      the turn's question, as submitted
//	SubagentStart / SubagentStop          around a delegated child
//	Stop                                  the run's terminal decision
//	Interrupt                             a cancelled tool batch
//
// The mutating phases are Pre* only, plus PostToolUse: a PreToolUse hook may
// rewrite the arguments or veto the call before it reaches the capability or
// the approval gate, and a PostToolUse hook may veto or replace what the tool
// returned. Everything else is observation.
//
// A registry with no hooks is a nil receiver away: every method is nil-safe,
// so the loop calls straight through without conditionals on the hot path.
package hooks

import (
	"context"
	"sort"
	"sync"
)

// Event is one lifecycle point.
type Event string

// The lifecycle points. They are stable API: a hook registered against them
// keeps working while the loop internals change.
const (
	PreModelRequest  Event = "pre_model_request"
	PostModelRequest Event = "post_model_request"
	PreToolUse       Event = "pre_tool_use"
	PostToolUse      Event = "post_tool_use"
	PreCompact       Event = "pre_compact"
	PostCompact      Event = "post_compact"
	SessionStart     Event = "session_start"
	SessionEnd       Event = "session_end"
	UserPromptSubmit Event = "user_prompt_submit"
	SubagentStart    Event = "subagent_start"
	SubagentStop     Event = "subagent_stop"
	Stop             Event = "stop"
	Interrupt        Event = "interrupt"
)

// AllEvents is the full set, for hooks that want to observe everything.
func AllEvents() []Event {
	return []Event{
		PreModelRequest, PostModelRequest,
		PreToolUse, PostToolUse,
		PreCompact, PostCompact,
		SessionStart, SessionEnd,
		UserPromptSubmit,
		SubagentStart, SubagentStop,
		Stop, Interrupt,
	}
}

// ToolContext describes one tool call before it runs.
type ToolContext struct {
	RunID     string
	Round     int
	Tool      string
	Arguments string
	Risk      string
}

// ToolResultContext describes one tool call after it ran.
type ToolResultContext struct {
	RunID     string
	Round     int
	Tool      string
	Arguments string
	Content   string
	IsError   bool
}

// ModelContext describes one model call before it is sent.
type ModelContext struct {
	RunID     string
	Round     int
	Model     string
	Messages  int
	EstTokens int
}

// ModelResultContext describes one model call after it returned.
type ModelResultContext struct {
	RunID     string
	Round     int
	Model     string
	TokensIn  int
	TokensOut int
	Error     string
}

// CompactContext describes one history fold.
type CompactContext struct {
	RunID    string
	Round    int
	Trigger  string // pressure tier that forced the fold
	Strategy string // fold | prune | clear
	Dropped  int
}

// RunContext is the payload of the run-level points: start, end, prompt, stop,
// interrupt and the sub-agent points.
type RunContext struct {
	RunID string
	Agent string
	Round int
	// Prompt carries the user's turn on UserPromptSubmit.
	Prompt string
	// Name carries the child's agent on SubagentStart / SubagentStop.
	Name string
	// Reason carries why the run stopped, or why the batch was interrupted.
	Reason string
}

// Mutation is what a hook changed. All fields are optional: an empty mutation
// is pure observation.
type Mutation struct {
	// Arguments replaces the tool arguments (PreToolUse). The replacement is
	// what the approval gate, the audit trail and the capability all see.
	Arguments string
	// Content replaces the tool result (PostToolUse).
	Content string
	// Veto stops the call. On PreToolUse the tool never runs; on PostToolUse
	// the result is discarded.
	Veto bool
	// VetoReason is the message the model reads when a veto fires.
	VetoReason string
}

// Hook receives every event it subscribed to.
type Hook interface {
	Events() []Event
	// Handle returns a Mutation for the mutating phases; nil elsewhere.
	Handle(ctx context.Context, event Event, payload any) *Mutation
}

// HookFunc adapts a function to Hook. Events must be provided by the caller:
// the func is called only for the events it registered.
type HookFunc struct {
	OnEvents []Event
	Fn       func(ctx context.Context, event Event, payload any) *Mutation
}

// Events implements Hook.
func (h HookFunc) Events() []Event { return h.OnEvents }

// Handle implements Hook.
func (h HookFunc) Handle(ctx context.Context, event Event, payload any) *Mutation {
	return h.Fn(ctx, event, payload)
}

// Match narrows a hook to a subset of tools or models. Empty means "all".
type Match struct {
	Events []Event
	Tools  []string
	Models []string
}

func (m Match) matches(event Event, tool, model string) bool {
	if len(m.Events) > 0 {
		found := false
		for _, candidate := range m.Events {
			if candidate == event {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	if len(m.Tools) > 0 && !containsFold(m.Tools, tool) {
		return false
	}
	if len(m.Models) > 0 && !containsFold(m.Models, model) {
		return false
	}
	return true
}

func containsFold(list []string, value string) bool {
	for _, candidate := range list {
		if equalFold(candidate, value) {
			return true
		}
	}
	return false
}

func equalFold(a, b string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := 0; i < len(a); i++ {
		ca, cb := a[i], b[i]
		if 'A' <= ca && ca <= 'Z' {
			ca += 'a' - 'A'
		}
		if 'A' <= cb && cb <= 'Z' {
			cb += 'a' - 'A'
		}
		if ca != cb {
			return false
		}
	}
	return true
}

type entry struct {
	match Match
	hook  Hook
}

// Registry dispatches lifecycle events to the registered hooks.
//
// It is safe for concurrent use: hooks may be added while a run executes (an
// operator-facing registry does exactly that), and the dispatch copies the
// entry list rather than holding the lock while calling out.
type Registry struct {
	mu      sync.Mutex
	entries []entry
}

// NewRegistry builds an empty registry.
func NewRegistry() *Registry { return &Registry{} }

// Add registers one hook under a match.
func (r *Registry) Add(match Match, hook Hook) {
	if r == nil || hook == nil || len(match.Events) == 0 {
		return
	}
	// A hook that never fires is a registration bug: fail it early instead of
	// letting an extension silently believe it is protected.
	subscribed := hook.Events()
	if len(subscribed) == 0 {
		return
	}
	r.mu.Lock()
	r.entries = append(r.entries, entry{match: match, hook: hook})
	r.mu.Unlock()
}

// AddFunc registers a function hook.
func (r *Registry) AddFunc(match Match, fn func(ctx context.Context, event Event, payload any) *Mutation) {
	if fn == nil {
		return
	}
	events := match.Events
	r.Add(match, HookFunc{OnEvents: events, Fn: fn})
}

// Count reports how many hooks are registered.
func (r *Registry) Count() int {
	if r == nil {
		return 0
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.entries)
}

func (r *Registry) snapshot() []entry {
	r.mu.Lock()
	entries := make([]entry, len(r.entries))
	copy(entries, r.entries)
	r.mu.Unlock()
	return entries
}

func (r *Registry) fire(ctx context.Context, event Event, tool, model string, payload any) *Mutation {
	if r == nil {
		return nil
	}
	var merged *Mutation
	for _, item := range r.snapshot() {
		if !item.match.matches(event, tool, model) {
			continue
		}
		if !containsEvent(item.hook.Events(), event) {
			continue
		}
		mutation := item.hook.Handle(ctx, event, payload)
		if mutation == nil {
			continue
		}
		if merged == nil {
			merged = &Mutation{}
		}
		if mutation.Arguments != "" {
			merged.Arguments = mutation.Arguments
		}
		if mutation.Content != "" {
			merged.Content = mutation.Content
		}
		if mutation.Veto {
			merged.Veto = true
			if mutation.VetoReason != "" {
				merged.VetoReason = mutation.VetoReason
			}
		}
	}
	return merged
}

func containsEvent(events []Event, event Event) bool {
	for _, candidate := range events {
		if candidate == event {
			return true
		}
	}
	return false
}

// PreToolUse runs before a tool call. A returned Mutation may rewrite the
// arguments or veto the call outright — a veto fires before the approval gate,
// so a policy hook never leaves a dangerous call waiting on a human.
func (r *Registry) PreToolUse(ctx context.Context, payload ToolContext) *Mutation {
	return r.fire(ctx, PreToolUse, payload.Tool, "", payload)
}

// PostToolUse runs after a tool call. A returned Mutation may replace the
// result's content (for example, a sanitizer) or discard it.
func (r *Registry) PostToolUse(ctx context.Context, payload ToolResultContext) *Mutation {
	return r.fire(ctx, PostToolUse, payload.Tool, "", payload)
}

// PreModelRequest runs before a model call.
func (r *Registry) PreModelRequest(ctx context.Context, payload ModelContext) *Mutation {
	return r.fire(ctx, PreModelRequest, "", payload.Model, payload)
}

// PostModelRequest runs after a model call.
func (r *Registry) PostModelRequest(ctx context.Context, payload ModelResultContext) *Mutation {
	return r.fire(ctx, PostModelRequest, "", payload.Model, payload)
}

// PreCompact runs before a history fold.
func (r *Registry) PreCompact(ctx context.Context, payload CompactContext) *Mutation {
	return r.fire(ctx, PreCompact, "", "", payload)
}

// PostCompact runs after a history fold.
func (r *Registry) PostCompact(ctx context.Context, payload CompactContext) *Mutation {
	return r.fire(ctx, PostCompact, "", "", payload)
}

// Notify delivers an observation-only event.
func (r *Registry) Notify(ctx context.Context, event Event, payload RunContext) {
	switch event {
	case SessionStart, SessionEnd, UserPromptSubmit, SubagentStart, SubagentStop, Stop, Interrupt:
	default:
		return
	}
	r.fire(ctx, event, "", "", payload)
}

// SortedEvents returns the event set of a hook in stable order, for logs.
func SortedEvents(events []Event) []Event {
	out := make([]Event, len(events))
	copy(out, events)
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}
