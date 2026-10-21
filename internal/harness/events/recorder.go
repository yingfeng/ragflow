package events

import (
	"context"
	"encoding/json"
	"sync"
	"sync/atomic"

	"ragflow/internal/harness/graph/pregel"
)

// ---- Context helpers for passing EventRecorder through context ----

type recorderContextKey struct{}

// ContextWithRecorder stores an EventRecorder in context for use by
// model wrappers and tool middlewares in the agent core.
func ContextWithRecorder(ctx context.Context, r *EventRecorder) context.Context {
	return context.WithValue(ctx, recorderContextKey{}, r)
}

// RecorderFromContext retrieves an EventRecorder from context.
// Returns nil when no recorder is present.
func RecorderFromContext(ctx context.Context) *EventRecorder {
	r, _ := ctx.Value(recorderContextKey{}).(*EventRecorder)
	return r
}

// RecorderOption configures an EventRecorder.
type RecorderOption func(*recorderOptions)

type recorderOptions struct {
	traceID  string
	threadID string
	allow    LayerFilter
}

// WithTraceID sets the trace ID for the recorder.
func WithTraceID(traceID string) RecorderOption {
	return func(o *recorderOptions) {
		o.traceID = traceID
	}
}

// WithThreadID sets the thread ID for the recorder.
func WithThreadID(threadID string) RecorderOption {
	return func(o *recorderOptions) {
		o.threadID = threadID
	}
}

// WithLayers restricts the recorder to the layers the filter accepts. A nil
// filter records every layer, which is what an audit-only log usually wants; the
// composition root decides whether to drop diagnostics.
func WithLayers(filter LayerFilter) RecorderOption {
	return func(o *recorderOptions) {
		o.allow = filter
	}
}

// EventRecorder records graph execution events as append-only Events.
// It implements pregel.GraphCallback and can be added to a CallbackManager.
// Additionally, RecordModelCall / RecordToolCall / etc. provide fine-grained
// event recording for LLM invocations and tool executions.
//
// Every event carries its data layer and a causal link: ParentID points at the
// previous event of the same trace and thread, and a paired result event also
// lists its start event in CausedBy. That chain is what makes a run auditable
// without replaying the whole log.
//
// Usage:
//
//	store := events.NewMemoryEventStore()
//	recorder := events.NewEventRecorder(store, events.WithTraceID("trace-001"))
//	cb := pregel.NewCallbackManager()
//	cb.AddCallback(recorder)
type EventRecorder struct {
	store    EventLog
	clock    *LogicalClock
	traceID  string
	threadID string
	allow    LayerFilter
	failed   atomic.Int64

	mu   sync.Mutex
	last map[string]EventID
}

// NewEventRecorder creates a new EventRecorder.
func NewEventRecorder(store EventLog, opts ...RecorderOption) *EventRecorder {
	o := &recorderOptions{}
	for _, opt := range opts {
		opt(o)
	}
	return &EventRecorder{
		store:    store,
		clock:    NewLogicalClock(),
		traceID:  o.traceID,
		threadID: o.threadID,
		allow:    o.allow,
		last:     make(map[string]EventID),
	}
}

// Failures reports how many appends the log rejected. A recorder never fails a
// run, so this counter is the only signal that the audit trail has gaps.
func (r *EventRecorder) Failures() int64 { return r.failed.Load() }

// emit creates one event, links it into the trace and appends it. It returns nil
// when the layer is filtered out, so callers can link a dropped event without a
// nil check of their own.
func (r *EventRecorder) emit(ctx context.Context, typ EventType, opts ...func(*Event)) *Event {
	layer := LayerOf(typ)
	if r.allow != nil && !r.allow(layer) {
		return nil
	}

	// The chain is advanced under the lock so two concurrent tool results still
	// form one sequence rather than a fork at the same parent.
	key := r.traceID + "|" + r.threadID
	r.mu.Lock()
	ev := NewEvent(typ, r.clock.Tick())
	ev.ParentID = r.last[key]
	r.last[key] = ev.ID
	r.mu.Unlock()

	ev.Layer = layer
	ev.TraceID = r.traceID
	ev.ThreadID = r.threadID
	for _, fn := range opts {
		fn(ev)
	}
	ev.Seal()

	if r.store != nil {
		if err := r.store.Append(ctx, ev); err != nil {
			r.failed.Add(1)
		}
	}
	return ev
}

// record appends an event and discards the handle.
func (r *EventRecorder) record(ctx context.Context, typ EventType, opts ...func(*Event)) {
	r.emit(ctx, typ, opts...)
}

// causedBy links a paired event to the event that opened it.
func causedBy(ev *Event, parent *Event) {
	if ev == nil || parent == nil {
		return
	}
	ev.CausedBy = append(ev.CausedBy, parent.ID)
}

// ---- Context-based recording (used by model/tool wrappers) ----

// ModelCall is one model invocation to record.
type ModelCall struct {
	Model      string
	Provider   string
	Messages   []any
	Content    string
	Tokens     TokenUsage
	DurationMs int64
	Cost       float64
	// PrefixHash identifies the stable prefix; CachedPromptTokens is what the
	// provider served from its prefix cache. Together they show whether prompt
	// caching is working, which is the difference between a cheap run and an
	// expensive one.
	PrefixHash         string
	CachedPromptTokens int
}

// RecordModelCall records an LLM model invocation with its result.
func (r *EventRecorder) RecordModelCall(ctx context.Context, call ModelCall) {
	start := r.emit(ctx, EventLLMCallStart, func(ev *Event) {
		ev.Deterministic = false
		ev.Metadata["model"] = call.Model
		ev.Metadata["provider"] = call.Provider
		ev.Metadata["prefix_hash"] = call.PrefixHash
	})
	r.emit(ctx, EventLLMCallEnd, func(ev *Event) {
		ev.Deterministic = false
		pl := LLMCallPayload{
			Model:              call.Model,
			Provider:           call.Provider,
			Messages:           call.Messages,
			Tokens:             call.Tokens,
			Content:            call.Content,
			DurationMs:         call.DurationMs,
			Cost:               call.Cost,
			PrefixHash:         call.PrefixHash,
			CachedPromptTokens: call.CachedPromptTokens,
		}
		ev.Payload, _ = json.Marshal(pl)
		causedBy(ev, start)
	})
}

// RecordLLMChunk records a single streaming chunk from an LLM call.
func (r *EventRecorder) RecordLLMChunk(ctx context.Context, model string, chunk string) {
	r.record(ctx, EventLLMCallChunk, func(ev *Event) {
		ev.Deterministic = false
		ev.Metadata["model"] = model
		ev.Metadata["chunk"] = chunk
	})
}

// RecordToolCall records a tool invocation with its result.
func (r *EventRecorder) RecordToolCall(ctx context.Context, toolName string, arguments map[string]any, result any, durationMs int64, retryCount int, errStr string) {
	start := r.emit(ctx, EventToolCallStart, func(ev *Event) {
		ev.Metadata["tool"] = toolName
	})
	r.emit(ctx, EventToolCallResult, func(ev *Event) {
		pl := ToolCallPayload{
			ToolName:   toolName,
			Arguments:  arguments,
			Result:     result,
			DurationMs: durationMs,
			RetryCount: retryCount,
			Error:      errStr,
		}
		ev.Payload, _ = json.Marshal(pl)
		if errStr != "" {
			ev.Deterministic = false
		}
		causedBy(ev, start)
	})
}

// RecordSubAgentCall records a sub-agent invocation with its result.
func (r *EventRecorder) RecordSubAgentCall(ctx context.Context, subAgentName string, input, output any, depth int, durationMs int64, errStr string) {
	start := r.emit(ctx, EventSubAgentCallStart, func(ev *Event) {
		pl := SubAgentCallPayload{
			SubAgentName: subAgentName,
			Input:        input,
			Depth:        depth,
		}
		ev.Payload, _ = json.Marshal(pl)
	})
	r.emit(ctx, EventSubAgentCallEnd, func(ev *Event) {
		pl := SubAgentCallPayload{
			SubAgentName: subAgentName,
			Output:       output,
			Depth:        depth,
			DurationMs:   durationMs,
			Error:        errStr,
		}
		ev.Payload, _ = json.Marshal(pl)
		if errStr != "" {
			ev.Deterministic = false
		}
		causedBy(ev, start)
	})
}

// RecordVerdict records one termination decision. Governance needs to know why a
// run stopped, not only that it stopped.
func (r *EventRecorder) RecordVerdict(ctx context.Context, action, reason string, round, evidence int) {
	r.record(ctx, EventVerdict, func(ev *Event) {
		pl := VerdictPayload{Action: action, Reason: reason, Round: round, Evidence: evidence}
		ev.Payload, _ = json.Marshal(pl)
	})
}

// RecordSessionValue records a session value change.
func (r *EventRecorder) RecordSessionValue(ctx context.Context, key string, value any) {
	r.record(ctx, EventSessionValueSet, func(ev *Event) {
		pl := SessionValuePayload{Key: key, Value: value}
		ev.Payload, _ = json.Marshal(pl)
	})
}

// RecordSessionTransfer records an agent transfer event.
func (r *EventRecorder) RecordSessionTransfer(ctx context.Context, fromAgent, toAgent, reason string, input any) {
	r.record(ctx, EventSessionTransfer, func(ev *Event) {
		pl := SessionTransferPayload{
			FromAgent: fromAgent,
			ToAgent:   toAgent,
			Reason:    reason,
			Input:     input,
		}
		ev.Payload, _ = json.Marshal(pl)
	})
}

// RecordStateWrite records a state transition.
func (r *EventRecorder) RecordStateWrite(ctx context.Context, channel string, oldValue, newValue any, reducer string) {
	r.record(ctx, EventStateWrite, func(ev *Event) {
		pl := StateTransitionPayload{
			Channel:  channel,
			OldValue: oldValue,
			NewValue: newValue,
			Reducer:  reducer,
		}
		ev.Payload, _ = json.Marshal(pl)
	})
}

// RecordMemoryWrite records a memory operation.
func (r *EventRecorder) RecordMemoryWrite(ctx context.Context, store, operation, key string, value any, score float64) {
	r.record(ctx, EventMemoryWrite, func(ev *Event) {
		pl := MemoryWritePayload{
			Store:     store,
			Operation: operation,
			Key:       key,
			Value:     value,
			Score:     score,
		}
		ev.Payload, _ = json.Marshal(pl)
	})
}

// RecordMemoryRead records a memory read operation.
func (r *EventRecorder) RecordMemoryRead(ctx context.Context, store, key string, score float64) {
	r.record(ctx, EventMemoryRead, func(ev *Event) {
		pl := MemoryWritePayload{
			Store: store,
			Key:   key,
			Score: score,
		}
		ev.Payload, _ = json.Marshal(pl)
	})
}

// RecordApproval records a human-in-the-loop decision: the request and the
// verdict are separate events, so a pending request stays visible when an
// operator never answers.
func (r *EventRecorder) RecordApproval(ctx context.Context, requestID, action string, context any, decision string, latencyMs int64) {
	start := r.emit(ctx, EventApprovalRequest, func(ev *Event) {
		pl := ApprovalPayload{
			RequestID: requestID,
			Action:    action,
			Context:   context,
		}
		ev.Payload, _ = json.Marshal(pl)
	})

	typ := EventApprovalGranted
	if decision != "" && decision != "granted" && decision != "allow" {
		typ = EventApprovalDenied
	}
	r.emit(ctx, typ, func(ev *Event) {
		pl := ApprovalPayload{
			RequestID: requestID,
			Action:    action,
			Decision:  decision,
			LatencyMs: latencyMs,
		}
		ev.Payload, _ = json.Marshal(pl)
		causedBy(ev, start)
	})
}

// RecordError records an execution error.
func (r *EventRecorder) RecordError(ctx context.Context, errMsg string) {
	r.record(ctx, EventError, func(ev *Event) {
		ev.Metadata["error"] = errMsg
	})
}

// RecordRetry records a retry event.
func (r *EventRecorder) RecordRetry(ctx context.Context, detail string) {
	r.record(ctx, EventRetry, func(ev *Event) {
		ev.Metadata["detail"] = detail
	})
}

// ---- GraphCallback implementation ----

// OnRunStart implements pregel.RunCallback.
func (r *EventRecorder) OnRunStart(ctx context.Context, graphName, threadID string) {
	r.record(ctx, EventGraphStart, func(ev *Event) {
		ev.Metadata["graph_name"] = graphName
	})
}

// OnRunEnd implements pregel.RunCallback.
func (r *EventRecorder) OnRunEnd(ctx context.Context, graphName, threadID string, err error) {
	r.record(ctx, EventGraphEnd, func(ev *Event) {
		ev.Metadata["graph_name"] = graphName
		if err != nil {
			ev.Metadata["error"] = err.Error()
		}
	})
}

// OnStepStart implements pregel.StepCallback.
func (r *EventRecorder) OnStepStart(ctx context.Context, step, taskCount int) {
	r.record(ctx, EventStepStart, func(ev *Event) {
		ev.Step = step
		ev.Metadata["task_count"] = taskCount
	})
}

// OnStepEnd implements pregel.StepCallback.
func (r *EventRecorder) OnStepEnd(ctx context.Context, step int, err error) {
	r.record(ctx, EventStepEnd, func(ev *Event) {
		ev.Step = step
		if err != nil {
			ev.Metadata["error"] = err.Error()
		}
	})
}

// OnNodeStart implements pregel.NodeCallback.
func (r *EventRecorder) OnNodeStart(ctx context.Context, nodeName string, step int) {
	r.record(ctx, EventNodeStart, func(ev *Event) {
		ev.Node = nodeName
		ev.Step = step
	})
}

// OnNodeEnd implements pregel.NodeCallback.
func (r *EventRecorder) OnNodeEnd(ctx context.Context, nodeName string, step int, output interface{}, err error) {
	r.record(ctx, EventNodeEnd, func(ev *Event) {
		ev.Node = nodeName
		ev.Step = step
		if err != nil {
			ev.Metadata["error"] = err.Error()
		}
	})
}

// OnCheckpointSave implements pregel.CheckpointCallback.
func (r *EventRecorder) OnCheckpointSave(ctx context.Context, threadID, checkpointID string, step int) {
	r.record(ctx, EventCheckpointCreated, func(ev *Event) {
		ev.ThreadID = threadID
		ev.Step = step
		ev.Metadata["checkpoint_id"] = checkpointID
	})
}

// OnCheckpointLoad implements pregel.CheckpointCallback.
func (r *EventRecorder) OnCheckpointLoad(ctx context.Context, threadID, checkpointID string, step int) {
	r.record(ctx, EventCheckpointRestored, func(ev *Event) {
		ev.ThreadID = threadID
		ev.Step = step
		ev.Metadata["checkpoint_id"] = checkpointID
	})
}

// OnCheckpointUpdate implements pregel.CheckpointCallback.
func (r *EventRecorder) OnCheckpointUpdate(ctx context.Context, threadID, asNode string) {
	r.record(ctx, EventStateWrite, func(ev *Event) {
		ev.ThreadID = threadID
		ev.Node = asNode
	})
}

// OnInterrupt implements pregel.InterruptCallback.
func (r *EventRecorder) OnInterrupt(ctx context.Context, nodeNames []string, step int) {
	r.record(ctx, EventInterrupt, func(ev *Event) {
		ev.Step = step
		ev.Metadata["interrupt_nodes"] = nodeNames
	})
}

// OnResume implements pregel.InterruptCallback.
func (r *EventRecorder) OnResume(ctx context.Context, threadID string) {
	r.record(ctx, EventResume, func(ev *Event) {
		ev.ThreadID = threadID
	})
}

// compile-time interface checks
var (
	_ pregel.GraphCallback = (*EventRecorder)(nil)
)
