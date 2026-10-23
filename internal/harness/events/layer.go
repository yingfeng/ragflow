package events

// Layer is the data class an event belongs to. The classes mirror the reference
// runtime's state layers, so retention, export and deletion run per class
// instead of per event type: control state is kept with the run, trace facts can
// be sampled, diagnostics are off by default.
type Layer string

const (
	// LayerControl is durable runtime control state: graph, step and node
	// lifecycle, checkpoints, interrupts, forks.
	LayerControl Layer = "C0"
	// LayerTranscript is user-visible conversation state.
	LayerTranscript Layer = "C1"
	// LayerAudit holds the durable facts an operator replays or audits: tool
	// calls, approvals, delegation, errors, memory writes.
	LayerAudit Layer = "C2"
	// LayerTrace is the structured execution trace: model rounds, retries,
	// state transitions.
	LayerTrace Layer = "C3"
	// LayerDebug holds short-lived raw captures. Off unless asked for.
	LayerDebug Layer = "C4"
	// LayerLearning holds consent-gated derived data. The runtime never emits
	// it; RSI derives artifacts separately.
	LayerLearning Layer = "C5"
)

// LayerOf reports the data class of an event type. Unknown types are treated as
// trace facts, which is the least destructive default: they are kept, but they
// are not mistaken for durable audit facts.
func LayerOf(typ EventType) Layer {
	switch typ {
	case EventGraphStart, EventGraphEnd,
		EventStepStart, EventStepEnd,
		EventNodeStart, EventNodeEnd,
		EventCheckpointCreated, EventCheckpointRestored,
		EventInterrupt, EventResume, EventFork:
		return LayerControl

	case EventSessionValueSet, EventSessionTransfer,
		EventInteractionRequest, EventInteractionResolved:
		return LayerTranscript

	case EventToolCallStart, EventToolCallResult, EventToolCallError,
		EventApprovalRequest, EventApprovalGranted, EventApprovalDenied,
		EventGrantPersisted,
		EventSubAgentCallStart, EventSubAgentCallEnd,
		EventError, EventMemoryWrite:
		return LayerAudit

	case EventLLMCallStart, EventLLMCallEnd,
		EventRetry, EventStateRead, EventStateWrite,
		EventMemoryRead, EventVerdict, EventCompaction:
		return LayerTrace

	case EventLLMCallChunk:
		return LayerDebug
	}
	return LayerTrace
}

// LayerFilter decides whether a layer is recorded. Nil records every layer.
type LayerFilter func(Layer) bool

// DefaultLayerFilter records the runtime facts and drops diagnostics. Raw model
// chunks are reproducible from the model call they belong to, and keeping them
// turns the audit log into a transcript copy.
func DefaultLayerFilter(layer Layer) bool {
	switch layer {
	case LayerDebug, LayerLearning:
		return false
	}
	return true
}
