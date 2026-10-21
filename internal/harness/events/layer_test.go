package events

import (
	"context"
	"errors"
	"testing"
)

func TestLayerOfCoversEveryEventType(t *testing.T) {
	// Every emitted type must land in a class; an unmapped type would silently
	// inherit the trace default and escape control-state retention.
	cases := map[EventType]Layer{
		EventGraphStart:        LayerControl,
		EventCheckpointCreated: LayerControl,
		EventInterrupt:         LayerControl,
		EventSessionValueSet:   LayerTranscript,
		EventSessionTransfer:   LayerTranscript,
		EventToolCallStart:     LayerAudit,
		EventToolCallResult:    LayerAudit,
		EventApprovalRequest:   LayerAudit,
		EventApprovalGranted:   LayerAudit,
		EventApprovalDenied:    LayerAudit,
		EventSubAgentCallStart: LayerAudit,
		EventError:             LayerAudit,
		EventMemoryWrite:       LayerAudit,
		EventLLMCallStart:      LayerTrace,
		EventLLMCallEnd:        LayerTrace,
		EventRetry:             LayerTrace,
		EventStateWrite:        LayerTrace,
		EventVerdict:           LayerTrace,
		EventLLMCallChunk:      LayerDebug,
	}
	for typ, want := range cases {
		if got := LayerOf(typ); got != want {
			t.Errorf("LayerOf(%s) = %s, want %s", typ, got, want)
		}
	}
}

func TestDefaultLayerFilterDropsDiagnostics(t *testing.T) {
	for _, layer := range []Layer{LayerControl, LayerTranscript, LayerAudit, LayerTrace} {
		if !DefaultLayerFilter(layer) {
			t.Errorf("DefaultLayerFilter(%s) = false, want true", layer)
		}
	}
	for _, layer := range []Layer{LayerDebug, LayerLearning} {
		if DefaultLayerFilter(layer) {
			t.Errorf("DefaultLayerFilter(%s) = true, want false", layer)
		}
	}
}

func TestRecorderAssignsLayerAndCausalChain(t *testing.T) {
	ctx := t.Context()
	store := NewMemoryEventStore()
	recorder := NewEventRecorder(store, WithTraceID("trace-1"), WithThreadID("thread-1"))

	recorder.RecordToolCall(ctx, "fs_cat", map[string]any{"path": "/a"}, "body", 12, 0, "")
	recorder.RecordModelCall(ctx, ModelCall{Model: "gpt", Provider: "openai", Content: "answer", Tokens: TokenUsage{TotalTokens: 10}, DurationMs: 30})

	events := collect(ctx, t, store, EventFilter{TraceID: "trace-1"})
	if len(events) != 4 {
		t.Fatalf("expected 4 events, got %d", len(events))
	}

	// Every emitted event carries its class.
	for _, event := range events {
		if event.Layer == "" {
			t.Errorf("%s has no layer", event.Type)
		}
	}

	// ParentID forms one sequence: each event points at its predecessor.
	for index := 1; index < len(events); index++ {
		if events[index].ParentID != events[index-1].ID {
			t.Errorf("%s parent = %s, want %s", events[index].Type, events[index].ParentID, events[index-1].ID)
		}
	}

	// The result of a call also names the event that opened it.
	if len(events[1].CausedBy) != 1 || events[1].CausedBy[0] != events[0].ID {
		t.Errorf("tool result caused by %v, want [%s]", events[1].CausedBy, events[0].ID)
	}
	if len(events[3].CausedBy) != 1 || events[3].CausedBy[0] != events[2].ID {
		t.Errorf("model result caused by %v, want [%s]", events[3].CausedBy, events[2].ID)
	}
}

func TestRecorderLayerFilterSkipsDebug(t *testing.T) {
	ctx := t.Context()
	store := NewMemoryEventStore()
	recorder := NewEventRecorder(store, WithTraceID("trace-2"), WithLayers(DefaultLayerFilter))

	recorder.RecordLLMChunk(ctx, "gpt", "par")
	recorder.RecordToolCall(ctx, "fs_ls", nil, "[]", 1, 0, "")

	events := collect(ctx, t, store, EventFilter{TraceID: "trace-2"})
	if len(events) != 2 {
		t.Fatalf("expected 2 events (chunk filtered), got %d", len(events))
	}
	for _, event := range events {
		if event.Layer == LayerDebug {
			t.Fatalf("debug event %s was recorded despite the filter", event.Type)
		}
	}

	// A filtered event must not become the parent of the next recorded one:
	// the chain has to skip it, not dangle.
	if events[1].ParentID != events[0].ID {
		t.Errorf("parent after a filtered event = %s, want %s", events[1].ParentID, events[0].ID)
	}
}

func TestRecorderApprovalRecordsDecision(t *testing.T) {
	ctx := t.Context()
	store := NewMemoryEventStore()
	recorder := NewEventRecorder(store, WithTraceID("trace-3"))

	recorder.RecordApproval(ctx, "req-1", "exec_shell", map[string]any{"cmd": "ls"}, "denied", 1200)

	events := collect(ctx, t, store, EventFilter{TraceID: "trace-3"})
	if len(events) != 2 {
		t.Fatalf("expected request and decision events, got %d", len(events))
	}
	if events[0].Type != EventApprovalRequest {
		t.Errorf("first event = %s, want %s", events[0].Type, EventApprovalRequest)
	}
	if events[1].Type != EventApprovalDenied {
		t.Errorf("second event = %s, want %s", events[1].Type, EventApprovalDenied)
	}
}

func TestRecorderCountsAppendFailures(t *testing.T) {
	ctx := t.Context()
	recorder := NewEventRecorder(failingLog{}, WithTraceID("trace-4"))

	recorder.RecordError(ctx, "boom")

	if got := recorder.Failures(); got != 1 {
		t.Fatalf("Failures() = %d, want 1", got)
	}
}

func TestEventFilterMatchesLayers(t *testing.T) {
	event := &Event{Type: EventToolCallStart, Layer: LayerAudit}
	if !(EventFilter{Layers: []Layer{LayerAudit}}).Matches(event) {
		t.Error("audit filter rejected an audit event")
	}
	if (EventFilter{Layers: []Layer{LayerTrace}}).Matches(event) {
		t.Error("trace filter accepted an audit event")
	}

	// Events written before the layer field existed are classified on read.
	legacy := &Event{Type: EventToolCallStart}
	if !(EventFilter{Layers: []Layer{LayerAudit}}).Matches(legacy) {
		t.Error("audit filter rejected a legacy event with no layer set")
	}
}

type failingLog struct{}

func (failingLog) Append(context.Context, ...*Event) error { return errors.New("append failed") }
func (failingLog) Stream(context.Context, EventFilter) EventIterator {
	return &sliceIterator{}
}
func (failingLog) Get(context.Context, EventID) (*Event, error) { return nil, nil }
func (failingLog) Range(context.Context, uint64, uint64, EventFilter) ([]*Event, error) {
	return nil, nil
}
func (failingLog) Seek(context.Context, uint64) (EventIterator, error) { return &sliceIterator{}, nil }
func (failingLog) Length(context.Context) (uint64, error)              { return 0, nil }

func collect(ctx context.Context, t *testing.T, store EventLog, filter EventFilter) []*Event {
	t.Helper()
	iterator := store.Stream(ctx, filter)
	collected := make([]*Event, 0, 8)
	for {
		event, ok := iterator.Next(ctx)
		if !ok {
			break
		}
		collected = append(collected, event)
	}
	if err := iterator.Close(); err != nil {
		t.Fatalf("close iterator: %v", err)
	}
	return collected
}
