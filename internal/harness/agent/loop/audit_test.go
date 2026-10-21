package loop

import (
	"context"
	"testing"

	"ragflow/internal/harness/agent/model"
	"ragflow/internal/harness/events"
	"ragflow/internal/harness/toolkit"
)

// TestLoopRecordsAuditEvents drives one tool round and checks that the audit log
// ends up with the model rounds, the tool call and a termination verdict — the
// three facts an operator needs to explain a run without reading the transcript.
func TestLoopRecordsAuditEvents(t *testing.T) {
	mdl := model.NewFake(
		model.CallTool("call-1", "fake_tool", `{"query":"x"}`),
		model.Reply("done"),
	)
	log := events.NewMemoryEventStore()
	recorder := events.NewEventRecorder(log, events.WithTraceID("run-audit"), events.WithThreadID("run-audit"))
	ctx := events.ContextWithRecorder(context.Background(), recorder)

	runner := buildRunner(t, mdl, &fakeCap{
		tool:   "fake_tool",
		result: toolkit.Result{Content: "hit"},
	}, Config{})

	outcome, err := runner.Invoke(ctx, &Input{RunID: "run-audit", UserMessage: "hi"}, nil)
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if outcome.Text != "done" {
		t.Fatalf("text = %q, want done", outcome.Text)
	}

	items, err := log.Range(ctx, 0, 0, events.EventFilter{TraceID: "run-audit"})
	if err != nil {
		t.Fatalf("range: %v", err)
	}
	counts := make(map[events.EventType]int)
	for _, item := range items {
		counts[item.Type]++
		if item.Layer == "" {
			t.Errorf("%s has no layer", item.Type)
		}
	}

	if counts[events.EventLLMCallEnd] != 2 {
		t.Errorf("model rounds = %d, want 2: %v", counts[events.EventLLMCallEnd], counts)
	}
	if counts[events.EventToolCallStart] != 1 || counts[events.EventToolCallResult] != 1 {
		t.Errorf("tool events = %d/%d, want 1/1: %v", counts[events.EventToolCallStart], counts[events.EventToolCallResult], counts)
	}
	if counts[events.EventVerdict] == 0 {
		t.Errorf("no verdict recorded: %v", counts)
	}
}

// TestLoopAuditSurvivesToolFailure keeps the failure in the log: a run that
// recovered is exactly the case an audit has to be able to explain.
func TestLoopAuditSurvivesToolFailure(t *testing.T) {
	mdl := model.NewFake(
		model.CallTool("call-1", "fake_tool", `{}`),
		model.Reply("recovered"),
	)
	log := events.NewMemoryEventStore()
	recorder := events.NewEventRecorder(log, events.WithTraceID("run-fail"))
	ctx := events.ContextWithRecorder(context.Background(), recorder)

	runner := buildRunner(t, mdl, &fakeCap{
		tool: "fake_tool",
		err:  context.DeadlineExceeded,
	}, Config{})

	if _, err := runner.Invoke(ctx, &Input{RunID: "run-fail", UserMessage: "hi"}, nil); err != nil {
		t.Fatalf("invoke: %v", err)
	}

	results, err := log.Range(ctx, 0, 0, events.EventFilter{TraceID: "run-fail", Types: []events.EventType{events.EventToolCallResult}})
	if err != nil {
		t.Fatalf("range: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("tool results = %d, want 1", len(results))
	}
	if results[0].Layer != events.LayerAudit {
		t.Errorf("tool result layer = %s, want %s", results[0].Layer, events.LayerAudit)
	}
}
