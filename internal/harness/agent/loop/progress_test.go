package loop

import (
	"errors"
	"testing"

	"ragflow/internal/harness/agent/model"
	"ragflow/internal/harness/toolkit"
)

// TestProgressReportsBlockedCallLifecycle pins the rule that every pending call
// reports a lifecycle. A call the loop refuses before it starts is still a call
// the model asked for and the user is waiting on: leaving it out of the stream
// keeps a spinner alive for work that was never dispatched.
func TestProgressReportsBlockedCallLifecycle(t *testing.T) {
	capability := &fakeCap{tool: "fake_tool", result: toolkit.Result{Content: "same"}}
	mdl := model.NewFake(
		model.CallTool("c1", "fake_tool", `{"query":"x"}`),
		model.CallTool("c2", "fake_tool", `{"query":"x"}`),
		model.CallTool("c3", "fake_tool", `{"query":"x"}`),
		model.CallTool("c4", "fake_tool", `{"query":"x"}`),
		model.CallTool("c5", "fake_tool", `{"query":"x"}`),
		model.Reply("done"),
	)
	runner := buildRunner(t, mdl, capability, Config{Deps: Deps{Policy: noProgressPolicy()}})

	var events []Progress
	ctx := WithProgressSink(t.Context(), func(progress Progress) { events = append(events, progress) })
	outcome, err := runner.Invoke(ctx, &Input{RunID: "run-blocked-progress", UserMessage: "go"}, nil)
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if outcome.BlockedCalls != 2 {
		t.Fatalf("blocked calls = %d, want 2", outcome.BlockedCalls)
	}

	if len(events) != 10 {
		t.Fatalf("progress events = %d (%+v), want one start and one end per call", len(events), events)
	}
	failed := 0
	for index := 0; index < len(events); index += 2 {
		if events[index].Phase != ProgressStart || events[index+1].Phase != ProgressEnd {
			t.Fatalf("call %d phases = %s, %s", index/2, events[index].Phase, events[index+1].Phase)
		}
		if events[index].Tool != "fake_tool" {
			t.Fatalf("call %d tool = %q", index/2, events[index].Tool)
		}
		if events[index+1].IsError {
			failed++
		}
	}
	if failed != 2 {
		t.Fatalf("error progress events = %d, want the two blocked calls", failed)
	}
}

// TestProgressReportsToolLevelError covers the other half of the same rule: a
// tool that fails by returning an error result (nil error from Invoke) must not
// report success, because the model already sees a <tool_error> in the
// transcript.
func TestProgressReportsToolLevelError(t *testing.T) {
	capability := &fakeCap{tool: "fake_tool", result: toolkit.ErrorResult(errors.New("boom"))}
	mdl := model.NewFake(
		model.CallTool("c1", "fake_tool", `{"query":"x"}`),
		model.Reply("done"),
	)
	runner := buildRunner(t, mdl, capability, Config{Deps: Deps{Policy: noProgressPolicy()}})

	var events []Progress
	ctx := WithProgressSink(t.Context(), func(progress Progress) { events = append(events, progress) })
	if _, err := runner.Invoke(ctx, &Input{RunID: "run-tool-error", UserMessage: "go"}, nil); err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if len(events) != 2 {
		t.Fatalf("progress events = %+v", events)
	}
	if !events[1].IsError {
		t.Fatalf("a tool-level failure reported success: %+v", events[1])
	}
}
