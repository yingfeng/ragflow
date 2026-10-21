package loop

import (
	"testing"

	"ragflow/internal/harness/agent/model"
	"ragflow/internal/harness/policy"
	"ragflow/internal/harness/toolkit"
)

// noProgressPolicy keeps the termination heuristic out of the way: these tests
// check the per-call guards, not when a run should stop.
func noProgressPolicy() *policy.Engine {
	return policy.New(policy.Budget{}, policy.Completion{})
}

func TestRepeatedToolCallsAreBlocked(t *testing.T) {
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

	outcome, err := runner.Invoke(t.Context(), &Input{RunID: "run-repeat", UserMessage: "go"}, nil)
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if capability.invoked != maxRepeatedToolCalls {
		t.Fatalf("tool ran %d times, want %d", capability.invoked, maxRepeatedToolCalls)
	}
	if outcome.BlockedCalls != 2 {
		t.Fatalf("blocked calls = %d, want 2", outcome.BlockedCalls)
	}
}

func TestDifferentArgumentsAreNotBlocked(t *testing.T) {
	capability := &fakeCap{tool: "fake_tool", result: toolkit.Result{Content: "hit"}}
	mdl := model.NewFake(
		model.CallTool("c1", "fake_tool", `{"query":"a"}`),
		model.CallTool("c2", "fake_tool", `{"query":"b"}`),
		model.CallTool("c3", "fake_tool", `{"query":"c"}`),
		model.CallTool("c4", "fake_tool", `{"query":"d"}`),
		model.Reply("done"),
	)
	runner := buildRunner(t, mdl, capability, Config{Deps: Deps{Policy: noProgressPolicy()}})

	outcome, err := runner.Invoke(t.Context(), &Input{RunID: "run-distinct", UserMessage: "go"}, nil)
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if capability.invoked != 4 {
		t.Fatalf("tool ran %d times, want 4", capability.invoked)
	}
	if outcome.BlockedCalls != 0 {
		t.Fatalf("blocked calls = %d, want 0", outcome.BlockedCalls)
	}
}

func TestPrefixStaysStableAcrossRounds(t *testing.T) {
	capability := &fakeCap{tool: "fake_tool", result: toolkit.Result{Content: "hit"}}
	mdl := model.NewFake(
		model.CallTool("c1", "fake_tool", `{"query":"a"}`),
		model.CallTool("c2", "fake_tool", `{"query":"b"}`),
		model.Reply("done"),
	)
	runner := buildRunner(t, mdl, capability, Config{Deps: Deps{Policy: noProgressPolicy()}})

	outcome, err := runner.Invoke(t.Context(), &Input{RunID: "run-prefix", UserMessage: "go"}, nil)
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if outcome.PrefixRebuilds != 0 {
		t.Fatalf("prefix rebuilt %d times with a fixed tool surface", outcome.PrefixRebuilds)
	}
	if outcome.FinalState == nil || outcome.FinalState.PrefixHash == "" {
		t.Fatal("the prefix hash was never recorded")
	}
}

func TestPrefixFingerprintTracksSystemPromptAndTools(t *testing.T) {
	base := prefixFingerprint("system", []model.ToolSpec{{Name: "fs_read"}})
	if base != prefixFingerprint("system", []model.ToolSpec{{Name: "fs_read"}}) {
		t.Fatal("the fingerprint is not stable for identical inputs")
	}
	if base == prefixFingerprint("system changed", []model.ToolSpec{{Name: "fs_read"}}) {
		t.Fatal("the fingerprint ignored a system prompt change, which invalidates the provider cache")
	}
	if base == prefixFingerprint("system", []model.ToolSpec{{Name: "fs_read"}, {Name: "fs_write"}}) {
		t.Fatal("the fingerprint ignored a tool surface change")
	}
}
