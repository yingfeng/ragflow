package loop

import (
	"context"
	"strings"
	"testing"

	"ragflow/internal/harness/agent/definition"
	"ragflow/internal/harness/agent/model"
	"ragflow/internal/harness/caps"
	"ragflow/internal/harness/policy"
)

type fakeCap struct {
	tool     string
	result   caps.Result
	err      error
	invoked  int
	failOnce bool
}

func (f *fakeCap) Name() string { return "fake" }

func (f *fakeCap) Tools() []model.ToolSpec {
	return []model.ToolSpec{{
		Name:       f.tool,
		Parameters: caps.ObjectSchema(map[string]any{"query": map[string]any{"type": "string"}}),
	}}
}

func (f *fakeCap) Invoke(context.Context, caps.Call) (caps.Result, error) {
	f.invoked++
	if f.err != nil {
		return caps.Result{}, f.err
	}
	return f.result, nil
}

func buildRunner(t *testing.T, mdl model.Model, capability caps.Capability, cfg Config) *Runner {
	t.Helper()
	registry := caps.NewRegistry()
	if capability != nil {
		if err := registry.Register(capability); err != nil {
			t.Fatalf("register: %v", err)
		}
	}
	cfg.Deps.Model = mdl
	cfg.Deps.Registry = registry
	cfg.Deps.SystemPrompt = "you are a test agent"
	runner, err := Build(cfg)
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	return runner
}

func TestLoop_FinalAnswerStops(t *testing.T) {
	mdl := model.NewFake(model.Reply("hello there"))
	runner := buildRunner(t, mdl, nil, Config{})

	outcome, err := runner.Invoke(context.Background(), &Input{RunID: "run-1", UserMessage: "hi"}, nil)
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if outcome.Text != "hello there" {
		t.Fatalf("text = %q", outcome.Text)
	}
	if outcome.Rounds != 1 {
		t.Fatalf("rounds = %d, want 1", outcome.Rounds)
	}
	if outcome.Verdict.Action != policy.ActionFinish {
		t.Fatalf("verdict = %s (%s)", outcome.Verdict.Action, outcome.Verdict.Reason)
	}
	if calls := mdl.Calls(); calls != 1 {
		t.Fatalf("model calls = %d, want 1 (loop did not terminate)", calls)
	}
}

func TestLoop_ToolRoundTrip(t *testing.T) {
	capability := &fakeCap{
		tool: "rag_search",
		result: caps.Result{
			Content:  "[1] chunk-a\ncontent",
			Evidence: []caps.Evidence{{Source: "rag", Ref: "chunk-a"}},
		},
	}
	mdl := model.NewFake(
		model.CallTool("call-1", "rag_search", `{"query":"x"}`),
		model.Reply("answer with [1]"),
	)
	runner := buildRunner(t, mdl, capability, Config{})

	outcome, err := runner.Invoke(context.Background(), &Input{RunID: "run-2", UserMessage: "q"}, nil)
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if outcome.Text != "answer with [1]" {
		t.Fatalf("text = %q", outcome.Text)
	}
	if outcome.ToolCalls != 1 || capability.invoked != 1 {
		t.Fatalf("tool calls = %d invoked = %d", outcome.ToolCalls, capability.invoked)
	}
	if len(outcome.Evidence) != 1 || outcome.Evidence[0].Ref != "chunk-a" {
		t.Fatalf("evidence = %+v", outcome.Evidence)
	}
	if outcome.Rounds != 2 {
		t.Fatalf("rounds = %d, want 2", outcome.Rounds)
	}
}

func TestLoop_RoundBudgetStops(t *testing.T) {
	capability := &fakeCap{tool: "rag_search", result: caps.Result{Content: "nothing"}}
	mdl := model.NewFake(
		model.CallTool("c1", "rag_search", `{"query":"a"}`),
		model.CallTool("c2", "rag_search", `{"query":"b"}`),
		model.CallTool("c3", "rag_search", `{"query":"c"}`),
		model.CallTool("c4", "rag_search", `{"query":"d"}`),
	)
	runner := buildRunner(t, mdl, capability, Config{PolicyConfig: definition.PolicyConfig{MaxRounds: 2}})

	outcome, err := runner.Invoke(context.Background(), &Input{RunID: "run-3", UserMessage: "q"}, nil)
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if outcome.Verdict.Action != policy.ActionFinish {
		t.Fatalf("verdict = %s", outcome.Verdict.Action)
	}
	if !strings.Contains(outcome.Stopped, "round budget") {
		t.Fatalf("stopped = %q", outcome.Stopped)
	}
	if outcome.Rounds != 2 {
		t.Fatalf("rounds = %d, want 2", outcome.Rounds)
	}
}

func TestLoop_ScopeViolationBlocks(t *testing.T) {
	capability := &fakeCap{
		tool: "rag_search",
		err:  errScope{},
	}
	mdl := model.NewFake(
		model.CallTool("c1", "rag_search", `{"query":"a"}`),
		model.Reply("should not be reached"),
	)
	runner := buildRunner(t, mdl, capability, Config{})

	outcome, err := runner.Invoke(context.Background(), &Input{RunID: "run-4", UserMessage: "q"}, nil)
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if outcome.Verdict.Action != policy.ActionBlock {
		t.Fatalf("verdict = %s (%s)", outcome.Verdict.Action, outcome.Verdict.Reason)
	}
}

type errScope struct{}

func (errScope) Error() string { return `dataset_id "kb9" is outside the run scope` }

func TestLoop_UnknownToolIsRecoverable(t *testing.T) {
	mdl := model.NewFake(
		model.CallTool("c1", "nope", `{}`),
		model.Reply("recovered"),
	)
	runner := buildRunner(t, mdl, &fakeCap{tool: "rag_search", result: caps.Result{Content: "ok"}}, Config{})

	outcome, err := runner.Invoke(context.Background(), &Input{RunID: "run-5", UserMessage: "q"}, nil)
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if outcome.Text != "recovered" {
		t.Fatalf("text = %q", outcome.Text)
	}
	if len(outcome.Errors) != 1 || !strings.Contains(outcome.Errors[0].Message, "unknown tool") {
		t.Fatalf("errors = %+v", outcome.Errors)
	}
}
