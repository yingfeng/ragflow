package loop

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"ragflow/internal/harness/agent/definition"
	"ragflow/internal/harness/agent/model"
	"ragflow/internal/harness/policy"
	"ragflow/internal/harness/toolkit"
	"ragflow/internal/harness/toolkit/builtin"
)

type fakeCap struct {
	tool     string
	result   toolkit.Result
	err      error
	invoked  int
	failOnce bool
}

func (f *fakeCap) Name() string { return "fake" }

func (f *fakeCap) Tools() []model.ToolSpec {
	return []model.ToolSpec{{
		Name:       f.tool,
		Parameters: toolkit.ObjectSchema(map[string]any{"query": map[string]any{"type": "string"}}),
	}}
}

func (f *fakeCap) Invoke(context.Context, toolkit.Call) (toolkit.Result, error) {
	f.invoked++
	if f.err != nil {
		return toolkit.Result{}, f.err
	}
	return f.result, nil
}

func buildRunner(t *testing.T, mdl model.Model, capability toolkit.Capability, cfg Config) *Runner {
	t.Helper()
	registry := toolkit.NewRegistry()
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
		result: toolkit.Result{
			Content:  "[1] chunk-a\ncontent",
			Evidence: []toolkit.Evidence{{Source: "rag", Ref: "chunk-a"}},
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
	capability := &fakeCap{tool: "rag_search", result: toolkit.Result{Content: "nothing"}}
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
	runner := buildRunner(t, mdl, &fakeCap{tool: "rag_search", result: toolkit.Result{Content: "ok"}}, Config{})

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

// A round that only calls tools leaves no assistant text behind. When the
// progress policy stops such a run, the collected evidence must still become an
// answer instead of an empty message.
func TestLoop_StagnantRunStillAnswers(t *testing.T) {
	capability := &fakeCap{
		tool: "rag_search",
		result: toolkit.Result{
			Content:  "<chunk chunk_id=\"c1\" doc_id=\"d1\"/>",
			Evidence: []toolkit.Evidence{{Source: "d1", Ref: "c1"}},
		},
	}
	mdl := model.NewFake(
		model.CallTool("c1", "rag_search", `{"query":"a"}`),
		model.CallTool("c2", "rag_search", `{"query":"b"}`),
		model.CallTool("c3", "rag_search", `{"query":"c"}`),
		model.CallTool("c4", "rag_search", `{"query":"d"}`),
		model.Reply("关羽斩了颜良、文丑等六将"),
	)
	runner := buildRunner(t, mdl, capability, Config{})

	outcome, err := runner.Invoke(context.Background(), &Input{RunID: "run-6", UserMessage: "关羽杀了谁"}, nil)
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if outcome.Text != "关羽斩了颜良、文丑等六将" {
		t.Fatalf("text = %q, want the synthesized answer", outcome.Text)
	}
	if !strings.Contains(outcome.Stopped, "no new evidence") {
		t.Fatalf("stopped = %q, want the stagnation verdict", outcome.Stopped)
	}
	requests := mdl.Requests()
	if len(requests) != 5 {
		t.Fatalf("model calls = %d, want 4 rounds + 1 answer", len(requests))
	}
	if len(requests[4].Tools) != 0 {
		t.Errorf("answer request carried %d tools, want none", len(requests[4].Tools))
	}
	if len(outcome.Errors) != 0 {
		t.Errorf("errors = %+v", outcome.Errors)
	}
}

// freshOutputCap returns different content on every call, which is what a
// read-only tool looks like when it is making progress. Evidence is empty on
// purpose: a run that keeps finding new material must not be stopped for
// lacking citations.
type freshOutputCap struct{ calls int }

func (f *freshOutputCap) Name() string { return "fresh" }

func (f *freshOutputCap) Tools() []model.ToolSpec {
	return []model.ToolSpec{{Name: "read_more", Parameters: toolkit.ObjectSchema(map[string]any{"page": map[string]any{"type": "integer"}}, "page")}}
}

func (f *freshOutputCap) Invoke(_ context.Context, call toolkit.Call) (toolkit.Result, error) {
	f.calls++
	return toolkit.Result{Content: fmt.Sprintf("page %s content: 关羽斩了%s", call.Arguments, strings.Repeat("x", 32))}, nil
}

// A plan is state, not transcript: the loop must keep it visible in the next
// round's prompt even though compaction may drop the call that created it.
func TestLoop_PlanSurvivesIntoNextPrompt(t *testing.T) {
	builtinCap, err := builtin.New(context.Background())
	if err != nil {
		t.Fatalf("builtin: %v", err)
	}
	mdl := model.NewFake(
		model.CallTool("c1", "todo_write", `{"task":"count","steps":[{"id":"1","description":"检索关羽斩将","status":"in_progress"},{"id":"2","description":"核对五关六将名单","status":"pending"}]}`),
		model.Reply("共 6 人"),
	)
	runner := buildRunner(t, mdl, builtinCap, Config{})

	outcome, err := runner.Invoke(context.Background(), &Input{RunID: "run-8", UserMessage: "关羽杀了谁"}, nil)
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if outcome.Text != "共 6 人" {
		t.Fatalf("text = %q", outcome.Text)
	}
	if len(outcome.FinalState.Plan) != 2 {
		t.Fatalf("plan = %+v, want the two steps the tool published", outcome.FinalState.Plan)
	}

	requests := mdl.Requests()
	if len(requests) != 2 {
		t.Fatalf("model calls = %d, want 2", len(requests))
	}
	second := ""
	for _, message := range requests[1].Messages {
		second += message.Content + "\n"
	}
	if !strings.Contains(second, "Plan state") || !strings.Contains(second, "核对五关六将名单") {
		t.Fatalf("second prompt has no plan section:\n%s", second)
	}
	if !strings.Contains(second, "2 step(s) still open") {
		t.Errorf("second prompt does not report the open steps:\n%s", second)
	}
}

func TestLoop_NewOutputCountsAsProgress(t *testing.T) {
	capability := &freshOutputCap{}
	mdl := model.NewFake(
		model.CallTool("c1", "read_more", `{"page":1}`),
		model.CallTool("c2", "read_more", `{"page":2}`),
		model.CallTool("c3", "read_more", `{"page":3}`),
		model.CallTool("c4", "read_more", `{"page":4}`),
		model.Reply("共 4 页，答案是 6 人"),
	)
	runner := buildRunner(t, mdl, capability, Config{})

	outcome, err := runner.Invoke(context.Background(), &Input{RunID: "run-7", UserMessage: "读完全部页面"}, nil)
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if outcome.Text != "共 4 页，答案是 6 人" {
		t.Fatalf("text = %q, want the model's own answer", outcome.Text)
	}
	if outcome.Rounds != 5 {
		t.Fatalf("rounds = %d, want 5 (the run must not be stopped as stagnant)", outcome.Rounds)
	}
	if strings.Contains(outcome.Stopped, "no new evidence") {
		t.Fatalf("stopped = %q, a productive run was called stagnant", outcome.Stopped)
	}
	if capability.calls != 4 {
		t.Fatalf("tool calls = %d, want 4", capability.calls)
	}
}

// A run that died between dispatch and result cannot know whether a
// state-changing call took effect. Replaying it would duplicate the effect, so
// the resumed batch reports the call as unresolved and lets the model decide.
func TestLoop_ResumeDoesNotReplayMutatingCall(t *testing.T) {
	capability := &fakeCap{tool: "write_thing", result: toolkit.Result{Content: "written"}}
	mdl := model.NewFake(
		model.CallTool("c1", "write_thing", `{"query":"x"}`),
		model.Reply("done"),
	)
	runner := buildRunner(t, mdl, capability, Config{})

	// A resumed context is what Runner.Resume hands the graph after a restart.
	outcome, err := runner.Invoke(markResumed(context.Background()), &Input{RunID: "run-resume", UserMessage: "write it"}, nil)
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if capability.invoked != 0 {
		t.Fatalf("a state-changing call was replayed after resume (%d calls)", capability.invoked)
	}
	noticed := false
	for _, message := range outcome.FinalState.History {
		if message.Role == model.RoleTool && strings.Contains(message.Content, "may already have taken effect") {
			noticed = true
		}
	}
	if !noticed {
		t.Fatal("the model was never told the call may have taken effect")
	}
}
