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
	// required names the tools this capability gates on; nil disables the gate.
	required []string
}

func (f *fakeCap) Name() string { return "fake" }

// RequiredTools implements toolkit.RequiredToolList when the test asks for a gate.
func (f *fakeCap) RequiredTools() []string { return f.required }

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
	// The progress judgement spans a window, so the stall needs three flat
	// rounds behind it (see evidenceWindowRounds) before it is reported.
	mdl := model.NewFake(
		model.CallTool("c1", "rag_search", `{"query":"a"}`),
		model.CallTool("c2", "rag_search", `{"query":"b"}`),
		model.CallTool("c3", "rag_search", `{"query":"c"}`),
		model.CallTool("c4", "rag_search", `{"query":"d"}`),
		model.CallTool("c5", "rag_search", `{"query":"e"}`),
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
	// The stall is described to the run instead of ending it: the round that
	// added no evidence is told so, and keeps the turn.
	if outcome.FinalState.Nudges != 1 {
		t.Fatalf("nudges = %d, want the stall reported once", outcome.FinalState.Nudges)
	}
	requests := mdl.Requests()
	if len(requests) != 6 {
		t.Fatalf("model calls = %d, want 5 rounds + 1 answer", len(requests))
	}
	if !requestMentions(requests[5], "retrieved nothing new") {
		t.Fatalf("the nudge never reached the model: %+v", requests[5].Messages)
	}
	// The post-nudge round is an ordinary round: it still holds the tools, and
	// the answer it wrote ended the run without a separate answer call.
	if len(requests[5].Tools) == 0 {
		t.Error("the post-nudge round was stripped of its tools")
	}
	if len(outcome.Errors) != 0 {
		t.Errorf("errors = %+v", outcome.Errors)
	}
}

// requestMentions reports whether any message in the request contains text.
func requestMentions(request model.Request, text string) bool {
	for _, message := range request.Messages {
		if strings.Contains(message.Content, text) {
			return true
		}
	}
	return false
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
		// The open steps keep the run alive for one grace round, so a second
		// reply is what finally ends it (see TestLoop_OpenPlanStopsAfterOneGraceRound).
		model.Reply("共 6 人"),
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
	if len(requests) != 3 {
		t.Fatalf("model calls = %d, want 3", len(requests))
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

// An open plan must not hold the run open. A model that cannot tell what it has
// already covered keeps its steps open forever, and while the plan was a gate
// the run never reached the answer call at all. It finishes now; the answer call
// rewrites the reply once with the coverage note in front of it, so the uncovered
// part is named instead of the run being held hostage.
func TestLoop_OpenPlanStillAnswersWithACoverageNote(t *testing.T) {
	builtinCap, err := builtin.New(context.Background())
	if err != nil {
		t.Fatalf("builtin: %v", err)
	}
	mdl := model.NewFake(
		model.CallTool("c1", "todo_write", `{"task":"count","steps":[{"id":"1","description":"检索关羽斩将","status":"completed"},{"id":"2","description":"核对五关六将名单","status":"pending"}]}`),
		model.Reply("共 12 人"),
		model.Reply("共 12 人（第 41-80 回未覆盖）"),
	)
	runner := buildRunner(t, mdl, builtinCap, Config{})

	outcome, err := runner.Invoke(context.Background(), &Input{RunID: "run-plan", UserMessage: "关羽杀了多少人"}, nil)
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if outcome.Text != "共 12 人（第 41-80 回未覆盖）" {
		t.Fatalf("text = %q, want the rewrite that names the uncovered range", outcome.Text)
	}
	if calls := mdl.Calls(); calls != 3 {
		t.Fatalf("model calls = %d, want 3 (plan, answer, rewrite)", calls)
	}
	if outcome.Verdict.Action != policy.ActionFinish {
		t.Fatalf("verdict = %s (%s)", outcome.Verdict.Action, outcome.Verdict.Reason)
	}
	if last := mdl.Requests()[2]; !requestMentions(last, "plan step(s) are still open") {
		t.Fatalf("the rewrite call was not told what is left open")
	}
	if !strings.Contains(outcome.Stopped, "1/2 plan steps open") {
		t.Fatalf("stopped = %q, want the coverage recorded", outcome.Stopped)
	}
}

// A new query over the same chunks is not progress. Retrieval tools publish
// chunk ids as evidence; when that pool stops growing the run is done searching
// even though every round returns fresh text with a different snippet.
type sameChunkCap struct{ calls int }

func (c *sameChunkCap) Name() string { return "same-chunk" }

func (c *sameChunkCap) Tools() []model.ToolSpec {
	return []model.ToolSpec{{
		Name:       "rag_search",
		Parameters: toolkit.ObjectSchema(map[string]any{"query": map[string]any{"type": "string"}}, "query"),
	}}
}

func (c *sameChunkCap) Invoke(_ context.Context, call toolkit.Call) (toolkit.Result, error) {
	c.calls++
	return toolkit.Result{
		Content:  fmt.Sprintf("<chunk chunk_id=\"chunk-1\">snippet for %s</chunk>", call.Arguments),
		Evidence: []toolkit.Evidence{{Source: "doc-1", Ref: "chunk-1"}},
	}, nil
}

func TestLoop_SameChunksAreNotProgress(t *testing.T) {
	capability := &sameChunkCap{}
	mdl := model.NewFake(
		model.CallTool("c1", "rag_search", `{"query":"a"}`),
		model.CallTool("c2", "rag_search", `{"query":"b"}`),
		model.CallTool("c3", "rag_search", `{"query":"c"}`),
		model.CallTool("c4", "rag_search", `{"query":"d"}`),
		model.CallTool("c5", "rag_search", `{"query":"e"}`),
		model.Reply("共 14 人"),
	)
	runner := buildRunner(t, mdl, capability, Config{})

	outcome, err := runner.Invoke(context.Background(), &Input{RunID: "run-stagnant", UserMessage: "关羽杀了多少人"}, nil)
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if calls := mdl.Calls(); calls != 6 {
		t.Fatalf("model calls = %d, want 6: the stall is reported, not acted on", calls)
	}
	if outcome.FinalState.Nudges != 1 {
		t.Fatalf("nudges = %d, want the flat round reported once", outcome.FinalState.Nudges)
	}
	if outcome.Verdict.Action != policy.ActionFinish {
		t.Fatalf("verdict = %+v, want the answer to finish the run", outcome.Verdict)
	}
}

// A stall never ends a run by itself. It is reported twice, then the answer call
// takes the evidence that was collected; the budgets stay the only hard stop.
func TestLoop_StallHandsOverToTheAnswerCall(t *testing.T) {
	capability := &sameChunkCap{}
	// The window needs three flat rounds per stall and two nudges are allowed, so
	// the hand-off lands on the eleventh flat round.
	script := make([]model.Step, 0, 12)
	for index := 0; index < 11; index++ {
		script = append(script, model.CallTool(fmt.Sprintf("c%d", index), "rag_search", fmt.Sprintf(`{"query":"%d"}`, index)))
	}
	script = append(script, model.Reply("共 14 人"))
	mdl := model.NewFake(script...)
	runner := buildRunner(t, mdl, capability, Config{})

	outcome, err := runner.Invoke(context.Background(), &Input{RunID: "run-stall", UserMessage: "关羽杀了多少人"}, nil)
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if outcome.Text != "共 14 人" {
		t.Fatalf("text = %q, want the answer written from the evidence", outcome.Text)
	}
	if outcome.FinalState.Nudges != 2 {
		t.Fatalf("nudges = %d, want the stall reported twice", outcome.FinalState.Nudges)
	}
	if outcome.Verdict.Code != "stalled" {
		t.Fatalf("verdict = %+v, want the stalled hand-off", outcome.Verdict)
	}
	if !strings.Contains(outcome.Stopped, "stalled") {
		t.Fatalf("stopped = %q, want the stalled cause recorded", outcome.Stopped)
	}
}

// A terminal round that only narrates the next step is not an answer. Delivering
// it left a real run replying "Let me see the final truncated chunk and continue
// searching for more kills." with no result at all.
func TestLoop_NarrationIsNotAnAnswer(t *testing.T) {
	capability := &fakeCap{
		tool:   "rag_search",
		result: toolkit.Result{Content: "关羽斩颜良", Evidence: []toolkit.Evidence{{Source: "doc", Ref: "chunk-1"}}},
	}
	mdl := model.NewFake(
		model.CallTool("c1", "rag_search", `{"query":"关羽 斩"}`),
		model.Reply("Let me see the final truncated chunk and continue searching for more kills."),
		model.Reply("已核实 14 人"),
	)
	runner := buildRunner(t, mdl, capability, Config{})

	outcome, err := runner.Invoke(context.Background(), &Input{RunID: "run-narration", UserMessage: "关羽杀了多少人"}, nil)
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if outcome.Text != "已核实 14 人" {
		t.Fatalf("text = %q, want the answer call's text, not the narration", outcome.Text)
	}
	if calls := mdl.Calls(); calls != 3 {
		t.Fatalf("model calls = %d, want 3 (retrieval, narration, answer)", calls)
	}
}

// TestLoop_AnchorIsOnlyForLostHistory pins that the question anchor is a repair,
// not a per-round reminder.
//
// A chat turn arrives with the whole thread in History, so the prompt carries
// every earlier question too, and compaction can drop the current one from the
// front. That is when the anchor earns its place. When nothing was dropped the
// transcript already ends with the question, and a restated "The question for
// this turn" only adds a section whose subject is the work still open.
func TestLoop_AnchorIsOnlyForLostHistory(t *testing.T) {
	capability := &fakeCap{tool: "rag_search", result: toolkit.Result{Content: "关羽斩颜良"}}
	mdl := model.NewFake(
		model.CallTool("c1", "rag_search", `{"query":"关羽 斩"}`),
		model.Reply("共 14 人"),
	)
	runner := buildRunner(t, mdl, capability, Config{})

	if _, err := runner.Invoke(context.Background(), &Input{RunID: "run-anchor", UserMessage: "关羽杀了多少人"}, nil); err != nil {
		t.Fatalf("invoke: %v", err)
	}
	requests := mdl.Requests()
	if len(requests) < 2 {
		t.Fatalf("model calls = %d, want at least 2", len(requests))
	}
	for index, request := range requests {
		if !requestMentions(request, "关羽杀了多少人") {
			t.Fatalf("request %d lost the question entirely", index)
		}
		if requestMentions(request, "The question for this turn") {
			t.Fatalf("request %d restates the question although nothing was compacted", index)
		}
	}
}

// Two prose-only rounds: the first gets a grace round, the second hands the
// write-up to the answer call, which must still deliver something.
func TestLoop_NarrationTwiceFallsBackToAnswer(t *testing.T) {
	capability := &fakeCap{
		tool:   "rag_search",
		result: toolkit.Result{Content: "关羽斩颜良", Evidence: []toolkit.Evidence{{Source: "doc", Ref: "chunk-1"}}},
	}
	mdl := model.NewFake(
		model.CallTool("c1", "rag_search", `{"query":"关羽 斩"}`),
		model.Reply("Let me look at the next chunk."),
		model.Reply("Let me look at the next chunk again."),
		model.Reply("兜底：已核实 14 人"),
	)
	runner := buildRunner(t, mdl, capability, Config{})

	outcome, err := runner.Invoke(context.Background(), &Input{RunID: "run-narration-2", UserMessage: "关羽杀了多少人"}, nil)
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if outcome.Text != "兜底：已核实 14 人" {
		t.Fatalf("text = %q, want the answer call's text", outcome.Text)
	}
	if calls := mdl.Calls(); calls != 4 {
		t.Fatalf("model calls = %d, want 4", calls)
	}
}

func TestLooksLikeNarration(t *testing.T) {
	for _, text := range []string{
		"",
		"Let me see the final truncated chunk and continue searching for more kills.",
		"let's check the next window",
		"I'll read the rest now.",
	} {
		if !looksLikeNarration(text) {
			t.Errorf("looksLikeNarration(%q) = false, want true", text)
		}
	}
	// Real answers must pass through: a short factual reply, a list, a CJK answer.
	for _, text := range []string{
		"共 16 人",
		"关羽一共斩了十六人：颜良、文丑、华雄……",
		"1. 颜良 2. 文丑",
	} {
		if looksLikeNarration(text) {
			t.Errorf("looksLikeNarration(%q) = true, want false", text)
		}
	}
}

// The concluding call can fail. What the model already said this turn is the
// only thing left to deliver, so the fallback prefers it over silence.
func TestLastAssistantText(t *testing.T) {
	history := []model.Message{
		{Role: model.RoleUser, Content: "关羽杀了多少人"},
		{Role: model.RoleAssistant, Content: ""},
		{Role: model.RoleAssistant, Content: "共 12 人"},
		{Role: model.RoleTool, Content: "tool output"},
		{Role: model.RoleAssistant, ToolCalls: []model.ToolCall{{ID: "c1", Name: "rag_search"}}},
	}
	if got := lastAssistantText(history); got != "共 12 人" {
		t.Fatalf("lastAssistantText = %q, want the last assistant prose", got)
	}
	if got := lastAssistantText(nil); got != "" {
		t.Fatalf("lastAssistantText(nil) = %q, want empty", got)
	}
}

// A pause used to end the run with no text at all: the caller got an empty
// message and no resume entry existed to continue it. The answer call must
// still run so the operator receives what was verified plus the gap.
func TestLoop_PauseWithoutTextStillAnswers(t *testing.T) {
	capability := &fakeCap{
		tool:   "rag_search",
		result: toolkit.Result{Content: "关羽斩颜良"},
	}
	// Four identical calls put three repeats on record (the first occurrence is
	// not a repeat), which is the progress policy's limit. The last script entry
	// is the answer the pause must still produce.
	mdl := model.NewFake(
		model.CallTool("c1", "rag_search", `{"query":"关羽 斩将"}`),
		model.CallTool("c2", "rag_search", `{"query":"关羽 斩将"}`),
		model.CallTool("c3", "rag_search", `{"query":"关羽 斩将"}`),
		model.CallTool("c4", "rag_search", `{"query":"关羽 斩将"}`),
		model.Reply("已核实颜良、文丑；名单可能不完整。"),
	)
	runner := buildRunner(t, mdl, capability, Config{})

	outcome, err := runner.Invoke(context.Background(), &Input{RunID: "run-pause", UserMessage: "关羽杀了多少人"}, nil)
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if outcome.Text == "" {
		t.Fatal("paused run returned no answer")
	}
	if !outcome.Interrupted {
		t.Fatalf("interrupted = false, verdict = %s (%s)", outcome.Verdict.Action, outcome.Verdict.Reason)
	}
	if !strings.Contains(outcome.Stopped, "pause") {
		t.Fatalf("stopped = %q, want the pause reason", outcome.Stopped)
	}
}
