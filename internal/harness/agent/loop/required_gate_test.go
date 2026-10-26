package loop

import (
	"context"
	"testing"

	"ragflow/internal/harness/agent/model"
	"ragflow/internal/harness/toolkit"
)

// gateCaptureModel records each request's ToolChoice, which is the whole thing
// the required-tool gate owns.
type gateCaptureModel struct {
	steps []model.Step
	seen  []string
	next  int
}

func (m *gateCaptureModel) Complete(_ context.Context, req model.Request) (model.Response, error) {
	m.seen = append(m.seen, req.ToolChoice)
	step := m.steps[m.next]
	m.next++
	return step(req)
}

// TestRequiredToolGateForcesThenLifts pins the gate's whole contract: the first
// sampling request of a turn names a required tool, and the request after the
// model actually called one is free again.
//
// Without the gate, an evidence agent's first round can go to pure planning
// and produce nothing the answer can stand on — observed on the kill-count
// benchmark, where round 1 was a todo_write and the answer carried fewer names
// the longer the run searched.
func TestRequiredToolGateForcesThenLifts(t *testing.T) {
	capability := &fakeCap{tool: "rag_search", required: []string{"rag_search"}, result: toolkit.Result{Content: "hit"}}
	mdl := &gateCaptureModel{steps: []model.Step{
		model.CallTool("c1", "rag_search", `{"query":"x"}`),
		model.Reply("done"),
	}}
	runner := buildRunner(t, mdl, capability, Config{Deps: Deps{Policy: noProgressPolicy()}})

	if _, err := runner.Invoke(t.Context(), &Input{RunID: "run-gate", UserMessage: "find it"}, nil); err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if len(mdl.seen) < 2 {
		t.Fatalf("model calls = %d, want at least 2", len(mdl.seen))
	}
	if mdl.seen[0] != "rag_search" {
		t.Errorf("first request tool_choice = %q, want the required tool forced", mdl.seen[0])
	}
	if mdl.seen[1] != "" {
		t.Errorf("second request tool_choice = %q, want the gate lifted after one real call", mdl.seen[1])
	}
}

// TestRequiredToolGateSilentWithoutDeclaration keeps the default untouched: a
// capability that declares nothing never forces a choice, and the provider's
// request falls back to "auto".
func TestRequiredToolGateSilentWithoutDeclaration(t *testing.T) {
	capability := &fakeCap{tool: "rag_search", result: toolkit.Result{Content: "hit"}}
	mdl := &gateCaptureModel{steps: []model.Step{
		model.Reply("no tools needed"),
	}}
	runner := buildRunner(t, mdl, capability, Config{Deps: Deps{Policy: noProgressPolicy()}})

	if _, err := runner.Invoke(t.Context(), &Input{RunID: "run-no-gate", UserMessage: "hello"}, nil); err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if len(mdl.seen) != 1 {
		t.Fatalf("model calls = %d, want 1", len(mdl.seen))
	}
	if mdl.seen[0] != "" {
		t.Errorf("request tool_choice = %q, want empty (auto) with no declared gate", mdl.seen[0])
	}
}

// TestRequiredToolChoiceUnit covers the scan itself: satisfaction is scoped to
// the round's start index, and a required name the request does not carry is
// never forced.
func TestRequiredToolChoiceUnit(t *testing.T) {
	tools := []model.ToolSpec{{Name: "grep_chunks"}, {Name: "search_bm25_chunks"}}
	required := []string{"grep_chunks", "search_bm25_chunks"}
	base := []model.Message{
		{Role: model.RoleUser, Content: "earlier question"},
		{Role: model.RoleAssistant, Content: "old", ToolCalls: []model.ToolCall{{ID: "a", Name: "grep_chunks"}}},
		{Role: model.RoleTool, Content: "old result"},
	}
	thisTurn := append(append([]model.Message{}, base...), model.Message{Role: model.RoleUser, Content: "this turn"})
	thisTurnWithCall := append(append([]model.Message{}, thisTurn...),
		model.Message{Role: model.RoleAssistant, ToolCalls: []model.ToolCall{{ID: "b", Name: "search_bm25_chunks"}}})
	// The harness appends user-role notes of its own after the model has already
	// searched. The nearest user message is then the note, not the question, and
	// reading it as the turn's edge forced another retrieval call every time a
	// note was injected.
	withInjectedNote := append(append([]model.Message{}, thisTurnWithCall...),
		model.Message{Role: model.RoleUser, Content: "## Plan\n- step one"})

	cases := []struct {
		name    string
		history []model.Message
		start   int
		want    string
	}{
		{"empty history", nil, 0, "grep_chunks"},
		{"prior turn's call does not satisfy this turn", thisTurn, 3, "grep_chunks"},
		{"this turn's call satisfies", thisTurnWithCall, 3, ""},
		{"an injected note does not restart the round", withInjectedNote, 3, ""},
		{"required tool missing from request", []model.Message{
			{Role: model.RoleUser, Content: "q"},
		}, 0, ""},
	}
	for _, tc := range cases {
		var visible []model.ToolSpec
		if tc.name == "required tool missing from request" {
			visible = []model.ToolSpec{{Name: "other_tool"}} // required names not offered
		} else {
			visible = tools
		}
		if got := requiredToolChoice(visible, required, tc.history, tc.start); got != tc.want {
			t.Errorf("%s: got %q, want %q", tc.name, got, tc.want)
		}
	}
}
