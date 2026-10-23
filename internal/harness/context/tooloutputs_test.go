package agentctx

import (
	"context"
	"testing"

	"ragflow/internal/harness/agent/model"
)

// A provider rejects an assistant message whose tool_calls no tool message
// answers, so the builder must never send one: a run cancelled or checkpointed
// mid-batch would otherwise fail every later request with a 400.
func TestBuildSynthesizesMissingToolOutputs(t *testing.T) {
	builder := New(Config{MaxInputTokens: 100000})
	history := []model.Message{
		{Role: model.RoleUser, Content: "count them"},
		{Role: model.RoleAssistant, ToolCalls: []model.ToolCall{
			{ID: "call_1", Name: "search", Arguments: `{"q":"a"}`},
			{ID: "call_2", Name: "search", Arguments: `{"q":"b"}`},
		}},
		{Role: model.RoleTool, ToolCallID: "call_1", Name: "search", Content: "first result"},
	}

	prompt := builder.Build(context.Background(), "system", nil, history)
	outputs := map[string]string{}
	for _, message := range prompt {
		if message.Role == model.RoleTool {
			outputs[message.ToolCallID] = message.Content
		}
	}
	if outputs["call_1"] != "first result" {
		t.Errorf("existing output was rewritten: %q", outputs["call_1"])
	}
	missing, ok := outputs["call_2"]
	if !ok {
		t.Fatal("the unanswered tool call got no result")
	}
	if missing == "" {
		t.Error("the synthesized result is empty, the model cannot tell what happened")
	}
}

// Complete transcripts must come back untouched: no duplicate tool messages and
// no reordering.
func TestBuildLeavesCompleteTranscriptAlone(t *testing.T) {
	builder := New(Config{MaxInputTokens: 100000})
	history := []model.Message{
		{Role: model.RoleUser, Content: "q"},
		{Role: model.RoleAssistant, ToolCalls: []model.ToolCall{{ID: "call_1", Name: "search"}}},
		{Role: model.RoleTool, ToolCallID: "call_1", Name: "search", Content: "ok"},
		{Role: model.RoleAssistant, Content: "done"},
	}

	prompt := builder.Build(context.Background(), "system", nil, history)
	tools := 0
	for _, message := range prompt {
		if message.Role == model.RoleTool {
			tools++
		}
	}
	if tools != 1 {
		t.Fatalf("tool messages = %d, want the single original result", tools)
	}
	if last := prompt[len(prompt)-1]; last.Role != model.RoleAssistant || last.Content != "done" {
		t.Fatalf("last message = %+v, want the final answer", last)
	}
}
