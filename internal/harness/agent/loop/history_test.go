package loop

import (
	"context"
	"strings"
	"testing"

	"ragflow/internal/harness/agent/model"
)

type capturingModel struct{ requests []model.Request }

func (c *capturingModel) Complete(_ context.Context, req model.Request) (model.Response, error) {
	c.requests = append(c.requests, req)
	return model.Response{Message: model.Message{Role: model.RoleAssistant, Content: "answered"}}, nil
}

func TestSeedHistoryKeepsOnlyConversationText(t *testing.T) {
	history := seedHistory(Input{
		Messages: []model.Message{
			{Role: model.RoleSystem, Content: "system is supplied by the loop, not the caller"},
			{Role: model.RoleUser, Content: "first question"},
			{Role: model.RoleAssistant, Content: "first answer"},
			{Role: model.RoleTool, Content: "raw tool output"},
			{Role: model.RoleAssistant, Content: "   "},
		},
		UserMessage: "second question",
	})

	want := []struct {
		role    model.Role
		content string
	}{
		{model.RoleUser, "first question"},
		{model.RoleAssistant, "first answer"},
		{model.RoleUser, "second question"},
	}
	if len(history) != len(want) {
		t.Fatalf("history = %+v", history)
	}
	for index, expected := range want {
		if history[index].Role != expected.role || history[index].Content != expected.content {
			t.Errorf("history[%d] = %+v, want %s/%q", index, history[index], expected.role, expected.content)
		}
	}
}

func TestSeedHistoryDoesNotDuplicateTheQuestion(t *testing.T) {
	history := seedHistory(Input{
		Messages:    []model.Message{{Role: model.RoleUser, Content: "same"}},
		UserMessage: "same",
	})
	if len(history) != 1 {
		t.Fatalf("history = %+v, want a single user turn", history)
	}
}

func TestSeedHistoryUsesMessagesWhenNoUserMessage(t *testing.T) {
	history := seedHistory(Input{Messages: []model.Message{
		{Role: model.RoleUser, Content: "q"},
		{Role: model.RoleAssistant, Content: "a"},
		{Role: model.RoleUser, Content: "q2"},
	}})
	if len(history) != 3 || history[2].Content != "q2" {
		t.Fatalf("history = %+v", history)
	}
}

func TestRunnerSendsConversationHistoryToTheModel(t *testing.T) {
	mdl := &capturingModel{}
	runner := buildRunner(t, mdl, nil, Config{})

	outcome, err := runner.Invoke(t.Context(), &Input{
		RunID: "run-history",
		Messages: []model.Message{
			{Role: model.RoleUser, Content: "who wrote it"},
			{Role: model.RoleAssistant, Content: "Ada wrote it [ID:1]"},
			{Role: model.RoleTool, Content: "should not be replayed"},
		},
		UserMessage: "and when",
	}, nil)
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if len(mdl.requests) == 0 {
		t.Fatal("the model was never called")
	}

	var sent []string
	for _, message := range mdl.requests[0].Messages {
		sent = append(sent, string(message.Role)+":"+message.Content)
	}
	joined := strings.Join(sent, "|")
	for _, want := range []string{"user:who wrote it", "assistant:Ada wrote it", "user:and when"} {
		if !strings.Contains(joined, want) {
			t.Errorf("model did not receive %q: %s", want, joined)
		}
	}
	// Durable knowledge stays in the transcript; the evidence handle does not,
	// because it only resolves against the turn that produced it.
	if strings.Contains(joined, "[ID:1]") {
		t.Errorf("an earlier turn's citation handle was replayed: %s", joined)
	}
	if strings.Contains(joined, "should not be replayed") {
		t.Errorf("an earlier tool result was replayed: %s", joined)
	}
	if outcome.Text != "answered" {
		t.Fatalf("outcome = %+v", outcome)
	}
}
