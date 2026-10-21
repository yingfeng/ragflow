package harness

import (
	"context"
	"strings"
	"sync"
	"testing"

	"ragflow/internal/harness/agent/definition"
	"ragflow/internal/harness/agent/loop"
	"ragflow/internal/harness/agent/model"
	"ragflow/internal/harness/toolkit"
	capsubagent "ragflow/internal/harness/toolkit/subagent"
)

// routingModel answers by inspecting the conversation instead of by call order,
// so parallel children cannot race the script.
type routingModel struct {
	mu    sync.Mutex
	calls []model.Request
}

func (m *routingModel) Complete(_ context.Context, req model.Request) (model.Response, error) {
	m.mu.Lock()
	m.calls = append(m.calls, req)
	m.mu.Unlock()

	last := req.Messages[len(req.Messages)-1]
	switch {
	case last.Role == model.RoleTool:
		return response("parent:" + last.Content), nil
	case strings.HasPrefix(last.Content, "child:"):
		return response(last.Content + " done"), nil
	default:
		return model.Response{
			Message: model.Message{
				Role:      model.RoleAssistant,
				ToolCalls: []model.ToolCall{{ID: "call-1", Name: capsubagent.SpawnManyName, Arguments: `{"tasks":["child:a","child:b"]}`}},
			},
			StopReason: "tool_calls",
		}, nil
	}
}

func (m *routingModel) requests() []model.Request {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]model.Request, len(m.calls))
	copy(out, m.calls)
	return out
}

func response(text string) model.Response {
	return model.Response{Message: model.Message{Role: model.RoleAssistant, Content: text}, StopReason: "stop"}
}

func toolNames(specs []model.ToolSpec) map[string]bool {
	names := make(map[string]bool, len(specs))
	for _, spec := range specs {
		names[spec.Name] = true
	}
	return names
}

func TestRegisterSubagentToolsExposesDelegationTools(t *testing.T) {
	registry := toolkit.NewRegistry()
	deps := loop.Deps{Model: &routingModel{}, Registry: registry}

	if err := RegisterSubagentTools(registry, deps, definition.PolicyConfig{}, 0); err != nil {
		t.Fatalf("RegisterSubagentTools: %v", err)
	}

	names := toolNames(registry.ToolSpecs())
	for _, want := range []string{capsubagent.ToolName, capsubagent.SpawnManyName} {
		if !names[want] {
			t.Errorf("registry does not offer %q: %v", want, names)
		}
	}
}

func TestRegisterSubagentToolsRequiresRegistry(t *testing.T) {
	err := RegisterSubagentTools(nil, loop.Deps{Model: &routingModel{}}, definition.PolicyConfig{}, 0)
	if err == nil || !strings.Contains(err.Error(), "registry") {
		t.Fatalf("error = %v, want a registry requirement", err)
	}
}

func TestRegisterSubagentToolsRequiresModel(t *testing.T) {
	registry := toolkit.NewRegistry()

	err := RegisterSubagentTools(registry, loop.Deps{Registry: registry}, definition.PolicyConfig{}, 0)
	if err == nil || !strings.Contains(err.Error(), "model is required") {
		t.Fatalf("error = %v, want a model requirement", err)
	}
}

func TestSubagentToolsRunChildrenInOneToolRound(t *testing.T) {
	mdl := &routingModel{}
	registry := toolkit.NewRegistry()
	deps := loop.Deps{
		Model:        mdl,
		Registry:     registry,
		SystemPrompt: "you are a test agent",
	}
	if err := RegisterSubagentTools(registry, deps, definition.PolicyConfig{}, 0); err != nil {
		t.Fatalf("RegisterSubagentTools: %v", err)
	}
	runner, err := BuildLoop(loop.Config{Deps: deps})
	if err != nil {
		t.Fatalf("BuildLoop: %v", err)
	}

	outcome, err := runner.Invoke(context.Background(), &loop.Input{RunID: "run-subagent", UserMessage: "delegate"}, nil)
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}

	for _, want := range []string{"child:a done", "child:b done"} {
		if !strings.Contains(outcome.Text, want) {
			t.Errorf("result missing %q: %s", want, outcome.Text)
		}
	}
	if outcome.ToolCalls != 1 {
		t.Errorf("parent tool calls = %d, want 1", outcome.ToolCalls)
	}

	var parentRequests, childRequests int
	for _, req := range mdl.requests() {
		last := req.Messages[len(req.Messages)-1]
		if last.Role == model.RoleUser && strings.HasPrefix(last.Content, "child:") {
			childRequests++
			if names := toolNames(req.Tools); names[capsubagent.ToolName] || names[capsubagent.SpawnManyName] {
				t.Errorf("child run can delegate again: %v", names)
			}
			continue
		}
		parentRequests++
	}
	if childRequests != 2 {
		t.Errorf("child model requests = %d, want 2 (one per task)", childRequests)
	}
	if parentRequests != 2 {
		t.Errorf("parent model requests = %d, want 2 (tool round + final answer)", parentRequests)
	}
}
