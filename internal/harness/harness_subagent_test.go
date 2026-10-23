package harness

import (
	"context"
	"strings"
	"sync"
	"testing"

	"ragflow/internal/harness/agent/definition"
	"ragflow/internal/harness/agent/loop"
	"ragflow/internal/harness/agent/model"
	agentsubagent "ragflow/internal/harness/agent/subagent"
	"ragflow/internal/harness/toolkit"
	capsubagent "ragflow/internal/harness/toolkit/subagent"
)

// anchorPrefix is how the loop restates the question at the end of a prompt.
// prepare appends it after the transcript so compaction cannot drop it, which
// means the freshest tool result is no longer the last element.
const anchorPrefix = "The question for this turn"

// routingTarget is the message these fixtures script against: the last one the
// loop addressed to the model, not the anchor restating the question.
func routingTarget(req model.Request) model.Message {
	for index := len(req.Messages) - 1; index >= 0; index-- {
		message := req.Messages[index]
		if message.Role == model.RoleUser && strings.HasPrefix(message.Content, anchorPrefix) {
			continue
		}
		return message
	}
	return req.Messages[len(req.Messages)-1]
}

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

	last := routingTarget(req)
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

// registerDelegation wires the delegation tools the way the runtime does: the
// spawner is built from the loop deps and its capability is registered on the
// registry the parent's graph reads.
func registerDelegation(t *testing.T, registry *toolkit.Registry, deps loop.Deps) {
	t.Helper()
	spawner, err := agentsubagent.New(deps, definition.PolicyConfig{}, 0)
	if err != nil {
		t.Fatalf("spawner: %v", err)
	}
	if err := registry.Register(capsubagent.New(spawner)); err != nil {
		t.Fatalf("register delegation: %v", err)
	}
}

func TestRegisterSubagentToolsExposesDelegationTools(t *testing.T) {
	registry := toolkit.NewRegistry()
	deps := loop.Deps{Model: &routingModel{}, Registry: registry}

	registerDelegation(t, registry, deps)

	names := toolNames(registry.ToolSpecs())
	for _, want := range []string{capsubagent.ToolName, capsubagent.SpawnManyName} {
		if !names[want] {
			t.Errorf("registry does not offer %q: %v", want, names)
		}
	}
}

func TestSubagentSpawnerRequiresModel(t *testing.T) {
	registry := toolkit.NewRegistry()

	_, err := agentsubagent.New(loop.Deps{Registry: registry}, definition.PolicyConfig{}, 0)
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
	registerDelegation(t, registry, deps)
	runner, err := loop.Build(loop.Config{Deps: deps})
	if err != nil {
		t.Fatalf("Build: %v", err)
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
		last := routingTarget(req)
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
