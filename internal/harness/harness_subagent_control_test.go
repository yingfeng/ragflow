package harness

import (
	"context"
	"strings"
	"testing"

	"github.com/glebarez/sqlite"
	"gorm.io/gorm"

	"ragflow/internal/entity"
	"ragflow/internal/harness/agent/definition"
	"ragflow/internal/harness/agent/loop"
	"ragflow/internal/harness/agent/model"
	agentsubagent "ragflow/internal/harness/agent/subagent"
	"ragflow/internal/harness/policy"
	"ragflow/internal/harness/store"
	"ragflow/internal/harness/toolkit"
	capsubagent "ragflow/internal/harness/toolkit/subagent"
)

// controlModel drives the asynchronous plane: start a child, wait for it, answer.
type controlModel struct{}

func (*controlModel) Complete(_ context.Context, req model.Request) (model.Response, error) {
	last := req.Messages[len(req.Messages)-1]
	switch {
	case last.Role == model.RoleTool && strings.Contains(last.Content, "<subagent_children"):
		return response("parent: " + last.Content), nil
	case last.Role == model.RoleTool && strings.Contains(last.Content, "<subagent_started"):
		return model.Response{
			Message: model.Message{
				Role:      model.RoleAssistant,
				ToolCalls: []model.ToolCall{{ID: "call-2", Name: capsubagent.StatusName, Arguments: `{"wait_seconds":30}`}},
			},
			StopReason: "tool_calls",
		}, nil
	case strings.HasPrefix(last.Content, "child:"):
		return response("child answer for " + last.Content), nil
	default:
		return model.Response{
			Message: model.Message{
				Role:      model.RoleAssistant,
				ToolCalls: []model.ToolCall{{ID: "call-1", Name: capsubagent.StartName, Arguments: `{"prompt":"child:a"}`}},
			},
			StopReason: "tool_calls",
		}, nil
	}
}

func newControlRunStore(t *testing.T) *store.RunStore {
	t.Helper()
	db, err := gorm.Open(sqlite.Open("file:"+t.Name()+"?mode=memory&cache=shared"), &gorm.Config{})
	if err != nil {
		t.Fatalf("open sqlite: %v", err)
	}
	if err := db.AutoMigrate(&entity.HarnessAgentRun{}, &entity.HarnessAgentStep{}); err != nil {
		t.Fatalf("migrate: %v", err)
	}
	return store.NewRunStore(db)
}

func TestSubagentControlPlaneStartAndStatus(t *testing.T) {
	runStore := newControlRunStore(t)
	registry := toolkit.NewRegistry()
	deps := loop.Deps{
		Model:        &controlModel{},
		Registry:     registry,
		SystemPrompt: "you are a test agent",
		// Delegation produces no evidence, so the progressive-stop policy would
		// end this run before the status round. The test targets the control
		// plane, not the termination heuristic.
		Policy: policy.New(policy.Budget{}, policy.Completion{}),
	}

	spawner, err := agentsubagent.New(deps, definition.PolicyConfig{}, 0)
	if err != nil {
		t.Fatalf("spawner: %v", err)
	}
	spawner = spawner.WithStore(runStore)
	if err := registry.Register(capsubagent.New(spawner)); err != nil {
		t.Fatalf("register: %v", err)
	}

	names := toolNames(registry.ToolSpecs())
	for _, want := range []string{capsubagent.StartName, capsubagent.StatusName} {
		if !names[want] {
			t.Fatalf("control tool %q is not offered: %v", want, names)
		}
	}

	runner, err := BuildLoop(loop.Config{Deps: deps})
	if err != nil {
		t.Fatalf("BuildLoop: %v", err)
	}
	outcome, err := runner.Invoke(context.Background(), &loop.Input{RunID: "run-control", UserMessage: "delegate"}, nil)
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if !strings.Contains(outcome.Text, "child answer for child:a") {
		t.Fatalf("status did not return the child answer: %q (rounds=%d toolcalls=%d stopped=%q)",
			outcome.Text, outcome.Rounds, outcome.ToolCalls, outcome.Stopped)
	}

	children, err := runStore.ListChildren(context.Background(), "run-control")
	if err != nil {
		t.Fatalf("list children: %v", err)
	}
	if len(children) != 1 {
		t.Fatalf("children = %d, want 1", len(children))
	}
	if children[0].Status != "done" {
		t.Fatalf("child status = %q, want done", children[0].Status)
	}
	if children[0].ParentRunID != "run-control" {
		t.Fatalf("child parent = %q", children[0].ParentRunID)
	}
}

func TestSubagentControlToolsHiddenWithoutStore(t *testing.T) {
	registry := toolkit.NewRegistry()
	deps := loop.Deps{Model: &controlModel{}, Registry: registry, SystemPrompt: "test"}
	spawner, err := agentsubagent.New(deps, definition.PolicyConfig{}, 0)
	if err != nil {
		t.Fatalf("spawner: %v", err)
	}
	if err := registry.Register(capsubagent.New(spawner)); err != nil {
		t.Fatalf("register: %v", err)
	}

	names := toolNames(registry.ToolSpecs())
	if names[capsubagent.StartName] {
		t.Fatalf("control tools offered without a store: %v", names)
	}
	if !names[capsubagent.ToolName] {
		t.Fatalf("blocking delegation tool disappeared: %v", names)
	}
}
