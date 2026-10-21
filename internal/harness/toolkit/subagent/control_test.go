package subagent

import (
	"context"
	"strings"
	"testing"
	"time"

	"ragflow/internal/harness/agent/model"
	"ragflow/internal/harness/toolkit"
)

// controlledSpawner adds the asynchronous control plane to the blocking fake.
type controlledSpawner struct {
	*fakeSpawner
	startRequests []Request
	children      []ChildState
	stopped       string
	message       string
	startErr      error
}

func (c *controlledSpawner) Start(_ context.Context, req Request) (string, error) {
	c.startRequests = append(c.startRequests, req)
	if c.startErr != nil {
		return "", c.startErr
	}
	return "child-1", nil
}

func (c *controlledSpawner) List(context.Context, []string) ([]ChildState, error) {
	return c.children, nil
}

func (c *controlledSpawner) Stop(_ context.Context, runID string) bool {
	c.stopped = runID
	return runID == "child-1"
}

func (c *controlledSpawner) Message(_ context.Context, runID, message string) bool {
	c.message = runID + ":" + message
	return runID == "child-1"
}

func newControlled() *controlledSpawner {
	return &controlledSpawner{fakeSpawner: &fakeSpawner{}}
}

func toolNames(specs []model.ToolSpec) map[string]bool {
	names := make(map[string]bool, len(specs))
	for _, spec := range specs {
		names[spec.Name] = true
	}
	return names
}

func TestControlToolsRegisteredOnlyWithController(t *testing.T) {
	blocking := toolNames(New(&fakeSpawner{}).Tools())
	for _, name := range []string{StartName, StatusName, StopName, MessageName} {
		if blocking[name] {
			t.Errorf("%s was offered without a control plane", name)
		}
	}

	controlled := toolNames(New(newControlled()).Tools())
	for _, name := range []string{ToolName, SpawnManyName, StartName, StatusName, StopName, MessageName} {
		if !controlled[name] {
			t.Errorf("%s is missing with a control plane", name)
		}
	}
}

func TestControlStartPassesDepthAndScope(t *testing.T) {
	spawner := newControlled()
	capability := New(spawner)
	ctx := toolkit.WithScope(context.Background(), toolkit.Scope{TenantID: "tenant-1", Datasets: []string{"kb-1"}})
	ctx = WithDepth(ctx, 1)

	result, err := capability.Invoke(ctx, toolkit.Call{
		Tool:      StartName,
		Arguments: `{"prompt":"summarise the report"}`,
	})
	if err != nil {
		t.Fatalf("start: %v", err)
	}
	if !strings.Contains(result.Content, "child-1") {
		t.Fatalf("result = %q", result.Content)
	}
	if len(spawner.startRequests) != 1 {
		t.Fatalf("start requests = %d", len(spawner.startRequests))
	}
	request := spawner.startRequests[0]
	if request.Depth != 2 {
		t.Errorf("depth = %d, want 2", request.Depth)
	}
	if request.Scope.TenantID != "tenant-1" || len(request.Scope.Datasets) != 1 {
		t.Errorf("scope = %+v", request.Scope)
	}
}

func TestControlStartRejectsBlankPrompt(t *testing.T) {
	capability := New(newControlled())
	if _, err := capability.Invoke(context.Background(), toolkit.Call{
		Tool:      StartName,
		Arguments: `{"prompt":"  "}`,
	}); err == nil {
		t.Fatal("expected an error for a blank prompt")
	}
}

func TestControlStatusRendersChildren(t *testing.T) {
	spawner := newControlled()
	spawner.children = []ChildState{
		{RunID: "child-1", Status: "done", Output: "answer"},
		{RunID: "child-2", Status: "failed", Error: "model unavailable"},
	}
	capability := New(spawner)

	result, err := capability.Invoke(context.Background(), toolkit.Call{
		Tool:      StatusName,
		Arguments: `{}`,
	})
	if err != nil {
		t.Fatalf("status: %v", err)
	}
	for _, want := range []string{"child-1", "done", "answer", "child-2", "failed", "model unavailable"} {
		if !strings.Contains(result.Content, want) {
			t.Errorf("status output is missing %q: %s", want, result.Content)
		}
	}
}

func TestControlStatusWithoutChildren(t *testing.T) {
	capability := New(newControlled())
	result, err := capability.Invoke(context.Background(), toolkit.Call{Tool: StatusName, Arguments: `{}`})
	if err != nil {
		t.Fatalf("status: %v", err)
	}
	if !strings.Contains(result.Content, `count="0"`) {
		t.Fatalf("result = %q", result.Content)
	}
}

func TestControlStatusWaitsOnlyWhileChildrenRun(t *testing.T) {
	spawner := newControlled()
	spawner.children = []ChildState{{RunID: "child-1", Status: "done", Output: "answer"}}
	capability := New(spawner)

	// A terminal child must return at once even when a wait was requested.
	done := make(chan struct{})
	go func() {
		defer close(done)
		if _, err := capability.Invoke(context.Background(), toolkit.Call{
			Tool:      StatusName,
			Arguments: `{"wait_seconds":120}`,
		}); err != nil {
			t.Errorf("status: %v", err)
		}
	}()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("status waited for a child that had already finished")
	}
}

func TestControlStopAndMessage(t *testing.T) {
	spawner := newControlled()
	capability := New(spawner)

	stopped, err := capability.Invoke(context.Background(), toolkit.Call{
		Tool:      StopName,
		Arguments: `{"run_id":"child-1"}`,
	})
	if err != nil {
		t.Fatalf("stop: %v", err)
	}
	if !strings.Contains(stopped.Content, "subagent_stopped") {
		t.Fatalf("stop result = %q", stopped.Content)
	}

	notRunning, err := capability.Invoke(context.Background(), toolkit.Call{
		Tool:      StopName,
		Arguments: `{"run_id":"child-9"}`,
	})
	if err != nil {
		t.Fatalf("stop unknown: %v", err)
	}
	if !strings.Contains(notRunning.Content, "not_running") {
		t.Fatalf("stop unknown result = %q", notRunning.Content)
	}

	sent, err := capability.Invoke(context.Background(), toolkit.Call{
		Tool:      MessageName,
		Arguments: `{"run_id":"child-1","message":"focus on 2019"}`,
	})
	if err != nil {
		t.Fatalf("message: %v", err)
	}
	if !strings.Contains(sent.Content, "delivered") {
		t.Fatalf("message result = %q", sent.Content)
	}
	if spawner.message != "child-1:focus on 2019" {
		t.Fatalf("message = %q", spawner.message)
	}
}
