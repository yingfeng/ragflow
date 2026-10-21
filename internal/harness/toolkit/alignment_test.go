package toolkit

import (
	"context"
	"strings"
	"testing"
	"time"

	"ragflow/internal/harness/agent/model"
)

type schemaCapability struct {
	name  string
	specs []model.ToolSpec
}

func (c *schemaCapability) Name() string            { return c.name }
func (c *schemaCapability) Tools() []model.ToolSpec { return c.specs }
func (c *schemaCapability) Invoke(context.Context, Call) (Result, error) {
	return Result{Content: c.name}, nil
}

func bigSpec(name string) model.ToolSpec {
	return model.ToolSpec{
		Name:        name,
		Description: strings.Repeat("d", 1200),
		Parameters:  map[string]any{"type": "object", "properties": map[string]any{"q": map[string]any{"type": "string"}}},
	}
}

func TestToolSpecsWithinBudgetDropsExpensiveTools(t *testing.T) {
	registry := NewRegistry()
	if err := registry.Register(&schemaCapability{
		name:  "many",
		specs: []model.ToolSpec{bigSpec("a_tool"), bigSpec("b_tool"), bigSpec("c_tool")},
	}); err != nil {
		t.Fatalf("register: %v", err)
	}

	if got := len(registry.ToolSpecs()); got != 3 {
		t.Fatalf("unbudgeted specs = %d, want 3", got)
	}
	budgeted := registry.ToolSpecsWithinBudget(SchemaBudgetTokens)
	if len(budgeted) == 0 {
		t.Fatal("budget dropped every tool; the first one must always fit")
	}
	if len(budgeted) >= 3 {
		t.Fatalf("budgeted specs = %d, want the expensive ones dropped", len(budgeted))
	}
	if len(registry.ToolSpecsWithinBudget(0)) != 3 {
		t.Error("a zero budget must mean no limit")
	}
}

func TestOnlyReadOnlyDropsMutatingCapabilities(t *testing.T) {
	registry := NewRegistry()
	if err := registry.Register(&schemaCapability{name: "reader", specs: []model.ToolSpec{{Name: "search_chunks"}}}); err != nil {
		t.Fatalf("register reader: %v", err)
	}
	if err := registry.Register(&schemaCapability{name: "writer", specs: []model.ToolSpec{{Name: "fs_edit"}}}); err != nil {
		t.Fatalf("register writer: %v", err)
	}

	readOnly := registry.OnlyReadOnly()

	if !readOnly.Has("search_chunks") {
		t.Error("read-only view lost a read tool")
	}
	if readOnly.Has("fs_edit") {
		t.Error("read-only view kept a mutating tool")
	}
	if !registry.Has("fs_edit") {
		t.Error("the parent registry must not be narrowed")
	}
}

type fixedApprover struct {
	decision ApprovalDecision
	calls    int
}

func (a *fixedApprover) Approve(context.Context, Call, Risk) ApprovalDecision {
	a.calls++
	return a.decision
}

func TestApprovalGateSkipsLowRisk(t *testing.T) {
	approver := &fixedApprover{decision: ApprovalDecision{Allow: false}}
	gate := NewApprovalGate(approver, RiskDangerous)

	decision := gate.Approve(context.Background(), Call{Tool: "search_chunks"}, RiskReadOnly)
	if !decision.Allow {
		t.Fatal("read-only call should pass without asking")
	}
	if approver.calls != 0 {
		t.Fatalf("approver was consulted %d times for a read", approver.calls)
	}
}

func TestApprovalGateDeniesWithoutApprover(t *testing.T) {
	gate := NewApprovalGate(nil, RiskDangerous)

	decision := gate.Approve(context.Background(), Call{Tool: "exec_shell"}, RiskDangerous)
	if decision.Allow {
		t.Fatal("a dangerous call must not run without an approver")
	}
	if !strings.Contains(decision.Reason, "requires approval") {
		t.Fatalf("reason = %q", decision.Reason)
	}
}

func TestApprovalBrokerRoundTrip(t *testing.T) {
	broker := NewApprovalBroker(2 * time.Second)
	ctx := WithRunID(context.Background(), "run-1")

	result := make(chan ApprovalDecision, 1)
	go func() {
		result <- broker.Approve(ctx, Call{ID: "call-1", Tool: "exec_shell"}, RiskDangerous)
	}()

	deadline := time.Now().Add(time.Second)
	for !broker.Resolve("run-1", "call-1", false, "needs review") {
		if time.Now().After(deadline) {
			t.Fatal("request never became pending")
		}
		time.Sleep(5 * time.Millisecond)
	}

	decision := <-result
	if decision.Allow {
		t.Fatal("operator rejected the call but it was allowed")
	}
	if decision.Reason != "needs review" {
		t.Fatalf("reason = %q", decision.Reason)
	}
}

func TestApprovalBrokerDeniesOnTimeout(t *testing.T) {
	broker := NewApprovalBroker(20 * time.Millisecond)
	ctx := WithRunID(context.Background(), "run-2")

	decision := broker.Approve(ctx, Call{ID: "call-2", Tool: "exec_shell"}, RiskDangerous)
	if decision.Allow {
		t.Fatal("a timed-out approval must deny")
	}
	if !strings.Contains(decision.Reason, "timed out") {
		t.Fatalf("reason = %q", decision.Reason)
	}
}

func TestApprovalBrokerUnknownRequest(t *testing.T) {
	broker := NewApprovalBroker(0)

	if broker.Resolve("run-x", "call-x", true, "") {
		t.Fatal("resolving an unknown request must report false")
	}
}
