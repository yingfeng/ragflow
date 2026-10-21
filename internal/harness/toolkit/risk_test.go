package toolkit

import (
	"context"
	"testing"

	"ragflow/internal/harness/agent/model"
)

type riskCapability struct {
	name string
	tool string
	risk Risk
}

func (c *riskCapability) Name() string { return c.name }

func (c *riskCapability) Tools() []model.ToolSpec {
	return []model.ToolSpec{{Name: c.tool}}
}

func (c *riskCapability) Invoke(context.Context, Call) (Result, error) {
	return Result{Content: "ok"}, nil
}

func (c *riskCapability) ToolRisk(string) Risk { return c.risk }

func TestRiskOfBuiltinTable(t *testing.T) {
	registry := NewRegistry()

	cases := map[string]Risk{
		"search_chunks":  RiskReadOnly,
		"fs_cat":         RiskReadOnly,
		"fs_write":       RiskMutating,
		"exec_shell":     RiskDangerous,
		"run_javascript": RiskDangerous,
		"unknown_tool":   RiskMutating,
	}
	for tool, want := range cases {
		if got := registry.RiskOf(tool); got != want {
			t.Errorf("RiskOf(%q) = %v, want %v", tool, got, want)
		}
	}
}

func TestRiskOfPrefersCapabilityClassification(t *testing.T) {
	registry := NewRegistry()
	if err := registry.Register(&riskCapability{name: "custom", tool: "read_spill", risk: RiskDangerous}); err != nil {
		t.Fatalf("register: %v", err)
	}

	// read_spill is read-only in the builtin table; the owning capability says
	// otherwise, and the capability wins.
	if got := registry.RiskOf("read_spill"); got != RiskDangerous {
		t.Fatalf("RiskOf = %v, want the capability's classification", got)
	}
}

func TestParallelizableOnlyForReadOnly(t *testing.T) {
	registry := NewRegistry()

	if !registry.Parallelizable("search_chunks") {
		t.Error("retrieval should be parallelizable")
	}
	if registry.Parallelizable("fs_edit") {
		t.Error("a write must not be parallelizable")
	}
	if registry.Parallelizable("exec_shell") {
		t.Error("shell must not be parallelizable")
	}
}

func TestNilRegistryIsSerial(t *testing.T) {
	var registry *Registry

	if registry.Parallelizable("search_chunks") {
		t.Error("a nil registry must not claim parallel safety")
	}
	if got := registry.RiskOf("exec_shell"); got != RiskDangerous {
		t.Errorf("RiskOf = %v, want dangerous", got)
	}
}
