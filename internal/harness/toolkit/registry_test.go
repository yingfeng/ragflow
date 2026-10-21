package toolkit

import (
	"context"
	"testing"

	"ragflow/internal/harness/agent/model"
)

type namedCapability struct {
	name  string
	tools []string
}

func (c *namedCapability) Name() string { return c.name }

func (c *namedCapability) Tools() []model.ToolSpec {
	specs := make([]model.ToolSpec, 0, len(c.tools))
	for _, tool := range c.tools {
		specs = append(specs, model.ToolSpec{Name: tool})
	}
	return specs
}

func (c *namedCapability) Invoke(context.Context, Call) (Result, error) {
	return Result{Content: c.name}, nil
}

func registered(t *testing.T, capabilities ...Capability) *Registry {
	t.Helper()
	registry := NewRegistry()
	for _, capability := range capabilities {
		if err := registry.Register(capability); err != nil {
			t.Fatalf("register %s: %v", capability.Name(), err)
		}
	}
	return registry
}

func toolNames(registry *Registry) map[string]bool {
	names := make(map[string]bool)
	for _, spec := range registry.ToolSpecs() {
		names[spec.Name] = true
	}
	return names
}

func TestRegistryWithoutDropsOwningCapabilities(t *testing.T) {
	registry := registered(t,
		&namedCapability{name: "rag", tools: []string{"search", "open"}},
		&namedCapability{name: "delegate", tools: []string{"subagent", "spawn_subagents"}},
		&namedCapability{name: "builtin", tools: []string{"think"}},
	)

	child := registry.Without("subagent", "spawn_subagents")

	names := toolNames(child)
	if len(names) != 3 || !names["search"] || !names["open"] || !names["think"] {
		t.Fatalf("child tools = %v, want search/open/think", names)
	}
	if child.Has("subagent") || child.Has("spawn_subagents") {
		t.Errorf("child still exposes delegation tools: %v", names)
	}
	if got := child.Names(); len(got) != 2 {
		t.Errorf("child capability names = %v, want 2", got)
	}
}

func TestRegistryWithoutLeavesParentIntact(t *testing.T) {
	registry := registered(t,
		&namedCapability{name: "rag", tools: []string{"search"}},
		&namedCapability{name: "delegate", tools: []string{"subagent"}},
	)

	_ = registry.Without("subagent")

	if !registry.Has("subagent") || !registry.Has("search") {
		t.Errorf("parent registry lost tools: %v", toolNames(registry))
	}
}

func TestRegistryWithoutKeepsDeferredState(t *testing.T) {
	registry := NewRegistry()
	if err := registry.RegisterDeferred(&namedCapability{name: "rag", tools: []string{"search"}}); err != nil {
		t.Fatalf("register deferred: %v", err)
	}

	child := registry.Without("nothing")

	if names := toolNames(child); len(names) != 0 {
		t.Errorf("deferred tool leaked into the offered list: %v", names)
	}
	if !child.Has("search") {
		t.Error("deferred tool should stay callable in the child")
	}
}

func TestRegistryWithoutMatchesNothing(t *testing.T) {
	registry := registered(t, &namedCapability{name: "rag", tools: []string{"search"}})

	child := registry.Without("unknown_tool")

	if names := toolNames(child); len(names) != 1 || !names["search"] {
		t.Fatalf("child tools = %v, want search", names)
	}
}

func TestRegistryWithoutNilRegistry(t *testing.T) {
	var registry *Registry

	if child := registry.Without("search"); child == nil || len(child.ToolSpecs()) != 0 {
		t.Fatalf("child = %#v, want an empty registry", child)
	}
}
