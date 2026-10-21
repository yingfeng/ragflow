package toolkit

import (
	"context"
	"strings"
	"testing"

	"ragflow/internal/harness/agent/model"
)

func TestAllowlistNarrowsProjectionAndAdmission(t *testing.T) {
	registry := registered(t,
		&namedCapability{name: "fs", tools: []string{"fs_read", "fs_write"}},
		&namedCapability{name: "search", tools: []string{"search_chunks"}},
	)

	ctx := WithAllowlist(context.Background())
	allowlist := AllowlistFrom(ctx)
	allowlist.Set([]string{"fs_read"})

	if !allowlist.Allows("fs_read") {
		t.Fatal("listed tool was refused")
	}
	if allowlist.Allows("fs_write") {
		t.Fatal("unlisted tool was allowed")
	}
	// Leaving the skill must stay possible, otherwise the restriction is a trap.
	if !allowlist.Allows("skill_load") {
		t.Fatal("skill_load was refused while a skill narrowed the surface")
	}

	specs := RestrictSpecs([]model.ToolSpec{{Name: "fs_read"}, {Name: "fs_write"}}, allowlist)
	if len(specs) != 1 || specs[0].Name != "fs_read" {
		t.Fatalf("restricted specs = %v, want only fs_read", specs)
	}

	result, err := registry.Invoke(ctx, Call{Tool: "fs_write"})
	if err != nil {
		t.Fatalf("admission returned an error instead of a tool result: %v", err)
	}
	if !result.IsError {
		t.Fatal("unlisted tool ran despite the allowlist")
	}
	if !strings.Contains(result.Content, "not available") {
		t.Fatalf("result = %q", result.Content)
	}

	allowlist.Clear()
	result, _ = registry.Invoke(ctx, Call{Tool: "fs_write"})
	if result.IsError {
		t.Fatalf("clearing the allowlist still refused the tool: %s", result.Content)
	}
}

func TestAllowlistInactiveAllowsEverything(t *testing.T) {
	registry := registered(t, &namedCapability{name: "fs", tools: []string{"fs_read"}})

	ctx := WithAllowlist(context.Background())
	result, _ := registry.Invoke(ctx, Call{Tool: "fs_read"})
	if result.IsError {
		t.Fatalf("inactive allowlist refused a tool: %s", result.Content)
	}

	// A context without a holder behaves the same.
	result, _ = registry.Invoke(context.Background(), Call{Tool: "fs_read"})
	if result.IsError {
		t.Fatalf("missing allowlist refused a tool: %s", result.Content)
	}
}
