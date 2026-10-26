package skill

import (
	"context"
	"testing"

	"ragflow/internal/harness/toolkit"
)

type stubLoader struct{ item Skill }

func (s stubLoader) List(context.Context) ([]Skill, error) { return []Skill{s.item}, nil }

func (s stubLoader) Load(context.Context, string) (Skill, error) { return s.item, nil }

func loadSkill(t *testing.T, item Skill) *toolkit.Allowlist {
	t.Helper()
	capability := New(context.Background(), stubLoader{item: item})
	ctx := toolkit.WithAllowlist(context.Background())
	if _, err := capability.Invoke(ctx, toolkit.Call{Tool: "skill_load", Arguments: `{"name":"` + item.Name + `"}`}); err != nil {
		t.Fatalf("load skill: %v", err)
	}
	return toolkit.AllowlistFrom(ctx)
}

func TestSkillLoadNarrowsTheToolSurface(t *testing.T) {
	allowlist := loadSkill(t, Skill{Name: "reporter", AllowedTools: []string{"search_chunks", "fs_read"}})

	if !allowlist.Active() {
		t.Fatal("allowed_tools did not narrow the surface")
	}
	for _, name := range []string{"search_chunks", "fs_read"} {
		if !allowlist.Allows(name) {
			t.Errorf("%s was refused but the skill declares it", name)
		}
	}
	if allowlist.Allows("exec_shell") {
		t.Error("exec_shell was allowed although the skill does not declare it")
	}
}

func TestSkillWithoutAllowedToolsLeavesSurfaceOpen(t *testing.T) {
	allowlist := loadSkill(t, Skill{Name: "notes"})

	if allowlist.Active() {
		t.Fatal("a skill without allowed_tools narrowed the surface")
	}
	if !allowlist.Allows("exec_shell") {
		t.Fatal("an unrestricted skill refused a tool")
	}
}
