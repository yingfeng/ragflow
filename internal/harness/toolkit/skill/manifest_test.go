package skill

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestFSLoaderParsesExtendedFrontMatter(t *testing.T) {
	root := t.TempDir()
	frontMatter := strings.Join([]string{
		"---",
		"name: retrieval",
		"description: How to search the corpus",
		"version: 2.1.0",
		"when_to_use: before answering a factual question",
		"max_tokens: 1500",
		"allowed_tools:",
		"  - search_chunks",
		"  - fs_cat",
		"---",
		"",
	}, "\n")
	writeSkill(t, root, "retrieval", frontMatter, "Search before answering.\n")

	skills, err := FSLoader{Root: root}.List(context.Background())
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(skills) != 1 {
		t.Fatalf("skills = %d, want 1", len(skills))
	}
	item := skills[0]
	if item.Version != "2.1.0" {
		t.Errorf("version = %q", item.Version)
	}
	if item.WhenToUse != "before answering a factual question" {
		t.Errorf("when_to_use = %q", item.WhenToUse)
	}
	if item.MaxTokens != 1500 {
		t.Errorf("max_tokens = %d", item.MaxTokens)
	}
	if len(item.AllowedTools) != 2 || item.AllowedTools[0] != "search_chunks" || item.AllowedTools[1] != "fs_cat" {
		t.Errorf("allowed_tools = %v", item.AllowedTools)
	}
}

func TestFSLoaderParsesInlineAllowedToolsAndHiddenFlag(t *testing.T) {
	root := t.TempDir()
	frontMatter := "---\nname: quiet\ndescription: hidden\ndisable_model_invocation: true\nallowed_tools: [fs_ls, fs_cat]\n---\n"
	writeSkill(t, root, "quiet", frontMatter, "body\n")

	skills, err := FSLoader{Root: root}.List(context.Background())
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	item := skills[0]
	if !item.DisableModelInvocation {
		t.Error("disable_model_invocation was not parsed")
	}
	if len(item.AllowedTools) != 2 || item.AllowedTools[1] != "fs_cat" {
		t.Errorf("allowed_tools = %v", item.AllowedTools)
	}
}

func TestCatalogHidesModelDisabledSkills(t *testing.T) {
	catalog := renderCatalog([]Skill{
		{Name: "visible", Description: "shown"},
		{Name: "hidden", Description: "not shown", DisableModelInvocation: true},
	})

	if !strings.Contains(catalog, "visible") {
		t.Errorf("catalog lost the visible skill: %s", catalog)
	}
	if strings.Contains(catalog, "hidden") {
		t.Errorf("catalog leaked a hidden skill: %s", catalog)
	}
}

func TestCatalogStopsAtBudget(t *testing.T) {
	skills := make([]Skill, 0, 40)
	for i := 0; i < 40; i++ {
		skills = append(skills, Skill{
			Name:        fmt.Sprintf("skill-%02d", i),
			Description: strings.Repeat("d", MaxDescriptionLength),
		})
	}

	catalog := renderCatalog(skills)

	if len(catalog) > MetadataBudgetBytes+512 {
		t.Fatalf("catalog = %d bytes, want it under the budget", len(catalog))
	}
	if !strings.Contains(catalog, "omitted") {
		t.Error("catalog did not announce the omitted entries")
	}
}

func TestRenderAppliesBodyBudget(t *testing.T) {
	body := strings.Repeat("b", MaxBodyChars*2)
	rendered := render(Skill{Name: "big", Description: "d", Body: body})

	if len(rendered) > MaxBodyChars+512 {
		t.Fatalf("rendered = %d bytes, want the body clipped", len(rendered))
	}
	if !strings.Contains(rendered, "omitted") {
		t.Error("body clipping was not announced")
	}
}

func TestRenderDeclaresAllowedTools(t *testing.T) {
	rendered := render(Skill{Name: "s", Body: "body", AllowedTools: []string{"search_chunks", "fs_cat"}})

	if !strings.Contains(rendered, "allowed tools: search_chunks, fs_cat") {
		t.Fatalf("allowed tools missing from the rendered skill: %s", rendered)
	}
}

func TestFSLoaderRefusesSymlinkEscape(t *testing.T) {
	outside := t.TempDir()
	writeSkill(t, outside, "evil", "---\nname: evil\ndescription: escapes\n---\n", "body\n")

	root := t.TempDir()
	if err := os.Symlink(filepath.Join(outside, "evil"), filepath.Join(root, "evil")); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}

	if _, err := (FSLoader{Root: root}).Load(context.Background(), "evil"); err == nil {
		t.Fatal("a skill reachable only through a symlink must be refused")
	} else if !strings.Contains(err.Error(), "escapes") {
		t.Fatalf("error should name the escape: %v", err)
	}
}

func TestFSLoaderRejectsOversizedSkillFile(t *testing.T) {
	root := t.TempDir()
	dir := filepath.Join(root, "big")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	content := "---\nname: big\ndescription: huge\n---\n" + strings.Repeat("x", MaxSkillFileBytes+1)
	if err := os.WriteFile(filepath.Join(dir, SkillFile), []byte(content), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}

	if _, err := (FSLoader{Root: root}).Load(context.Background(), "big"); err == nil {
		t.Fatal("an oversized SKILL.md must be rejected")
	}
}
