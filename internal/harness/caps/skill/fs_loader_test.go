package skill

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

func writeSkill(t *testing.T, root, name, frontMatter, body string) {
	t.Helper()
	dir := filepath.Join(root, name)
	if err := os.MkdirAll(filepath.Join(dir, "resources"), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	content := frontMatter + body
	if err := os.WriteFile(filepath.Join(dir, SkillFile), []byte(content), 0o644); err != nil {
		t.Fatalf("write %s: %v", name, err)
	}
	if err := os.WriteFile(filepath.Join(dir, "resources", "schema.json"), []byte("{}"), 0o644); err != nil {
		t.Fatalf("write resource: %v", err)
	}
}

func TestFSLoaderListAndLoad(t *testing.T) {
	root := t.TempDir()
	writeSkill(t, root, "retrieval", "---\nname: retrieval\ndescription: How to search the corpus\n---\n", "Search before answering.\n")
	writeSkill(t, root, "citation", "---\nname: citation\ndescription: How to cite chunks\n---\n", "Cite every claim.\n")

	loader := FSLoader{Root: root}

	skills, err := loader.List(context.Background())
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(skills) != 2 {
		t.Fatalf("skills = %d, want 2", len(skills))
	}
	if skills[0].Name != "citation" || skills[1].Name != "retrieval" {
		t.Fatalf("skills = %+v, want sorted by name", skills)
	}
	if skills[1].Description != "How to search the corpus" {
		t.Fatalf("description = %q", skills[1].Description)
	}
	if skills[0].Body != "" {
		t.Fatalf("List must not read bodies, got %q", skills[0].Body)
	}

	loaded, err := loader.Load(context.Background(), "retrieval")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if loaded.Body != "Search before answering." {
		t.Fatalf("body = %q", loaded.Body)
	}
	if len(loaded.Resources) != 1 || loaded.Resources[0] != "schema.json" {
		t.Fatalf("resources = %v, want [schema.json]", loaded.Resources)
	}
}

func TestFSLoaderRejectsTraversal(t *testing.T) {
	loader := FSLoader{Root: t.TempDir()}
	for _, name := range []string{"", ".", "..", "../etc", "a/b"} {
		if _, err := loader.Load(context.Background(), name); err == nil {
			t.Fatalf("Load(%q) accepted an invalid name", name)
		}
	}
}

func TestFSLoaderSkipsMalformedSkill(t *testing.T) {
	root := t.TempDir()
	writeSkill(t, root, "good", "---\nname: good\ndescription: fine\n---\n", "body\n")
	// A directory without SKILL.md must not break the listing.
	if err := os.MkdirAll(filepath.Join(root, "broken"), 0o755); err != nil {
		t.Fatal(err)
	}

	skills, err := FSLoader{Root: root}.List(context.Background())
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(skills) != 1 || skills[0].Name != "good" {
		t.Fatalf("skills = %+v, want only the healthy one", skills)
	}
}
