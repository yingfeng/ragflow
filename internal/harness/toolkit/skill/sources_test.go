package skill

import (
	"context"
	"strings"
	"testing"
)

// mapLoader is a loader over memory, which is what a tenant's skillspace looks
// like to the capability: a set of names with metadata and bodies.
type mapLoader map[string]Skill

func (m mapLoader) List(context.Context) ([]Skill, error) {
	out := make([]Skill, 0, len(m))
	for _, item := range m {
		out = append(out, item)
	}
	return out, nil
}

func (m mapLoader) Load(_ context.Context, name string) (Skill, error) {
	if item, ok := m[name]; ok {
		return item, nil
	}
	return Skill{}, errNotFound(name)
}

type notFound string

func (n notFound) Error() string { return "not found: " + string(n) }

func errNotFound(name string) error { return notFound(name) }

// TestBundledCarriesTheRetrievalPolicy is the reason the built-in set exists: a
// deployment with no skills directory still runs under the retrieval policy,
// and that policy is a file (bundled/retrieval/SKILL.md), not a constant.
func TestBundledCarriesTheRetrievalPolicy(t *testing.T) {
	loader := Bundled()
	skills, err := loader.List(context.Background())
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	var found *Skill
	for index := range skills {
		if skills[index].Name == "retrieval" {
			found = &skills[index]
		}
	}
	if found == nil {
		t.Fatalf("bundled skills = %v, want retrieval", skills)
	}
	if !found.AutoLoad {
		t.Error("retrieval must auto_load: a policy the model has to ask for is a policy it will not read")
	}
	if found.Description == "" || found.WhenToUse == "" {
		t.Errorf("retrieval manifest is incomplete: %+v", found)
	}

	loaded, err := loader.Load(context.Background(), "retrieval")
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if strings.Contains(loaded.Body, "auto_load:") {
		t.Error("body still carries the front matter")
	}
	for _, tool := range []string{"grep_chunks", "list_chunks", "navigate_tree", "search_chunks"} {
		if !strings.Contains(loaded.Body, tool) {
			t.Errorf("policy does not mention %s", tool)
		}
	}
}

// TestCatalogToolsNeedSomethingToDiscover: with only auto_load policies the
// catalog answers "no skills installed", so the discovery tools are not offered
// at all — their schemas and the round spent calling them buy nothing.
func TestCatalogToolsNeedSomethingToDiscover(t *testing.T) {
	autoOnly := New(context.Background(), mapLoader{
		"policy": {Name: "policy", AutoLoad: true, Body: "always apply this"},
	})
	if tools := autoOnly.Tools(); len(tools) != 0 {
		t.Fatalf("tools = %v, want none when every skill is auto_load", tools)
	}

	withProcedure := New(context.Background(), mapLoader{
		"policy":    {Name: "policy", AutoLoad: true, Body: "always apply this"},
		"procedure": {Name: "procedure", Body: "load me on demand"},
	})
	offered := make(map[string]struct{}, 2)
	for _, spec := range withProcedure.Tools() {
		offered[spec.Name] = struct{}{}
	}
	for _, want := range []string{"skill_list", "skill_load"} {
		if _, ok := offered[want]; !ok {
			t.Errorf("tools = %v, want %s once a procedure is installed", offered, want)
		}
	}
}

// TestBundledSetIsPolicyWithoutCatalog pins the deployment's own skill set: the
// retrieval policy stays in the system prompt, and nothing else is offered. A
// chat turn measured 16 rounds this way is the baseline this guards.
func TestBundledSetIsPolicyWithoutCatalog(t *testing.T) {
	capability := New(context.Background(), MergedLoader{Layers: []Loader{Bundled()}})
	if tools := capability.Tools(); len(tools) != 0 {
		t.Fatalf("tools = %v, want none for the bundled set alone", tools)
	}
	note := capability.SystemNote(context.Background())
	if !strings.Contains(note, "## Retrieval tools") || !strings.Contains(note, "list_chunks") {
		t.Fatalf("note lost the bundled retrieval policy: %q", note)
	}
}

func TestBundledRejectsTraversal(t *testing.T) {
	for _, name := range []string{"../bundled/retrieval", "a/b", "", ".."} {
		if _, err := Bundled().Load(context.Background(), name); err == nil {
			t.Errorf("Load(%q) succeeded", name)
		}
	}
}

// TestMergedLoaderLayersOverBundled: an installed skill with a bundled name
// replaces it, which is how a tenant changes policy without a rebuild.
func TestMergedLoaderLayersOverBundled(t *testing.T) {
	merged := MergedLoader{Layers: []Loader{
		mapLoader{"retrieval": {Name: "retrieval", Description: "bundled", AutoLoad: true, Body: "bundled body"}},
		mapLoader{"retrieval": {Name: "retrieval", Description: "tenant", AutoLoad: true, Body: "tenant body"}},
	}}

	skills, err := merged.List(context.Background())
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	if len(skills) != 1 {
		t.Fatalf("skills = %v, want one entry", skills)
	}
	if skills[0].Description != "tenant" {
		t.Errorf("description = %q, want the later layer", skills[0].Description)
	}

	loaded, err := merged.Load(context.Background(), "retrieval")
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if loaded.Body != "tenant body" {
		t.Errorf("body = %q, want the later layer", loaded.Body)
	}
}

// TestMergedLoaderSurvivesABrokenLayer: one unreachable source must not take
// the others down.
func TestMergedLoaderSurvivesABrokenLayer(t *testing.T) {
	merged := MergedLoader{Layers: []Loader{
		mapLoader{"retrieval": {Name: "retrieval", AutoLoad: true, Body: "bundled"}},
		failingLoader{},
	}}
	skills, err := merged.List(context.Background())
	if err != nil || len(skills) != 1 {
		t.Fatalf("skills = %v, err = %v", skills, err)
	}
	loaded, err := merged.Load(context.Background(), "retrieval")
	if err != nil || loaded.Body != "bundled" {
		t.Fatalf("load = %+v, err = %v", loaded, err)
	}
}

type failingLoader struct{}

func (failingLoader) List(context.Context) ([]Skill, error) { return nil, errNotFound("boom") }
func (failingLoader) Load(context.Context, string) (Skill, error) {
	return Skill{}, errNotFound("boom")
}

// TestSelectedLoaderIsACatalogGate: an agent's skillspace is a catalog, not a
// prompt. Selecting nothing must offer nothing, and selecting a name must not
// expose its neighbours.
func TestSelectedLoaderIsACatalogGate(t *testing.T) {
	inner := mapLoader{
		"memo":    {Name: "memo", Description: "how to keep notes"},
		"finance": {Name: "finance", Description: "how to read a balance sheet"},
	}

	empty := SelectedLoader{Inner: inner}
	if skills, err := empty.List(context.Background()); err != nil || len(skills) != 0 {
		t.Fatalf("empty selection = %v, err = %v", skills, err)
	}

	selected := SelectedLoader{Inner: inner, Names: []string{"memo"}}
	skills, err := selected.List(context.Background())
	if err != nil || len(skills) != 1 || skills[0].Name != "memo" {
		t.Fatalf("selection = %v, err = %v", skills, err)
	}
	if _, err := selected.Load(context.Background(), "finance"); err == nil {
		t.Error("loading an unselected skill succeeded")
	}
	if item, err := selected.Load(context.Background(), "memo"); err != nil || item.Name != "memo" {
		t.Errorf("load memo = %+v, err = %v", item, err)
	}
}

// TestSystemNoteCarriesOnlyAutoLoadedBodies: the note is paid for on every
// round, so a procedure must stay behind skill_load.
func TestSystemNoteCarriesOnlyAutoLoadedBodies(t *testing.T) {
	capability := New(context.Background(), mapLoader{
		"policy":    {Name: "policy", AutoLoad: true, Body: "always apply this"},
		"procedure": {Name: "procedure", Body: "load me on demand"},
	})

	note := capability.SystemNote(context.Background())
	if !strings.Contains(note, "always apply this") {
		t.Errorf("note = %q, want the auto_load body", note)
	}
	if strings.Contains(note, "load me on demand") {
		t.Errorf("note = %q, must not inline a procedure", note)
	}
}

// TestSystemNoteIsBoundedAndStable: a careless skill cannot crowd the window
// out, and the order does not depend on map iteration.
func TestSystemNoteIsBoundedAndStable(t *testing.T) {
	huge := strings.Repeat("x", MaxSystemNoteChars*2)
	capability := New(context.Background(), mapLoader{
		"a": {Name: "a", AutoLoad: true, Body: huge},
		"b": {Name: "b", AutoLoad: true, Body: huge},
	})

	note := capability.SystemNote(context.Background())
	if len(note) > MaxSystemNoteChars {
		t.Fatalf("note = %d chars, want <= %d", len(note), MaxSystemNoteChars)
	}
	if note != capability.SystemNote(context.Background()) {
		t.Error("note is not stable across calls")
	}
}
