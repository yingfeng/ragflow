package service

import "testing"

// A request for an abstract class must reach the instances typed with its
// descendants: this expansion is the whole point of the declared hierarchy, and
// without it `agent` matches nothing while `person` holds the data.
func TestOntologyStructureFilterExpandsDeclaredDescendants(t *testing.T) {
	declared := parseDeclaredOntology(ontologyTestConfig(
		[]map[string]interface{}{
			{"type": "agent"},
			{"type": "person", "parent": "agent"},
			{"type": "organization", "parent": "agent"},
			{"type": "place"},
		},
		nil,
	))

	filter := OntologyStructureFilter(declared, OntologyStructureConstraints{
		EntityTypes: []string{"agent"},
	})

	got, ok := filter["entity_type_kwd"].([]string)
	if !ok {
		t.Fatalf("entity_type_kwd = %#v, want a string slice", filter["entity_type_kwd"])
	}
	assertSameStrings(t, got, []string{"agent", "person", "organization"})
	if kinds, _ := filter["knowledge_graph_kwd"].([]string); len(kinds) != 1 || kinds[0] != "entity" {
		t.Fatalf("knowledge_graph_kwd = %#v, want [entity]", filter["knowledge_graph_kwd"])
	}
}

// A constraint on a property or an endpoint means compiled assertions; without
// pinning the row kind, `prop_kwd` would also match hypernode rows, which carry
// a leaf attribute in the same column.
func TestOntologyStructureFilterPinsRowKinds(t *testing.T) {
	declared := parseDeclaredOntology(ontologyTestConfig(
		[]map[string]interface{}{{"type": "person"}, {"type": "organization"}},
		[]map[string]interface{}{{"type": "member_of", "kind": "object", "domain": "person", "range": "organization"}},
	))

	filter := OntologyStructureFilter(declared, OntologyStructureConstraints{
		EntityTypes: []string{"person"},
		Properties:  []string{"member_of"},
	})

	if kinds, _ := filter["knowledge_graph_kwd"].([]string); len(kinds) != 2 || kinds[0] != "entity" || kinds[1] != "relation" {
		t.Fatalf("knowledge_graph_kwd = %#v, want both kinds", filter["knowledge_graph_kwd"])
	}
	if props, _ := filter["prop_kwd"].([]string); len(props) != 1 || props[0] != "member_of" {
		t.Fatalf("prop_kwd = %#v, want [member_of]", filter["prop_kwd"])
	}
}

// Endpoint constraints expand the same way the subject side does, and they are
// assertion constraints, not entity ones.
func TestOntologyStructureFilterExpandsEndpointSides(t *testing.T) {
	declared := parseDeclaredOntology(ontologyTestConfig(
		[]map[string]interface{}{
			{"type": "agent"},
			{"type": "person", "parent": "agent"},
		},
		nil,
	))

	filter := OntologyStructureFilter(declared, OntologyStructureConstraints{
		FromTypes: []string{"agent"},
		ToTypes:   []string{"agent"},
	})

	assertSameStrings(t, filter["from_type_kwd"].([]string), []string{"agent", "person"})
	assertSameStrings(t, filter["to_type_kwd"].([]string), []string{"agent", "person"})
	if kinds, _ := filter["knowledge_graph_kwd"].([]string); len(kinds) != 1 || kinds[0] != "relation" {
		t.Fatalf("knowledge_graph_kwd = %#v, want [relation]", filter["knowledge_graph_kwd"])
	}
}

// A class the template does not declare is kept: the caller named it, and a
// filter that matches nothing is more honest than one that silently searches for
// something else.
func TestOntologyStructureFilterKeepsUnknownClass(t *testing.T) {
	declared := parseDeclaredOntology(ontologyTestConfig(
		[]map[string]interface{}{{"type": "person"}},
		nil,
	))

	filter := OntologyStructureFilter(declared, OntologyStructureConstraints{
		EntityTypes: []string{"ghost"},
	})

	assertSameStrings(t, filter["entity_type_kwd"].([]string), []string{"ghost"})
}

// A cyclic parent chain is a template defect the declaration validator rejects,
// but this walk must not hang on a config edited outside the service.
func TestExpandDeclaredClassesTerminatesOnCycle(t *testing.T) {
	declared := parseDeclaredOntology(ontologyTestConfig(
		[]map[string]interface{}{
			{"type": "a", "parent": "b"},
			{"type": "b", "parent": "a"},
		},
		nil,
	))

	got := expandDeclaredClasses([]string{"a"}, declaredChildren(declared))
	assertSameStrings(t, got, []string{"a", "b"})
}

// No constraints must leave retrieval exactly as it was.
func TestOntologyStructureFilterIgnoresNothing(t *testing.T) {
	declared := parseDeclaredOntology(ontologyTestConfig(
		[]map[string]interface{}{{"type": "person"}},
		nil,
	))

	if filter := OntologyStructureFilter(declared, OntologyStructureConstraints{}); filter != nil {
		t.Fatalf("filter = %#v, want nil for an empty request", filter)
	}
	if !(OntologyStructureConstraints{}).IsZero() {
		t.Fatal("an empty constraint set must report itself as zero")
	}
}

func assertSameStrings(t *testing.T, got, want []string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got %v, want %v", got, want)
		}
	}
}
