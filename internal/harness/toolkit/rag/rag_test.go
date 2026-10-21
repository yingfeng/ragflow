package rag

import (
	"context"
	"testing"
)

// A tool whose schema is declared but not rendered reaches the model with no
// arguments at all: the model then invents an envelope such as
// {"queries":{"item":[...]}} and every call fails to parse. The spec must carry
// the tool's real JSON Schema.
func TestDescribeRendersJSONSchema(t *testing.T) {
	specs, err := describe(context.Background(), map[string]toolFactory{
		"search_bm25_chunks": retrievalTools["search_bm25_chunks"],
		"search_metadata":    retrievalTools["search_metadata"],
	}, "tenant", nil)
	if err != nil {
		t.Fatalf("describe: %v", err)
	}
	if len(specs) != 2 {
		t.Fatalf("specs = %d, want 2", len(specs))
	}

	properties := func(specIndex int) map[string]any {
		t.Helper()
		params := specs[specIndex].Parameters
		if params["type"] != "object" {
			t.Fatalf("%s parameters = %#v, want an object schema", specs[specIndex].Name, params)
		}
		props, ok := params["properties"].(map[string]any)
		if !ok {
			t.Fatalf("%s properties = %#v", specs[specIndex].Name, params["properties"])
		}
		return props
	}

	// describe sorts by name: search_bm25_chunks comes first.
	queries, ok := properties(0)["queries"].(map[string]any)
	if !ok || queries["type"] != "array" {
		t.Fatalf("search_bm25_chunks queries = %#v, want an array", properties(0)["queries"])
	}
	filters, ok := properties(1)["filters"].(map[string]any)
	if !ok || filters["type"] != "array" {
		t.Fatalf("search_metadata filters = %#v, want an array", properties(1)["filters"])
	}
}
