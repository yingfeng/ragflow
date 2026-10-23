package rag

import (
	"context"
	"strings"
	"testing"
)

// A tool whose schema is declared but not rendered reaches the model with no
// arguments at all: the model then invents an envelope such as
// {"queries":{"item":[...]}} and every call fails to parse. The spec must carry
// the tool's real JSON Schema.
// TestParseEvidenceCarriesThePassage is what makes the evidence pool usable as
// memory: the entry has to say WHAT the chunk contains, not only which document
// it came from. A pool of "三国演义" forty times over tells a run nothing about
// which names it has already found.
func TestParseEvidenceCarriesThePassage(t *testing.T) {
	cases := []struct {
		name    string
		content string
		want    string
	}{
		{
			name:    "grep matched window",
			content: `<search_results><chunk chunk_id="c1" doc_id="d1" doc_name="三国演义"><match_snippet>关公手起一刀，斩夏侯存于马下</match_snippet></chunk></search_results>`,
			want:    "三国演义 · 关公手起一刀，斩夏侯存于马下",
		},
		{
			name:    "list_chunks body text",
			content: `<chunk chunk_id="c2" doc_id="d1" doc_name="三国演义"><content>云长曰：吾今斩颜良，以报曹公</content></chunk>`,
			want:    "三国演义 · 云长曰：吾今斩颜良，以报曹公",
		},
		{
			name:    "self closing element falls back to the document name",
			content: `<chunk chunk_id="c3" doc_id="d1" doc_name="三国演义"/>`,
			want:    "三国演义",
		},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			evidences := ParseEvidence(testCase.content)
			if len(evidences) != 1 {
				t.Fatalf("evidences = %d, want 1 (%+v)", len(evidences), evidences)
			}
			if evidences[0].Snippet != testCase.want {
				t.Fatalf("snippet = %q, want %q", evidences[0].Snippet, testCase.want)
			}
		})
	}
}

// TestParseEvidenceTrimsLongPassages keeps one pool line bounded: the pool is
// re-sent every round, so it cannot carry whole passages.
func TestParseEvidenceTrimsLongPassages(t *testing.T) {
	body := strings.Repeat("斩", evidenceExcerptRunes*2)
	evidences := ParseEvidence(`<chunk chunk_id="c1" doc_id="d1">` + body + `</chunk>`)
	if len(evidences) != 1 {
		t.Fatalf("evidences = %d, want 1", len(evidences))
	}
	if runes := []rune(evidences[0].Snippet); len(runes) > evidenceExcerptRunes+1 {
		t.Fatalf("snippet has %d runes, want at most %d plus the ellipsis", len(runes), evidenceExcerptRunes)
	}
}

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
