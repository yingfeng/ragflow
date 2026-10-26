package loop

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	"ragflow/internal/harness/agent/model"
	agentctx "ragflow/internal/harness/context"
	"ragflow/internal/harness/toolkit"
)

// A re-issued query is one entry with a count, not two entries: four identical
// lines read like four different attempts and hide the repetition the log exists
// to expose.
func TestRecordSearchCollapsesRepeats(t *testing.T) {
	state := &State{Entry: EntryMeta{Round: 3}}
	result := toolkit.Result{Evidence: []toolkit.Evidence{{Ref: "c1"}, {Ref: "c2"}}}
	recordSearch(state, PendingCall{Tool: "grep_chunks", Arguments: `{"query":"关羽.*斩","intent":"first try"}`}, result)
	// The reworded intent is the point: the model changes its own narration and
	// re-sends the same regex, which is exactly what the old idempotency key
	// treated as new work.
	recordSearch(state, PendingCall{Tool: "grep_chunks", Arguments: `{"query":"关羽.*斩","intent":"different words"}`}, result)

	if len(state.Searches) != 1 {
		t.Fatalf("searches = %+v, want one entry for one query", state.Searches)
	}
	if state.Searches[0].Count != 2 {
		t.Fatalf("count = %d, want 2", state.Searches[0].Count)
	}
	if state.Searches[0].Hits != 2 {
		t.Fatalf("hits = %d, want the last call's 2", state.Searches[0].Hits)
	}
}

func TestSearchQueryReadsEveryCallShape(t *testing.T) {
	cases := []struct {
		name      string
		arguments string
		want      string
	}{
		{"single query, whitespace collapsed", `{"query":"  华雄   纪灵 "}`, "华雄 纪灵"},
		{"several queries", `{"queries":["a","b"]}`, "a | b"},
		{"anchored read", `{"doc_id":"d1","anchor_chunk_ids":["c1","c2"]}`, "d1 chunks c1,c2"},
		{"no query at all", `{}`, ""},
		{"malformed", `not json`, ""},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			if got := searchQuery(testCase.arguments); got != testCase.want {
				t.Fatalf("searchQuery(%s) = %q, want %q", testCase.arguments, got, testCase.want)
			}
		})
	}
}

// The section is a record of work, never of findings: it may name a query and a
// chunk id, and it must not name anything the run found. A findings list here is
// what stopped an earlier run at 8 rounds with 9 names.
func TestRenderSearchLogCarriesWorkNotFindings(t *testing.T) {
	state := &State{
		Searches:   []SearchLog{{Tool: "grep_chunks", Query: "关羽.*斩", Hits: 9, Count: 3}},
		ReadChunks: []string{"c1", "c2"},
	}
	rendered := renderSearchLog(state)
	for _, want := range []string{"## Search log", "grep_chunks", "关羽.*斩", "ran 3 times", "9 hit(s)", "c1, c2"} {
		if !strings.Contains(rendered, want) {
			t.Fatalf("section missing %q:\n%s", want, rendered)
		}
	}
	if strings.Contains(rendered, "华雄") {
		t.Fatalf("section leaked a finding:\n%s", rendered)
	}
}

func TestRenderSearchLogKeepsTheRecentWindow(t *testing.T) {
	state := &State{}
	for index := 0; index < searchLogShown+3; index++ {
		state.Searches = append(state.Searches, SearchLog{Tool: "grep_chunks", Query: fmt.Sprintf("q%d", index), Count: 1})
	}
	rendered := renderSearchLog(state)
	if !strings.Contains(rendered, "3 earlier queries not shown") {
		t.Fatalf("section did not announce the dropped queries:\n%s", rendered)
	}
	if strings.Contains(rendered, "q0") {
		t.Fatalf("section kept the oldest query:\n%s", rendered)
	}
}

func TestRenderSearchLogWithoutHistory(t *testing.T) {
	if rendered := renderSearchLog(&State{}); rendered != "" {
		t.Fatalf("empty run rendered a section: %q", rendered)
	}
}

// A renderer that works in isolation proves nothing: the log has to reach the
// prompt on the round after the search. Compaction is what erases the tool
// messages it is derived from, so this checks the wiring, not the formatting.
// The search log is a repair for lost history, so it appears exactly when the
// builder dropped some. Injected into a transcript that lost nothing it reads as
// a list of what has not been covered yet: a ten-round run answered only after
// five rounds of "one final sweep", each restarted by this section.
func TestLoop_SearchLogOnlyAfterCompaction(t *testing.T) {
	run := func(t *testing.T, cfg Config, script ...model.Step) []model.Request {
		t.Helper()
		if len(script) == 0 {
			script = []model.Step{
				model.CallTool("c1", "rag_search", `{"query":"关羽.*斩","intent":"first"}`),
				model.CallTool("c2", "rag_search", `{"query":"关羽.*杀","intent":"second"}`),
				model.Reply("共 14 人"),
			}
		}
		mdl := model.NewFake(script...)
		cfg.Deps.Model = mdl
		runner := buildRunner(t, mdl, &sameChunkCap{}, cfg)
		if _, err := runner.Invoke(context.Background(), &Input{RunID: "run-log", UserMessage: "关羽杀了多少人"}, nil); err != nil {
			t.Fatalf("invoke: %v", err)
		}
		return mdl.Requests()
	}

	// Nothing was dropped: the transcript itself still carries both queries.
	for index, request := range run(t, Config{}) {
		if requestMentions(request, "## Search log") {
			t.Fatalf("request %d carries a search log although nothing was compacted", index)
		}
	}

	// A prompt-too-long rejection is what makes the builder drop the earlier
	// rounds, and that is when naming the queries is the only way to keep from
	// paying for them twice. An estimate cannot cause it: a count reports.
	//
	// The refusal lands on the second call, after one search has already run, so
	// the log has something to list. The log is injected into the retry's prompt
	// and nowhere else: it repairs one compaction, it is not a per-round
	// instruction to keep searching.
	requests := run(t, Config{Deps: Deps{Builder: agentctx.New(agentctx.Config{})}},
		model.CallTool("c1", "rag_search", `{"query":"关羽.*斩","intent":"first"}`),
		model.Fail(errors.New("prompt is too long: 300000 tokens")),
		model.CallTool("c2", "rag_search", `{"query":"关羽.*杀","intent":"second"}`),
		model.Reply("共 14 人"),
	)
	logged := -1
	for index, request := range requests {
		if requestMentions(request, "## Search log") {
			logged = index
			break
		}
	}
	if logged < 0 {
		t.Fatal("no request carries a search log after the refusal; the run would repeat its own queries")
	}
	for _, want := range []string{"关羽.*斩", "chunk-1"} {
		if !requestMentions(requests[logged], want) {
			t.Fatalf("search log does not carry %q", want)
		}
	}
	for index := logged + 1; index < len(requests); index++ {
		if requestMentions(requests[index], "## Search log") {
			t.Fatalf("request %d carries the search log again: it must repair one compaction, not steer every round", index)
		}
	}
}

// The same query with a reworded intent is a repeat, not a new attempt: that is
// the exact shape that let a run spend twenty rounds on one regex. The third copy
// makes the round flat at once, without waiting for the evidence window.
func TestLoop_RepeatedQueryCountsAsFlat(t *testing.T) {
	capability := &sameChunkCap{}
	mdl := model.NewFake(
		model.CallTool("c1", "rag_search", `{"query":"关羽","intent":"first"}`),
		model.CallTool("c2", "rag_search", `{"query":"关羽","intent":"second"}`),
		model.CallTool("c3", "rag_search", `{"query":"关羽","intent":"third"}`),
		model.Reply("共 14 人"),
	)
	runner := buildRunner(t, mdl, capability, Config{})

	outcome, err := runner.Invoke(context.Background(), &Input{RunID: "run-repeat", UserMessage: "关羽杀了多少人"}, nil)
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if outcome.FinalState.Nudges != 1 {
		t.Fatalf("nudges = %d, want the repeat reported on the third identical query", outcome.FinalState.Nudges)
	}
	if len(outcome.FinalState.Searches) != 1 || outcome.FinalState.Searches[0].Count != 3 {
		t.Fatalf("searches = %+v, want one entry with count 3", outcome.FinalState.Searches)
	}
}

func TestNoteReadChunksDeduplicates(t *testing.T) {
	state := &State{}
	noteReadChunks(state, []toolkit.Evidence{{Ref: "c1"}, {Ref: "c2"}, {Ref: "c1"}})
	noteReadChunks(state, []toolkit.Evidence{{Ref: "c2"}, {Ref: " "}})
	if len(state.ReadChunks) != 2 || state.ReadChunks[0] != "c1" || state.ReadChunks[1] != "c2" {
		t.Fatalf("read chunks = %v, want c1 and c2 in first-seen order", state.ReadChunks)
	}
}
