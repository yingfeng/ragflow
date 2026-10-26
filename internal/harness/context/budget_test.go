package agentctx

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"ragflow/internal/harness/agent/model"
)

func TestEffectiveInputReservesOutputAndHonoursCap(t *testing.T) {
	budget := Budget{ContextWindow: 200_000, InputBudgetRatio: 0.80, OutputReserveTokens: 16_384}
	if got := budget.EffectiveInput(); got != 143_616 {
		t.Fatalf("effective input = %d, want 143616", got)
	}

	budget.MaxInputTokens = 96_000
	if got := budget.EffectiveInput(); got != 96_000 {
		t.Fatalf("effective input = %d, want the absolute cap 96000", got)
	}
}

func TestEffectiveInputSurvivesTinyWindow(t *testing.T) {
	budget := Budget{ContextWindow: 1_000, InputBudgetRatio: 0.80, OutputReserveTokens: 16_384}
	if got := budget.EffectiveInput(); got <= 0 {
		t.Fatalf("effective input = %d, want a positive limit", got)
	}
}

// TestDefaultBudgetKeepsTheWholeWindow pins the production ceiling: the prompt
// may fill the model's window minus the room one completion needs, and nothing
// else. A ratio below 1 is a second reserve over the same gap — on a 200k window
// the two together capped input at 143k, and a run whose prompt had reached
// 129,725 tokens was folded down to 23,526, after which it restarted from
// "locating the key passages" with eight rounds of retrieval thrown away.
func TestDefaultBudgetKeepsTheWholeWindow(t *testing.T) {
	builder := New(Config{})
	if got := builder.budget.MaxInputTokens; got != 0 {
		t.Fatalf("default MaxInputTokens = %d, want 0 (the window is the bound)", got)
	}
	if got := builder.budget.EffectiveInput(); got != 183_616 {
		t.Fatalf("effective input = %d, want the window minus the output reserve 183616", got)
	}
}

// A transcript the window still holds must reach the model untouched: no tier is
// derived from an estimate any more, so only a prompt-too-long rejection
// escalates one.
func TestBuildKeepsTheTranscriptUntilTheProviderRefuses(t *testing.T) {
	builder := New(Config{CountTokens: countingTokenizer(4)})
	chunk := strings.Repeat("关羽提刀出阵，斩了颜良文丑。", 1000) // ~36 KB
	history := []model.Message{{Role: model.RoleUser, Content: "关羽杀了多少人"}}
	const rounds = 2
	for round := 0; round < rounds; round++ {
		history = append(history, model.Message{
			Role:      model.RoleAssistant,
			ToolCalls: []model.ToolCall{{ID: "c", Name: "grep_chunks", Arguments: `{"query":"关羽"}`}},
		})
		for i := 0; i < 3; i++ {
			history = append(history, model.Message{
				Role: model.RoleTool, ToolCallID: "c", Name: "grep_chunks", Content: chunk,
			})
		}
	}

	prompt := builder.BuildWithTier(context.Background(), "system", nil, history, TierNormal)
	if len(prompt) != len(history)+1 {
		t.Fatalf("prompt messages = %d, want the whole transcript %d", len(prompt), len(history)+1)
	}
	if builder.Compacted() {
		t.Fatal("builder reports compaction although the transcript fit")
	}
}

// TestTierForThresholds is gone with TierFor: the tier is now argued by the
// provider's rejection, not by an estimate of the fill.

// enumeration returns a transcript of retrieval rounds, each one assistant
// tool call answered by three passages of the given byte size. It is the shape
// of a counting run: the evidence accumulates across rounds and every earlier
// round still matters at the end.
func enumeration(rounds, chunksPerRound, chunkBytes int) ([]model.Message, string) {
	sentence := "关羽提刀出阵，斩了颜良文丑。" // 42 bytes
	chunk := strings.Repeat(sentence, chunkBytes/len(sentence))
	history := []model.Message{{Role: model.RoleUser, Content: "三国演义中，关羽杀了多少有姓名的人物？"}}
	for round := 0; round < rounds; round++ {
		history = append(history, model.Message{
			Role:      model.RoleAssistant,
			ToolCalls: []model.ToolCall{{ID: "c", Name: "grep_chunks", Arguments: `{"query":"关羽"}`}},
		})
		for i := 0; i < chunksPerRound; i++ {
			history = append(history, model.Message{
				Role: model.RoleTool, ToolCallID: "c", Name: "grep_chunks", Content: chunk,
			})
		}
	}
	return history, chunk
}

// The fold line used to be pressure at 0.75 of the window (150k), and the
// estimate reads about 16% above the provider's own count, so the two together
// folded a prompt the provider had accepted. Measured on run 8320046f: round 8
// reached 129,725 provider tokens while the estimate was already past 150k, the
// fold replaced eight rounds of retrieval with a summary, and round 9 began
// "Starting with locating the key passages" again — the same question, from
// nothing. A prompt at that size must reach the model whole.
func TestBuildKeepsTheMeasuredEnumerationWhole(t *testing.T) {
	builder := New(Config{CountTokens: countingTokenizer(4)})
	history, chunk := enumeration(4, 3, 52_500) // ~157k estimated tokens, ~135k provider

	if got := builder.EstimateTokens(history); got <= 150_000 {
		t.Fatalf("fixture is wrong: estimate = %d, must be past the retired 150k fold line", got)
	}

	prompt := builder.BuildWithTier(context.Background(), "system", nil, history, TierNormal)
	if got, want := len(prompt), len(history)+1; got != want {
		t.Fatalf("prompt messages = %d, want the whole transcript %d: only a provider refusal may fold", got, want)
	}
	verbatim := 0
	for _, message := range prompt {
		if message.Role == model.RoleTool && message.Content == chunk {
			verbatim++
		}
	}
	if verbatim != 12 {
		t.Fatalf("kept %d of 12 passages verbatim, want all of them", verbatim)
	}
	if builder.Compacted() {
		t.Fatal("builder reports compaction although the window still had room")
	}
}

// An estimate above the ceiling must not touch the transcript either. The
// estimate is a tokenizer difference away from the truth, so it can only
// report: the reduction is reserved for a prompt the provider actually
// refused. Measured on run 140e27ad, which is this test's shape — the provider
// accepted 176,810 tokens while the local estimate read ~205k, crossed the
// 183,616 ceiling, and the transcript of ten retrieval rounds was folded to a
// summary; the next round began searching from nothing.
func TestBuildLeavesTheTranscriptAloneAboveTheEstimatedCeiling(t *testing.T) {
	builder := New(Config{CountTokens: countingTokenizer(4)})
	const rounds = 8
	history, chunk := enumeration(rounds, 3, 52_500)

	if got := builder.EstimateTokens(history); got <= builder.budget.EffectiveInput() {
		t.Fatalf("fixture is wrong: estimate = %d, must exceed the ceiling %d", got, builder.budget.EffectiveInput())
	}

	prompt := builder.BuildWithTier(context.Background(), "system", nil, history, TierNormal)
	if got, want := len(prompt), len(history)+1; got != want {
		t.Fatalf("prompt messages = %d, want the whole transcript %d: only a provider refusal may reduce it", got, want)
	}
	verbatim := 0
	for _, message := range prompt {
		if message.Role == model.RoleTool && message.Content == chunk {
			verbatim++
		}
	}
	if verbatim != rounds*3 {
		t.Fatalf("kept %d of %d passages verbatim, want all of them", verbatim, rounds*3)
	}
	if builder.Compacted() {
		t.Fatal("builder reports compaction although the provider was never refused")
	}
}

// TestToolResultByteBudgetShrinksWithPressure pins the retention budget: no
// clearing while the transcript still fits, and only the tiers that already fold
// turns away start dropping the older rounds. It is a byte budget rather than a
// result count because an enumeration accumulates its evidence over rounds.
func TestToolResultByteBudgetShrinksWithPressure(t *testing.T) {
	normal := toolResultByteBudget(TierNormal, DefaultKeepToolResultBytes)
	trim := toolResultByteBudget(TierTrimSchemas, DefaultKeepToolResultBytes)
	compact := toolResultByteBudget(TierCompactHistory, DefaultKeepToolResultBytes)
	prune := toolResultByteBudget(TierAggressivePrune, DefaultKeepToolResultBytes)

	if normal != 0 {
		t.Errorf("normal budget = %d, want 0 (no clearing at zero pressure)", normal)
	}
	// Trimming schemas is not a reason to lose a passage: the transcript still
	// fits, so dropping tool output only makes the model search again.
	if trim != 0 {
		t.Errorf("trim budget = %d, want 0 (evidence is only dropped when turns are folded)", trim)
	}
	if !(compact > prune) {
		t.Errorf("budgets must shrink with pressure: compact=%d prune=%d", compact, prune)
	}
	if prune < 8<<10 {
		t.Errorf("prune budget = %d, must still hold one spilled retrieval page (8 KiB)", prune)
	}
}

// TestToolResultCeilingStaysAboveReadWindow pins the ordering that keeps C from
// cutting what spill just stored: the per-result rendering ceiling must never
// fall below one read_spill window (16 KiB), at any tier.
func TestToolResultCeilingStaysAboveReadWindow(t *testing.T) {
	for _, tier := range []Tier{TierNormal, TierTrimSchemas, TierCompactHistory, TierAggressivePrune} {
		if got := toolResultBudget(tier, 64<<10); got < 16<<10 {
			t.Errorf("tier %v ceiling = %d, must stay at or above the 16 KiB read window", tier, got)
		}
	}
}

// A run must not lose passages it already read while the window still has room.
// Clearing the older results at zero pressure left a 29-round enumeration
// answering from the six results that were still visible.
func TestBuildKeepsEveryToolResultAtNormalPressure(t *testing.T) {
	builder := New(Config{MaxInputTokens: 100_000})
	history := []model.Message{{Role: model.RoleUser, Content: "go"}}
	for i := 0; i < 8; i++ {
		history = append(history, model.Message{Role: model.RoleTool, Name: "search", Content: fmt.Sprintf("result-%d", i)})
	}

	out := builder.Build(context.Background(), "system", nil, history)

	cleared, kept := 0, 0
	for _, message := range out {
		if message.Role != model.RoleTool {
			continue
		}
		if message.Content == ClearedToolPlaceholder {
			cleared++
			continue
		}
		kept++
	}
	if kept != 8 || cleared != 0 {
		t.Fatalf("kept=%d cleared=%d, want every result kept at normal pressure", kept, cleared)
	}
}

func TestBuildPrunesUnderCriticalPressure(t *testing.T) {
	builder := New(Config{
		MaxInputTokens:         300,
		KeepRecentMessages:     2,
		AggressiveKeepMessages: 2,
	})
	history := make([]model.Message, 0, 12)
	for i := 0; i < 12; i++ {
		history = append(history, model.Message{
			Role:    model.RoleUser,
			Content: strings.Repeat("x", 400),
		})
	}
	history = append(history, model.Message{Role: model.RoleUser, Content: "final question"})

	out := builder.BuildWithTier(context.Background(), "system", nil, history, TierAggressivePrune)

	if len(out) > 5 {
		t.Fatalf("pruned output = %d messages, want a small trailing window", len(out))
	}
	found := false
	for _, message := range out {
		if strings.Contains(message.Content, "earlier context pruned") ||
			strings.Contains(message.Content, "earlier turns compacted") {
			found = true
		}
	}
	if !found {
		t.Fatalf("expected a compaction placeholder, got %+v", out)
	}
	if out[len(out)-1].Content != "final question" {
		t.Fatalf("last message = %q, want the final question", out[len(out)-1].Content)
	}
}

func TestBuildTruncatesLongToolResultInTheMiddle(t *testing.T) {
	builder := New(Config{MaxInputTokens: 100_000, MaxToolResultBytes: 200})
	long := "HEAD-" + strings.Repeat("m", 4_000) + "-TAIL"
	out := builder.Build(context.Background(), "system", nil, []model.Message{
		{Role: model.RoleTool, Name: "cat", Content: long},
	})

	toolContent := out[len(out)-1].Content
	if len(toolContent) > 400 {
		t.Fatalf("tool content = %d bytes, want it truncated", len(toolContent))
	}
	if !strings.HasPrefix(toolContent, "HEAD-") || !strings.HasSuffix(toolContent, "-TAIL") {
		t.Fatalf("truncation did not keep head and tail: %q", toolContent)
	}
	if !strings.Contains(toolContent, "truncated") {
		t.Fatalf("truncation was not announced: %q", toolContent)
	}
}

// The injected counter is the only measure of size, and it is used even when it
// says the transcript does not fit: a count reports, it does not decide. Only a
// provider refusal reduces the transcript.
func TestBuildUsesInjectedTokenCounter(t *testing.T) {
	calls := 0
	builder := New(Config{
		MaxInputTokens: 100,
		CountTokens: func(string) int {
			calls++
			return 1_000
		},
	})

	out := builder.Build(context.Background(), "system", nil, []model.Message{
		{Role: model.RoleUser, Content: "one"},
		{Role: model.RoleUser, Content: "two"},
	})
	if calls == 0 {
		t.Fatal("injected counter was not used")
	}
	if len(out) != 3 {
		t.Fatalf("output = %d messages, want system + both messages", len(out))
	}
}

func TestBuildRecoversFromPanickingCounter(t *testing.T) {
	builder := New(Config{
		MaxInputTokens: 100_000,
		CountTokens:    func(string) int { panic("tokenizer not loaded") },
	})

	out := builder.Build(context.Background(), "system", nil, []model.Message{{Role: model.RoleUser, Content: "hello"}})
	if len(out) != 2 {
		t.Fatalf("output = %d messages, want system + user", len(out))
	}
}

func TestTruncateMiddleMarksTheCut(t *testing.T) {
	value := strings.Repeat("a", 500) + "TAIL"
	got := truncateMiddle(value, 200)
	if len(got) > 300 {
		t.Fatalf("length = %d, want near 200", len(got))
	}
	if !strings.HasSuffix(got, "TAIL") {
		t.Fatalf("tail lost: %q", got)
	}
	if !strings.Contains(got, "truncated") {
		t.Fatalf("cut not announced: %q", got)
	}
}
