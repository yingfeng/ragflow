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

func TestTierForThresholds(t *testing.T) {
	budget := Budget{MaxInputTokens: 1_000}
	cases := []struct {
		used int
		want Tier
	}{
		{0, TierNormal},
		{599, TierNormal},
		{600, TierTrimSchemas},
		{749, TierTrimSchemas},
		{750, TierCompactHistory},
		{899, TierCompactHistory},
		{900, TierAggressivePrune},
		{5_000, TierAggressivePrune},
	}
	for _, testCase := range cases {
		if got := budget.TierFor(testCase.used); got != testCase.want {
			t.Errorf("TierFor(%d) = %v, want %v", testCase.used, got, testCase.want)
		}
	}
}

// TestToolResultByteBudgetShrinksWithPressure pins the retention budget: normal
// pressure clears nothing, and only real pressure starts dropping the older
// rounds. It is a byte budget rather than a result count because an enumeration
// accumulates its evidence over rounds.
func TestToolResultByteBudgetShrinksWithPressure(t *testing.T) {
	normal := toolResultByteBudget(TierNormal, DefaultKeepToolResultBytes)
	trim := toolResultByteBudget(TierTrimSchemas, DefaultKeepToolResultBytes)
	compact := toolResultByteBudget(TierCompactHistory, DefaultKeepToolResultBytes)
	prune := toolResultByteBudget(TierAggressivePrune, DefaultKeepToolResultBytes)

	if normal != 0 {
		t.Errorf("normal budget = %d, want 0 (no clearing at zero pressure)", normal)
	}
	if !(trim > compact && compact > prune) {
		t.Errorf("budgets must shrink with pressure: trim=%d compact=%d prune=%d", trim, compact, prune)
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

	out := builder.Build(context.Background(), "system", nil, history)

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
	if len(out) != 2 {
		t.Fatalf("output = %d messages, want system + the last message", len(out))
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
