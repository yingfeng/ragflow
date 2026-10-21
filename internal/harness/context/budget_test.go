package agentctx

import (
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

func TestKeepRecentToolResultsShrinksWithPressure(t *testing.T) {
	if got := keepRecentToolResults(TierNormal, 4); got != 6 {
		t.Errorf("normal = %d, want 6", got)
	}
	if got := keepRecentToolResults(TierCompactHistory, 6); got != 2 {
		t.Errorf("compact = %d, want 2", got)
	}
	if got := keepRecentToolResults(TierAggressivePrune, 6); got != 1 {
		t.Errorf("prune = %d, want 1", got)
	}
}

func TestBuildKeepsRecentToolResultsAtNormalPressure(t *testing.T) {
	builder := New(Config{MaxInputTokens: 100_000})
	history := []model.Message{{Role: model.RoleUser, Content: "go"}}
	for i := 0; i < 8; i++ {
		history = append(history, model.Message{Role: model.RoleTool, Name: "search", Content: fmt.Sprintf("result-%d", i)})
	}

	out := builder.Build("system", nil, history)

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
	if kept != 6 || cleared != 2 {
		t.Fatalf("kept=%d cleared=%d, want 6/2", kept, cleared)
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

	out := builder.Build("system", nil, history)

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
	out := builder.Build("system", nil, []model.Message{
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

	out := builder.Build("system", nil, []model.Message{
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

	out := builder.Build("system", nil, []model.Message{{Role: model.RoleUser, Content: "hello"}})
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
