package agentctx

import (
	"context"
	"strings"
	"testing"

	"ragflow/internal/harness/agent/model"
)

// countingTokenizer stands in for a tokenizer whose rate on the corpus differs
// from the provider's. charsPerToken = 1.7 is what the cl100k counter measured
// on the Chinese corpus, and the provider counted the same text at ~4x less.
func countingTokenizer(bytesPerToken int) func(string) int {
	return func(text string) int {
		if text == "" {
			return 0
		}
		return len(text)/bytesPerToken + 1
	}
}

// The budget is calibrated from the prompt size the provider reports, so a
// local tokenizer that over-counts a non-English corpus reports a fill near the
// provider's. Measured live: a 30k-token prompt read as ~120k locally, a 4x
// difference that made every pressure number meaningless.
//
// The estimate is a report rather than a decision, so this pins the report and
// not a compaction side effect: after one observation the estimate tracks the
// provider's count of the same prompt.
func TestObservedPromptSizeCalibratesTheBudget(t *testing.T) {
	builder := New(Config{CountTokens: countingTokenizer(1)})
	chunk := strings.Repeat("关羽提刀出阵，斩了颜良文丑。", 1000) // ~36 KB
	history := []model.Message{{Role: model.RoleUser, Content: "三国演义中，关羽杀了多少有姓名的人物？"}}
	for round := 0; round < 4; round++ {
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
	local := builder.EstimateTokens(prompt)
	if local <= 100_000 {
		t.Fatalf("fixture is wrong: the local count is %d, must be far above the provider's", local)
	}

	// The provider counted this corpus at a quarter of the local rate.
	builder.ObservePromptTokens(local / 4)

	calibrated := builder.EstimateTokens(prompt)
	if diff := calibrated - local/4; diff > local/100 || diff < -local/100 {
		t.Fatalf("calibrated estimate = %d, want about the reported %d", calibrated, local/4)
	}
}

// A report that cannot be a tokenizer difference is ignored: one provider that
// reports only its uncached prefix must not re-scale the whole run.
func TestObservePromptTokensIgnoresImplausibleReports(t *testing.T) {
	builder := New(Config{CountTokens: countingTokenizer(4)})
	builder.BuildWithTier(context.Background(), "system", nil, []model.Message{{Role: model.RoleUser, Content: "q"}}, TierNormal)
	raw := builder.lastRaw

	builder.ObservePromptTokens(raw * 100)
	if builder.scale != 0 {
		t.Fatalf("scale = %v, want no calibration from an implausible report", builder.scale)
	}
	builder.ObservePromptTokens(-1)
	if builder.scale != 0 {
		t.Fatalf("scale = %v, want no calibration without a count", builder.scale)
	}
	builder.ObservePromptTokens(raw / 2)
	want := float64(raw/2) / float64(raw)
	if diff := builder.scale - want; diff > 0.001 || diff < -0.001 {
		t.Fatalf("scale = %v, want %v", builder.scale, want)
	}
}
