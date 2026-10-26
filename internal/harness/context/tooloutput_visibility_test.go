package agentctx

import (
	"context"
	"strings"
	"testing"

	"ragflow/internal/harness/agent/model"
)

// toolHistory is a transcript with many tool results: the shape of an
// enumeration run that greps its way through a corpus.
func toolHistory(count int) []model.Message {
	history := make([]model.Message, 0, count*2)
	for index := 0; index < count; index++ {
		history = append(history,
			model.Message{
				Role:      model.RoleAssistant,
				ToolCalls: []model.ToolCall{{ID: "c", Name: "grep_chunks", Arguments: `{"query":"关羽"}`}},
			},
			model.Message{Role: model.RoleTool, ToolCallID: "c", Name: "grep_chunks", Content: "passage that names a general"},
		)
	}
	return history
}

// Evidence the model already read stays in the window while the window has room.
// Clearing it at zero pressure made a 29-round run answer from the last six
// results it could still see.
func TestBuild_KeepsToolOutputWhileTheWindowHasRoom(t *testing.T) {
	builder := New(Config{
		MaxInputTokens: 100_000,
		CountTokens:    func(text string) int { return len(text)/4 + 1 },
	})
	history := toolHistory(12)

	prompt := builder.Build(context.Background(), "system", nil, history)

	for _, message := range prompt {
		if message.Role != model.RoleTool {
			continue
		}
		if strings.Contains(message.Content, "cleared") {
			t.Fatalf("tool output was cleared at %.1f%% of the budget", 0.0)
		}
		if message.Content != "passage that names a general" {
			t.Fatalf("tool content = %q", message.Content)
		}
	}
}

// Once the transcript has to shrink (a prompt-too-long rejection escalated the
// tier, which is now the only thing that does), the oldest results go, and the
// placeholder names the tool so the model can fetch the result again instead of
// assuming it was empty.
func TestBuild_ClearsOldToolOutputWhenTheTranscriptMustShrink(t *testing.T) {
	builder := New(Config{
		MaxInputTokens:      60_000,
		KeepToolResultBytes: 8 << 10,
		CountTokens:         func(text string) int { return len(text)/4 + 1 },
	})
	history := toolHistory(12)
	// Real-size results: the byte budget only bites on output that is actually
	// large, which is what a retrieval result is.
	for index := range history {
		if history[index].Role == model.RoleTool {
			history[index].Content = strings.Repeat("关羽提刀出阵，斩了颜良文丑。", 200)
		}
	}

	prompt := builder.BuildWithTier(context.Background(), "system", nil, history, TierCompactHistory)

	cleared := 0
	verbatim := 0
	pruned := false
	for _, message := range prompt {
		if message.Role != model.RoleTool {
			if strings.HasPrefix(message.Content, "[earlier context pruned") {
				pruned = true
			}
			continue
		}
		if strings.Contains(message.Content, "cleared") {
			cleared++
			if !strings.Contains(message.Content, "grep_chunks") {
				t.Fatalf("placeholder does not name the tool: %q", message.Content)
			}
			continue
		}
		verbatim++
	}
	// The contract is that a window this small does not keep twelve results
	// verbatim. How it frees them is tier-dependent: trimming replaces each one
	// with a placeholder, and critical pressure then windows the transcript, which
	// can drop those placeholders again. Both are correct outcomes; keeping all
	// twelve is not.
	if verbatim >= 12 {
		t.Fatalf("pressure kept all %d tool results verbatim; the window cannot hold them", verbatim)
	}
	if cleared == 0 && !pruned {
		t.Fatalf("pressure dropped tool output with no signal to the model: %+v", prompt)
	}
}

// The fold summary carries the task thread, so the critical-pressure prune keeps
// it: a window of recent reads with no question in it is how a run answers the
// last passage it read.
func TestBuild_PruneKeepsTheFoldSummary(t *testing.T) {
	builder := New(Config{
		MaxInputTokens: 4_000,
		CountTokens:    func(text string) int { return len(text)/4 + 1 },
	})
	summary := model.Message{Role: model.RoleUser, Content: foldSummaryPrefix + "4 messages]\nthe question is a count of generals"}
	history := append([]model.Message{summary}, toolHistory(24)...)

	prompt := builder.Build(context.Background(), "system", nil, history)

	if len(prompt) < 2 {
		t.Fatalf("prompt = %+v", prompt)
	}
	kept := false
	for _, message := range prompt {
		if strings.Contains(message.Content, "count of generals") {
			kept = true
		}
	}
	if !kept {
		t.Fatalf("prune dropped the summary: %+v", prompt)
	}
}
