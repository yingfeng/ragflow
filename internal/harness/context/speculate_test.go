package agentctx

import (
	"strings"
	"sync"
	"testing"

	"ragflow/internal/harness/agent/model"
)

// speculationBuilder counts summarizer calls so a test can tell a prepared
// summary from one written inline.
func speculationBuilder(calls *int) *Builder {
	var mu sync.Mutex
	return New(Config{
		MaxInputTokens:     700,
		KeepRecentMessages: 2,
		Summarize: func(dropped []model.Message) string {
			mu.Lock()
			*calls++
			mu.Unlock()
			return "briefing"
		},
		CountTokens: func(text string) int { return len(text)/4 + 1 },
	})
}

// speculationHistory is a transcript large enough that the next round has to
// fold it.
func speculationHistory() []model.Message {
	body := strings.Repeat("x", 400)
	history := make([]model.Message, 0, 8)
	for index := 0; index < 4; index++ {
		history = append(history,
			model.Message{Role: model.RoleUser, Content: body},
			model.Message{Role: model.RoleAssistant, Content: body},
		)
	}
	return history
}

// The summary is written while the run is still working, and the fold adopts it
// instead of paying for the model call inline.
func TestSpeculator_SummaryIsReadyBeforeTheFold(t *testing.T) {
	calls := 0
	builder := speculationBuilder(&calls)
	history := speculationHistory()

	speculator := NewSpeculator(builder)
	speculator.MaybeStart("system", nil, history)
	speculator.WaitForIdle()

	hint := speculator.Hint()
	if hint == nil {
		t.Fatal("no summary was prepared")
	}
	if hint.Covered <= 0 || hint.Summary != "briefing" {
		t.Fatalf("hint = %+v", hint)
	}
	if calls != 1 {
		t.Fatalf("summarizer calls = %d, want 1 in the background", calls)
	}

	prompt, used := builder.BuildWithFoldHint("system", nil, history, TierCompactHistory, hint)
	if !used {
		t.Fatal("the fold ignored the prepared summary")
	}
	if calls != 1 {
		t.Fatalf("summarizer calls = %d, want no second call on the fold", calls)
	}
	if len(prompt) < 2 || !strings.Contains(prompt[1].Content, "briefing") {
		t.Fatalf("folded prompt = %+v, want the prepared summary", prompt)
	}
}

// A hint that covers far less than the fold needs is not worth folding early:
// the fold summarizes synchronously instead of keeping most of the history.
func TestSpeculator_IgnoresAHintThatCoversTooLittle(t *testing.T) {
	calls := 0
	builder := speculationBuilder(&calls)
	history := speculationHistory()

	prompt, used := builder.BuildWithFoldHint("system", nil, history, TierCompactHistory, &FoldHint{Covered: 1, Summary: "briefing"})
	if used {
		t.Fatal("a one-message hint was adopted for a six-message fold")
	}
	if calls != 1 {
		t.Fatalf("summarizer calls = %d, want the synchronous summary", calls)
	}
	if len(prompt) < 2 || !strings.Contains(prompt[1].Content, "briefing") {
		t.Fatalf("folded prompt = %+v, want the summary", prompt)
	}
}

// A run with no model summarizer folds mechanically, so there is nothing to
// prepare in the background.
func TestSpeculator_DoesNotStartWithoutAModelSummarizer(t *testing.T) {
	builder := New(Config{
		MaxInputTokens:     700,
		KeepRecentMessages: 2,
		CountTokens:        func(text string) int { return len(text)/4 + 1 },
	})
	speculator := NewSpeculator(builder)
	speculator.MaybeStart("system", nil, speculationHistory())
	speculator.WaitForIdle()

	if hint := speculator.Hint(); hint != nil {
		t.Fatalf("hint = %+v, want none for a mechanical fold", hint)
	}
}
