package loop

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"

	"ragflow/internal/harness/agent/model"
	"ragflow/internal/harness/toolkit"
)

// streamingModel emits fixed chunks and reports a full response, like a provider
// whose deltas have already been sent.
type streamingModel struct {
	mu       sync.Mutex
	chunks   []string
	thoughts []string
	failNext error
	failWith error
	calls    int
	// emitBeforeFail decides whether a failing attempt still streamed text.
	emitBeforeFail bool
}

func (m *streamingModel) Complete(ctx context.Context, req model.Request) (model.Response, error) {
	return m.Stream(ctx, req, nil)
}

func (m *streamingModel) Stream(ctx context.Context, req model.Request, sender model.StreamSender) (model.Response, error) {
	m.mu.Lock()
	m.calls++
	attempt := m.calls
	m.mu.Unlock()

	if attempt == 1 && m.failNext != nil {
		if m.emitBeforeFail && sender != nil {
			partial := "par"
			_ = sender(&partial, nil)
		}
		return model.Response{}, m.failNext
	}
	// Reasoning arrives before the visible text, exactly as a provider reports
	// it: the model thinks, then writes.
	for _, thought := range m.thoughts {
		value := thought
		if sender != nil {
			if err := sender(nil, &value); err != nil {
				return model.Response{}, err
			}
		}
	}
	answer := strings.Builder{}
	for _, chunk := range m.chunks {
		value := chunk
		if sender != nil {
			if err := sender(&value, nil); err != nil {
				return model.Response{}, err
			}
		}
		answer.WriteString(value)
	}
	return model.Response{
		Message:    model.Message{Role: model.RoleAssistant, Content: answer.String()},
		StopReason: "stop",
	}, nil
}

func TestStreamingDeltasReachTheSink(t *testing.T) {
	mdl := &streamingModel{chunks: []string{"Hel", "lo"}}
	runner := buildRunner(t, mdl, nil, Config{})

	var mu sync.Mutex
	var deltas []string
	ctx := WithDeltaSink(t.Context(), func(text string) {
		mu.Lock()
		deltas = append(deltas, text)
		mu.Unlock()
	})

	outcome, err := runner.Invoke(ctx, &Input{RunID: "run-stream", UserMessage: "hi"}, nil)
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if strings.Join(deltas, "") != "Hello" {
		t.Fatalf("deltas = %v", deltas)
	}
	if outcome.Text != "Hello" {
		t.Fatalf("outcome text = %q", outcome.Text)
	}
}

// Reasoning is working material, not answer text: it reaches the panel that
// explains a round and never the answer area. It used to be dropped in the
// adapter outright — the sender accepted only its first argument — which is why
// a run could search twenty rounds with nothing on screen about why.
func TestStreamingReasoningReachesTheWorkingSink(t *testing.T) {
	mdl := &streamingModel{chunks: []string{"答案"}, thoughts: []string{"先查华雄", "再看五关"}}
	runner := buildRunner(t, mdl, nil, Config{})

	var mu sync.Mutex
	var thoughts []string
	var answers []string
	ctx := WithDeltaSink(t.Context(), func(text string) {
		mu.Lock()
		answers = append(answers, text)
		mu.Unlock()
	})
	ctx = WithReasoningSink(ctx, func(text string) {
		mu.Lock()
		thoughts = append(thoughts, text)
		mu.Unlock()
	})

	outcome, err := runner.Invoke(ctx, &Input{RunID: "run-thought", UserMessage: "hi"}, nil)
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if got := strings.Join(thoughts, ""); got != "先查华雄再看五关" {
		t.Fatalf("thoughts = %q, want the model's reasoning", got)
	}
	if got := strings.Join(answers, ""); got != "答案" {
		t.Fatalf("answers = %q, want only the visible text", got)
	}
	if outcome.Text != "答案" {
		t.Fatalf("outcome text = %q", outcome.Text)
	}
}

func TestStreamingWithoutSinkDoesNotStream(t *testing.T) {
	mdl := &streamingModel{chunks: []string{"Hello"}}
	runner := buildRunner(t, mdl, nil, Config{})

	outcome, err := runner.Invoke(t.Context(), &Input{RunID: "run-nosink", UserMessage: "hi"}, nil)
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if outcome.Text != "Hello" {
		t.Fatalf("outcome text = %q", outcome.Text)
	}
}

func TestStreamingRetriesOnlyBeforeTheFirstDelta(t *testing.T) {
	// Nothing was shown: a retry is invisible to the user.
	mdl := &streamingModel{chunks: []string{"ok"}, failNext: errors.New("rate limit exceeded")}
	runner := buildRunner(t, mdl, nil, Config{})
	outcome, err := runner.Invoke(WithDeltaSink(t.Context(), func(string) {}), &Input{RunID: "run-retry", UserMessage: "hi"}, nil)
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if mdl.calls != 2 {
		t.Fatalf("calls = %d, want a retry", mdl.calls)
	}
	if outcome.Text != "ok" {
		t.Fatalf("outcome text = %q", outcome.Text)
	}
}

func TestStreamingDoesNotRetryAfterShowingText(t *testing.T) {
	mdl := &streamingModel{
		chunks:         []string{"ok"},
		failNext:       errors.New("rate limit exceeded"),
		emitBeforeFail: true,
	}
	runner := buildRunner(t, mdl, nil, Config{})

	var seen []string
	_, err := runner.Invoke(WithDeltaSink(t.Context(), func(text string) { seen = append(seen, text) }),
		&Input{RunID: "run-noretry", UserMessage: "hi"}, nil)
	if err == nil {
		t.Fatal("expected the failure to surface")
	}
	if mdl.calls != 1 {
		t.Fatalf("calls = %d, want no retry after text was shown", mdl.calls)
	}
	if len(seen) != 1 || seen[0] != "par" {
		t.Fatalf("deltas = %v", seen)
	}
}

func TestProgressReportsToolLifecycle(t *testing.T) {
	capability := &fakeCap{tool: "fake_tool", result: toolkit.Result{Content: "hit"}}
	mdl := model.NewFake(
		model.CallTool("c1", "fake_tool", `{"query":"x"}`),
		model.Reply("done"),
	)
	runner := buildRunner(t, mdl, capability, Config{Deps: Deps{Policy: noProgressPolicy()}})

	var events []Progress
	ctx := WithProgressSink(t.Context(), func(progress Progress) { events = append(events, progress) })

	if _, err := runner.Invoke(ctx, &Input{RunID: "run-progress", UserMessage: "go"}, nil); err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if len(events) != 2 {
		t.Fatalf("progress events = %+v", events)
	}
	if events[0].Phase != ProgressStart || events[1].Phase != ProgressEnd {
		t.Fatalf("phases = %s, %s", events[0].Phase, events[1].Phase)
	}
	if events[0].Tool != "fake_tool" || !strings.Contains(events[0].Summary, "query") {
		t.Fatalf("start event = %+v", events[0])
	}
	if events[1].IsError {
		t.Fatalf("end event reported an error: %+v", events[1])
	}
}

func TestSummarizeArgumentsBoundsTheExcerpt(t *testing.T) {
	long := `{"query":"` + strings.Repeat("a", 400) + `"}`
	summary := summarizeArguments(long)
	if len(summary) > progressSummaryLimit+3 {
		t.Fatalf("summary length = %d", len(summary))
	}
	if strings.Contains(summary, "\n") {
		t.Fatal("summary is not single-line")
	}
	if summarizeArguments("") != "" {
		t.Fatal("empty arguments produced a summary")
	}
	// Non-JSON arguments still produce a readable excerpt.
	if got := summarizeArguments("plain text"); got != "plain text" {
		t.Fatalf("summary = %q", got)
	}
}
