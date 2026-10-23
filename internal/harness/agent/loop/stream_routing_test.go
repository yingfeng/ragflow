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

// workingModel talks before it acts: round one streams "让我先查一下", then asks
// for a tool; round two answers. A model that calls a tool usually emits no
// text, which is exactly why this case went unnoticed until a draft showed up
// in the answer area.
type workingModel struct {
	mu    sync.Mutex
	calls int
}

func (m *workingModel) Complete(ctx context.Context, req model.Request) (model.Response, error) {
	return m.Stream(ctx, req, nil)
}

func (m *workingModel) Stream(_ context.Context, _ model.Request, sender model.StreamSender) (model.Response, error) {
	m.mu.Lock()
	m.calls++
	round := m.calls
	m.mu.Unlock()

	emit := func(text string) {
		if sender != nil {
			_ = sender(&text, nil)
		}
	}
	if round == 1 {
		emit("让我先查一下")
		return model.Response{Message: model.Message{Role: model.RoleAssistant, ToolCalls: []model.ToolCall{{
			ID: "c1", Name: "rag_search", Arguments: `{"query":"关羽"}`,
		}}}, StopReason: "tool_calls"}, nil
	}
	emit("关羽斩了")
	emit("十六人")
	return model.Response{Message: model.Message{Role: model.RoleAssistant, Content: "关羽斩了十六人"}, StopReason: "stop"}, nil
}

// The answer area belongs to the round that ends the turn. Text a round printed
// before calling a tool is working text and goes to the step panel, which is
// what stops one round from overwriting the next one's answer.
func TestLoop_WorkingTextGoesToTheStepPanel(t *testing.T) {
	capability := &fakeCap{
		tool:   "rag_search",
		result: toolkit.Result{Content: "[1] chunk-a", Evidence: []toolkit.Evidence{{Source: "rag", Ref: "chunk-a"}}},
	}
	runner := buildRunner(t, &workingModel{}, capability, Config{})

	var answers []string
	var steps []string
	ctx := WithReasoningSink(WithDeltaSink(t.Context(), func(text string) {
		answers = append(answers, text)
	}), func(text string) {
		steps = append(steps, text)
	})

	outcome, err := runner.Invoke(ctx, &Input{RunID: "run-routing", UserMessage: "关羽杀了多少人"}, nil)
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if got := strings.Join(answers, ""); got != "关羽斩了十六人" {
		t.Fatalf("answer deltas = %q, want only the final round's text", got)
	}
	if len(steps) != 1 || steps[0] != "让我先查一下" {
		t.Fatalf("step text = %q, want the working round's text", steps)
	}
	if outcome.Text != "关羽斩了十六人" {
		t.Fatalf("text = %q", outcome.Text)
	}
}

// A round that produced text and then failed to finish still shows it: the
// retry path treats that text as already delivered.
func TestLoop_HeldTextSurvivesAFailedRound(t *testing.T) {
	mdl := &failingStreamModel{}
	runner := buildRunner(t, mdl, nil, Config{})

	var answers []string
	ctx := WithDeltaSink(t.Context(), func(text string) { answers = append(answers, text) })
	if _, err := runner.Invoke(ctx, &Input{RunID: "run-failed", UserMessage: "hi"}, nil); err == nil {
		t.Fatal("expected the failure to surface")
	}
	if len(answers) != 1 || answers[0] != "半个回答" {
		t.Fatalf("answer deltas = %q, want the text produced before the failure", answers)
	}
	if mdl.calls != 1 {
		t.Fatalf("model calls = %d, want no retry after streaming", mdl.calls)
	}
}

// narratingModel ends its first turn with prose that announces the next step
// instead of answering, then answers. That is the round the policy gives one
// grace turn, not the round that owns the answer area.
type narratingModel struct{ calls int }

func (m *narratingModel) Complete(ctx context.Context, req model.Request) (model.Response, error) {
	return m.Stream(ctx, req, nil)
}

func (m *narratingModel) Stream(_ context.Context, _ model.Request, sender model.StreamSender) (model.Response, error) {
	m.calls++
	emit := func(text string) {
		if sender != nil {
			_ = sender(&text, nil)
		}
	}
	if m.calls == 1 {
		emit("Let me check the next chunk.")
		return model.Response{Message: model.Message{Role: model.RoleAssistant, Content: "Let me check the next chunk."}, StopReason: "stop"}, nil
	}
	emit("关羽斩了十六人")
	return model.Response{Message: model.Message{Role: model.RoleAssistant, Content: "关羽斩了十六人"}, StopReason: "stop"}, nil
}

// Narration is working text even when the round ends the turn without a tool
// call: the run continues for a grace round or hands over to the answer call,
// and either one writes the real reply into the answer area afterwards.
func TestLoop_NarrationNeverReachesTheAnswerArea(t *testing.T) {
	runner := buildRunner(t, &narratingModel{}, nil, Config{})

	var answers []string
	var steps []string
	ctx := WithReasoningSink(WithDeltaSink(t.Context(), func(text string) {
		answers = append(answers, text)
	}), func(text string) {
		steps = append(steps, text)
	})

	outcome, err := runner.Invoke(ctx, &Input{RunID: "run-narration-stream", UserMessage: "关羽杀了多少人"}, nil)
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if got := strings.Join(answers, ""); got != "关羽斩了十六人" {
		t.Fatalf("answer deltas = %q, want only the round that answered", got)
	}
	if len(steps) != 1 || steps[0] != "Let me check the next chunk." {
		t.Fatalf("step text = %q, want the narrating round", steps)
	}
	if outcome.Text != "关羽斩了十六人" {
		t.Fatalf("text = %q", outcome.Text)
	}
}

// A prompt-too-long round is replayed on a smaller prompt. Its streamed text is
// a draft: leaving it in the answer area would put it in front of the answer the
// replay writes there.
func TestLoop_RecompactDraftStaysOutOfTheAnswerArea(t *testing.T) {
	mdl := &streamingModel{
		chunks:         []string{"关羽斩了十六人"},
		failNext:       errors.New("prompt is too long: reduce the length"),
		emitBeforeFail: true,
	}
	runner := buildRunner(t, mdl, nil, Config{})

	var answers []string
	var steps []string
	ctx := WithReasoningSink(WithDeltaSink(t.Context(), func(text string) {
		answers = append(answers, text)
	}), func(text string) {
		steps = append(steps, text)
	})

	if _, err := runner.Invoke(ctx, &Input{RunID: "run-ptl-stream", UserMessage: "关羽杀了多少人"}, nil); err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if got := strings.Join(answers, ""); got != "关羽斩了十六人" {
		t.Fatalf("answer deltas = %q, want only the replayed round", got)
	}
	if len(steps) != 1 || steps[0] != "par" {
		t.Fatalf("step text = %q, want the draft of the rejected round", steps)
	}
}

type failingStreamModel struct{ calls int }

func (m *failingStreamModel) Complete(ctx context.Context, req model.Request) (model.Response, error) {
	return m.Stream(ctx, req, nil)
}

func (m *failingStreamModel) Stream(_ context.Context, _ model.Request, sender model.StreamSender) (model.Response, error) {
	m.calls++
	partial := "半个回答"
	if sender != nil {
		_ = sender(&partial, nil)
	}
	return model.Response{}, errTestProvider
}

type testProviderError struct{}

func (testProviderError) Error() string { return "provider unavailable" }

var errTestProvider = testProviderError{}
