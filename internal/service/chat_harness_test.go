package service

import (
	"context"
	"errors"
	"strings"
	"testing"

	"ragflow/internal/common"
	"ragflow/internal/entity"
	providermodel "ragflow/internal/entity/models"
	"ragflow/internal/harness/agent/definition"
	"ragflow/internal/harness/agent/loop"
	harnessmodel "ragflow/internal/harness/agent/model"
)

// fakeHarnessTurn stands in for the runtime: it records the request the chat
// path built and replays a scripted turn.
type fakeHarnessTurn struct {
	calls     []harnessTurnRequest
	deltas    []string
	approvals []harnessApprovalEvent
	result    harnessTurnResult
	err       error
}

func (f *fakeHarnessTurn) Run(ctx context.Context, req harnessTurnRequest) (harnessTurnResult, error) {
	f.calls = append(f.calls, req)
	if req.OnApproval != nil {
		for _, event := range f.approvals {
			req.OnApproval(event)
		}
	}
	if sink := loop.DeltaSinkFromContext(ctx); sink != nil {
		for _, delta := range f.deltas {
			sink(delta)
		}
	}
	return f.result, f.err
}

func harnessTestChat() *entity.Chat {
	return &entity.Chat{
		ID:       "chat-1",
		TenantID: "tenant-1",
		LLMID:    "model-x",
		AgentID:  "agent-1",
		KBIDs:    entity.JSONSlice{"kb-1", "", "kb-2"},
	}
}

func collectChatResults(t *testing.T, out <-chan AsyncChatResult) []AsyncChatResult {
	t.Helper()
	var results []AsyncChatResult
	for result := range out {
		results = append(results, result)
	}
	return results
}

func TestHarnessTurnStreamsDeltasThenAuthoritativeAnswer(t *testing.T) {
	executor := &fakeHarnessTurn{
		deltas: []string{"Apple ", "Inc."},
		result: harnessTurnResult{
			Answer:         "Apple Inc. was founded in 1976.",
			Rounds:         3,
			TokensIn:       1200,
			TokensOut:      80,
			CostUSD:        0.002,
			PrefixRebuilds: 1,
		},
	}
	svc := &ChatPipelineService{harness: executor}

	out, err := svc.harnessTurn(context.Background(), "user-1", harnessTestChat(), []map[string]interface{}{
		{"role": "user", "content": "who founded Apple?"},
		{"role": "assistant", "content": "Apple is a company."},
		{"role": "user", "content": "when was it founded?"},
	}, true, map[string]interface{}{"conversation_id": "conv-9"})
	if err != nil {
		t.Fatalf("harnessTurn: %v", err)
	}
	results := collectChatResults(t, out)

	if len(executor.calls) != 1 {
		t.Fatalf("executor ran %d time(s), want 1", len(executor.calls))
	}
	req := executor.calls[0]
	if req.Question != "when was it founded?" {
		t.Errorf("question = %q", req.Question)
	}
	if len(req.History) != 2 {
		t.Fatalf("history = %+v, want the two turns before the question", req.History)
	}
	if req.Datasets[0] != "kb-1" || len(req.Datasets) != 2 {
		t.Errorf("datasets = %v, want the thread's non-empty KBs", req.Datasets)
	}
	if req.SessionID != "conv-9" || req.ModelRef != "model-x" || req.AgentID != "agent-1" || req.TenantID != "tenant-1" {
		t.Errorf("request identity = %+v", req)
	}
	// The actor travels with the turn: model access resolves per user, and a
	// tenant-only lookup silently picks a provider the user does not own.
	if req.UserID != "user-1" {
		t.Errorf("user = %q, want the turn's actor", req.UserID)
	}

	if len(results) != 4 {
		t.Fatalf("results = %d, want two deltas, one usage step and one final", len(results))
	}
	for i, want := range []string{"Apple ", "Inc."} {
		if results[i].Answer != want || results[i].Final {
			t.Errorf("delta %d = %+v", i, results[i])
		}
	}
	// The turn's accounting rides the step stream: rounds, tokens, cost and
	// prefix rebuilds are not owned by any tool call, so the client would have
	// no way to show what the turn cost without this event.
	usage := results[2]
	if usage.ThinkEvent == nil || usage.ThinkEvent.Kind != "usage" {
		t.Fatalf("third result = %+v, want the usage step", usage)
	}
	if usage.ThinkEvent.Rounds != 3 || usage.ThinkEvent.TokensIn != 1200 ||
		usage.ThinkEvent.TokensOut != 80 || usage.ThinkEvent.CostUSD != 0.002 ||
		usage.ThinkEvent.PrefixRebuilds != 1 {
		t.Errorf("usage step = %+v, want the turn's accounting", usage.ThinkEvent)
	}
	final := results[len(results)-1]
	if !final.Final || !final.AnswerIsAuthoritative {
		t.Errorf("final result = %+v", final)
	}
	if final.Answer != "Apple Inc. was founded in 1976." {
		t.Errorf("final answer = %q", final.Answer)
	}
	// No citations were stamped on this run, so the reference stays the empty
	// payload the SSE shape expects rather than inventing chunks.
	if len(final.Reference) != 0 {
		t.Errorf("reference = %v", final.Reference)
	}
}

func TestHarnessTurnWithoutStreamingSendsOnlyTheFinalAnswer(t *testing.T) {
	executor := &fakeHarnessTurn{
		deltas: []string{"Apple "},
		result: harnessTurnResult{Answer: "Apple Inc."},
	}
	svc := &ChatPipelineService{harness: executor}

	out, err := svc.harnessTurn(context.Background(), "user-1", harnessTestChat(), []map[string]interface{}{
		{"role": "user", "content": "who founded Apple?"},
	}, false, nil)
	if err != nil {
		t.Fatalf("harnessTurn: %v", err)
	}
	results := collectChatResults(t, out)
	if len(results) != 1 {
		t.Fatalf("results = %d, want only the final", len(results))
	}
	if !results[0].Final || results[0].Answer != "Apple Inc." {
		t.Errorf("final = %+v", results[0])
	}
}

func TestHarnessTurnFallsBackToStreamedTextWhenTheRunReportsNone(t *testing.T) {
	executor := &fakeHarnessTurn{deltas: []string{"streamed ", "text"}}
	svc := &ChatPipelineService{harness: executor}

	out, err := svc.harnessTurn(context.Background(), "user-1", harnessTestChat(), []map[string]interface{}{
		{"role": "user", "content": "hi"},
	}, true, nil)
	if err != nil {
		t.Fatalf("harnessTurn: %v", err)
	}
	results := collectChatResults(t, out)
	final := results[len(results)-1]
	if final.Answer != "streamed text" {
		t.Errorf("final answer = %q, want the accumulated deltas", final.Answer)
	}
}

func TestHarnessTurnReportsExecutorFailure(t *testing.T) {
	svc := &ChatPipelineService{harness: &fakeHarnessTurn{err: errors.New("provider down")}}

	out, err := svc.harnessTurn(context.Background(), "user-1", harnessTestChat(), []map[string]interface{}{
		{"role": "user", "content": "hi"},
	}, true, nil)
	if err != nil {
		t.Fatalf("harnessTurn: %v", err)
	}
	results := collectChatResults(t, out)
	if len(results) != 1 || !results[0].Final {
		t.Fatalf("results = %+v, want one final error result", results)
	}
	if !strings.Contains(results[0].Answer, "provider down") {
		t.Errorf("answer = %q", results[0].Answer)
	}
}

func TestHarnessTurnRejectsAConversationWithoutAQuestion(t *testing.T) {
	svc := &ChatPipelineService{harness: &fakeHarnessTurn{}}

	_, err := svc.harnessTurn(context.Background(), "user-1", harnessTestChat(), []map[string]interface{}{
		{"role": "assistant", "content": "anything else?"},
	}, true, nil)
	if err == nil {
		t.Fatal("a conversation ending in an assistant turn was accepted as a question")
	}
}

func TestAsyncChatRoutesAnAgentThreadToTheHarness(t *testing.T) {
	executor := &fakeHarnessTurn{result: harnessTurnResult{Answer: "agent answer"}}
	svc := &ChatPipelineService{harness: executor}

	out, err := svc.AsyncChat(context.Background(), "user-1", harnessTestChat(), []map[string]interface{}{
		{"role": "user", "content": "hi"},
	}, false, nil)
	if err != nil {
		t.Fatalf("AsyncChat: %v", err)
	}
	results := collectChatResults(t, out)
	if len(executor.calls) != 1 {
		t.Fatalf("the harness ran %d time(s), want the thread's agent to own the turn", len(executor.calls))
	}
	final := results[len(results)-1]
	if !final.Final || final.Answer != "agent answer" {
		t.Errorf("final = %+v", final)
	}
}

func TestHarnessTurnHonoursQuoteOff(t *testing.T) {
	executor := &fakeHarnessTurn{result: harnessTurnResult{Answer: "Apple Inc. [ID:0]"}}
	svc := &ChatPipelineService{harness: executor}

	out, err := svc.harnessTurn(context.Background(), "user-1", harnessTestChat(), []map[string]interface{}{
		{"role": "user", "content": "who founded Apple?"},
	}, false, map[string]interface{}{"quote": false})
	if err != nil {
		t.Fatalf("harnessTurn: %v", err)
	}
	final := collectChatResults(t, out)[0]
	if len(final.Reference) != 0 {
		t.Errorf("reference = %v, want none when citations are off", final.Reference)
	}
	if final.Answer != "Apple Inc. [ID:0]" {
		t.Errorf("answer = %q, want the agent's text untouched", final.Answer)
	}
}

func TestSplitHarnessConversationKeepsOnlySpeakableTurns(t *testing.T) {
	history, question := splitHarnessConversation([]map[string]interface{}{
		{"role": "system", "content": "ignored"},
		{"role": "user", "content": "first"},
		{"role": "tool", "content": "ignored"},
		{"role": "assistant", "content": "  "},
		{"role": "user", "content": "second"},
	})
	if question != "second" {
		t.Errorf("question = %q", question)
	}
	if len(history) != 1 || history[0].Content != "first" {
		t.Errorf("history = %+v, want only the substantive prior turn", history)
	}
}

func TestHarnessTurnStreamsApprovalRequestsAndVerdicts(t *testing.T) {
	executor := &fakeHarnessTurn{
		approvals: []harnessApprovalEvent{
			{RequestKey: "run-1:call-1", Tool: "exec_shell", Args: `{"cmd":"ls"}`, Risk: "dangerous"},
			{RequestKey: "run-1:call-1", Tool: "exec_shell", Args: `{"cmd":"ls"}`, Risk: "dangerous",
				Decided: true, Allowed: false, Reason: "not on production"},
		},
		result: harnessTurnResult{Answer: "done"},
	}
	svc := &ChatPipelineService{harness: executor}

	out, err := svc.harnessTurn(context.Background(), "user-1", harnessTestChat(), []map[string]interface{}{
		{"role": "user", "content": "list the files"},
	}, false, nil)
	if err != nil {
		t.Fatalf("harnessTurn: %v", err)
	}
	results := collectChatResults(t, out)
	if len(results) != 3 {
		t.Fatalf("results = %d, want wait, verdict and final", len(results))
	}

	// An approval must reach the client even on a non-streaming turn: it is a
	// question, and a turn that blocks on one has nothing else to send.
	pending := results[0].ThinkEvent
	if pending == nil || pending.Kind != "approval" || pending.Status != "pending" {
		t.Fatalf("pending event = %+v", results[0])
	}
	if pending.RequestID != "run-1:call-1" || pending.Tool != "exec_shell" {
		t.Errorf("pending event = %+v", pending)
	}
	verdict := results[1].ThinkEvent
	if verdict == nil || verdict.Status != "denied" || verdict.Reason != "not on production" {
		t.Fatalf("verdict event = %+v", results[1])
	}
	if verdict.RequestID != pending.RequestID {
		t.Errorf("verdict id = %q, want the id the client already holds (%q)", verdict.RequestID, pending.RequestID)
	}
	if !results[2].Final {
		t.Errorf("last result = %+v", results[2])
	}
}

// scriptedCompleter answers with its name, or fails when err is set.
type scriptedCompleter struct {
	name string
	err  error
}

func (c scriptedCompleter) ChatWithMessages(
	_ context.Context,
	_ []providermodel.Message,
	_ *providermodel.ChatConfig,
	_ *common.ModelUsage,
) (*providermodel.ChatResponse, error) {
	if c.err != nil {
		return nil, c.err
	}
	answer := c.name
	return &providermodel.ChatResponse{Answer: &answer}, nil
}

func TestFailoverCompleterAnswersFromTheFirstWorkingModel(t *testing.T) {
	chain := failoverCompleter{models: []harnessmodel.ChatCompleter{
		scriptedCompleter{name: "primary", err: errors.New("insufficient balance")},
		scriptedCompleter{name: "fallback"},
	}}

	response, err := chain.ChatWithMessages(context.Background(), nil, nil, nil)
	if err != nil {
		t.Fatalf("chat: %v", err)
	}
	if response == nil || response.Answer == nil || *response.Answer != "fallback" {
		t.Fatalf("response = %+v, want the fallback model's answer", response)
	}
}

func TestFailoverCompleterReportsTheLastErrorWhenEveryModelFails(t *testing.T) {
	chain := failoverCompleter{models: []harnessmodel.ChatCompleter{
		scriptedCompleter{err: errors.New("first down")},
		scriptedCompleter{err: errors.New("second down")},
	}}

	if _, err := chain.ChatWithMessages(context.Background(), nil, nil, nil); err == nil ||
		!strings.Contains(err.Error(), "second down") {
		t.Fatalf("error = %v, want the last failure", err)
	}
}

func TestHarnessTurnCarriesTheChatFailoverModels(t *testing.T) {
	executor := &fakeHarnessTurn{result: harnessTurnResult{Answer: "ok"}}
	svc := &ChatPipelineService{harness: executor}
	chat := harnessTestChat()
	chat.LLMSetting = entity.JSONMap{"failover_llm_ids": []interface{}{"model-y", "model-z"}}

	out, err := svc.harnessTurn(context.Background(), "user-1", chat, []map[string]interface{}{
		{"role": "user", "content": "hi"},
	}, false, nil)
	if err != nil {
		t.Fatalf("harnessTurn: %v", err)
	}
	collectChatResults(t, out)

	got := executor.calls[0].FallbackModelRefs
	if len(got) != 2 || got[0] != "model-y" || got[1] != "model-z" {
		t.Fatalf("fallbacks = %v, want the chat's configured chain", got)
	}
}

// TestHarnessTurnCarriesTheChatGenerationSettings pins that the conversation's
// llm_setting reaches the harness run. It used to stop at the naive and agentic
// paths, so turning thinking off in the UI changed nothing here and the catalog
// default (adaptive for MiniMax-M3) kept every round reasoning.
func TestHarnessTurnCarriesTheChatGenerationSettings(t *testing.T) {
	executor := &fakeHarnessTurn{result: harnessTurnResult{Answer: "ok"}}
	svc := &ChatPipelineService{harness: executor}
	chat := harnessTestChat()
	chat.LLMSetting = entity.JSONMap{
		"temperature": 0.5,
		"max_tokens":  512,
		"thinking":    false,
	}

	out, err := svc.harnessTurn(context.Background(), "user-1", chat, []map[string]interface{}{
		{"role": "user", "content": "hi"},
	}, false, nil)
	if err != nil {
		t.Fatalf("harnessTurn: %v", err)
	}
	collectChatResults(t, out)

	req := executor.calls[0]
	if req.Sampling == nil {
		t.Fatalf("Sampling = nil, want the chat's generation settings")
	}
	if req.Sampling.Temperature == nil || *req.Sampling.Temperature != 0.5 {
		t.Errorf("temperature = %v, want 0.5", req.Sampling.Temperature)
	}
	if req.Sampling.Thinking == nil || *req.Sampling.Thinking {
		t.Errorf("thinking = %v, want an explicit false", req.Sampling.Thinking)
	}
	if req.MaxOutputTokens != 512 {
		t.Errorf("max output tokens = %d, want 512", req.MaxOutputTokens)
	}
}

// TestHarnessTurnLeavesGenerationUnpinnedWithoutChatSettings: a chat that pins
// nothing must keep the catalog defaults rather than pinning zero values.
func TestHarnessTurnLeavesGenerationUnpinnedWithoutChatSettings(t *testing.T) {
	executor := &fakeHarnessTurn{result: harnessTurnResult{Answer: "ok"}}
	svc := &ChatPipelineService{harness: executor}

	out, err := svc.harnessTurn(context.Background(), "user-1", harnessTestChat(), []map[string]interface{}{
		{"role": "user", "content": "hi"},
	}, false, nil)
	if err != nil {
		t.Fatalf("harnessTurn: %v", err)
	}
	collectChatResults(t, out)

	req := executor.calls[0]
	if req.Sampling != nil || req.MaxOutputTokens != 0 {
		t.Fatalf("sampling = %+v, max output tokens = %d, want both unset", req.Sampling, req.MaxOutputTokens)
	}
}

// TestMergeSamplingLetsTheAgentPinWinPerField: pinning one field on the agent
// must not freeze the controls its users still own.
func TestMergeSamplingLetsTheAgentPinWinPerField(t *testing.T) {
	chatTemperature := 0.4
	pinnedTemperature := 0.0
	chatThinking := false
	merged := mergeSampling(
		&definition.SamplingConfig{Temperature: &chatTemperature, Thinking: &chatThinking},
		&definition.SamplingConfig{Temperature: &pinnedTemperature},
	)
	if merged.Temperature == nil || *merged.Temperature != pinnedTemperature {
		t.Errorf("temperature = %v, want the agent's pin", merged.Temperature)
	}
	if merged.Thinking == nil || *merged.Thinking {
		t.Errorf("thinking = %v, want the chat's setting", merged.Thinking)
	}

	if got := mergeSampling(nil, nil); got != nil {
		t.Errorf("mergeSampling(nil, nil) = %+v, want nil", got)
	}
	if got := mergeSampling(&definition.SamplingConfig{}, nil); got == nil {
		t.Error("mergeSampling(chat, nil) dropped the chat settings")
	}
}

func TestHarnessTurnErrorMessageExplainsAProviderFailure(t *testing.T) {
	// The production text: internal node names with the provider's own message
	// buried inside.
	err := errors.New(`all 1 tasks failed in step 1:  think=*pregel.RetryExhaustedError(node think failed after 1 attempts: loop: model call failed: failed to scan response body: minimax API error: insufficient balance)`)

	message := harnessTurnErrorMessage(err)
	if !strings.Contains(message, "no remaining quota") {
		t.Errorf("message = %q, want the cause", message)
	}
	if !strings.Contains(message, "insufficient balance") {
		t.Errorf("message = %q, want the provider's own words kept", message)
	}
	if strings.Contains(message, "pregel") || strings.Contains(message, "tasks failed") {
		t.Errorf("message = %q, leaked harness internals", message)
	}
	if !strings.Contains(message, "settings") {
		t.Errorf("message = %q, want an actionable next step", message)
	}
}

func TestHarnessTurnErrorMessageAsksResendWhenTheStreamStops(t *testing.T) {
	// The production text: the provider closed the stream before its terminal
	// frame, so nothing is wrong with the chat's configuration and switching
	// models is not the fix.
	err := errors.New(`all 1 tasks failed in step 8:  think=*pregel.RetryExhaustedError(node think failed after 3 attempts: loop: model call failed: stream ended before [DONE] or finish_reason (events=4 choices_events=3 content_chunks=0))`)

	message := harnessTurnErrorMessage(err)
	if !strings.Contains(message, "stopped before the answer was complete") {
		t.Errorf("message = %q, want the cause", message)
	}
	if !strings.Contains(message, "Send the message again") {
		t.Errorf("message = %q, want a resend as the next step", message)
	}
	if strings.Contains(message, "settings") {
		t.Errorf("message = %q, did not stop a dropped stream from reading as a config problem", message)
	}
	if strings.Contains(message, "pregel") || strings.Contains(message, "tasks failed") {
		t.Errorf("message = %q, leaked harness internals", message)
	}
}

func TestHarnessTurnErrorMessageKeepsUnrecognisedFailuresVerbatim(t *testing.T) {
	err := errors.New("some unexpected failure")
	if got := harnessTurnErrorMessage(err); got != err.Error() {
		t.Fatalf("message = %q, want the original text unchanged", got)
	}
}
