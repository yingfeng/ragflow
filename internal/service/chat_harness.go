package service

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"go.uber.org/zap"

	"ragflow/internal/agentic_rag"
	"ragflow/internal/common"
	"ragflow/internal/entity"
	providermodel "ragflow/internal/entity/models"
	"ragflow/internal/harness"
	"ragflow/internal/harness/agent/definition"
	"ragflow/internal/harness/agent/loop"
	harnessmodel "ragflow/internal/harness/agent/model"
	"ragflow/internal/harness/agent/run"
	"ragflow/internal/harness/store"
	"ragflow/internal/harness/toolkit"
)

// harnessModelResolver adapts ModelFactory to the harness port, so neither the
// harness nor its callers construct a chat client themselves. An empty reference
// means "the tenant default", which is what a thread with no model selected
// asked for.
type harnessModelResolver struct {
	factory *ModelFactory
}

// NewHarnessModelResolver is the one resolver every harness entry point uses:
// the API manager and chat turns resolve models by the same rule.
func NewHarnessModelResolver() harness.ModelResolver {
	return harnessModelResolver{factory: NewModelFactory()}
}

func (r harnessModelResolver) Resolve(ctx context.Context, tenantID, userID, modelRef string) (harnessmodel.ChatCompleter, error) {
	// The same access the pipeline builds for its own model calls: the user is
	// part of it, because provider selection and API keys resolve per user.
	access := ModelAccess{UserID: userID, TenantID: tenantID}
	if modelRef == "" {
		return r.factory.NewDefaultChatModel(ctx, access)
	}
	return r.factory.NewChatModel(ctx, access, modelRef)
}

// harnessApprovalEvent is one tool call waiting for, or leaving, human approval.
type harnessApprovalEvent struct {
	// RequestKey settles the request; it is the broker's handle and the only id
	// a client needs to answer it.
	RequestKey string
	Tool       string
	Args       string
	Risk       string
	// Diff is what a pending write will change: the reviewer decides on the
	// change, not on the arguments that produced it.
	Diff string
	// Decided is false for the wait, true once a verdict (or a timeout) arrived.
	Decided bool
	Allowed bool
	Reason  string
}

// harnessApprover puts the process approval broker behind the dangerous-call
// gate and reports each wait, so a client can render and answer it. Everything
// below the threshold passes without a report: a progress-free approval for a
// read-only call would train people to click through prompts.
type harnessApprover struct {
	broker *toolkit.ApprovalBroker
	report func(harnessApprovalEvent)
	// auto is the agent's approval mode "auto": dangerous calls run without a
	// human. Explicit deny rules still apply, because they are evaluated outside
	// this approver.
	auto bool
}

func (a harnessApprover) Approve(ctx context.Context, call toolkit.Call, risk toolkit.Risk) toolkit.ApprovalDecision {
	if a.broker == nil || a.auto || risk < toolkit.RiskDangerous {
		return toolkit.ApprovalDecision{Allow: true}
	}
	event := harnessApprovalEvent{
		RequestKey: toolkit.RunIDFrom(ctx) + ":" + call.ID,
		Tool:       call.Tool,
		Args:       call.Arguments,
		Risk:       risk.String(),
		Diff:       call.Preview,
	}
	if a.report != nil {
		a.report(event)
	}
	decision := a.broker.Approve(ctx, call, risk)
	event.Decided = true
	event.Allowed = decision.Allow
	event.Reason = decision.Reason
	if a.report != nil {
		a.report(event)
	}
	return decision
}

// harnessInteractionEvent is one question the run asked the user and what came
// back. The chat is the interactive surface: every wait and every answer is a
// think event, so the client can render the question and post the answer.
type harnessInteractionEvent struct {
	// RequestKey settles the question on the process broker.
	RequestKey string
	Question   string
	Context    string
	// Answered is false while the question waits, and true only when the user
	// replied in time; a timeout is reported with Answered=false and empty text.
	Answered bool
	Answer   string
}

// harnessTurnRequest is one agent turn inside a chat thread. The thread supplies
// the model, the datasets and the conversation; the agent supplies the prompt,
// its capabilities and its policy.
type harnessTurnRequest struct {
	TenantID string
	// UserID is the actor: the model is resolved for this user, not just the
	// tenant, so their provider and key are the ones used.
	UserID    string
	AgentID   string
	SessionID string
	ModelRef  string
	// FallbackModelRefs are the chat's configured failover models, tried in
	// order when the primary one fails.
	FallbackModelRefs []string
	// Sampling is the conversation's generation settings, read from the chat's
	// llm_setting. It is the base the agent definition's own pin overrides, the
	// same rule the agentic path applies to a template pin: one conversation
	// switch has to mean the same thing on every path.
	Sampling *definition.SamplingConfig
	// MaxOutputTokens caps one completion, from the same llm_setting.
	MaxOutputTokens int
	Question        string
	History         []harnessmodel.Message
	Datasets        []string
	WebSearch       agentic_rag.WebSearchFunc
	// OnApproval reports a tool call that needs a human. The run blocks while the
	// verdict is outstanding, so the callback must return immediately.
	OnApproval func(event harnessApprovalEvent)
	// OnInteraction reports a question the run asked the user, and the answer
	// when one arrives. The run blocks while the question waits, so the callback
	// must return immediately.
	OnInteraction func(event harnessInteractionEvent)
}

// failoverCompleter tries each model in order and answers from the first one that
// works.
//
// The chain is a user setting on the chat. One broken account — no balance,
// revoked key, provider outage — should cost the next call rather than the whole
// turn: the agentic path has carried a chain for a while, and an agent thread
// dying where a plain chat survives would be a regression against it.
type failoverCompleter struct {
	models []harnessmodel.ChatCompleter
}

func (f failoverCompleter) ChatWithMessages(
	ctx context.Context,
	messages []providermodel.Message,
	config *providermodel.ChatConfig,
	usage *common.ModelUsage,
) (*providermodel.ChatResponse, error) {
	var lastErr error
	for _, model := range f.models {
		// A failed attempt can still report tokens; zeroing keeps the accounting
		// tied to the call whose answer is actually used.
		if usage != nil {
			*usage = common.ModelUsage{}
		}
		response, err := model.ChatWithMessages(ctx, messages, config, usage)
		if err == nil {
			return response, nil
		}
		lastErr = err
	}
	if lastErr == nil {
		lastErr = errors.New("no model configured")
	}
	return nil, lastErr
}

// harnessModelChain resolves the turn's model plus the chat's configured
// fallbacks. An unresolvable fallback is skipped: a deleted model must not take
// down a turn that a working primary can still answer.
func harnessModelChain(ctx context.Context, resolver harness.ModelResolver, req harnessTurnRequest) (harnessmodel.ChatCompleter, error) {
	primary, err := resolver.Resolve(ctx, req.TenantID, req.UserID, req.ModelRef)
	if err != nil {
		return nil, fmt.Errorf("harness turn: resolve model %q: %w", req.ModelRef, err)
	}
	models := []harnessmodel.ChatCompleter{primary}
	for _, ref := range req.FallbackModelRefs {
		if ref == "" || ref == req.ModelRef {
			continue
		}
		next, resolveErr := resolver.Resolve(ctx, req.TenantID, req.UserID, ref)
		if resolveErr != nil || next == nil {
			common.WarnCtx(ctx, "harness turn: skipping unresolvable fallback model",
				zap.String("model_ref", ref), zap.Error(resolveErr))
			continue
		}
		models = append(models, next)
	}
	if len(models) == 1 {
		return primary, nil
	}
	return failoverCompleter{models: models}, nil
}

// chatGeneration reads the conversation's generation settings for a harness run.
// It goes through BuildChatConfig so the chat's llm_setting means the same thing
// here as on the naive and agentic paths; a switch that only moves two of three
// paths is a switch the user cannot trust.
func chatGeneration(chat *entity.Chat) (*definition.SamplingConfig, int) {
	cfg := BuildChatConfig(chat, nil)
	if cfg == nil {
		return nil, 0
	}
	sampling := &definition.SamplingConfig{
		Temperature: cfg.Temperature,
		TopP:        cfg.TopP,
		Thinking:    cfg.Thinking,
	}
	if sampling.Temperature == nil && sampling.TopP == nil && sampling.Thinking == nil {
		sampling = nil
	}
	maxTokens := 0
	if cfg.MaxTokens != nil {
		maxTokens = *cfg.MaxTokens
	}
	return sampling, maxTokens
}

// mergeSampling layers the conversation's settings under the agent definition's
// own pin: a field the agent pins is the agent's, everything else follows the
// chat. Without this, pinning sampling on an agent would also freeze the
// controls its users expect to own.
func mergeSampling(chat, pinned *definition.SamplingConfig) *definition.SamplingConfig {
	if pinned == nil {
		return chat
	}
	if chat == nil {
		return pinned
	}
	merged := *pinned
	if merged.Temperature == nil {
		merged.Temperature = chat.Temperature
	}
	if merged.TopP == nil {
		merged.TopP = chat.TopP
	}
	if merged.Thinking == nil {
		merged.Thinking = chat.Thinking
	}
	return &merged
}

// harnessTurnErrorMessage turns a run failure into something a user can act on.
//
// The raw text names the harness's own nodes ("all 1 tasks failed in step 1:
// think=*pregel.RetryExhaustedError(...)") with the provider's message buried
// inside, which reads as an internal crash. What the user controls is the model,
// so say what went wrong with it and what to do.
func harnessTurnErrorMessage(err error) string {
	if err == nil {
		return ""
	}
	message := err.Error()
	lower := strings.ToLower(message)

	var cause string
	// advice is what the user should do next. It follows the cause because a
	// dropped stream is worth resending, while a quota or credential failure is
	// not: resending only spends the same error again.
	advice := "Choose another model in this chat's settings and send again."
	switch {
	case strings.Contains(lower, "insufficient balance"), strings.Contains(lower, "insufficient_quota"),
		strings.Contains(lower, "quota exceeded"), strings.Contains(lower, "no remaining"):
		cause = "the selected model has no remaining quota"
	case strings.Contains(lower, "invalid api key"), strings.Contains(lower, "unauthorized"),
		strings.Contains(lower, "authentication"):
		cause = "the selected model rejected its credentials"
	case strings.Contains(lower, "429"), strings.Contains(lower, "rate limit"):
		cause = "the selected model is rate limited"
	case strings.Contains(lower, "timeout"), strings.Contains(lower, "deadline exceeded"):
		cause = "the selected model did not answer in time"
	case strings.Contains(lower, "stream ended before"), strings.Contains(lower, "stream ended prematurely"),
		strings.Contains(lower, "incomplete stream"):
		cause = "the selected model's response stream stopped before the answer was complete"
		advice = "Send the message again."
	}
	if cause == "" {
		// Nothing recognisable: hand over the original rather than hide it.
		return message
	}

	if detail := providerErrorText(message); detail != "" {
		cause = fmt.Sprintf("%s (%s)", cause, detail)
	}
	return cause + ". " + advice
}

// providerErrorText keeps the provider's own wording: it is what tells an
// operator whether to top up, rotate a key, or wait.
func providerErrorText(message string) string {
	const marker = "api error:"
	// Providers capitalise the marker differently ("minimax API error:"), and the
	// index has to come from a same-length scan so it still points into the
	// original text.
	index := strings.LastIndex(strings.ToLower(message), marker)
	if index < 0 {
		return ""
	}
	return strings.TrimSpace(message[index+len(marker):])
}

// harnessReasoningLimit bounds one round's working text in the step panel. The
// panel is a log, not the transcript: a long monologue belongs in the run's own
// trace, and shipping it whole would grow every message payload.
const harnessReasoningLimit = 4000

// clipReasoning trims a round's working text to what the panel can show.
func clipReasoning(text string) string {
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return ""
	}
	runes := []rune(trimmed)
	if len(runes) <= harnessReasoningLimit {
		return trimmed
	}
	return string(runes[:harnessReasoningLimit]) + "…"
}

// harnessTurnResult is what one turn produced, already shaped for the chat
// response.
type harnessTurnResult struct {
	Answer    string
	Rounds    int
	TokensIn  int
	TokensOut int
	CostUSD   float64
	Stopped   string
	// PrefixRebuilds counts how often the cached prompt prefix changed in this
	// turn. It is the one measurable outcome of prompt layering: with a fixed
	// tool surface it stays at zero, and a number that grows with rounds means
	// something in the run is rewriting the prefix every round.
	PrefixRebuilds int
	// Review is the independent reviewer's verdict on the turn's file writes.
	// Empty means nothing was written or the review found no issues.
	Review string
}

// harnessTurnExecutor runs one turn. It is an interface so a test can drive the
// chat path without a model provider.
type harnessTurnExecutor interface {
	Run(ctx context.Context, req harnessTurnRequest) (harnessTurnResult, error)
}

// harnessSettings is read once per process; the harness owns the parsing so chat
// turns and the harness API cannot disagree about the workspace.
var harnessSettings = harness.LoadSettings()

// harnessRuntimeExecutor builds a runtime per turn and runs it.
//
// The runtime is per turn because the tool surface depends on the thread's
// dataset scope; sharing one would leak one chat's datasets into another's tool
// list. Everything else comes from the process-wide state, so a chat turn and an
// API run share one broker, one breaker registry and one audit log.
type harnessRuntimeExecutor struct{}

// Run implements harnessTurnExecutor.
func (harnessRuntimeExecutor) Run(ctx context.Context, req harnessTurnRequest) (harnessTurnResult, error) {
	proc := SharedHarnessProcess()

	resolved, err := proc.Agents.Resolve(ctx, req.TenantID, req.AgentID, "")
	if err != nil {
		return harnessTurnResult{}, fmt.Errorf("harness turn: resolve agent: %w", err)
	}
	prompt := resolved.Definition.Prompt
	if addendum := strings.TrimSpace(resolved.Definition.Addendum); addendum != "" {
		prompt += "\n\n" + addendum
	}

	chatModel, err := harnessModelChain(ctx, proc.Resolver, req)
	if err != nil {
		return harnessTurnResult{}, err
	}

	scope := toolkit.Scope{TenantID: req.TenantID, Datasets: req.Datasets}
	runtime, err := harness.NewRuntime(ctx, harness.RuntimeConfig{
		Store:          proc.Store,
		Checkpointer:   proc.Checkpointer(),
		Resolver:       proc.Resolver,
		Model:          chatModel,
		ModelRef:       req.ModelRef,
		Scope:          scope,
		ActorUserID:    req.UserID,
		AgentID:        resolved.AgentID,
		AgentVersionID: resolved.VersionID,
		AgentName:      resolved.Definition.Name,
		SystemPrompt:   prompt,
		// The conversation's settings are the base; the agent's own pin wins
		// field by field, so a pinned agent stays reproducible while an
		// unpinned one follows the chat.
		Sampling:        mergeSampling(req.Sampling, resolved.Definition.Sampling),
		MaxOutputTokens: req.MaxOutputTokens,
		PolicyConfig:    resolved.Definition.Policy,
		MaxRounds:       resolved.Definition.Policy.MaxRounds,
		SkillsRoot:      harnessSettings.SkillsRoot,
		// Skills come from the caller's own skillspace, narrowed to what this
		// agent selected. The built-in retrieval policy applies regardless.
		SkillLoader:   NewHarnessSkillLoader(req.UserID, req.TenantID),
		EnabledSkills: resolved.Definition.Skills,
		ExecRoot:      harnessWorkspace(req.SessionID),
		// Lessons close the self-improvement loop: the store records them, the
		// loop renders them, and without this the loop never read them back.
		Lessons:   HarnessLessons(proc.Store, req.TenantID, resolved.AgentID),
		WebSearch: req.WebSearch,
		Approver: harnessApprover{
			broker: proc.Approvals,
			report: req.OnApproval,
			auto:   resolved.Definition.Policy.Approval == definition.ApprovalAuto,
		},
		// The chat is the surface a question is actually asked on: the wrapper
		// reports the wait to the stream, the process broker settles it, and a
		// client answers by the request id through the harness interaction API.
		Interactor: toolkit.ReportInteractor{
			Next: proc.Interactions,
			OnAsk: func(requestID, question, contextText string) {
				if req.OnInteraction != nil {
					req.OnInteraction(harnessInteractionEvent{
						RequestKey: requestID,
						Question:   question,
						Context:    contextText,
					})
				}
			},
			OnAnswer: func(requestID, question, answer string, answered bool) {
				if req.OnInteraction != nil {
					req.OnInteraction(harnessInteractionEvent{
						RequestKey: requestID,
						Question:   question,
						Answered:   answered,
						Answer:     answer,
					})
				}
			},
		},
		Grants:      proc.Grants,
		Breakers:    proc.Breakers,
		Retry:       harnessSettings.RetryPolicy(),
		EventLog:    proc.Events,
		EventLayers: EventLayers(),
	})
	if err != nil {
		return harnessTurnResult{}, err
	}
	for _, note := range runtime.Notes {
		common.WarnCtx(ctx, "harness capability degraded", zap.String("agent_id", resolved.AgentID), zap.String("note", note))
	}

	outcome, runErr := runtime.Service.Invoke(ctx, run.Request{
		RunID:          uuid.NewString(),
		AgentID:        resolved.AgentID,
		AgentVersionID: resolved.VersionID,
		TenantID:       req.TenantID,
		UserID:         req.UserID,
		AgentName:      resolved.Definition.Name,
		ModelRef:       req.ModelRef,
		SessionID:      req.SessionID,
		History:        req.History,
		UserMessage:    req.Question,
		Trigger:        "chat",
		Scope:          scope,
	})
	result := harnessTurnResult{
		Answer:         outcome.Text,
		Rounds:         outcome.Rounds,
		TokensIn:       outcome.TokensIn,
		TokensOut:      outcome.TokensOut,
		CostUSD:        outcome.CostUSD,
		Stopped:        outcome.Stopped,
		Review:         outcome.Review,
		PrefixRebuilds: outcome.PrefixRebuilds,
	}
	if runErr != nil {
		return result, runErr
	}
	return result, nil
}

// harnessTurn drives one agent turn for a chat thread and streams it in the shape
// the chat SSE handler already understands.
//
// The run's evidence registry is installed here, not inside the harness: it is
// what stamps every retrieved passage with the `[ID:n]` handle the answer cites,
// and it is the same registry the reference payload is resolved against, so the
// handles the model wrote and the chunks the UI renders cannot disagree.
func (s *ChatPipelineService) harnessTurn(
	ctx context.Context,
	userID string,
	chat *entity.Chat,
	messages []map[string]interface{},
	stream bool,
	kwargs map[string]interface{},
) (<-chan AsyncChatResult, error) {
	history, question := splitHarnessConversation(messages)
	if question == "" {
		return nil, fmt.Errorf("harness chat: the last message is not a user question")
	}

	datasets := make([]string, 0, len(chat.KBIDs))
	for _, raw := range chat.KBIDs {
		if id, ok := raw.(string); ok && id != "" {
			datasets = append(datasets, id)
		}
	}
	// nil hides web_search entirely, which is the same rule the agentic path
	// applies: a provider that is not configured has no tool.
	var webSearch agentic_rag.WebSearchFunc
	if s.shouldUseWebSearch(chat, kwargs["internet"]) {
		webSearch = s.webSearchFunc(chat.PromptConfig)
	}

	registry := agentic_rag.NewEvidenceRegistry()
	turnCtx := agentic_rag.WithEvidenceRegistry(ctx, registry)
	quote := quoteEnabled(kwargs, chat.PromptConfig)
	sessionID := conversationIDOf(kwargs)
	turnStarted := time.Now()

	executor := s.harness
	if executor == nil {
		executor = harnessRuntimeExecutor{}
	}

	out := make(chan AsyncChatResult, 16)
	go func() {
		defer close(out)

		var mu sync.Mutex
		streamed := strings.Builder{}
		emit := func(result AsyncChatResult) {
			select {
			case out <- result:
			case <-ctx.Done():
			}
		}

		turnCtx = loop.WithDeltaSink(turnCtx, func(text string) {
			mu.Lock()
			streamed.WriteString(text)
			mu.Unlock()
			if stream {
				emit(AsyncChatResult{Answer: text, Final: false})
			}
		})
		// A round that ends in a tool call is the model working, not the answer:
		// it goes to the step panel. Only the round that ends the turn writes the
		// answer area, so no round can overwrite text a user was reading.
		turnCtx = loop.WithReasoningSink(turnCtx, func(text string) {
			if !stream {
				return
			}
			reasoning := clipReasoning(text)
			if reasoning == "" {
				return
			}
			emit(AsyncChatResult{Final: false, ThinkEvent: &ThinkEvent{
				Kind:    "reasoning",
				Summary: reasoning,
			}})
		})
		turnCtx = loop.WithProgressSink(turnCtx, func(progress loop.Progress) {
			if progress.Phase != loop.ProgressStart {
				return
			}
			emit(AsyncChatResult{Final: false, ThinkEvent: &ThinkEvent{
				Kind:    "tool_call",
				Tool:    progress.Tool,
				Status:  "running",
				Args:    progress.Summary,
				Summary: fmt.Sprintf("%s(%s)", progress.Tool, progress.Summary),
			}})
		})

		// Cost accounting rides on the turn: the run table stores what happened
		// but not what it cost, and this is the only place the token counts of a
		// chat-driven harness run are visible.
		lfClient := LangfuseClientFromTenant(ctx, chat.TenantID, userID, chat.ID, chat.LLMID)
		lfTraceID := ""
		if lfClient != nil {
			defer func() {
				shutdownCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
				defer cancel()
				_ = lfClient.Shutdown(shutdownCtx)
			}()
			lfTraceID = fmt.Sprintf("trace-%d", time.Now().UnixNano())
			_ = lfClient.PostTrace(ctx, LangfuseTrace{
				ID:        lfTraceID,
				Name:      "harness_chat",
				UserID:    userID,
				SessionID: chat.ID,
				Metadata: map[string]interface{}{
					"agent_id":  chat.AgentID,
					"stream":    stream,
					"kb_count":  len(datasets),
					"session":   sessionID,
					"model_ref": chat.LLMID,
				},
				Timestamp: time.Now().UTC().Format(time.RFC3339Nano),
			})
		}

		// The turn runs under the chat's generation settings, not the model
		// catalog's defaults: a switch the naive and agentic paths honour has to
		// bind this path too.
		sampling, maxOutputTokens := chatGeneration(chat)
		result, err := executor.Run(turnCtx, harnessTurnRequest{
			TenantID:  chat.TenantID,
			UserID:    userID,
			AgentID:   chat.AgentID,
			SessionID: sessionID,
			ModelRef:  chat.LLMID,
			// The chat's own failover list, so a dead provider costs one call
			// here exactly as it does on the agentic path.
			FallbackModelRefs: agenticFailoverModelIDs(chat),
			Sampling:          sampling,
			MaxOutputTokens:   maxOutputTokens,
			Question:          question,
			History:           history,
			Datasets:          datasets,
			WebSearch:         webSearch,
			OnApproval: func(event harnessApprovalEvent) {
				emit(AsyncChatResult{Final: false, ThinkEvent: approvalThinkEvent(event)})
			},
			OnInteraction: func(event harnessInteractionEvent) {
				emit(AsyncChatResult{Final: false, ThinkEvent: interactionThinkEvent(event)})
			},
		})
		if err != nil {
			// A failed turn still spent tokens; log them so the cost of a timeout
			// or a dead provider stays visible.
			common.InfoCtx(ctx, "harness turn failed", zap.String("agent_id", chat.AgentID),
				zap.String("session_id", sessionID),
				zap.Int64("elapsed_ms", time.Since(turnStarted).Milliseconds()),
				zap.Int("tokens_in", result.TokensIn),
				zap.Int("tokens_out", result.TokensOut),
				zap.String("error", err.Error()))
			postHarnessGeneration(ctx, lfClient, lfTraceID, chat, turnStarted, result, "error: "+err.Error())
			emit(AsyncChatResult{Answer: fmt.Sprintf("**ERROR**: %s", harnessTurnErrorMessage(err)), Final: true})
			return
		}
		if result.Review != "" {
			// The review is a fact about the turn, not part of the answer: it lands
			// in the step panel so the answer stays the answer.
			emit(AsyncChatResult{Final: false, ThinkEvent: &ThinkEvent{
				Kind:    "review",
				Status:  "flagged",
				Summary: result.Review,
			}})
		}
		// The turn's own accounting, once, after the work is done: rounds, tokens,
		// cost, and how often the prompt prefix was rebuilt. Before this, only the
		// log carried them, so "why was this turn expensive" could not be answered
		// from the conversation.
		if stream {
			emit(AsyncChatResult{Final: false, ThinkEvent: usageThinkEvent(result)})
		}

		answer := result.Answer
		if strings.TrimSpace(answer) == "" {
			mu.Lock()
			answer = streamed.String()
			mu.Unlock()
		}
		// Citations off ships neither the markers nor the payload, exactly as the
		// pipeline's own quote rule does; the answer text stays as the agent wrote it.
		reference := map[string]interface{}{}
		if quote {
			marked := ""
			reference, marked = s.buildAgenticReference(turnCtx, chat.TenantID, datasets, answer, registry)
			if strings.TrimSpace(marked) != "" {
				answer = marked
			}
		}
		common.InfoCtx(ctx, "harness turn done", zap.String("agent_id", chat.AgentID),
			zap.String("session_id", sessionID),
			zap.Bool("stream", stream),
			zap.Int64("elapsed_ms", time.Since(turnStarted).Milliseconds()),
			zap.Int("rounds", result.Rounds),
			zap.Int("prefix_rebuilds", result.PrefixRebuilds),
			zap.Int("tokens_in", result.TokensIn),
			zap.Int("tokens_out", result.TokensOut),
			zap.Float64("cost_usd", result.CostUSD),
			zap.String("stopped", result.Stopped),
			zap.Int("final_bytes", len(answer)))
		postHarnessGeneration(ctx, lfClient, lfTraceID, chat, turnStarted, result, answer)
		emit(AsyncChatResult{
			Answer:                answer,
			Reference:             reference,
			Final:                 true,
			AnswerIsAuthoritative: true,
		})
	}()
	return out, nil
}

// postHarnessGeneration records one turn's usage. A nil client (no Langfuse
// configured) makes it a no-op: observability never fails a turn.
func postHarnessGeneration(ctx context.Context, client *LangfuseClient, traceID string, chat *entity.Chat, started time.Time, result harnessTurnResult, output string) {
	if client == nil || traceID == "" {
		return
	}
	_ = client.PostGeneration(ctx, LangfuseGeneration{
		ID:        fmt.Sprintf("gen-%s", traceID),
		TraceID:   traceID,
		Name:      "harness_chat",
		Model:     chat.LLMID,
		StartTime: started.UTC().Format(time.RFC3339Nano),
		EndTime:   time.Now().UTC().Format(time.RFC3339Nano),
		Output:    output,
		Usage: &LangfuseUsage{
			PromptTokens:     result.TokensIn,
			CompletionTokens: result.TokensOut,
			TotalTokens:      result.TokensIn + result.TokensOut,
		},
		Metadata: map[string]interface{}{
			"rounds":          result.Rounds,
			"stopped":         result.Stopped,
			"cost_usd":        result.CostUSD,
			"prefix_rebuilds": result.PrefixRebuilds,
			"agent_id":        chat.AgentID,
		},
	})
}

// approvalThinkEvent renders one approval state for the chat SSE stream. The
// request_id is what the client posts back; a wait and its verdict carry the
// same id so a UI can replace the card instead of stacking two.
// usageThinkEvent renders a turn's accounting as a step the client can show
// beside the work. Summary stays a plain fallback for clients that render only
// text; the structured fields are what a UI should format, so the line can be
// localized and can omit numbers that do not apply.
//
// Cost and prefix rebuilds are omitted when zero: a turn that cost nothing to
// account for should not print "$0.0000", and a zero rebuild count is the
// healthy case, not a number worth a row.
func usageThinkEvent(result harnessTurnResult) *ThinkEvent {
	summary := fmt.Sprintf("%d rounds · %d in / %d out tokens", result.Rounds, result.TokensIn, result.TokensOut)
	if result.CostUSD > 0 {
		summary += fmt.Sprintf(" · $%.4f", result.CostUSD)
	}
	if result.PrefixRebuilds > 0 {
		summary += fmt.Sprintf(" · prefix rebuilt %d time(s)", result.PrefixRebuilds)
	}
	return &ThinkEvent{
		Kind:           "usage",
		Stage:          "usage",
		Summary:        summary,
		Rounds:         result.Rounds,
		TokensIn:       result.TokensIn,
		TokensOut:      result.TokensOut,
		CostUSD:        result.CostUSD,
		PrefixRebuilds: result.PrefixRebuilds,
	}
}

func approvalThinkEvent(event harnessApprovalEvent) *ThinkEvent {
	status := "pending"
	switch {
	case event.Decided && event.Allowed:
		status = "granted"
	case event.Decided:
		status = "denied"
	}
	summary := fmt.Sprintf("Tool %q requires approval (%s risk).", event.Tool, event.Risk)
	if event.Decided {
		summary = fmt.Sprintf("Tool %q was %s.", event.Tool, status)
	}
	return &ThinkEvent{
		Kind:      "approval",
		Tool:      event.Tool,
		Args:      event.Args,
		Diff:      event.Diff,
		Status:    status,
		Reason:    event.Reason,
		RequestID: event.RequestKey,
		Summary:   summary,
	}
}

// interactionThinkEvent renders one model→user question for the chat SSE
// stream. Pending carries the question; answered carries what came back — or
// nothing, when the wait expired. The request_id is the handle the client posts
// the answer with, and the same id on both events lets a UI replace the card.
func interactionThinkEvent(event harnessInteractionEvent) *ThinkEvent {
	status := "pending"
	summary := event.Question
	if event.Answered {
		status = "answered"
		summary = fmt.Sprintf("%s → %s", event.Question, event.Answer)
	} else if event.Context != "" {
		summary = fmt.Sprintf("%s (%s)", event.Question, event.Context)
	}
	return &ThinkEvent{
		Kind:      "interaction",
		Status:    status,
		Reason:    event.Context,
		RequestID: event.RequestKey,
		Summary:   summary,
	}
}

// splitHarnessConversation separates the question being asked from the turns that
// preceded it.
func splitHarnessConversation(messages []map[string]interface{}) ([]harnessmodel.Message, string) {
	history := make([]harnessmodel.Message, 0, len(messages))
	question := ""
	for index, raw := range messages {
		role, _ := raw["role"].(string)
		content, _ := raw["content"].(string)
		if role != string(harnessmodel.RoleUser) && role != string(harnessmodel.RoleAssistant) {
			continue
		}
		if strings.TrimSpace(content) == "" {
			continue
		}
		if index == len(messages)-1 && role == string(harnessmodel.RoleUser) {
			question = content
			continue
		}
		history = append(history, harnessmodel.Message{Role: harnessmodel.Role(role), Content: content})
	}
	return history, question
}

// conversationIDOf reads the conversation id a completion request carries.
func conversationIDOf(kwargs map[string]interface{}) string {
	if kwargs == nil {
		return ""
	}
	id, _ := kwargs["conversation_id"].(string)
	return id
}

// HarnessLessons supplies the lessons this agent has earned.
//
// The store records them, the loop renders them as "## Relevant experience",
// and until this was wired nothing read them back — the self-improvement loop
// was open at its last step. Returning a closure keeps the read lazy: a turn
// that never reaches the model (a rejected request) does not pay for it.
func HarnessLessons(runs *store.RunStore, tenantID, agentID string) func(context.Context) []string {
	if runs == nil {
		return nil
	}
	return func(ctx context.Context) []string {
		lessons, err := runs.ListLessons(ctx, tenantID, agentID, 8)
		if err != nil {
			return nil
		}
		out := make([]string, 0, len(lessons))
		for _, lesson := range lessons {
			title := strings.TrimSpace(lesson.Title)
			body := strings.TrimSpace(lesson.Body)
			switch {
			case title != "" && body != "":
				out = append(out, title+": "+body)
			case title != "":
				out = append(out, title)
			case body != "":
				out = append(out, body)
			}
		}
		return out
	}
}

// harnessWorkspace resolves the shell workspace for one conversation.
//
// The configured root is per worker; the subdirectory is per conversation, so
// one thread cannot read another's files and a long task can pick up on the
// next turn where it stopped. A directory that cannot be created disables the
// shell rather than falling back to the shared root: two conversations sharing
// a workspace is a correctness bug, not a degradation.
func harnessWorkspace(conversationID string) string {
	base := harnessSettings.ExecRoot
	if base == "" {
		return ""
	}
	segment := strings.Map(func(r rune) rune {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '-', r == '_':
			return r
		default:
			return -1
		}
	}, conversationID)
	if segment == "" {
		return base
	}
	dir := filepath.Join(base, segment)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		common.Warn("harness: conversation workspace unavailable, shell disabled", zap.Error(err))
		return ""
	}
	return dir
}
