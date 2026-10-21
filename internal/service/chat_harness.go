package service

import (
	"context"
	"errors"
	"fmt"
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
	Question          string
	History           []harnessmodel.Message
	Datasets          []string
	WebSearch         agentic_rag.WebSearchFunc
	// OnApproval reports a tool call that needs a human. The run blocks while the
	// verdict is outstanding, so the callback must return immediately.
	OnApproval func(event harnessApprovalEvent)
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
	}
	if cause == "" {
		// Nothing recognisable: hand over the original rather than hide it.
		return message
	}

	if detail := providerErrorText(message); detail != "" {
		cause = fmt.Sprintf("%s (%s)", cause, detail)
	}
	return cause + ". Choose another model in this chat's settings and send again."
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

// harnessTurnResult is what one turn produced, already shaped for the chat
// response.
type harnessTurnResult struct {
	Answer    string
	Rounds    int
	TokensIn  int
	TokensOut int
	CostUSD   float64
	Stopped   string
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
		PolicyConfig:   resolved.Definition.Policy,
		MaxRounds:      resolved.Definition.Policy.MaxRounds,
		SkillsRoot:     harnessSettings.SkillsRoot,
		ExecRoot:       harnessSettings.ExecRoot,
		WebSearch:      req.WebSearch,
		Approver: harnessApprover{
			broker: proc.Approvals,
			report: req.OnApproval,
			auto:   resolved.Definition.Policy.Approval == definition.ApprovalAuto,
		},
		Grants:      proc.Grants,
		Breakers:    proc.Breakers,
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
		Answer:    outcome.Text,
		Rounds:    outcome.Rounds,
		TokensIn:  outcome.TokensIn,
		TokensOut: outcome.TokensOut,
		CostUSD:   outcome.CostUSD,
		Stopped:   outcome.Stopped,
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

		result, err := executor.Run(turnCtx, harnessTurnRequest{
			TenantID:  chat.TenantID,
			UserID:    userID,
			AgentID:   chat.AgentID,
			SessionID: sessionID,
			ModelRef:  chat.LLMID,
			// The chat's own failover list, so a dead provider costs one call
			// here exactly as it does on the agentic path.
			FallbackModelRefs: agenticFailoverModelIDs(chat),
			Question:          question,
			History:           history,
			Datasets:          datasets,
			WebSearch:         webSearch,
			OnApproval: func(event harnessApprovalEvent) {
				emit(AsyncChatResult{Final: false, ThinkEvent: approvalThinkEvent(event)})
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
			"rounds":   result.Rounds,
			"stopped":  result.Stopped,
			"cost_usd": result.CostUSD,
			"agent_id": chat.AgentID,
		},
	})
}

// approvalThinkEvent renders one approval state for the chat SSE stream. The
// request_id is what the client posts back; a wait and its verdict carry the
// same id so a UI can replace the card instead of stacking two.
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
		Status:    status,
		Reason:    event.Reason,
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
