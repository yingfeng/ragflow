// Package loop is the built-in agent loop: one fixed StateGraph executed by the
// harness graph engine.
//
//	prepare → think → act → observe → verify → prepare
//	              ↘ verify → finish
//
// Policies decide termination; round counts are only a backstop.
package loop

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"reflect"
	"strings"
	"sync"
	"time"

	"ragflow/internal/harness/agent/definition"
	"ragflow/internal/harness/agent/model"
	agentctx "ragflow/internal/harness/context"
	"ragflow/internal/harness/events"
	"ragflow/internal/harness/graph/channels"
	"ragflow/internal/harness/graph/checkpoint"
	"ragflow/internal/harness/graph/constants"
	"ragflow/internal/harness/graph/graph"
	"ragflow/internal/harness/toolkit"
	"ragflow/internal/harness/toolkit/spill"
	"ragflow/internal/harness/toolkit/toolsearch"
	// Blank import installs the Pregel engine into the graph package.
	_ "ragflow/internal/harness/graph/pregel"
	"ragflow/internal/harness/graph/types"
	"ragflow/internal/harness/policy"
)

// StepRecord is one trace line. It carries no message content.
type StepRecord struct {
	RunID      string
	Phase      string
	Round      int
	Tool       string
	ArgsHash   string
	Decision   string
	Reason     string
	DurationMS int64
	TokensIn   int
	TokensOut  int
}

// Deps are the runtime dependencies captured by node closures. They are never
// stored in State so the state stays serializable.
type Deps struct {
	Model    model.Model
	Registry *toolkit.Registry
	Policy   *policy.Engine
	Builder  *agentctx.Builder
	// ModelName and Provider label model-call audit events. They are labels
	// only: the model handle decides where the call goes.
	ModelName    string
	Provider     string
	SystemPrompt string
	// Breaker protects other runs from a provider that is down. Nil disables the
	// check, which is what a single-run test wants.
	Breaker *model.Breaker
	// Lessons returns experience entries injected into the system prompt (RSI P0).
	Lessons func(ctx context.Context) []string
	// Sink receives trace records; nil disables tracing.
	Sink func(StepRecord)
	// Concurrency bounds parallel tool execution.
	Concurrency int
	// MaxOutputTokens caps one completion. Zero leaves the provider default,
	// which is how a turn silently overruns the context window.
	MaxOutputTokens int
	// Scope is the fallback scope when the context carries none.
	Scope toolkit.Scope
	// PromptCostPerToken / CompletionCostPerToken enable cost accounting.
	PromptCostPerToken     float64
	CompletionCostPerToken float64
	// Approver gates tool calls before execution. Nil runs everything the
	// registry offers; a non-nil gate is where approvals attach.
	Approver toolkit.Approver
	// ToolSchemaBudget caps the schemas sent with each request. Zero uses
	// toolkit.SchemaBudgetTokens.
	ToolSchemaBudget int
}

// Config builds a runner.
type Config struct {
	Deps         Deps
	PolicyConfig definition.PolicyConfig
	MaxRounds    int
	Checkpointer checkpoint.BaseCheckpointer
	// InterruptBeforeAct pauses before tool execution for approval.
	InterruptBeforeAct bool
	// RequireCitations makes Completion pause when a final answer has no evidence.
	RequireCitations bool
}

// Runner executes the built-in loop.
type Runner struct {
	compiled     types.CompiledGraph
	deps         Deps
	policyConfig definition.PolicyConfig
	maxRounds    int
}

// Input starts a run.
type Input struct {
	RunID       string
	AgentName   string
	UserMessage string
	Scope       toolkit.Scope
}

const defaultMaxRounds = 40

// Build compiles the loop graph.
func Build(cfg Config) (*Runner, error) {
	deps := cfg.Deps
	if deps.Model == nil {
		return nil, fmt.Errorf("loop: model is required")
	}
	if deps.Registry == nil {
		deps.Registry = toolkit.NewRegistry()
	}
	if deps.Builder == nil {
		deps.Builder = agentctx.New(agentctx.DefaultConfig())
	}
	if deps.Concurrency <= 0 {
		deps.Concurrency = 4
	}
	engine := deps.Policy
	if engine == nil {
		engine = policy.New(policy.Budget{}, policy.Progress{}, policy.Completion{RequireCitations: cfg.RequireCitations}, policy.Safety{})
	}
	deps.Policy = engine
	registerRuntimeCapabilities(deps.Registry)

	maxRounds := cfg.MaxRounds
	if maxRounds <= 0 {
		maxRounds = cfg.PolicyConfig.MaxRounds
	}
	if maxRounds <= 0 {
		maxRounds = defaultMaxRounds
	}

	sg := graph.NewStateGraph(&State{})

	// Every node returns the whole State snapshot, so each channel must replace
	// its value. The schema-derived defaults accumulate slice fields element by
	// element, which would turn History/Pending into []interface{} and break the
	// map→struct round trip. Accumulation and de-duplication live in the nodes.
	for name := range stateChannelNames() {
		sg.AddChannel(name, channels.NewLastValue(nil))
	}

	// ── prepare: budget clock, prompt assembly, per-round counters ──
	sg.AddNode("prepare", func(ctx context.Context, raw interface{}) (interface{}, error) {
		state, err := asState(raw)
		if err != nil {
			return nil, err
		}
		start := time.Now()
		if state.StartedMS == 0 {
			state.StartedMS = start.UnixMilli()
		}
		state.Entry.Round++
		state.Budget.ElapsedMS = start.UnixMilli() - state.StartedMS
		state.RetryPrepare = false
		state.Pending = nil
		state.NewEvidence = 0
		state.ThinkTerminal = false
		state.EvidenceBefore = len(state.Evidence)

		// Messages injected while the run was executing join the transcript
		// before the prompt is assembled, so the model sees them this round.
		if inbox := InboxFrom(ctx); inbox != nil {
			for _, message := range inbox.Drain() {
				state.History = append(state.History, model.Message{Role: model.RoleUser, Content: message})
				state.Steers++
			}
		}

		var lessons []string
		if deps.Lessons != nil {
			lessons = deps.Lessons(ctx)
		}
		state.Prompt = deps.Builder.BuildWithTier(deps.SystemPrompt, lessons, state.History, compactionTierFor(state.CompactionBoost))
		record(deps, StepRecord{
			RunID:      state.RunID,
			Phase:      "prepare",
			Round:      state.Entry.Round,
			DurationMS: time.Since(start).Milliseconds(),
		})
		return state, nil
	})

	// ── think: one model call ──
	sg.AddNode("think", func(ctx context.Context, raw interface{}) (interface{}, error) {
		state, err := asState(raw)
		if err != nil {
			return nil, err
		}
		start := time.Now()
		tools := toolkit.RestrictSpecs(deps.Registry.ToolSpecsWithinBudget(deps.ToolSchemaBudget), toolkit.AllowlistFrom(ctx))
		// The stable prefix is what a provider caches. Counting rebuilds tells an
		// operator whether the expensive part of the prompt is actually reused.
		prefix := prefixFingerprint(deps.SystemPrompt, tools)
		if state.PrefixHash != "" && state.PrefixHash != prefix {
			state.PrefixRebuilds++
		}
		state.PrefixHash = prefix
		response, err := callModel(ctx, deps.Model, model.Request{
			Messages:  state.Prompt,
			Tools:     tools,
			MaxTokens: deps.MaxOutputTokens,
		}, deps.Breaker)
		if err != nil {
			// A prompt-too-long rejection means the estimate was wrong. Shrink
			// the prompt and try again instead of failing the run; after
			// ptlAttempts recoveries the run genuinely cannot fit.
			if isPromptTooLongError(err) && state.PtLErrors < ptlAttempts {
				state.PtLErrors++
				state.CompactionBoost++
				state.RetryPrepare = true
				record(deps, StepRecord{
					RunID:      state.RunID,
					Phase:      "prepare",
					Round:      state.Entry.Round,
					Decision:   "recompact",
					Reason:     "prompt too long",
					DurationMS: time.Since(start).Milliseconds(),
				})
				return state, nil
			}
			if recorder := events.RecorderFromContext(ctx); recorder != nil {
				recorder.RecordError(ctx, err.Error())
			}
			return nil, fmt.Errorf("loop: model call failed: %w", err)
		}
		recordModelCall(ctx, deps, response, prefix, time.Since(start))

		state.Budget.TokensIn += response.Usage.PromptTokens
		state.Budget.TokensOut += response.Usage.CompletionTokens
		if deps.PromptCostPerToken > 0 || deps.CompletionCostPerToken > 0 {
			state.Budget.CostUSD += float64(response.Usage.PromptTokens)*deps.PromptCostPerToken +
				float64(response.Usage.CompletionTokens)*deps.CompletionCostPerToken
		}

		assistant := response.Message
		assistant.Role = model.RoleAssistant
		state.History = append(state.History, assistant)
		state.Pending = nil
		for _, call := range assistant.ToolCalls {
			key := idempotencyKey(call.Name, call.Arguments)
			state.Pending = append(state.Pending, PendingCall{
				ID:             call.ID,
				Tool:           call.Name,
				Arguments:      call.Arguments,
				IdempotencyKey: key,
			})
			if _, seen := state.SeenCalls[key]; seen {
				state.RepeatedCalls++
			} else {
				if state.SeenCalls == nil {
					state.SeenCalls = make(map[string]struct{})
				}
				state.SeenCalls[key] = struct{}{}
			}
		}
		state.ThinkTerminal = len(assistant.ToolCalls) == 0
		if state.ThinkTerminal {
			state.FinalText = strings.TrimSpace(assistant.Content)
		}

		record(deps, StepRecord{
			RunID:      state.RunID,
			Phase:      "think",
			Round:      state.Entry.Round,
			Tool:       toolNames(assistant.ToolCalls),
			DurationMS: time.Since(start).Milliseconds(),
			TokensIn:   response.Usage.PromptTokens,
			TokensOut:  response.Usage.CompletionTokens,
		})
		return state, nil
	})

	// ── act: execute the pending batch ──
	sg.AddNode("act", func(ctx context.Context, raw interface{}) (interface{}, error) {
		state, err := asState(raw)
		if err != nil {
			return nil, err
		}
		if len(state.Pending) == 0 {
			return state, nil
		}
		if !toolkit.HasScope(ctx) {
			ctx = toolkit.WithScope(ctx, deps.Scope)
		}
		// Bind the run id: approval requests and child runs are addressed by it.
		ctx = toolkit.WithRunID(ctx, state.RunID)
		// Publish the remaining budget so a spawned child cannot outlive this
		// run and leave it without turns to converge.
		if remaining := maxRounds - state.Entry.Round; remaining > 0 {
			ctx = toolkit.WithRemainingRounds(ctx, remaining)
		}
		start := time.Now()

		calls := state.Pending
		results := make([]toolkit.Result, len(calls))
		failures := make([]error, len(calls))

		// A graded breaker on repetition. The first repeats are allowed — a model
		// retrying after a transient failure is doing the right thing — but past
		// maxRepeatedToolCalls the same call is refused with an explanation, so a
		// loop costs a sentence instead of the rest of the budget. Counting runs
		// before the parallel batch so the map needs no lock.
		if state.CallRepeats == nil {
			state.CallRepeats = make(map[string]int)
		}
		blocked := make([]bool, len(calls))
		for index, call := range calls {
			key := call.IdempotencyKey
			if key == "" {
				key = idempotencyKey(call.Tool, call.Arguments)
			}
			state.CallRepeats[key]++
			if state.CallRepeats[key] > maxRepeatedToolCalls {
				blocked[index] = true
				state.BlockedCalls++
			}
		}
		if state.BlockedCalls > 0 {
			if recorder := events.RecorderFromContext(ctx); recorder != nil {
				recorder.RecordRetry(ctx, fmt.Sprintf("%d repeated tool call(s) blocked", state.BlockedCalls))
			}
		}

		invoke := func(index int) {
			call := calls[index]
			if blocked[index] {
				results[index] = toolkit.ErrorResult(fmt.Errorf(
					"identical call to %q has already run %d times; change the arguments or answer with what you have",
					call.Tool, maxRepeatedToolCalls))
				return
			}
			approvalCall := toolkit.Call{ID: call.ID, Tool: call.Tool, Arguments: call.Arguments}
			recorder := events.RecorderFromContext(ctx)
			if deps.Approver != nil {
				risk := deps.Registry.RiskOf(call.Tool)
				approvalStart := time.Now()
				decision := deps.Approver.Approve(ctx, approvalCall, risk)
				if recorder != nil {
					// The audit trail needs the verdict even when the call is
					// allowed: "who let this run" is the question asked later.
					verdict := "granted"
					if !decision.Allow {
						verdict = "denied"
					}
					recorder.RecordApproval(ctx, call.ID, call.Tool, map[string]any{
						"risk":   risk.String(),
						"reason": decision.Reason,
					}, verdict, time.Since(approvalStart).Milliseconds())
				}
				if !decision.Allow {
					// A rejection is a tool-level failure: the model sees why and
					// can choose another path instead of the run dying here.
					reason := decision.Reason
					if reason == "" {
						reason = "rejected"
					}
					failures[index] = fmt.Errorf("tool %q was not approved: %s", call.Tool, reason)
					return
				}
			}
			toolStart := time.Now()
			result, err := deps.Registry.Invoke(ctx, approvalCall)
			results[index] = result
			failures[index] = err
			if recorder != nil {
				recorder.RecordToolCall(ctx, call.Tool, decodeArguments(call.Arguments),
					clip(result.Content, auditResultLimit), time.Since(toolStart).Milliseconds(), 0, errorText(err))
			}
		}

		// Read-only calls may overlap; anything that changes state runs alone,
		// after the concurrent buffer ahead of it has drained. Batch order stays
		// the model's order, so a write still follows the reads it asked for.
		for index := 0; index < len(calls); {
			if !deps.Registry.Parallelizable(calls[index].Tool) {
				invoke(index)
				index++
				continue
			}
			batchStart := index
			for index < len(calls) && deps.Registry.Parallelizable(calls[index].Tool) {
				index++
			}
			semaphore := make(chan struct{}, deps.Concurrency)
			var wait sync.WaitGroup
			for position := batchStart; position < index; position++ {
				wait.Add(1)
				go func(position int) {
					defer wait.Done()
					select {
					case semaphore <- struct{}{}:
						defer func() { <-semaphore }()
					case <-ctx.Done():
						failures[position] = ctx.Err()
						return
					}
					invoke(position)
				}(position)
			}
			wait.Wait()
		}

		inlineBudget := aggregateInlineBudget(results, spill.DefaultInlineBytes)

		for index, call := range calls {
			result := results[index]
			if failures[index] != nil {
				result = toolkit.ErrorResult(failures[index])
			}
			state.Budget.ToolCalls++
			content := result.Content
			if strings.TrimSpace(content) == "" {
				content = "(no output)"
			}
			// Oversized output leaves the context window here; the model gets a
			// preview and a handle instead of paying for the whole payload.
			content = spill.MaybeSpill(ctx, call.Tool, content, inlineBudget)
			state.History = append(state.History, model.Message{
				Role:       model.RoleTool,
				ToolCallID: call.ID,
				Name:       call.Tool,
				Content:    content,
			})
			if result.IsError {
				state.Errors = append(state.Errors, StepError{Tool: call.Tool, Message: clip(content, 256)})
				if strings.Contains(content, "outside the run scope") {
					state.ScopeViolations++
				}
			}
			state.Evidence = append(state.Evidence, result.Evidence...)
			record(deps, StepRecord{
				RunID:    state.RunID,
				Phase:    "act",
				Round:    state.Entry.Round,
				Tool:     call.Tool,
				ArgsHash: call.IdempotencyKey,
				Decision: ternary(result.IsError, "error", "ok"),
			})
		}
		state.Pending = nil
		state.UnresolvedCalls = 0
		record(deps, StepRecord{
			RunID:      state.RunID,
			Phase:      "act_batch",
			Round:      state.Entry.Round,
			Decision:   fmt.Sprintf("%d calls", len(calls)),
			DurationMS: time.Since(start).Milliseconds(),
		})
		return state, nil
	})

	// ── observe: evidence accounting ──
	sg.AddNode("observe", func(ctx context.Context, raw interface{}) (interface{}, error) {
		state, err := asState(raw)
		if err != nil {
			return nil, err
		}
		seen := make(map[string]struct{}, len(state.Evidence))
		unique := make([]toolkit.Evidence, 0, len(state.Evidence))
		for _, item := range state.Evidence {
			key := item.Source + "|" + item.Ref
			if _, ok := seen[key]; ok {
				continue
			}
			seen[key] = struct{}{}
			unique = append(unique, item)
		}
		state.Evidence = unique
		state.Citations = len(unique)
		state.NewEvidence = len(unique) - state.EvidenceBefore
		if state.NewEvidence < 0 {
			state.NewEvidence = 0
		}
		if state.NewEvidence == 0 {
			state.StagnantRounds++
		} else {
			state.StagnantRounds = 0
		}
		return state, nil
	})

	// ── verify: policies decide ──
	sg.AddNode("verify", func(ctx context.Context, raw interface{}) (interface{}, error) {
		state, err := asState(raw)
		if err != nil {
			return nil, err
		}
		state.Verdict = deps.Policy.Decide(snapshotOf(state))
		if recorder := events.RecorderFromContext(ctx); recorder != nil {
			recorder.RecordVerdict(ctx, string(state.Verdict.Action), state.Verdict.Reason, state.Entry.Round, len(state.Evidence))
		}
		record(deps, StepRecord{
			RunID:    state.RunID,
			Phase:    "verify",
			Round:    state.Entry.Round,
			Decision: string(state.Verdict.Action),
			Reason:   state.Verdict.Reason,
		})
		return state, nil
	})

	// ── finish: terminal ──
	sg.AddNode("finish", func(ctx context.Context, raw interface{}) (interface{}, error) {
		state, err := asState(raw)
		if err != nil {
			return nil, err
		}
		if strings.TrimSpace(state.FinalText) == "" {
			for index := len(state.History) - 1; index >= 0; index-- {
				message := state.History[index]
				if message.Role == model.RoleAssistant && strings.TrimSpace(message.Content) != "" {
					state.FinalText = strings.TrimSpace(message.Content)
					break
				}
			}
		}
		state.Done = true
		state.Budget.ElapsedMS = time.Now().UnixMilli() - state.StartedMS
		if state.Verdict.Reason != "" {
			state.Stopped = string(state.Verdict.Action) + ": " + state.Verdict.Reason
		} else if state.Verdict.Action != "" && state.Verdict.Action != policy.ActionContinue {
			state.Stopped = string(state.Verdict.Action)
		}
		record(deps, StepRecord{
			RunID:    state.RunID,
			Phase:    "finish",
			Round:    state.Entry.Round,
			Decision: string(state.Verdict.Action),
			Reason:   state.Verdict.Reason,
		})
		return state, nil
	})

	// ── edges ──
	if err := sg.AddEdge(constants.Start, "prepare"); err != nil {
		return nil, fmt.Errorf("loop: start edge: %w", err)
	}
	if err := sg.AddEdge("prepare", "think"); err != nil {
		return nil, fmt.Errorf("loop: prepare→think: %w", err)
	}
	if err := sg.AddConditionalEdges("think", func(_ context.Context, raw interface{}) (interface{}, error) {
		state, err := asState(raw)
		if err != nil {
			return nil, err
		}
		if state.RetryPrepare {
			return "prepare", nil
		}
		if len(state.Pending) > 0 {
			return "act", nil
		}
		return "verify", nil
	}, map[string]string{"prepare": "prepare", "act": "act", "verify": "verify"}); err != nil {
		return nil, fmt.Errorf("loop: think branch: %w", err)
	}
	if err := sg.AddEdge("act", "observe"); err != nil {
		return nil, fmt.Errorf("loop: act→observe: %w", err)
	}
	// Every round passes through verify so that budgets and stagnation are
	// checked even when think never produces terminal text.
	if err := sg.AddEdge("observe", "verify"); err != nil {
		return nil, fmt.Errorf("loop: observe→verify: %w", err)
	}
	if err := sg.AddConditionalEdges("verify", func(_ context.Context, raw interface{}) (interface{}, error) {
		state, err := asState(raw)
		if err != nil {
			return nil, err
		}
		if state.Verdict.Action == policy.ActionContinue {
			return "prepare", nil
		}
		return "finish", nil
	}, map[string]string{"prepare": "prepare", "finish": "finish"}); err != nil {
		return nil, fmt.Errorf("loop: verify branch: %w", err)
	}
	if err := sg.AddEdge("finish", constants.End); err != nil {
		return nil, fmt.Errorf("loop: finish→end: %w", err)
	}

	compileOptions := []interface{}{
		graph.WithRecursionLimit((maxRounds + 1) * 6),
	}
	if cfg.Checkpointer != nil {
		compileOptions = append(compileOptions, graph.WithCheckpointer(cfg.Checkpointer))
	}
	if cfg.InterruptBeforeAct {
		compileOptions = append(compileOptions, graph.WithInterrupts("act"))
	}
	compiled, err := sg.Compile(compileOptions...)
	if err != nil {
		return nil, fmt.Errorf("loop: compile: %w", err)
	}

	return &Runner{compiled: compiled, deps: deps, policyConfig: cfg.PolicyConfig, maxRounds: maxRounds}, nil
}

// Invoke runs the loop. A nil input resumes from the checkpoint named by config.
func (r *Runner) Invoke(ctx context.Context, input *Input, config *types.RunnableConfig) (Outcome, error) {
	var state interface{}
	runID := ""
	if input != nil {
		state = newState(*input, r.policyConfig, r.maxRounds)
		runID = input.RunID
		ctx = toolkit.WithScope(ctx, input.Scope)
	}
	// One allowlist per run: a skill that narrows the tools during round three
	// must still narrow them in round four, and the holder is shared through the
	// context every node receives.
	ctx = toolkit.WithAllowlist(ctx)
	if config == nil {
		config = types.NewRunnableConfig()
	}
	if runID != "" {
		if config.Configurable == nil {
			config.Configurable = make(map[string]interface{})
		}
		config.ThreadID = runID
		config.Configurable[constants.ConfigKeyThreadID] = runID
	}

	result, err := r.compiled.Invoke(ctx, state, config)
	if err != nil {
		return Outcome{}, err
	}
	final, ok := result.(*State)
	if !ok {
		return Outcome{}, fmt.Errorf("loop: unexpected final state %T", result)
	}
	return outcomeOf(final), nil
}

// Resume loads the checkpoint for runID and continues.
func (r *Runner) Resume(ctx context.Context, runID string) (Outcome, error) {
	config := types.NewRunnableConfig()
	config.ThreadID = runID
	config.Configurable[constants.ConfigKeyThreadID] = runID
	return r.Invoke(ctx, nil, config)
}

// registerRuntimeCapabilities installs the capabilities the loop itself depends
// on. A host-provided tool_search wins, so a deployment can replace the search
// strategy without forking the loop.
func registerRuntimeCapabilities(registry *toolkit.Registry) {
	if registry == nil {
		return
	}
	if !registry.Has(toolsearch.ToolName) {
		_ = registry.Register(toolsearch.New(registry))
	}
	if !registry.Has("read_spill") {
		_ = registry.Register(spill.NewCapability())
	}
}

func newState(input Input, policyConfig definition.PolicyConfig, maxRounds int) *State {
	state := &State{
		RunID:     input.RunID,
		AgentName: input.AgentName,
		StartedMS: time.Now().UnixMilli(),
		History: []model.Message{{
			Role:    model.RoleUser,
			Content: input.UserMessage,
		}},
		SeenCalls:   make(map[string]struct{}),
		CallRepeats: make(map[string]int),
		Budget: Budget{
			MaxRounds:    maxRounds,
			MaxTokens:    policyConfig.MaxTokens,
			MaxToolCalls: policyConfig.MaxToolCalls,
			MaxWallMS:    int64(policyConfig.MaxWallSec) * 1000,
			MaxCostUSD:   policyConfig.MaxCostUSD,
		},
	}
	return state
}

func snapshotOf(state *State) policy.Snapshot {
	return policy.Snapshot{
		RunID:           state.RunID,
		Round:           state.Entry.Round,
		MaxRounds:       state.Budget.MaxRounds,
		MaxTokens:       state.Budget.MaxTokens,
		TokensIn:        state.Budget.TokensIn,
		TokensOut:       state.Budget.TokensOut,
		MaxToolCalls:    state.Budget.MaxToolCalls,
		ToolCalls:       state.Budget.ToolCalls,
		Elapsed:         time.Duration(state.Budget.ElapsedMS) * time.Millisecond,
		MaxWall:         time.Duration(state.Budget.MaxWallMS) * time.Millisecond,
		CostUSD:         state.Budget.CostUSD,
		MaxCostUSD:      state.Budget.MaxCostUSD,
		NewEvidence:     state.NewEvidence,
		StagnantRounds:  state.StagnantRounds,
		RepeatedCalls:   state.RepeatedCalls,
		HasFinalText:    strings.TrimSpace(state.FinalText) != "",
		Citations:       state.Citations,
		UnresolvedCalls: state.UnresolvedCalls,
		ScopeViolations: state.ScopeViolations,
		ThinkTerminal:   state.ThinkTerminal,
	}
}

func outcomeOf(state *State) Outcome {
	return Outcome{
		RunID:          state.RunID,
		Text:           state.FinalText,
		Rounds:         state.Entry.Round,
		TokensIn:       state.Budget.TokensIn,
		TokensOut:      state.Budget.TokensOut,
		CostUSD:        state.Budget.CostUSD,
		ToolCalls:      state.Budget.ToolCalls,
		Evidence:       state.Evidence,
		Errors:         state.Errors,
		Citations:      state.Citations,
		PrefixRebuilds: state.PrefixRebuilds,
		BlockedCalls:   state.BlockedCalls,
		Verdict:        state.Verdict,
		Stopped:        state.Stopped,
		Interrupted:    state.Verdict.Action == policy.ActionPause,
		FinalState:     state,
	}
}

// stateChannelNames lists every serialized State field. Channels are keyed by
// the Go field name so mapToStateSchema can rebuild the struct from them.
func stateChannelNames() map[string]struct{} {
	stateType := reflect.TypeOf(State{})
	names := make(map[string]struct{}, stateType.NumField())
	for i := 0; i < stateType.NumField(); i++ {
		field := stateType.Field(i)
		if field.PkgPath != "" {
			continue
		}
		names[field.Name] = struct{}{}
	}
	return names
}

func asState(raw interface{}) (*State, error) {
	switch typed := raw.(type) {
	case *State:
		return typed, nil
	case State:
		return &typed, nil
	case map[string]interface{}:
		payload, err := json.Marshal(typed)
		if err != nil {
			return nil, fmt.Errorf("loop: encode state: %w", err)
		}
		var state State
		if err := json.Unmarshal(payload, &state); err != nil {
			return nil, fmt.Errorf("loop: decode state: %w", err)
		}
		return &state, nil
	default:
		return nil, fmt.Errorf("loop: unexpected state type %T", raw)
	}
}

// compactionTierFor maps the prompt-too-long recovery count onto a forced
// compaction tier: one retry compacts history, a second keeps only the tail.
func compactionTierFor(boost int) agentctx.Tier {
	switch boostTier(boost) {
	case 0:
		return agentctx.TierNormal
	case 1:
		return agentctx.TierCompactHistory
	default:
		return agentctx.TierAggressivePrune
	}
}

func idempotencyKey(tool, arguments string) string {
	sum := sha256.Sum256([]byte(tool + "\x00" + arguments))
	return hex.EncodeToString(sum[:8])
}

func toolNames(calls []model.ToolCall) string {
	if len(calls) == 0 {
		return ""
	}
	names := make([]string, 0, len(calls))
	for _, call := range calls {
		names = append(names, call.Name)
	}
	return strings.Join(names, ",")
}

// Aggregate tool-output budget for one batch, borrowed from Astra's
// AGGREGATE_OUTPUT_BUDGET: a batch that returns many large results costs the
// same context as one huge result, so the per-result inline allowance shrinks as
// the batch total grows. Below the soft limit nothing changes.
const (
	aggregateSoftLimitBytes = 64 << 10
	aggregateFloorBytes     = 1 << 10
)

func aggregateInlineBudget(results []toolkit.Result, perCall int) int {
	if perCall <= 0 {
		perCall = spill.DefaultInlineBytes
	}
	total := 0
	for _, result := range results {
		total += len(result.Content)
	}
	if total <= aggregateSoftLimitBytes {
		return perCall
	}
	over := float64(total-aggregateSoftLimitBytes) / float64(aggregateSoftLimitBytes)
	if over > 1 {
		over = 1
	}
	scaled := int(float64(perCall) * (1 - 0.75*over))
	if scaled < aggregateFloorBytes {
		return aggregateFloorBytes
	}
	return scaled
}

func clip(value string, limit int) string {
	runes := []rune(value)
	if len(runes) <= limit {
		return value
	}
	return string(runes[:limit]) + "..."
}

func record(deps Deps, item StepRecord) {
	if deps.Sink != nil {
		deps.Sink(item)
	}
}

// auditResultLimit bounds the tool output copied into the audit log. The full
// result is already in the transcript, and the audit trail must stay a fact log
// rather than a second copy of every payload.
const auditResultLimit = 4 << 10

// recordModelCall writes one model round to the event log. The prompt is left
// out on purpose: it is reconstructible from the transcript, and copying it here
// would double the size of the log for every round.
func recordModelCall(ctx context.Context, deps Deps, response model.Response, prefix string, elapsed time.Duration) {
	recorder := events.RecorderFromContext(ctx)
	if recorder == nil {
		return
	}
	cost := float64(response.Usage.PromptTokens)*deps.PromptCostPerToken +
		float64(response.Usage.CompletionTokens)*deps.CompletionCostPerToken
	recorder.RecordModelCall(ctx, events.ModelCall{
		Model:      deps.ModelName,
		Provider:   deps.Provider,
		Content:    response.Message.Content,
		PrefixHash: prefix,
		Tokens: events.TokenUsage{
			PromptTokens:     response.Usage.PromptTokens,
			CompletionTokens: response.Usage.CompletionTokens,
			TotalTokens:      response.Usage.PromptTokens + response.Usage.CompletionTokens,
		},
		DurationMs:         elapsed.Milliseconds(),
		Cost:               cost,
		CachedPromptTokens: response.Usage.CachedPromptTokens,
	})
}

// prefixFingerprint identifies the part of the prompt a provider can cache: the
// system prompt and the tool schemas. Conversation content is deliberately left
// out — it changes every round, and including it would make every round look like
// a cache rebuild.
func prefixFingerprint(systemPrompt string, tools []model.ToolSpec) string {
	digest := sha256.New()
	io.WriteString(digest, systemPrompt)
	for _, spec := range tools {
		io.WriteString(digest, "\x00")
		io.WriteString(digest, spec.Name)
	}
	return hex.EncodeToString(digest.Sum(nil))[:16]
}

// decodeArguments turns the model's raw JSON arguments into the map the audit
// payload expects. Malformed JSON is preserved as a single raw field so the
// record still shows what the model actually sent.
func decodeArguments(raw string) map[string]any {
	if strings.TrimSpace(raw) == "" {
		return nil
	}
	decoded := make(map[string]any)
	if err := json.Unmarshal([]byte(raw), &decoded); err != nil {
		return map[string]any{"_raw": raw}
	}
	return decoded
}

func errorText(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

func ternary(condition bool, whenTrue, whenFalse string) string {
	if condition {
		return whenTrue
	}
	return whenFalse
}
