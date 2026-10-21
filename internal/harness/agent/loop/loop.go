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
	"reflect"
	"strings"
	"sync"
	"time"

	"ragflow/internal/harness/agent/definition"
	"ragflow/internal/harness/agent/model"
	"ragflow/internal/harness/caps"
	agentctx "ragflow/internal/harness/context"
	"ragflow/internal/harness/graph/channels"
	"ragflow/internal/harness/graph/checkpoint"
	"ragflow/internal/harness/graph/constants"
	"ragflow/internal/harness/graph/graph"
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
	Model        model.Model
	Registry     *caps.Registry
	Policy       *policy.Engine
	Builder      *agentctx.Builder
	SystemPrompt string
	// Lessons returns experience entries injected into the system prompt (RSI P0).
	Lessons func(ctx context.Context) []string
	// Sink receives trace records; nil disables tracing.
	Sink func(StepRecord)
	// Concurrency bounds parallel tool execution.
	Concurrency int
	// Scope is the fallback scope when the context carries none.
	Scope caps.Scope
	// PromptCostPerToken / CompletionCostPerToken enable cost accounting.
	PromptCostPerToken     float64
	CompletionCostPerToken float64
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
	Scope       caps.Scope
}

const defaultMaxRounds = 40

// Build compiles the loop graph.
func Build(cfg Config) (*Runner, error) {
	deps := cfg.Deps
	if deps.Model == nil {
		return nil, fmt.Errorf("loop: model is required")
	}
	if deps.Registry == nil {
		deps.Registry = caps.NewRegistry()
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
		state.Pending = nil
		state.NewEvidence = 0
		state.ThinkTerminal = false
		state.EvidenceBefore = len(state.Evidence)

		var lessons []string
		if deps.Lessons != nil {
			lessons = deps.Lessons(ctx)
		}
		state.Prompt = deps.Builder.Build(deps.SystemPrompt, lessons, state.History)
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
		response, err := deps.Model.Complete(ctx, model.Request{
			Messages: state.Prompt,
			Tools:    deps.Registry.ToolSpecs(),
		})
		if err != nil {
			return nil, fmt.Errorf("loop: model call failed: %w", err)
		}

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
		if !caps.HasScope(ctx) {
			ctx = caps.WithScope(ctx, deps.Scope)
		}
		start := time.Now()

		calls := state.Pending
		results := make([]caps.Result, len(calls))
		failures := make([]error, len(calls))
		semaphore := make(chan struct{}, deps.Concurrency)
		var wait sync.WaitGroup
		for index, call := range calls {
			wait.Add(1)
			go func(index int, call PendingCall) {
				defer wait.Done()
				semaphore <- struct{}{}
				defer func() { <-semaphore }()
				result, err := deps.Registry.Invoke(ctx, caps.Call{
					ID:        call.ID,
					Tool:      call.Tool,
					Arguments: call.Arguments,
				})
				results[index] = result
				failures[index] = err
			}(index, call)
		}
		wait.Wait()

		for index, call := range calls {
			result := results[index]
			if failures[index] != nil {
				result = caps.ErrorResult(failures[index])
			}
			state.Budget.ToolCalls++
			content := result.Content
			if strings.TrimSpace(content) == "" {
				content = "(no output)"
			}
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
		unique := make([]caps.Evidence, 0, len(state.Evidence))
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
		if len(state.Pending) > 0 {
			return "act", nil
		}
		return "verify", nil
	}, map[string]string{"act": "act", "verify": "verify"}); err != nil {
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
		ctx = caps.WithScope(ctx, input.Scope)
	}
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

func newState(input Input, policyConfig definition.PolicyConfig, maxRounds int) *State {
	state := &State{
		RunID:     input.RunID,
		AgentName: input.AgentName,
		StartedMS: time.Now().UnixMilli(),
		History: []model.Message{{
			Role:    model.RoleUser,
			Content: input.UserMessage,
		}},
		SeenCalls: make(map[string]struct{}),
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
		RunID:       state.RunID,
		Text:        state.FinalText,
		Rounds:      state.Entry.Round,
		TokensIn:    state.Budget.TokensIn,
		TokensOut:   state.Budget.TokensOut,
		ToolCalls:   state.Budget.ToolCalls,
		Evidence:    state.Evidence,
		Errors:      state.Errors,
		Citations:   state.Citations,
		Verdict:     state.Verdict,
		Stopped:     state.Stopped,
		Interrupted: state.Verdict.Action == policy.ActionPause,
		FinalState:  state,
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

func ternary(condition bool, whenTrue, whenFalse string) string {
	if condition {
		return whenTrue
	}
	return whenFalse
}
