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
	"errors"
	"fmt"
	"io"
	"reflect"
	"regexp"
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
	RunID     string
	AgentName string
	// Messages is the conversation the run continues: earlier user and assistant
	// turns plus the question being asked. UserMessage is the convenience form
	// for a single-turn run and is appended when Messages does not already end
	// with it. Tool messages from earlier turns are never replayed — a tool
	// result out of its turn is misleading context.
	Messages    []model.Message
	UserMessage string
	Scope       toolkit.Scope
}

// defaultMaxRounds is a runaway guard, not a budget. The context window plus
// compaction is what ends a long turn — the same bound codex uses — so this
// number only has to be large enough that no legitimate investigation reaches
// it. A caller that wants a tighter loop sets policy.max_rounds explicitly.
const defaultMaxRounds = 1000

// answerInstruction closes a run that stopped collecting evidence without
// producing an answer. Without it the run ends with whatever the last think
// emitted, which for a tool-calling round is nothing at all: the user paid for
// the retrieval and got an empty message.
const answerInstruction = `Stop searching and answer the user's question now, in the question's own language.
Use only the conversation above, including the tool results already collected. Reply with the final answer text directly: no tool calls, no plan, no description of what you searched, and no mention of this instruction.
Before giving a count or a list, walk the tool results above and include every matching item found there — a count is only as complete as that list, and the last passage you read is not the whole search.
If the collected evidence genuinely cannot answer the question, say what is missing and state the best-supported partial answer instead of leaving the reply empty.`

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
		state.NewOutputs = 0
		state.ThinkTerminal = false
		state.EvidenceBefore = len(state.Evidence)
		state.EvidenceCalls = 0

		// Messages injected while the run was executing join the transcript
		// before the prompt is assembled, so the model sees them this round.
		if inbox := InboxFrom(ctx); inbox != nil {
			for _, message := range inbox.Drain() {
				state.History = append(state.History, model.Message{Role: model.RoleUser, Content: message})
				state.Steers++
			}
		}
		// Budget pressure is reported as fact, once. What the model does with the
		// fact is the model's decision: the limits themselves are already
		// enforced by policy, and restating them as instructions only teaches a
		// weak model to give up before the budget is actually spent.
		if !state.BudgetNotified {
			if notice := budgetNotice(state.Budget); notice != "" {
				state.BudgetNotified = true
				state.History = append(state.History, model.Message{Role: model.RoleUser, Content: notice})
			}
		}

		var lessons []string
		if deps.Lessons != nil {
			lessons = deps.Lessons(ctx)
		}
		speculator := speculatorFrom(ctx)
		var hint *agentctx.FoldHint
		if speculator != nil {
			hint = speculator.Hint()
		}
		var usedHint bool
		state.Prompt, usedHint = deps.Builder.BuildWithFoldHint(deps.SystemPrompt, lessons, state.History, compactionTierFor(state.CompactionBoost), hint)
		if usedHint {
			speculator.NoteHintAdopted()
		}
		// The plan rides after the transcript: the cacheable system prefix stays
		// byte-identical across rounds while the model still sees what is left.
		//
		// Only when it CHANGES. A step list restated byte-for-byte every round is
		// harness-only noise — the loop already carries the model's own plan
		// message in History — and a standing reminder of the open steps reads as
		// a standing reminder that they are the whole job.
		if plan := renderPlan(state.Plan, false); plan != "" {
			if fingerprint := planFingerprint(state.Plan); fingerprint != state.PlanShown {
				state.PlanShown = fingerprint
				state.Prompt = append(state.Prompt, model.Message{Role: model.RoleUser, Content: plan})
			}
		}
		// The evidence pool is NOT injected while the loop is searching. It reads
		// as a list of what has already been found, and a model handed that list
		// stops searching: measured on one enumeration, adding it to prepare cut
		// the run from 29 rounds to 8 and the answer from 15 names to 9 — the
		// model answered from the pool instead of continuing to search outside it.
		// The pool belongs to the answer call, where a list is what the output
		// needs (see the answer node).
		//
		// The question is re-stated every round, and it has to be. A chat turn
		// arrives with the whole thread in History, so the prompt carries every
		// earlier question too; without this line the model lost track of which
		// one it was answering and answered an older turn's question instead
		// (observed: a question about 三国 answered a list of 秦 and 霸王别姬
		// questions from earlier turns). The wording names the turn's question and
		// says the passages are evidence for it — it does NOT say "answer now",
		// which is what made runs stop searching after a few rounds.
		if anchor := questionAnchor(state.Question); anchor != "" {
			state.Prompt = append(state.Prompt, model.Message{Role: model.RoleUser, Content: anchor})
		}
		// Speculate now, on the same transcript this prompt was built from: the
		// summary is written while think and act do their work, so the round
		// that has to fold finds it ready instead of paying for it inline.
		if speculator != nil {
			speculator.MaybeStart(deps.SystemPrompt, lessons, state.History)
		}
		record(deps, StepRecord{
			RunID:      state.RunID,
			Phase:      "prepare",
			Round:      state.Entry.Round,
			Decision:   ternary(usedHint, "speculated", ""),
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
		request := model.Request{
			Messages:  state.Prompt,
			Tools:     tools,
			MaxTokens: deps.MaxOutputTokens,
			// The required-tool gate: until the transcript shows one required
			// tool call after the turn's last user message, the model is told
			// which tool to use instead of asked. Without it, the first round
			// can go to pure planning and produce no evidence at all.
			ToolChoice: requiredToolChoice(tools, deps.Registry.RequiredTools(), state.History),
		}
		// A round's text is held until the round is over, because only then does
		// the loop know what it was. A round that ends the turn produced the
		// answer; a round that goes on to call a tool produced working text that
		// belongs in the step panel. Sending both to the answer area is what let
		// one round print a draft the next round overwrote.
		answerSink := DeltaSinkFromContext(ctx)
		workingSink := ReasoningSinkFromContext(ctx)
		var held []string
		var response model.Response
		if streamer, ok := deps.Model.(model.Streamer); ok && (answerSink != nil || workingSink != nil) {
			response, err = callModelStreaming(ctx, streamer, request, deps.Breaker, func(text string) {
				held = append(held, text)
			})
		} else {
			response, err = callModel(ctx, deps.Model, request, deps.Breaker)
		}
		if err != nil {
			// A prompt-too-long rejection means the estimate was wrong. Shrink
			// the prompt and try again instead of failing the run; after
			// ptlAttempts recoveries the run genuinely cannot fit.
			if isPromptTooLongError(err) && state.PtLErrors < ptlAttempts {
				// The round is about to be replayed on a smaller prompt, so what
				// it streamed is a draft: sending it to the answer area would
				// leave it in front of the text the replay writes there.
				emitReasoning(held, workingSink)
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
			// The provider produced this text before it failed. It counts as
			// shown — the retry path refuses to replay a call that streamed — so
			// it reaches the answer area instead of vanishing with the error.
			flushAnswerText(held, answerSink)
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
			// A turn that ends without a tool call is not automatically an answer.
			// "Let me see the final truncated chunk and continue searching" is the
			// model narrating its next step; delivering that as the reply leaves
			// the user with a sentence about work that never happened. Treat it as
			// no answer so the answer call writes one from the evidence already
			// collected.
			state.FinalText = ""
			if text := strings.TrimSpace(assistant.Content); !looksLikeNarration(text) {
				state.FinalText = text
			}
			// Consecutive rounds that produced only prose. Policies read it to
			// tell "still working on an open plan step" from "restating a partial
			// answer": only the first deserves another round.
			state.TerminalRounds++
		} else {
			state.TerminalRounds = 0
			// A round that goes on to call tools has concluded nothing. Keeping an
			// earlier draft here is how a run that wrote an answer and then kept
			// researching delivered the pre-research answer: the loop finished with
			// text it had already invalidated.
			state.FinalText = ""
		}
		// Now that the round is known to be the turn's end or not, the held text
		// has exactly one home: the answer area for the round that ends the turn,
		// the step panel for the round that went on to call a tool.
		//
		// A terminal round whose text was rejected as narration wrote no answer,
		// so its words are working text as well: flushing them would park "let me
		// check that" in the answer area, where the grace round or the answer
		// call then streams the real reply underneath it.
		if state.ThinkTerminal && state.FinalText != "" {
			flushAnswerText(held, answerSink)
		} else {
			emitReasoning(held, workingSink)
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

		// A resumed run inherits the batch the dead process had dispatched.
		// Running a read-only call again costs nothing and recovers the progress;
		// running a state-changing call again can duplicate the effect it may have
		// already produced before the crash. Those are reported as unresolved so
		// the model decides, which is also what the reference runtime does: recovery
		// restores history, it does not grant replay authority.
		resumeGuard := IsResumed(ctx) && !state.ResumeHandled
		if resumeGuard {
			state.ResumeHandled = true
		}

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
			// Every pending call reports its lifecycle, including the ones that
			// never reach the tool. A blocked or non-replayed call is still an
			// event in the transcript, and a step panel left on "searching…"
			// because the call was refused before it started describes work the
			// run never did.
			finish := trackTool(ctx, toolCallView{tool: call.Tool, arguments: call.Arguments})
			defer func() {
				// A tool-level failure is reported in the result, not as an error
				// from Invoke: the call ran and failed, and the progress stream
				// has to show that the same way it shows a call that never
				// returned.
				finish(failures[index] != nil || results[index].IsError)
			}()
			if resumeGuard && !deps.Registry.Parallelizable(call.Tool) {
				results[index] = toolkit.ErrorResult(fmt.Errorf(
					"the %q call was dispatched before this run was interrupted and may already have taken effect, so it was not repeated; check the outcome or retry with different arguments",
					call.Tool))
				return
			}
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
				result = toolkit.ErrorResult(describeAbort(failures[index]))
			}
			state.Budget.ToolCalls++
			// Progress is measured on what the tool returned, not on what one
			// policy happens to call evidence: fresh content from a read-only tool
			// is progress even when it carries no citation, and a run that keeps
			// finding new material must not be mistaken for a stuck one.
			fingerprint := outputFingerprint(call.Tool, result.Content)
			if state.SeenOutputs == nil {
				state.SeenOutputs = make(map[string]struct{})
			}
			if _, seen := state.SeenOutputs[fingerprint]; !seen {
				state.SeenOutputs[fingerprint] = struct{}{}
				state.NewOutputs++
			}
			content := result.Content
			if strings.TrimSpace(content) == "" {
				content = "(no output)"
			}
			if !result.Inline {
				// Oversized output leaves the context window here; the model gets
				// a preview and a handle instead of paying for the whole payload.
				// An Inline result is exempt: it is a tool that already bounded
				// its own output (a retrieval window — grep's 30 matches,
				// list_chunks' one chunk window). Re-spilling it does not shrink
				// the payload, it only moves it behind a handle: the preview is a
				// preview of the preview, and the model has to spend one read_spill
				// round to get back to the text it just asked for. That round comes
				// out of the same budget the searches need. The per-result ceiling
				// in the context builder (MaxToolResultBytes, 64 KiB) is the only
				// guard a bounded window still needs.
				content = spill.MaybeSpill(ctx, content, inlineBudget)
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
			if len(result.Evidence) > 0 {
				state.EvidenceCalls++
			}
			if plan, ok := result.Meta[toolkit.MetaPlan].([]toolkit.PlanStep); ok && len(plan) > 0 {
				// Only a plan that actually changed counts as progress: a model
				// republishing the same three open steps every round would
				// otherwise reset the idle clock forever.
				if fingerprint := planFingerprint(plan); fingerprint != planFingerprint(state.Plan) {
					state.PlanRound = state.Entry.Round
				}
				state.Plan = plan
			}
			record(deps, StepRecord{
				RunID:    state.RunID,
				Phase:    "act",
				Round:    state.Entry.Round,
				Tool:     call.Tool,
				ArgsHash: call.IdempotencyKey,
				Decision: ternary(result.IsError, "error", "ok"),
				Reason:   describeCall(call, result),
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
		// Progress is new evidence units, not new output text.
		//
		// A different regex, a rephrased BM25 query or a second slice of the same
		// chunk all return fresh text over the same chunks, so an output
		// fingerprint advances every round while the pool stops growing. That is
		// how a retrieval run spent 47 rounds re-reading its own hits: every
		// round looked productive, so nothing stopped it.
		//
		// Tools that publish no evidence units (shell, files, planning) keep the
		// output-based signal, and a run that has never produced evidence has no
		// pool to judge by.
		switch {
		case state.NewEvidence > 0:
			state.StagnantRounds = 0
		case state.EvidenceCalls > 0:
			// Searched and the pool did not grow. This is the round that stops a
			// run grinding through rewrite after rewrite of the same query.
			state.StagnantRounds++
		case state.NewOutputs > 0 && len(state.Evidence) == 0:
			// No evidence model at all (shell, files): output is the only signal.
			state.StagnantRounds = 0
		default:
			// read_spill and planning: reading what was already retrieved is
			// neither progress nor stagnation, so the counter stays put.
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
		if state.Verdict.Action == policy.ActionNudge {
			// The correction joins the transcript as a fact from the run's own
			// instrumentation, and the counter it reports restarts: the next
			// stall is a new stall, not the same one counted twice.
			if advice := strings.TrimSpace(state.Verdict.Advice); advice != "" {
				state.History = append(state.History, model.Message{Role: model.RoleUser, Content: advice})
			}
			state.Nudges++
			state.StagnantRounds = 0
		}
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

	// ── answer: turn collected evidence into the final answer ──
	//
	// Policies decide when to stop; they cannot know that stopping mid-search
	// leaves nothing to show. This node is the one place that spends a model call
	// after the loop, and only when the run would otherwise end with no text.
	sg.AddNode("answer", func(ctx context.Context, raw interface{}) (interface{}, error) {
		state, err := asState(raw)
		if err != nil {
			return nil, err
		}
		// Only a run that already holds text may skip this call. A pause used to
		// skip it too, on the assumption that a human would resume the run — but
		// no resume entry exists on the chat path, so a paused run handed the
		// caller an empty message. One call for a partial answer that names its
		// own truncation is worth more than silence.
		if strings.TrimSpace(state.FinalText) != "" {
			return state, nil
		}
		start := time.Now()
		// Prompt is the view prepare built before the last round's tool results
		// existed. Rebuilding from History is what puts the evidence the loop just
		// paid for in front of the model.
		var lessons []string
		if deps.Lessons != nil {
			lessons = deps.Lessons(ctx)
		}
		messages := deps.Builder.BuildWithTier(deps.SystemPrompt, lessons, state.History, compactionTierFor(state.CompactionBoost))
		if plan := renderPlan(state.Plan, true); plan != "" {
			messages = append(messages, model.Message{Role: model.RoleUser, Content: plan})
		}
		// The evidence pool lives outside the transcript. Compaction trims
		// History as the context fills, so a long run reaches this call holding
		// only its last few rounds: the passages that proved the earliest items
		// are gone from the prompt even though the loop still holds them. That is
		// how a 50-round run answered with fewer names than a 6-round one. Feeding
		// the pool here is what makes "answer from what you collected" true.
		if evidence := renderEvidencePool(state.Evidence); evidence != "" {
			messages = append(messages, model.Message{Role: model.RoleUser, Content: evidence})
		}
		if anchor := questionAnchor(state.Question); anchor != "" {
			messages = append(messages, model.Message{Role: model.RoleUser, Content: anchor})
		}
		if note := stoppingNote(state); note != "" {
			messages = append(messages, model.Message{Role: model.RoleUser, Content: note})
		}
		messages = append(messages, model.Message{Role: model.RoleUser, Content: answerInstruction})
		request := model.Request{
			Messages:  messages,
			MaxTokens: deps.MaxOutputTokens,
		}
		var response model.Response
		if streamer, ok := deps.Model.(model.Streamer); ok && DeltaSinkFromContext(ctx) != nil {
			response, err = callModelStreaming(ctx, streamer, request, deps.Breaker, DeltaSinkFromContext(ctx))
		} else {
			response, err = callModel(ctx, deps.Model, request, deps.Breaker)
		}
		if err != nil {
			// The evidence is already paid for; failing the run here would throw
			// it away. Record the failure and let finish report what it has.
			state.Errors = append(state.Errors, StepError{Tool: "answer", Message: clip(err.Error(), 256)})
			if recorder := events.RecorderFromContext(ctx); recorder != nil {
				recorder.RecordError(ctx, err.Error())
			}
			// A failed concluding call must not become an empty message: the run
			// already paid for the evidence and may already have written prose
			// this turn. Fall back to it rather than deliver nothing.
			if strings.TrimSpace(state.FinalText) == "" {
				state.FinalText = lastAssistantText(state.History)
			}
			record(deps, StepRecord{
				RunID:      state.RunID,
				Phase:      "answer",
				Round:      state.Entry.Round,
				Decision:   "error",
				Reason:     clip(err.Error(), 200),
				DurationMS: time.Since(start).Milliseconds(),
			})
			return state, nil
		}
		recordModelCall(ctx, deps, response, state.PrefixHash, time.Since(start))

		state.Budget.TokensIn += response.Usage.PromptTokens
		state.Budget.TokensOut += response.Usage.CompletionTokens
		if deps.PromptCostPerToken > 0 || deps.CompletionCostPerToken > 0 {
			state.Budget.CostUSD += float64(response.Usage.PromptTokens)*deps.PromptCostPerToken +
				float64(response.Usage.CompletionTokens)*deps.CompletionCostPerToken
		}
		assistant := response.Message
		assistant.Role = model.RoleAssistant
		state.History = append(state.History, assistant)
		state.FinalText = strings.TrimSpace(assistant.Content)
		if state.FinalText == "" {
			state.Errors = append(state.Errors, StepError{Tool: "answer", Message: "answer model returned no text"})
			state.FinalText = lastAssistantText(state.History)
		}
		record(deps, StepRecord{
			RunID:      state.RunID,
			Phase:      "answer",
			Round:      state.Entry.Round,
			Decision:   ternary(state.FinalText != "", "ok", "empty"),
			DurationMS: time.Since(start).Milliseconds(),
			TokensIn:   response.Usage.PromptTokens,
			TokensOut:  response.Usage.CompletionTokens,
		})
		return state, nil
	})

	// ── finish: terminal ──
	sg.AddNode("finish", func(ctx context.Context, raw interface{}) (interface{}, error) {
		state, err := asState(raw)
		if err != nil {
			return nil, err
		}
		state.Done = true
		state.Budget.ElapsedMS = time.Now().UnixMilli() - state.StartedMS
		label := string(state.Verdict.Action)
		if state.Verdict.Code != "" {
			// A stable label beats the action: "budget_limited" says why the run
			// ended, "finish" only says that it did.
			label = state.Verdict.Code
		}
		if state.Verdict.Reason != "" {
			state.Stopped = label + ": " + state.Verdict.Reason
		} else if state.Verdict.Action != "" && state.Verdict.Action != policy.ActionContinue {
			state.Stopped = label
		}
		// The model's own plan says how much of its stated work was left when the
		// run ended. Without it a stopped record cannot be told apart from a
		// complete one, which is what made "12 of 16" invisible in the log.
		if total, open := planCounts(state.Plan); open > 0 {
			state.Stopped += fmt.Sprintf("; %d/%d plan steps open", open, total)
		}
		record(deps, StepRecord{
			RunID:    state.RunID,
			Phase:    "finish",
			Round:    state.Entry.Round,
			Decision: string(state.Verdict.Action),
			// Stopped, not Verdict.Reason: it carries the plan coverage too, so a
			// stopped record can be told apart from a complete one.
			Reason: clip(state.Stopped, 400),
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
		if state.Verdict.Action == policy.ActionContinue || state.Verdict.Action == policy.ActionNudge {
			return "prepare", nil
		}
		return "answer", nil
	}, map[string]string{"prepare": "prepare", "answer": "answer"}); err != nil {
		return nil, fmt.Errorf("loop: verify branch: %w", err)
	}
	if err := sg.AddEdge("answer", "finish"); err != nil {
		return nil, fmt.Errorf("loop: answer→finish: %w", err)
	}
	if err := sg.AddEdge("finish", constants.End); err != nil {
		return nil, fmt.Errorf("loop: finish→end: %w", err)
	}

	// The loop owns retries for the nodes that reach the outside world: think
	// retries the model with a classified backoff (and knows whether text already
	// reached the user), and act must never replay a tool. The engine's default
	// policy retries every error, which would emit a streamed answer twice and run
	// a write tool twice.
	noRetry := &types.RetryPolicy{MaxAttempts: 1}
	for _, name := range []string{"think", "act", "answer"} {
		if node, ok := sg.GetNodes()[name]; ok {
			node.RetryPolicy = noRetry
		}
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
	// context every node receives. It is replaced rather than reused because a
	// delegated child arrives with its parent's context, and a shared holder
	// would let either run narrow the other's tool surface.
	ctx = toolkit.ReplaceAllowlist(ctx)
	// Compaction speculation is per run: its hint indexes this run's history.
	if r.deps.Builder != nil {
		ctx = withSpeculator(ctx, agentctx.NewSpeculator(r.deps.Builder))
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
//
// The resumed marker tells act that its pending batch was dispatched by a run
// that died: a call may have taken effect without recording a result, so
// re-running it is not the same as retrying it.
func (r *Runner) Resume(ctx context.Context, runID string) (Outcome, error) {
	config := types.NewRunnableConfig()
	config.ThreadID = runID
	config.Configurable[constants.ConfigKeyThreadID] = runID
	return r.Invoke(markResumed(ctx), nil, config)
}

type speculatorKey struct{}

// withSpeculator attaches the run's compaction speculation. It lives in the
// context rather than in State because it owns a goroutine, and State is
// serialized for checkpoints.
func withSpeculator(ctx context.Context, speculator *agentctx.Speculator) context.Context {
	return context.WithValue(ctx, speculatorKey{}, speculator)
}

func speculatorFrom(ctx context.Context) *agentctx.Speculator {
	speculator, _ := ctx.Value(speculatorKey{}).(*agentctx.Speculator)
	return speculator
}

type resumedKey struct{}

func markResumed(ctx context.Context) context.Context {
	return context.WithValue(ctx, resumedKey{}, true)
}

// IsResumed reports whether this process took over the run from a checkpoint
// written by an earlier one.
func IsResumed(ctx context.Context) bool {
	resumed, _ := ctx.Value(resumedKey{}).(bool)
	return resumed
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

// seedHistory builds the turn's conversation from the caller's history.
//
// Earlier tool calls and results are dropped: they were executed against a
// corpus snapshot and a question that is already answered, so replaying them
// only spends context. A trailing assistant message is kept as well — it is the
// answer the user is following up on.
func seedHistory(input Input) []model.Message {
	history := make([]model.Message, 0, len(input.Messages)+1)
	for _, message := range input.Messages {
		switch message.Role {
		case model.RoleUser, model.RoleAssistant:
			content := strings.TrimSpace(message.Content)
			if content == "" {
				continue
			}
			if message.Role == model.RoleAssistant {
				// A citation handle resolves against the turn that produced it.
				// Replaying it would present an earlier, unverified answer as if
				// this run had collected the evidence.
				content = stripCitationHandles(content)
			}
			history = append(history, model.Message{Role: message.Role, Content: content})
		}
	}
	if input.UserMessage != "" {
		last := len(history) - 1
		if last < 0 || history[last].Role != model.RoleUser || history[last].Content != input.UserMessage {
			history = append(history, model.Message{Role: model.RoleUser, Content: input.UserMessage})
		}
	}
	return history
}

func newState(input Input, policyConfig definition.PolicyConfig, maxRounds int) *State {
	state := &State{
		RunID:       input.RunID,
		AgentName:   input.AgentName,
		StartedMS:   time.Now().UnixMilli(),
		History:     seedHistory(input),
		Question:    turnQuestion(input),
		SeenCalls:   make(map[string]struct{}),
		SeenOutputs: make(map[string]struct{}),
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
	planTotal, planOpen := planCounts(state.Plan)
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
		Nudges:          state.Nudges,
		HasFinalText:    strings.TrimSpace(state.FinalText) != "",
		Citations:       state.Citations,
		UnresolvedCalls: state.UnresolvedCalls,
		ScopeViolations: state.ScopeViolations,
		ThinkTerminal:   state.ThinkTerminal,
		PlanSteps:       planTotal,
		OpenPlanSteps:   planOpen,
		PlanIdleRounds:  planIdleRounds(state),
		TerminalRounds:  state.TerminalRounds,
	}
}

// planIdleRounds is how many rounds have passed since the plan last changed. A
// plan with no open steps is not holding the run open, so it reports zero.
func planIdleRounds(state *State) int {
	if _, open := planCounts(state.Plan); open == 0 {
		return 0
	}
	if state.PlanRound <= 0 {
		return 0
	}
	idle := state.Entry.Round - state.PlanRound
	if idle < 0 {
		return 0
	}
	return idle
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

// lastAssistantText is the most recent thing the model said that was not a tool
// call. It is the fallback when the concluding call fails or returns nothing: a
// partial answer the model already wrote beats an empty message, and the caller
// has no way to recover a run that ends without text.
func lastAssistantText(history []model.Message) string {
	for index := len(history) - 1; index >= 0; index-- {
		message := history[index]
		if message.Role != model.RoleAssistant {
			continue
		}
		if text := strings.TrimSpace(message.Content); text != "" && !looksLikeNarration(text) {
			return text
		}
	}
	return ""
}

// narrationPrefixes are how a model announces the next action instead of
// answering. A terminal round that starts with one of these and produced no
// tool call ended without an answer.
var narrationPrefixes = []string{
	"let me", "let's", "let us", "i'll", "i will", "i should", "i need to",
	"now let me", "now i ", "next,", "next i", "continuing", "i'm going to",
	"let me see", "i am going to",
}

// looksLikeNarration reports that a terminal message is a statement of intent
// rather than an answer. Empty text counts: nothing was answered.
func looksLikeNarration(text string) bool {
	trimmed := strings.ToLower(strings.TrimSpace(text))
	if trimmed == "" {
		return true
	}
	for _, prefix := range narrationPrefixes {
		if strings.HasPrefix(trimmed, prefix) {
			return true
		}
	}
	return false
}

// turnQuestion is what this run was asked: the explicit user message, else the
// last user turn of the seeded history.
func turnQuestion(input Input) string {
	if question := strings.TrimSpace(input.UserMessage); question != "" {
		return question
	}
	for index := len(input.Messages) - 1; index >= 0; index-- {
		if input.Messages[index].Role != model.RoleUser {
			continue
		}
		if text := strings.TrimSpace(input.Messages[index].Content); text != "" {
			return text
		}
	}
	return ""
}

// questionAnchor restates the request at the end of the prompt, where compaction
// cannot reach it.
//
// Compaction drops messages from the front of the history, and the request is
// the first thing a long run loses. Without an anchor the run answers whatever
// it just read: a search for Guan Yu's kills came back with a lecture on Liu
// Bei's teacher, because a passage about Zheng Xuan was the most recent context
// and nothing in the prompt still said what had been asked.
// requiredToolChoice decides whether this sampling request must name a tool.
//
// The gate follows the agentic pipeline's retrieval gate: scan back through the
// transcript to the turn's last user message; if no required tool was called
// after it, force the first required tool that is visible in this request. One
// real call lifts the gate — after that the model chooses freely again, and a
// turn that already used one is never re-forced.
func requiredToolChoice(tools []model.ToolSpec, required []string, history []model.Message) string {
	if len(required) == 0 || len(tools) == 0 {
		return ""
	}
	visible := make(map[string]bool, len(tools))
	for _, spec := range tools {
		visible[spec.Name] = true
	}
	needed := make(map[string]bool, len(required))
	for _, name := range required {
		needed[name] = true
	}
	for index := len(history) - 1; index >= 0; index-- {
		message := history[index]
		if message.Role == model.RoleUser {
			break // the turn's question; nothing before it counts
		}
		if message.Role != model.RoleAssistant {
			continue
		}
		for _, call := range message.ToolCalls {
			if visible[call.Name] && needed[call.Name] {
				return ""
			}
		}
	}
	for _, name := range required {
		if visible[name] {
			return name
		}
	}
	return ""
}

func questionAnchor(question string) string {
	trimmed := strings.TrimSpace(question)
	if trimmed == "" {
		return ""
	}
	return "The question for this turn, unchanged:\n" + trimmed +
		"\n\nThe messages above are earlier turns and retrieved passages. Answer the question above; the passages are evidence for it, not a new question to answer."
}

// stoppingNote tells the answer call that the search was cut short, so the reply
// can say so instead of implying it covered everything.
//
// Only a real ceiling counts: an exhausted budget, a pause, or unresolved calls.
// "completion has no final text" is NOT a ceiling — it means the model narrated
// its next step instead of writing the answer, and the search itself may be
// complete. Treating that as truncation and telling the model not to present a
// partial count made it drop names it had already found: the same run went from
// 16 names to 6.
func stoppingNote(state *State) string {
	reason := strings.TrimSpace(state.Verdict.Reason)
	cutShort := state.Verdict.Action == policy.ActionPause ||
		strings.Contains(reason, "exhausted") ||
		strings.Contains(reason, "no result")
	if !cutShort {
		return ""
	}
	if reason == "" {
		reason = string(state.Verdict.Action)
	}
	var b strings.Builder
	b.WriteString("The search ended early: " + reason + ".\n")
	b.WriteString("Answer from everything already collected and cover all of it — do not drop items found earlier just because the search stopped. Add one sentence saying the list may not be complete.\n")
	if total, open := planCounts(state.Plan); open > 0 {
		fmt.Fprintf(&b, "%d of %d plan step(s) were left open.\n", open, total)
	}
	return b.String()
}

// planCounts reports how many steps a plan holds and how many are not done.
//
// renderPlan and the policy snapshot both read these numbers, so "the plan says
// this is unfinished" cannot disagree between the prompt and the verdict: the
// model is told to keep working on an open step and the loop refuses to call a
// run finished while one remains.
func planCounts(plan []toolkit.PlanStep) (total, open int) {
	for _, step := range plan {
		if strings.TrimSpace(step.Description) == "" {
			continue
		}
		total++
		if strings.TrimSpace(step.Status) != "completed" {
			open++
		}
	}
	return total, open
}

// renderPlan turns the model's own step list into a prompt section.
//
// It is appended after the transcript rather than into the system prompt: the
// system prefix stays byte-identical so the provider cache still hits, while
// the plan stays current even when compaction dropped the tool call that
// created it. concluding marks the forced final answer, where open steps must
// be reported instead of worked on.
func renderPlan(plan []toolkit.PlanStep, concluding bool) string {
	if len(plan) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("Plan state (from your plan tool, kept across rounds):\n")
	written := 0
	for _, step := range plan {
		description := strings.TrimSpace(step.Description)
		if description == "" {
			continue
		}
		status := strings.TrimSpace(step.Status)
		if status == "" {
			status = "pending"
		}
		written++
		fmt.Fprintf(&b, "%d. [%s] %s\n", written, status, description)
	}
	_, open := planCounts(plan)
	if written == 0 {
		return ""
	}
	switch {
	case concluding && open > 0:
		fmt.Fprintf(&b, "%d step(s) were left open. Answer from the evidence already collected: include every item you found and mention the gap in one sentence.\n", open)
	case concluding:
		b.WriteString("Every step is complete.\n")
	case open > 0:
		fmt.Fprintf(&b, "%d step(s) still open. Work them before concluding.\n", open)
	default:
		b.WriteString("Every step is complete: answer the question now.\n")
	}
	return b.String()
}

// describeAbort turns a cancellation into a fact the model can act on.
//
// A raw "context canceled" reads like a tool failure, and failures invite
// retries of calls that were stopped on purpose. The truth is uncertainty: the
// call may have completed, and the model must not assume either way.
func describeAbort(err error) error {
	switch {
	case errors.Is(err, context.DeadlineExceeded):
		return errors.New("the run's time budget ran out before this call returned; it may or may not have completed — do not assume it failed")
	case errors.Is(err, context.Canceled):
		return errors.New("this run was stopped before the call returned; it may or may not have completed — do not assume it failed")
	}
	return err
}

// budgetNotice reports the spend as fact when any budget is at least four
// fifths gone. Facts only: the model decides what to do with them.
func budgetNotice(budget Budget) string {
	var parts []string
	if budget.MaxTokens > 0 {
		used := budget.TokensIn + budget.TokensOut
		if used*5 >= budget.MaxTokens*4 {
			parts = append(parts, fmt.Sprintf("%d of %d tokens used", used, budget.MaxTokens))
		}
	}
	if budget.MaxToolCalls > 0 && budget.ToolCalls*5 >= budget.MaxToolCalls*4 {
		parts = append(parts, fmt.Sprintf("%d of %d tool calls made", budget.ToolCalls, budget.MaxToolCalls))
	}
	if budget.MaxWallMS > 0 && budget.ElapsedMS*5 >= budget.MaxWallMS*4 {
		parts = append(parts, fmt.Sprintf("%d of %d ms elapsed", budget.ElapsedMS, budget.MaxWallMS))
	}
	if len(parts) == 0 {
		return ""
	}
	return "Budget status: " + strings.Join(parts, "; ") + "."
}

func idempotencyKey(tool, arguments string) string {
	sum := sha256.Sum256([]byte(tool + "\x00" + arguments))
	return hex.EncodeToString(sum[:8])
}

// citationHandlePattern matches the [ID:...] handle a tool prints for a passage.
var citationHandlePattern = regexp.MustCompile(`\s*\[ID:[^\]\s]*\]`)

// stripCitationHandles removes handles from replayed history.
func stripCitationHandles(text string) string {
	if !strings.Contains(text, "[ID:") {
		return text
	}
	return strings.TrimSpace(citationHandlePattern.ReplaceAllString(text, ""))
}

// outputFingerprint identifies a tool result by content. It answers "has this
// run already seen this exact answer", which is what separates a tool making
// progress from one being asked the same thing again.
func outputFingerprint(tool, content string) string {
	sum := sha256.Sum256([]byte(tool + "\x00" + content))
	return hex.EncodeToString(sum[:8])
}

// planFingerprint identifies a plan's shape and progress: the same steps with
// the same statuses hash the same, so republishing it is not progress.
func planFingerprint(plan []toolkit.PlanStep) string {
	if len(plan) == 0 {
		return ""
	}
	var b strings.Builder
	for _, step := range plan {
		b.WriteString(step.ID)
		b.WriteString("\x00")
		b.WriteString(step.Description)
		b.WriteString("\x00")
		b.WriteString(step.Status)
		b.WriteString("\x01")
	}
	sum := sha256.Sum256([]byte(b.String()))
	return hex.EncodeToString(sum[:8])
}

// Evidence pool bounds for the answer call. The list is the run's own record of
// what it retrieved, so it is worth re-sending even when History still carries
// the same passages; the caps keep that from turning into the whole corpus.
const (
	// The pool is re-sent on every round, so its size is paid per round: these
	// caps keep one run's memory bounded without dropping the passages that
	// proved the earliest findings — the ones compaction removes first.
	evidencePoolMaxItems = 160
	evidencePoolMaxLine  = 320
	evidencePoolMaxBytes = 24 << 10
)

// renderEvidencePool lists the evidence units a run collected, keyed by their
// citation handle. It exists because History is a lossy view of the same
// material: compaction drops old tool results first, and the passages that
// proved the earliest findings are exactly the oldest.
//
// The wording matters in both directions. It has to tell the model this list is
// its memory (so it stops re-fetching what it already has), and it must NOT read
// as a complete corpus (a list-shaped question would otherwise be answered from
// whatever happened to be collected by then).
func renderEvidencePool(evidence []toolkit.Evidence) string {
	if len(evidence) == 0 {
		return ""
	}
	seen := make(map[string]struct{}, len(evidence))
	var b strings.Builder
	b.WriteString("## Collected evidence\n")
	b.WriteString("Passages this run has already retrieved. The transcript above may have been compacted, so this list is your memory of what you have found so far: cite these handles as they are, and do not retrieve the same passage again unless you need its full text. An item that is NOT listed here has simply not been retrieved yet — its absence is not evidence that it does not exist.\n")
	written := 0
	for _, item := range evidence {
		ref := strings.TrimSpace(item.Ref)
		line := strings.TrimSpace(item.Snippet)
		if ref == "" && line == "" {
			continue
		}
		key := ref
		if key == "" {
			key = clip(line, 64)
		}
		if _, duplicate := seen[key]; duplicate {
			continue
		}
		seen[key] = struct{}{}
		if written >= evidencePoolMaxItems || b.Len() >= evidencePoolMaxBytes {
			b.WriteString("- … more evidence was collected than fits here; answer from what is listed and say what you could not restate.\n")
			break
		}
		b.WriteString("- ")
		if ref != "" {
			b.WriteString(ref)
			if line != "" {
				b.WriteString(": ")
			}
		}
		if line != "" {
			b.WriteString(clip(line, evidencePoolMaxLine))
		} else if item.Source != "" {
			b.WriteString(item.Source)
		}
		b.WriteString("\n")
		written++
	}
	if written == 0 {
		return ""
	}
	return b.String()
}

// describeCall records what a call actually asked for and how much came back.
//
// The step log otherwise keeps only a tool name and an argument hash, which is
// enough to spot a repeated call but not enough to explain a short answer: the
// pattern behind it, and whether its output was large enough to be spilled, are
// exactly the two facts that cannot be recovered after the run.
func describeCall(call PendingCall, result toolkit.Result) string {
	reason := strings.TrimSpace(call.Arguments)
	if reason == "" {
		reason = "-"
	}
	if size := len(result.Content); size > 0 {
		reason = fmt.Sprintf("%s | %d bytes", reason, size)
	}
	return clip(reason, 180)
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
