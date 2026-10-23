// Package policy decides whether a run continues, stops, is blocked, or pauses.
// Policies are read-only over a snapshot: they never rewrite state.
package policy

import (
	"fmt"
	"time"
)

// Action is the loop-level decision.
type Action string

const (
	ActionContinue Action = "continue"
	// ActionNudge continues the run with one factual correction injected into
	// the transcript. A policy that can only describe a stall has no business
	// ending the run: what it knows is "no progress yet", not "no progress is
	// possible", and the run may be in the middle of reading evidence it has
	// already paid for. Ending stays with the budgets.
	ActionNudge  Action = "nudge"
	ActionFinish Action = "finish"
	ActionBlock  Action = "block"
	ActionPause  Action = "pause"
)

// CodeBudgetLimited labels a run that stopped because a configured limit ran
// out, as opposed to one that finished its work.
const CodeBudgetLimited = "budget_limited"

// Severity is the operator-facing weight of a verdict.
type Severity string

const (
	SeverityInfo    Severity = "info"
	SeverityWarning Severity = "warning"
	SeverityError   Severity = "error"
	SeverityFatal   Severity = "fatal"
)

// Verdict is one decision.
type Verdict struct {
	Action            Action
	Severity          Severity
	Reason            string
	RecoveryThreshold int
	// Code is a stable machine-readable label for the terminal cause, e.g.
	// "budget_limited". The reference runtime reports a run that ran out of
	// budget as a distinct terminal state rather than as a generic stop, so a
	// caller can tell "the work finished" from "the money ran out".
	Code string
	// Advice is the factual correction ActionNudge injects into the transcript.
	// It describes what the run observed, never what to answer.
	Advice string
}

// Snapshot is the read-only view policies see at verify time.
type Snapshot struct {
	RunID string
	Round int

	MaxRounds int
	MaxTokens int
	TokensIn  int
	TokensOut int

	MaxToolCalls int
	ToolCalls    int

	Elapsed time.Duration
	MaxWall time.Duration

	CostUSD    float64
	MaxCostUSD float64

	// NewEvidence counts evidence added in the round that just finished.
	NewEvidence int
	// StagnantRounds counts consecutive rounds with no new evidence.
	StagnantRounds int
	// RepeatedCalls counts tool calls identical to an earlier call this run.
	RepeatedCalls int
	// Nudges counts the course corrections this run already received, which
	// bounds how often a stalled run is reminded of the same fact.
	Nudges int

	HasFinalText    bool
	Citations       int
	UnresolvedCalls int
	ScopeViolations int
	// ThinkTerminal reports that the model ended its turn instead of asking
	// for more tools.
	ThinkTerminal bool
	// PlanSteps is how many steps the model's own plan holds, and OpenPlanSteps
	// how many of them are still pending or in progress. The plan is the model's
	// own statement of what the answer still owes, so an open step is a
	// completion signal policies can read instead of inferring it from prose.
	PlanSteps     int
	OpenPlanSteps int
	// PlanIdleRounds counts rounds since the plan last changed. Open steps alone
	// keep a run alive because an open step is unfinished work; a plan that stops
	// moving is not unfinished work, it is the model holding the run open without
	// doing any of it.
	PlanIdleRounds int
	// TerminalRounds counts consecutive rounds that produced prose and no tool
	// call. It bounds how long an open plan may hold a run that has stopped
	// working: one round to resume the steps, after that the answer is due.
	TerminalRounds int
}

// Policy checks one snapshot.
type Policy interface {
	Name() string
	Check(snapshot Snapshot) []Verdict
}

// Engine evaluates every policy and returns the strongest verdict.
type Engine struct {
	policies []Policy
}

// New builds an engine. A nil policy is ignored.
func New(policies ...Policy) *Engine {
	engine := &Engine{}
	for _, item := range policies {
		if item != nil {
			engine.policies = append(engine.policies, item)
		}
	}
	return engine
}

// Default builds the four policies every run uses.
func Default() *Engine {
	return New(Budget{}, Progress{}, Completion{}, Safety{})
}

// Policies returns the configured policy names.
func (e *Engine) Policies() []string {
	out := make([]string, 0, len(e.policies))
	for _, item := range e.policies {
		out = append(out, item.Name())
	}
	return out
}

// Decide returns the strongest verdict. No policy firing means "continue".
func (e *Engine) Decide(snapshot Snapshot) Verdict {
	best := Verdict{Action: ActionContinue, Severity: SeverityInfo}
	for _, item := range e.policies {
		for _, verdict := range item.Check(snapshot) {
			if stronger(verdict, best) {
				best = verdict
			}
		}
	}
	return best
}

func stronger(candidate, current Verdict) bool {
	candidateRank := rank(candidate.Action)
	currentRank := rank(current.Action)
	if candidateRank != currentRank {
		return candidateRank > currentRank
	}
	return severityRank(candidate.Severity) > severityRank(current.Severity)
}

func rank(action Action) int {
	switch action {
	case ActionBlock:
		return 5
	case ActionPause:
		return 4
	case ActionFinish:
		return 3
	case ActionNudge:
		return 2
	case ActionContinue:
		return 1
	default:
		return 0
	}
}

func severityRank(severity Severity) int {
	switch severity {
	case SeverityFatal:
		return 4
	case SeverityError:
		return 3
	case SeverityWarning:
		return 2
	case SeverityInfo:
		return 1
	default:
		return 0
	}
}

// Budget stops a run that exhausted any configured limit.
type Budget struct{}

// Name implements Policy.
func (Budget) Name() string { return "budget" }

// Check implements Policy.
func (Budget) Check(snapshot Snapshot) []Verdict {
	var out []Verdict
	if snapshot.MaxRounds > 0 && snapshot.Round >= snapshot.MaxRounds {
		out = append(out, Verdict{
			Action:   ActionFinish,
			Severity: SeverityWarning,
			Code:     CodeBudgetLimited,
			Reason:   fmt.Sprintf("round budget exhausted (%d/%d)", snapshot.Round, snapshot.MaxRounds),
		})
	}
	if snapshot.MaxTokens > 0 && snapshot.TokensIn+snapshot.TokensOut >= snapshot.MaxTokens {
		out = append(out, Verdict{
			Action:   ActionFinish,
			Severity: SeverityWarning,
			Code:     CodeBudgetLimited,
			Reason:   fmt.Sprintf("token budget exhausted (%d/%d)", snapshot.TokensIn+snapshot.TokensOut, snapshot.MaxTokens),
		})
	}
	if snapshot.MaxToolCalls > 0 && snapshot.ToolCalls >= snapshot.MaxToolCalls {
		out = append(out, Verdict{
			Action:   ActionFinish,
			Severity: SeverityWarning,
			Code:     CodeBudgetLimited,
			Reason:   fmt.Sprintf("tool-call budget exhausted (%d/%d)", snapshot.ToolCalls, snapshot.MaxToolCalls),
		})
	}
	if snapshot.MaxWall > 0 && snapshot.Elapsed >= snapshot.MaxWall {
		out = append(out, Verdict{
			Action:   ActionFinish,
			Severity: SeverityWarning,
			Code:     CodeBudgetLimited,
			Reason:   fmt.Sprintf("wall-clock budget exhausted (%s/%s)", snapshot.Elapsed.Truncate(time.Second), snapshot.MaxWall),
		})
	}
	if snapshot.MaxCostUSD > 0 && snapshot.CostUSD >= snapshot.MaxCostUSD {
		out = append(out, Verdict{
			Action:   ActionFinish,
			Severity: SeverityWarning,
			Code:     CodeBudgetLimited,
			Reason:   fmt.Sprintf("cost budget exhausted ($%.4f/$%.4f)", snapshot.CostUSD, snapshot.MaxCostUSD),
		})
	}
	return out
}

// Progress corrects a run that keeps spending calls without learning anything.
//
// It never ends a run on its own. The reference runtime drives a turn to
// completion with the model's own decision and bounds it with budgets only, and
// the case that motivated this rule here was a run cut off in the middle of
// reading the evidence it had just collected: a round that adds no evidence is
// not proof the work is over. So a stall produces a nudge — the observed facts,
// twice — and after that the answer call takes over; the budgets remain the only
// hard stop.
type Progress struct {
	// StagnantRounds is how many consecutive no-new-evidence rounds are allowed
	// before the run is told about it.
	StagnantRounds int
	// RepeatedCalls is how many identical calls are allowed before pausing.
	RepeatedCalls int
	// MaxNudges bounds how often the same fact is restated.
	MaxNudges int
	// PlanIdleRounds is how long an open plan may stay unchanged before it counts
	// as a stall. Without it an open plan is an unbounded licence to continue:
	// a run that keeps retrieving new-but-irrelevant passages never accumulates
	// StagnantRounds, so the plan is the only thing keeping it alive and nothing
	// ever moves the plan.
	PlanIdleRounds int
	// PlanIdleRoundsWhileProgressing overrides PlanIdleRounds while the evidence
	// pool is still growing. The default (zero) does not judge an idle plan at all
	// in that case: a step stays open for as long as its enumeration is
	// incomplete, so plan idleness alone says nothing while evidence keeps
	// arriving. Set it to bound a run that keeps finding new-but-irrelevant
	// material before the round budget does.
	PlanIdleRoundsWhileProgressing int
}

// planKeepingNudges extends the nudge allowance while the model's own plan still
// has open steps.
//
// An enumeration's last item is what closes its step, so an open step is the
// run's own statement that the list is incomplete. Each extra correction is one
// stall cycle (the counter resets after a nudge, so this is not four extra
// rounds but four extra stalls); the bound exists so a plan the model never
// closes still ends on a budget rather than on a promise.
const planKeepingNudges = 4

// Name implements Policy.
func (Progress) Name() string { return "progress" }

// Check implements Policy.
func (p Progress) Check(snapshot Snapshot) []Verdict {
	limit := p.StagnantRounds
	if limit <= 0 {
		// Three flat rounds, not two: one flat round is normal while a tool's
		// output is being digested, and the budget policies already bound the
		// total spend of a run that never converges.
		limit = 3
	}
	repeats := p.RepeatedCalls
	if repeats <= 0 {
		repeats = 3
	}
	nudges := p.MaxNudges
	if nudges <= 0 {
		nudges = 2
	}
	planIdle := p.PlanIdleRounds
	if planIdle <= 0 {
		// Six rounds is long enough for a run to work through several batches
		// against the same open steps, and short enough that a run which has
		// stopped doing its own plan is handed to the answer call instead of
		// burning the whole round budget.
		planIdle = 6
	}
	planIdleProgressing := p.PlanIdleRoundsWhileProgressing
	if planIdleProgressing < 0 {
		planIdleProgressing = 0
	}
	var out []Verdict
	stalled := snapshot.StagnantRounds >= limit
	reason := ""
	planIdleStall := false
	if stalled {
		reason = fmt.Sprintf("no new evidence for %d consecutive rounds", snapshot.StagnantRounds)
	}
	// A run whose evidence pool grew this round is making progress, whatever its
	// plan says. An enumerating step cannot tick until its last item is found, so
	// it is idle by construction for as long as the search keeps adding evidence;
	// blocking it there ended runs at round 10 and round 20 while they were still
	// pulling fresh greps and spills per round, and the list came back short.
	//
	// When the pool stops growing the plan is judged again at the original limit,
	// which is the case the rule was written for; a run that finds new but
	// irrelevant material is bounded by the round and token budgets instead.
	planLimit := planIdle
	if snapshot.StagnantRounds == 0 {
		planLimit = planIdleProgressing
	}
	if planLimit > 0 && snapshot.OpenPlanSteps > 0 && snapshot.PlanIdleRounds >= planLimit {
		if !stalled {
			stalled = true
			planIdleStall = true
			reason = fmt.Sprintf("%d plan step(s) unchanged for %d rounds", snapshot.OpenPlanSteps, snapshot.PlanIdleRounds)
		}
	}
	if stalled {
		stalledRounds := snapshot.StagnantRounds
		if snapshot.OpenPlanSteps > 0 && snapshot.PlanIdleRounds > stalledRounds {
			stalledRounds = snapshot.PlanIdleRounds
		}
		// The two stalls need different words: one says the search stopped adding
		// evidence, the other says the plan stopped moving while it still was.
		observation := "retrieved nothing new"
		if planIdleStall {
			observation = "did not move your plan"
		}
		if snapshot.Nudges < nudges {
			out = append(out, Verdict{
				Action:   ActionNudge,
				Severity: SeverityInfo,
				Reason:   reason,
				Advice: fmt.Sprintf(
					"The last %d rounds %s; the evidence pool holds %d units. "+
						"If those units cover the question, write the answer now. If they do not, change the search — "+
						"different terms, a different tool, or read a passage that was retrieved but not yet read. "+
						"An open step is not a reason to stop: keep it open while it is still adding evidence.",
					stalledRounds, observation, snapshot.Citations),
			})
		} else if snapshot.OpenPlanSteps > 0 && !planIdleStall && snapshot.Nudges < nudges+planKeepingNudges {
			// The nudges are spent, but the run's own plan still lists open
			// steps — and for an enumeration the last item is exactly the one
			// that closes them. Handing such a run to the answer call is how a
			// long enumeration comes back one name short per stalled round: the
			// plan said "enumerate all" the whole time.
			//
			// So keep it running and restate the correction instead. It costs
			// one injected message per stall cycle (the counter resets, so this
			// fires about every StagnantRounds rounds), and the round and token
			// budgets remain the hard stop — planKeepingNudges only bounds how
			// long a plan can hold a run open without evidence.
			//
			// A plan that stopped moving is different: there the run is not
			// enumerating, it is holding itself open (see planIdleStall), and
			// that still goes to the answer call.
			out = append(out, Verdict{
				Action:   ActionNudge,
				Severity: SeverityInfo,
				Reason:   reason,
				Advice: fmt.Sprintf(
					"Your plan still lists %d open step(s) and the last %d rounds %s. "+
						"If every item is now found, close the steps and answer. If not, change the axis of the search — "+
						"another chapter, another set of names, a wider read — not the same query again.",
					snapshot.OpenPlanSteps, stalledRounds, observation),
			})
		} else {
			// The fact has been restated as often as it is useful, and the plan
			// no longer claims anything is left. Hand the run to the answer
			// call instead of ending it: the evidence already collected is what
			// the answer is written from.
			out = append(out, Verdict{
				Action:   ActionBlock,
				Severity: SeverityWarning,
				Reason:   reason,
				Code:     "stalled",
			})
		}
	}
	// Repeated calls are only evidence of being stuck when the model has not
	// said what is left. With open plan steps the run is not claiming to be
	// finished, and an enumeration needs the same query more than once as it
	// walks a list; stopping it there interrupts work that is going somewhere.
	// A run that is genuinely spinning still ends on StagnantRounds or Budget.
	if snapshot.RepeatedCalls >= repeats && snapshot.OpenPlanSteps == 0 {
		out = append(out, Verdict{
			Action:            ActionPause,
			Severity:          SeverityWarning,
			Reason:            fmt.Sprintf("%d repeated tool calls", snapshot.RepeatedCalls),
			RecoveryThreshold: repeats,
		})
	}
	return out
}

// Completion refuses success-shaped terminal states.
type Completion struct {
	// RequireCitations rejects a final answer with no citation-bearing evidence.
	RequireCitations bool
}

// Name implements Policy.
func (Completion) Name() string { return "completion" }

// Check implements Policy. A final answer finishes the run once the model's own
// plan is closed; an answer-shaped empty terminal state is refused.
func (c Completion) Check(snapshot Snapshot) []Verdict {
	var out []Verdict
	if snapshot.UnresolvedCalls > 0 {
		out = append(out, Verdict{
			Action:   ActionBlock,
			Severity: SeverityError,
			Reason:   fmt.Sprintf("%d tool calls have no result", snapshot.UnresolvedCalls),
		})
		return out
	}
	if snapshot.HasFinalText {
		if c.RequireCitations && snapshot.Citations == 0 {
			out = append(out, Verdict{
				Action:   ActionPause,
				Severity: SeverityWarning,
				Reason:   "final answer has no resolvable citation",
			})
			return out
		}
		// A plan is the model's own list of what the answer still needs. Writing
		// the answer while steps of that list are open is the "12 of 16"
		// failure: the prose reads as a conclusion while the work is unfinished.
		//
		// One round of grace: the model may have written the answer before reading
		// the plan section, so it gets a single round to reopen the work. A
		// second prose-only round is not convergence — it finishes here and the
		// answer call is told what is still open.
		if snapshot.OpenPlanSteps > 0 && snapshot.TerminalRounds <= 1 {
			out = append(out, Verdict{
				Action:   ActionContinue,
				Severity: SeverityInfo,
				Reason:   fmt.Sprintf("%d plan step(s) still open", snapshot.OpenPlanSteps),
			})
			return out
		}
		out = append(out, Verdict{
			Action:   ActionFinish,
			Severity: SeverityInfo,
			Reason:   "final answer produced",
		})
		return out
	}
	// A model that ended its turn without an answer — it narrated the next step,
	// or said nothing — gets one more round. That round is worth more than a
	// summary call over the transcript: the model still holds the ReAct context
	// that produced its best partial result, while a fresh answer call sees the
	// same text without the thread of reasoning and tends to answer only the
	// part it read last (a 16-name search came back as the six generals of one
	// chapter). After that grace round the answer call takes over.
	if snapshot.ThinkTerminal {
		if snapshot.TerminalRounds <= 1 {
			return []Verdict{{
				Action:   ActionContinue,
				Severity: SeverityInfo,
				Reason:   "no answer written yet",
			}}
		}
		out = append(out, Verdict{
			Action:   ActionBlock,
			Severity: SeverityError,
			Reason:   "completion has no final text",
		})
	}
	return out
}

// Safety blocks scope escapes and unapproved high-risk actions.
type Safety struct{}

// Name implements Policy.
func (Safety) Name() string { return "safety" }

// Check implements Policy.
func (Safety) Check(snapshot Snapshot) []Verdict {
	if snapshot.ScopeViolations <= 0 {
		return nil
	}
	return []Verdict{{
		Action:   ActionBlock,
		Severity: SeverityFatal,
		Reason:   fmt.Sprintf("%d scope violations", snapshot.ScopeViolations),
	}}
}
