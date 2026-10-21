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
	ActionFinish   Action = "finish"
	ActionBlock    Action = "block"
	ActionPause    Action = "pause"
)

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

	HasFinalText    bool
	Citations       int
	UnresolvedCalls int
	ScopeViolations int
	// ThinkTerminal reports that the model ended its turn instead of asking
	// for more tools.
	ThinkTerminal bool
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
		return 4
	case ActionPause:
		return 3
	case ActionFinish:
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
			Reason:   fmt.Sprintf("round budget exhausted (%d/%d)", snapshot.Round, snapshot.MaxRounds),
		})
	}
	if snapshot.MaxTokens > 0 && snapshot.TokensIn+snapshot.TokensOut >= snapshot.MaxTokens {
		out = append(out, Verdict{
			Action:   ActionFinish,
			Severity: SeverityWarning,
			Reason:   fmt.Sprintf("token budget exhausted (%d/%d)", snapshot.TokensIn+snapshot.TokensOut, snapshot.MaxTokens),
		})
	}
	if snapshot.MaxToolCalls > 0 && snapshot.ToolCalls >= snapshot.MaxToolCalls {
		out = append(out, Verdict{
			Action:   ActionFinish,
			Severity: SeverityWarning,
			Reason:   fmt.Sprintf("tool-call budget exhausted (%d/%d)", snapshot.ToolCalls, snapshot.MaxToolCalls),
		})
	}
	if snapshot.MaxWall > 0 && snapshot.Elapsed >= snapshot.MaxWall {
		out = append(out, Verdict{
			Action:   ActionFinish,
			Severity: SeverityWarning,
			Reason:   fmt.Sprintf("wall-clock budget exhausted (%s/%s)", snapshot.Elapsed.Truncate(time.Second), snapshot.MaxWall),
		})
	}
	if snapshot.MaxCostUSD > 0 && snapshot.CostUSD >= snapshot.MaxCostUSD {
		out = append(out, Verdict{
			Action:   ActionFinish,
			Severity: SeverityWarning,
			Reason:   fmt.Sprintf("cost budget exhausted ($%.4f/$%.4f)", snapshot.CostUSD, snapshot.MaxCostUSD),
		})
	}
	return out
}

// Progress stops a run that keeps spending calls without learning anything.
type Progress struct {
	// StagnantRounds is how many consecutive no-new-evidence rounds are allowed.
	StagnantRounds int
	// RepeatedCalls is how many identical calls are allowed before pausing.
	RepeatedCalls int
}

// Name implements Policy.
func (Progress) Name() string { return "progress" }

// Check implements Policy.
func (p Progress) Check(snapshot Snapshot) []Verdict {
	limit := p.StagnantRounds
	if limit <= 0 {
		limit = 2
	}
	repeats := p.RepeatedCalls
	if repeats <= 0 {
		repeats = 3
	}
	var out []Verdict
	if snapshot.StagnantRounds >= limit {
		out = append(out, Verdict{
			Action:   ActionFinish,
			Severity: SeverityWarning,
			Reason:   fmt.Sprintf("no new evidence for %d consecutive rounds", snapshot.StagnantRounds),
		})
	}
	if snapshot.RepeatedCalls >= repeats {
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

// Check implements Policy. A final answer finishes the run; an answer-shaped
// empty terminal state is refused.
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
		out = append(out, Verdict{
			Action:   ActionFinish,
			Severity: SeverityInfo,
			Reason:   "final answer produced",
		})
		return out
	}
	// Only a model that ended its turn without producing anything is a
	// refusal-worthy empty completion. A round interrupted mid-flight (tool
	// results pending, budget exhausted) is a normal loop iteration.
	if snapshot.ThinkTerminal {
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
