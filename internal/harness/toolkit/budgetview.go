package toolkit

import (
	"context"
	"fmt"
	"strings"

	"ragflow/internal/harness/agent/model"
)

// RunStats is what the loop knows about a run's consumption at the moment a
// tool runs. It exists so the model can ask the framework a question only the
// framework can answer: how much of the budget is left.
//
// Zero means "not known" for every field: a deployment without a token budget
// still has a context window, and a run without a tool-call cap still has
// rounds. The tool renders what it has instead of inventing numbers.
type RunStats struct {
	Round        int
	MaxRounds    int
	ToolCalls    int
	MaxToolCalls int
	TokensIn     int
	TokensOut    int
	TokenBudget  int
	// ContextWindow is the model's full window; ContextUsed is the estimated
	// size of the prompt the round is working with.
	ContextWindow int
	ContextUsed   int
	// CompactAt is the share of the window where folding starts, 0..1. Zero is
	// reported as the framework default (3/4).
	CompactAt float64
}

type runStatsKey struct{}

// WithRunStats binds the run's accounting to a context for one round.
func WithRunStats(ctx context.Context, stats RunStats) context.Context {
	return context.WithValue(ctx, runStatsKey{}, stats)
}

// RunStatsFrom reads the accounting bound by the loop.
func RunStatsFrom(ctx context.Context) (RunStats, bool) {
	stats, ok := ctx.Value(runStatsKey{}).(RunStats)
	return stats, ok
}

// NewBudgetCapability exposes context_remaining. The loop binds the numbers
// with WithRunStats before invoking tools, so the tool is a view over the same
// accounting the policy engine enforces, not a second counter that can drift.
func NewBudgetCapability() Capability { return &budgetCapability{} }

const contextRemainingDescription = "Report how much of this run's budget is still available: " +
	"rounds, tool calls, tokens and the context window. Call it when the answer " +
	"depends on how much room is left — a long enumeration that needs several more " +
	"passes, or a run that has read a lot and is deciding whether one more deep read " +
	"fits. The numbers are the framework's own accounting, not an estimate you could " +
	"derive from the transcript."

type budgetCapability struct{}

// Tools implements Capability.
func (*budgetCapability) Tools() []model.ToolSpec {
	return []model.ToolSpec{{
		Name:        "context_remaining",
		Description: contextRemainingDescription,
		Parameters:  ObjectSchema(map[string]any{}, ""),
	}}
}

// ToolRisk implements RiskClassifier: reading the accounting changes nothing.
func (*budgetCapability) ToolRisk(string) Risk { return RiskReadOnly }

// Name implements Capability.
func (*budgetCapability) Name() string { return "budget" }

// Invoke implements Capability.
func (*budgetCapability) Invoke(ctx context.Context, _ Call) (Result, error) {
	stats, ok := RunStatsFrom(ctx)
	if !ok {
		return Result{Content: "budget accounting is not bound to this run; no limits are being enforced"}, nil
	}
	return Result{Content: renderRunStats(stats)}, nil
}

// renderRunStats formats the accounting as facts, one per line. A limit the
// run does not have is left out rather than printed as zero: "tool calls: 0/0"
// reads as exhausted when it means unlimited.
func renderRunStats(stats RunStats) string {
	var lines []string
	if stats.MaxRounds > 0 {
		lines = append(lines, fmt.Sprintf("rounds: %d/%d used", stats.Round, stats.MaxRounds))
	}
	if stats.MaxToolCalls > 0 {
		lines = append(lines, fmt.Sprintf("tool calls: %d/%d used", stats.ToolCalls, stats.MaxToolCalls))
	}
	if stats.ContextWindow > 0 {
		pct := 0
		if stats.ContextWindow > 0 {
			pct = stats.ContextUsed * 100 / stats.ContextWindow
		}
		compactAt := stats.CompactAt
		if compactAt <= 0 {
			compactAt = 0.75
		}
		lines = append(lines, fmt.Sprintf(
			"context: ~%d of %d tokens in this round's prompt (%d%%); older turns start folding at %d%%",
			stats.ContextUsed, stats.ContextWindow, pct, int(compactAt*100)))
	}
	if stats.TokenBudget > 0 {
		lines = append(lines, fmt.Sprintf("tokens: %d in / %d out against a %d budget",
			stats.TokensIn, stats.TokensOut, stats.TokenBudget))
	} else if stats.TokensIn > 0 || stats.TokensOut > 0 {
		lines = append(lines, fmt.Sprintf("tokens: %d in / %d out (no run budget)", stats.TokensIn, stats.TokensOut))
	}
	if len(lines) == 0 {
		return "no limits are being enforced on this run"
	}
	return strings.Join(lines, "\n")
}
