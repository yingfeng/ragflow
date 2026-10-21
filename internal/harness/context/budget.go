package agentctx

// Context-window budgeting, modelled on the reference runtimes:
//
//   - Astra: DEFAULT_CONTEXT_WINDOW_TOKENS=200k, MODEL_CONTEXT_INPUT_BUDGET_RATIO=0.80,
//     graded pressure tiers 0.60 / 0.75 / 0.90.
//   - codex: auto-compact ceiling = 90% of the window, 20k tokens of recent user
//     turns kept across a compaction.
//   - pi: keepRecentTokens=20k, reserveTokens=16k.
//
// The tiers are ordered by severity, and a higher tier always implies the
// cheaper ones: trimming schemas, then compacting history, then pruning.
const (
	// DefaultContextWindow is the fallback window when the caller does not know
	// the model's real one.
	DefaultContextWindow = 200_000
	// DefaultInputBudgetRatio is the share of the window spent on input. The
	// rest covers the generated answer and protocol framing.
	DefaultInputBudgetRatio = 0.80
	// DefaultOutputReserveTokens is the room kept for one completion.
	DefaultOutputReserveTokens = 16_384
	// DefaultAggressiveKeepMessages is the trailing window kept once pressure is
	// critical.
	DefaultAggressiveKeepMessages = 6

	// PressureTrimSchemas starts clearing old tool output.
	PressureTrimSchemas = 0.60
	// PressureCompactHistory folds older turns into a summary.
	PressureCompactHistory = 0.75
	// PressureAggressivePrune keeps only the trailing turns.
	PressureAggressivePrune = 0.90
)

// Tier is the compaction level selected from context pressure.
type Tier int

const (
	// TierNormal keeps recent tool output verbatim.
	TierNormal Tier = iota
	// TierTrimSchemas clears tool output beyond the recent window.
	TierTrimSchemas
	// TierCompactHistory folds older turns into one summary message.
	TierCompactHistory
	// TierAggressivePrune keeps only the trailing messages.
	TierAggressivePrune
)

// Budget decides how much input the model may receive.
type Budget struct {
	// ContextWindow is the model's full window. Zero means DefaultContextWindow.
	ContextWindow int
	// InputBudgetRatio is the input share of the window.
	InputBudgetRatio float64
	// OutputReserveTokens is the room reserved for one completion.
	OutputReserveTokens int
	// MaxInputTokens is an absolute cap regardless of the window.
	MaxInputTokens int
}

// EffectiveInput is the token ceiling for one assembled prompt.
func (b Budget) EffectiveInput() int {
	window := b.ContextWindow
	if window <= 0 {
		window = DefaultContextWindow
	}
	ratio := b.InputBudgetRatio
	if ratio <= 0 || ratio >= 1 {
		ratio = DefaultInputBudgetRatio
	}
	reserve := b.OutputReserveTokens
	if reserve < 0 {
		reserve = 0
	}

	limit := int(float64(window)*ratio) - reserve
	if limit <= 0 {
		// A window smaller than the output reserve still has to carry input.
		limit = int(float64(window) * ratio / 2)
	}
	if b.MaxInputTokens > 0 && b.MaxInputTokens < limit {
		return b.MaxInputTokens
	}
	return limit
}

// TierFor maps the current input size onto a compaction tier. Pressure is
// measured against the effective input ceiling, so a model with a small window
// compacts earlier for the same transcript.
func (b Budget) TierFor(usedTokens int) Tier {
	limit := b.EffectiveInput()
	if limit <= 0 {
		return TierNormal
	}
	pressure := float64(usedTokens) / float64(limit)
	switch {
	case pressure >= PressureAggressivePrune:
		return TierAggressivePrune
	case pressure >= PressureCompactHistory:
		return TierCompactHistory
	case pressure >= PressureTrimSchemas:
		return TierTrimSchemas
	default:
		return TierNormal
	}
}

// keepRecentToolResults is the adaptive counterpart of Astra's microcompact:
// the more pressure, the fewer tool results stay inline.
func keepRecentToolResults(tier Tier, base int) int {
	switch tier {
	case TierAggressivePrune:
		return 1
	case TierCompactHistory:
		if base > 2 {
			return 2
		}
		return base
	case TierTrimSchemas:
		if base > 4 {
			return 4
		}
		return base
	default:
		if base < 6 {
			return 6
		}
		return base
	}
}

// toolResultBudget shrinks the per-result inline limit under pressure.
func toolResultBudget(tier Tier, base int) int {
	if base <= 0 {
		base = 16 << 10
	}
	shrink := func(factor int, floor int) int {
		limit := base / factor
		if limit < floor {
			return floor
		}
		return limit
	}
	switch tier {
	case TierAggressivePrune:
		return shrink(8, 2<<10)
	case TierCompactHistory:
		return shrink(2, 4<<10)
	case TierTrimSchemas:
		return shrink(1, 8<<10)
	default:
		return base
	}
}
