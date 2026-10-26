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
	// DefaultInputBudgetRatio is the share of the window spent on input. It is
	// 1.0 because DefaultOutputReserveTokens already subtracts the room one
	// completion needs: a ratio below 1 is a second reserve over the same gap,
	// and on a 200k window the two together capped input at 143k while the
	// provider had accepted 130k and still had room. Trimming the transcript at a
	// number nobody measured is what made a run fold eight rounds of retrieval
	// into a summary and start over.
	DefaultInputBudgetRatio = 1.0
	// DefaultOutputReserveTokens is the room kept for one completion.
	DefaultOutputReserveTokens = 16_384
	// DefaultAggressiveKeepMessages is the trailing window kept once pressure is
	// critical.
	DefaultAggressiveKeepMessages = 6
)

// Tier is the compaction level selected from context pressure.
type Tier int

// String names the tier for compaction records.
func (t Tier) String() string {
	switch t {
	case TierNormal:
		return "normal"
	case TierTrimSchemas:
		return "trim_schemas"
	case TierCompactHistory:
		return "compact_history"
	case TierAggressivePrune:
		return "aggressive_prune"
	}
	return "unknown"
}

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
	if ratio <= 0 || ratio > 1 {
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

// DefaultKeepToolResultBytes is the byte budget for verbatim tool output kept in
// the transcript at zero pressure.
//
// It replaces a fixed result COUNT (six results), and the difference is the whole
// reason an enumeration came out short: the evidence for a list is accumulated
// across many rounds, so a count cap deletes exactly the rounds that carry it —
// a 29-round enumeration was observed answering from the six results it could
// still see. A byte budget keeps every round that actually fits instead of
// counting rounds as if each one cost the same.
// The number is sized from the workload it has to survive: a 24-round
// enumeration pulls roughly 8 KiB of retrieval output per round, and the earlier
// 96 KiB budget kept only the last ~12 of them. The rounds it dropped were the
// ones holding the names already found, so the run re-read the same passages
// (34 spill reads in that run) and the longer it searched the shorter its list
// got — 14 names at 10 rounds, 9 names at 24. This budget holds a whole
// enumeration instead, which is what removes the reason to re-read at all.
const DefaultKeepToolResultBytes = 160 << 10

// toolResultByteBudget is the adaptive counterpart of Astra's microcompact: the
// more pressure, the fewer bytes of tool output stay verbatim.
//
// Zero clears NOTHING, and the zero is deliberate rather than an unlimited
// setting. The agentic loop these tools come from keeps every tool result in the
// transcript until the window itself fills, and it answers a counting question in
// a few rounds because of it: one broad search returns thirty passages at once,
// and every later round still sees all thirty. Clearing on a lower tier removes
// exactly that advantage — the older passages are the ones holding the first
// names found, so the answer gets shorter the longer the run searches.
//
// TrimSchemas also clears nothing now. It used to halve the budget at 60%
// pressure, which on a live enumeration meant that the second search round
// replaced the first one's passages with a placeholder; the model, which can
// only see what the prompt holds, then re-issued the same grep under a new
// intent instead of reading what it already had. Dropping evidence while the
// transcript still fits does not buy room, it buys a re-search. Clearing starts
// at the tier that also folds turns away, where the space is actually needed.
func toolResultByteBudget(tier Tier, base int) int {
	if tier <= TierTrimSchemas {
		return 0
	}
	if base <= 0 {
		base = DefaultKeepToolResultBytes
	}
	shrink := func(factor int, floor int) int {
		budget := base / factor
		if budget < floor {
			return floor
		}
		return budget
	}
	switch tier {
	case TierAggressivePrune:
		return shrink(8, 8<<10)
	default:
		return shrink(4, 16<<10)
	}
}

// toolResultBudget shrinks the per-result ceiling under pressure.
//
// This is a rendering guard, not memory management: the content volume is
// already bounded by the loop's spill threshold, which replaces an oversized
// result with a preview plus a handle. The ceiling must therefore stay ABOVE one
// read_spill window (16 KiB) — cutting there would truncate the very bytes the
// model just asked to read back.
func toolResultBudget(tier Tier, base int) int {
	if base <= 0 {
		base = 64 << 10
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
		return shrink(4, 20<<10)
	case TierCompactHistory:
		return shrink(2, 24<<10)
	case TierTrimSchemas:
		return shrink(1, 32<<10)
	default:
		return base
	}
}
