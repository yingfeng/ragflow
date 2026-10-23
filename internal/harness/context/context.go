// Package agentctx assembles and compacts the model input for one run.
//
// Compaction happens before each model call and never touches loop state: it
// only decides what the model sees. The strategy is graded by context pressure
// (see budget.go), so an almost-empty window keeps tool output verbatim while a
// nearly full one keeps only the trailing turns.
package agentctx

import (
	"context"
	"fmt"
	"strings"
	"unicode/utf8"

	"ragflow/internal/harness/agent/model"
)

// Config bounds the assembled prompt.
type Config struct {
	// MaxInputTokens is an absolute cap on the assembled prompt, regardless of
	// the model window.
	MaxInputTokens int
	// ContextWindow is the model's full window; zero means DefaultContextWindow.
	ContextWindow int
	// InputBudgetRatio is the share of the window spent on input.
	InputBudgetRatio float64
	// OutputReserveTokens is the room kept for one completion.
	OutputReserveTokens int
	// KeepToolResultBytes is the byte budget for tool output that stays verbatim
	// at zero pressure. It is a budget, not a count: an enumeration accumulates
	// evidence across rounds, and a count cap drops exactly the rounds carrying
	// the names it still needs.
	KeepToolResultBytes int
	// MaxToolResultBytes is the rendering ceiling for a single tool result. It
	// sits above one read_spill window on purpose: the loop's spill threshold
	// already bounds content volume, so truncating here would only cut the window
	// the model just asked to read.
	MaxToolResultBytes int
	// MaxLessons caps injected experience entries.
	MaxLessons int
	// KeepRecentMessages is the minimum number of trailing messages kept when
	// older turns are folded into a summary.
	KeepRecentMessages int
	// AggressiveKeepMessages is the trailing window kept at critical pressure.
	AggressiveKeepMessages int
	// Summarize folds dropped messages into one line. A nil function uses a
	// mechanical summary, which never calls a model.
	Summarize func(dropped []model.Message) string
	// CompactionRatio is the fraction of the input budget the retained window
	// may occupy after folding (0.7 leaves room for the next turns).
	CompactionRatio float64
	// CountTokens optionally counts a string with the model's real tokenizer.
	// It must be safe for a partially-loaded tokenizer; a zero or panicking
	// result falls back to the byte estimate.
	CountTokens func(text string) int
	// SystemBudgetRatio caps the system prompt (identity, constraints, lessons)
	// as a share of the input budget.
	SystemBudgetRatio float64
	// SystemBudgetCapTokens is the absolute ceiling of that share. Astra caps
	// identity and constraints at 10% / 2k tokens each; one combined cap is
	// enough here because the harness has a single system section. The default
	// below is the window's 5%, not Astra's 2k: the harness appends its runtime
	// rules and environment block at the END of the system section, and a 2k
	// cap clipped them on any real operator prompt — the newest rules were the
	// first to go.
	SystemBudgetCapTokens int
	// OnCompact observes every compaction that actually dropped material. It
	// fires from inside Build, so it runs on the caller's context and must not
	// call back into the builder. Nil is a no-op.
	OnCompact func(ctx context.Context, info CompactInfo)
}

// CompactInfo describes one compaction action that changed what the model sees.
type CompactInfo struct {
	// Trigger is the pressure tier that forced the action.
	Trigger string
	// Strategy is what was dropped: fold | prune | clear.
	Strategy string
	// Dropped counts the messages (fold, prune) or tool results (clear) removed.
	Dropped int
	// Reason is the human-readable cause, for compaction analytics.
	Reason string
}

// DefaultConfig returns production defaults.
func DefaultConfig() Config {
	return Config{
		MaxInputTokens:         96000,
		ContextWindow:          DefaultContextWindow,
		InputBudgetRatio:       DefaultInputBudgetRatio,
		OutputReserveTokens:    DefaultOutputReserveTokens,
		KeepToolResultBytes:    DefaultKeepToolResultBytes,
		MaxToolResultBytes:     64 << 10,
		MaxLessons:             5,
		KeepRecentMessages:     8,
		AggressiveKeepMessages: DefaultAggressiveKeepMessages,
		CompactionRatio:        0.7,
		SystemBudgetRatio:      0.10,
		SystemBudgetCapTokens:  10_000,
	}
}

// Builder assembles prompts.
type Builder struct {
	cfg    Config
	budget Budget
}

// New builds a builder, filling zero fields with defaults.
func New(cfg Config) *Builder {
	defaults := DefaultConfig()
	if cfg.MaxInputTokens <= 0 {
		cfg.MaxInputTokens = defaults.MaxInputTokens
	}
	if cfg.KeepToolResultBytes <= 0 {
		cfg.KeepToolResultBytes = defaults.KeepToolResultBytes
	}
	if cfg.MaxToolResultBytes <= 0 {
		cfg.MaxToolResultBytes = defaults.MaxToolResultBytes
	}
	if cfg.MaxLessons <= 0 {
		cfg.MaxLessons = defaults.MaxLessons
	}
	if cfg.KeepRecentMessages <= 0 {
		cfg.KeepRecentMessages = defaults.KeepRecentMessages
	}
	if cfg.AggressiveKeepMessages <= 0 {
		cfg.AggressiveKeepMessages = defaults.AggressiveKeepMessages
	}
	if cfg.CompactionRatio <= 0 || cfg.CompactionRatio >= 1 {
		cfg.CompactionRatio = defaults.CompactionRatio
	}
	if cfg.ContextWindow <= 0 {
		cfg.ContextWindow = defaults.ContextWindow
	}
	if cfg.InputBudgetRatio <= 0 || cfg.InputBudgetRatio >= 1 {
		cfg.InputBudgetRatio = defaults.InputBudgetRatio
	}
	if cfg.OutputReserveTokens <= 0 {
		cfg.OutputReserveTokens = defaults.OutputReserveTokens
	}
	return &Builder{
		cfg: cfg,
		budget: Budget{
			ContextWindow:       cfg.ContextWindow,
			InputBudgetRatio:    cfg.InputBudgetRatio,
			OutputReserveTokens: cfg.OutputReserveTokens,
			MaxInputTokens:      cfg.MaxInputTokens,
		},
	}
}

// ClearedToolPlaceholder replaces a tool result the model no longer needs.
const ClearedToolPlaceholder = "[previous tool output cleared]"

// clearedToolPlaceholder names the tool that produced the cleared result, so the
// model knows what it lost and how to get it back instead of assuming the result
// was empty.
func clearedToolPlaceholder(tool string) string {
	if tool == "" {
		return ClearedToolPlaceholder
	}
	return fmt.Sprintf("[output of %s cleared to save context — call %s again if you need it]", tool, tool)
}

// PrintedPlaceholder replaces a dropped span of history at critical pressure.
const PrunedPlaceholder = "[earlier context pruned: %d messages]"

// foldSummaryPrefix marks the message a fold left in place of the older turns.
const foldSummaryPrefix = "[earlier turns compacted: "

// Build assembles the prompt: system (with lessons), then compacted history.
func (b *Builder) Build(ctx context.Context, system string, lessons []string, history []model.Message) []model.Message {
	return b.BuildWithTier(ctx, system, lessons, history, TierNormal)
}

// BuildWithTier forces at least minTier. A prompt-too-long rejection from the
// provider is the one signal that must override the estimate: the provider just
// proved the prompt does not fit, so the next attempt compacts harder instead of
// failing the run.
func (b *Builder) BuildWithTier(ctx context.Context, system string, lessons []string, history []model.Message, minTier Tier) []model.Message {
	prompt, _ := b.BuildWithFoldHint(ctx, system, lessons, history, minTier, nil)
	return prompt
}

// BuildWithFoldHint builds with a summary that was prepared ahead of time. A nil
// hint is the ordinary path and summarizes synchronously if a fold is due. The
// second result reports that the hint replaced a model call.
func (b *Builder) BuildWithFoldHint(ctx context.Context, system string, lessons []string, history []model.Message, minTier Tier, hint *FoldHint) ([]model.Message, bool) {
	history = ensureToolOutputs(history)
	systemContent := b.renderSystemWithinBudget(system, lessons)
	limit := b.limit()
	headroom := limit - b.tokens(systemContent)
	if headroom < 0 {
		headroom = 0
	}

	compacted := make([]model.Message, len(history))
	copy(compacted, history)

	tier := b.budget.TierFor(b.tokens(systemContent) + b.estimate(compacted))
	if tier < minTier {
		tier = minTier
	}
	// A single oversized result is bounded at any pressure: that is rendering, not
	// memory management. Losing the older results outright is memory management,
	// and only pressure justifies it — clearing them at zero pressure made a
	// 29-round enumeration answer from the six results it could still see.
	compacted = b.truncateToolResults(compacted, tier)
	if tier >= TierTrimSchemas {
		compacted = b.notifyClear(ctx, tier, compacted)
	}

	usedHint := false
	if tier >= TierCompactHistory {
		compacted, usedHint = b.notifyFold(ctx, tier, compacted, headroom, hint)
	}
	if tier >= TierAggressivePrune {
		compacted = b.notifyPrune(ctx, tier, compacted)
	}
	compacted = b.fitToBudget(compacted, headroom)

	out := make([]model.Message, 0, len(compacted)+1)
	out = append(out, model.Message{Role: model.RoleSystem, Content: systemContent})
	out = append(out, compacted...)
	return out, usedHint
}

// notifyClear wraps clearOldToolResults with observation. How many results were
// cleared is the fact compaction analytics needs: the model that answers from
// the six results it can still see is the failure mode this exists to surface.
func (b *Builder) notifyClear(ctx context.Context, tier Tier, messages []model.Message) []model.Message {
	before := 0
	for _, message := range messages {
		if message.Role == model.RoleTool {
			before++
		}
	}
	out := b.clearOldToolResults(messages, tier)
	after := 0
	for _, message := range out {
		if message.Role == model.RoleTool && message.Content != "" &&
			!strings.HasPrefix(message.Content, "[output of ") {
			after++
		}
	}
	if dropped := before - after; dropped > 0 {
		b.notifyCompact(ctx, CompactInfo{Trigger: tier.String(), Strategy: "clear", Dropped: dropped, Reason: "tool result byte budget under pressure"})
	}
	return out
}

// notifyFold wraps foldWithSummary with observation.
func (b *Builder) notifyFold(ctx context.Context, tier Tier, messages []model.Message, headroom int, hint *FoldHint) ([]model.Message, bool) {
	out, usedHint := b.foldWithSummary(messages, headroom, hint)
	if len(out) < len(messages) {
		b.notifyCompact(ctx, CompactInfo{Trigger: tier.String(), Strategy: "fold", Dropped: len(messages) - len(out), Reason: "older turns folded into a summary"})
	}
	return out, usedHint
}

// notifyPrune wraps pruneToRecent with observation.
func (b *Builder) notifyPrune(ctx context.Context, tier Tier, messages []model.Message) []model.Message {
	out := b.pruneToRecent(messages)
	if len(out) < len(messages) {
		b.notifyCompact(ctx, CompactInfo{Trigger: tier.String(), Strategy: "prune", Dropped: len(messages) - len(out), Reason: "critical pressure: only the trailing window kept"})
	}
	return out
}

func (b *Builder) notifyCompact(ctx context.Context, info CompactInfo) {
	if b.cfg.OnCompact != nil {
		b.cfg.OnCompact(ctx, info)
	}
}

// clearOldToolResults replaces tool output the model no longer needs and
// truncates the results that stay. Under pressure both the byte budget for
// verbatim output and the per-result ceiling shrink.
//
// Retention is measured in BYTES, not in results. A fixed count is wrong for the
// one shape that matters most here: an enumeration gathers its evidence over many
// rounds, so "keep the last six" deletes precisely the earlier rounds that hold
// the names it still has to account for — an observed 29-round enumeration ended
// up answering from the six results it could still see.
func (b *Builder) clearOldToolResults(messages []model.Message, tier Tier) []model.Message {
	toolIndexes := make([]int, 0, len(messages))
	for index, message := range messages {
		if message.Role == model.RoleTool {
			toolIndexes = append(toolIndexes, index)
		}
	}
	budget := toolResultByteBudget(tier, b.cfg.KeepToolResultBytes)
	// Zero means "no clearing at this pressure": the transcript keeps every tool
	// result, exactly as the agentic loop it mirrors does. Only a real pressure
	// tier budgets the output, and only then does the window cap below apply.
	if budget <= 0 {
		return messages
	}
	// A byte budget is meaningless next to a window that cannot hold it: without
	// this cap, pressure on a small window would keep everything and let the final
	// fit do the cutting, which drops the wrong rounds. Three bytes per token is a
	// conservative estimate for the mixed CJK/latin text retrieval returns.
	if limit := b.budget.EffectiveInput(); limit > 0 {
		windowBytes := limit * 3
		ceiling := windowBytes / 2
		if tier >= TierAggressivePrune {
			ceiling = windowBytes / 4
		}
		if budget > ceiling {
			budget = ceiling
		}
	}
	keep := make(map[int]struct{}, len(toolIndexes))
	used := 0
	// Newest first: a result is kept while it fits the remaining budget. The
	// newest is kept even when it alone exceeds the budget — clearing the output
	// the model just received would leave it answering from nothing.
	for position := len(toolIndexes) - 1; position >= 0; position-- {
		index := toolIndexes[position]
		size := len(messages[index].Content)
		if len(keep) > 0 && used+size > budget {
			break
		}
		keep[index] = struct{}{}
		used += size
	}

	out := make([]model.Message, len(messages))
	copy(out, messages)
	for index, message := range out {
		if message.Role != model.RoleTool {
			continue
		}
		if _, ok := keep[index]; !ok {
			message.Content = clearedToolPlaceholder(message.Name)
			out[index] = message
		}
	}
	return out
}

// truncateToolResults bounds one result's size. Every tier does this: a single
// 400 KB tool result is a rendering problem, not a memory-management decision,
// and the spill capability already keeps the full text addressable.
func (b *Builder) truncateToolResults(messages []model.Message, tier Tier) []model.Message {
	resultLimit := toolResultBudget(tier, b.cfg.MaxToolResultBytes)
	out := make([]model.Message, len(messages))
	copy(out, messages)
	for index, message := range out {
		if message.Role != model.RoleTool || len(message.Content) <= resultLimit {
			continue
		}
		message.Content = truncateMiddle(message.Content, resultLimit)
		out[index] = message
	}
	return out
}

// foldWithSummary replaces the oldest turns with one summary message once the
// history exceeds the budget. Dropping whole turns (fitToBudget) loses the
// question that started the work; a summary keeps the thread of the task while
// freeing the tokens.
func (b *Builder) foldWithSummary(messages []model.Message, limit int, hint *FoldHint) ([]model.Message, bool) {
	cut := b.foldCut(messages, limit)
	if cut <= 0 {
		return messages, false
	}

	// A hint was computed a round earlier, over a shorter prefix than the fold
	// is about to drop. Using it folds less and saves the model call; waiting
	// for a fresh summary is the pause this whole mechanism exists to remove.
	summary := ""
	usedHint := false
	if hint != nil && hint.Covered >= cut/2 && hint.Covered < len(messages) {
		cut = hint.Covered
		summary = strings.TrimSpace(hint.Summary)
		usedHint = summary != ""
	}
	if summary == "" {
		summary = b.summarize(messages[:cut])
	}
	if strings.TrimSpace(summary) == "" {
		return messages, false
	}

	out := make([]model.Message, 0, len(messages)-cut+1)
	out = append(out, model.Message{
		Role:    model.RoleUser,
		Content: foldSummaryPrefix + fmt.Sprintf("%d messages]\n%s", cut, summary),
	})
	out = append(out, messages[cut:]...)
	return out, usedHint
}

// foldCut is how many leading messages a fold would replace, or zero when the
// history still fits. Splitting it out lets speculation prepare the same span the
// fold will later take.
func (b *Builder) foldCut(messages []model.Message, limit int) int {
	if b.estimate(messages) <= limit {
		return 0
	}
	keep := b.cfg.KeepRecentMessages
	if keep < 1 || keep >= len(messages) {
		return 0
	}

	cut := len(messages) - keep
	for cut < len(messages) && messages[cut].Role == model.RoleTool {
		cut++
	}
	if cut >= len(messages) {
		return 0
	}

	// Tighten the retained window to the target ratio so the next turns fit
	// without compacting again immediately.
	target := int(float64(limit) * b.cfg.CompactionRatio)
	for cut < len(messages)-keep && b.estimate(messages[cut:]) > target {
		cut++
		for cut < len(messages) && messages[cut].Role == model.RoleTool {
			cut++
		}
	}
	if cut <= 0 || cut >= len(messages) {
		return 0
	}
	return cut
}

func (b *Builder) summarize(dropped []model.Message) string {
	if b.cfg.Summarize != nil {
		if summary := strings.TrimSpace(b.cfg.Summarize(dropped)); summary != "" {
			return summary
		}
		// A failed model summary must still free the window; the mechanical
		// summary is lossy but never wrong about the task thread.
	}
	return mechanicalSummary(dropped)
}

// mechanicalSummary keeps the questions asked so far and the tools already used.
// It is intentionally lossy: the point is to preserve the task thread, not the
// transcript.
func mechanicalSummary(dropped []model.Message) string {
	questions := make([]string, 0, 6)
	tools := make([]string, 0, 6)
	seenTools := make(map[string]struct{}, 6)

	for _, message := range dropped {
		switch message.Role {
		case model.RoleUser:
			if content := strings.TrimSpace(message.Content); content != "" {
				questions = append(questions, truncate(content, 200))
			}
		case model.RoleTool:
			if message.Name == "" {
				continue
			}
			if _, ok := seenTools[message.Name]; !ok {
				seenTools[message.Name] = struct{}{}
				tools = append(tools, message.Name)
			}
		}
	}

	var builder strings.Builder
	if len(questions) > 0 {
		if len(questions) > 5 {
			questions = questions[len(questions)-5:]
		}
		builder.WriteString("Questions asked so far: ")
		builder.WriteString(strings.Join(questions, " | "))
	}
	if len(tools) > 0 {
		if builder.Len() > 0 {
			builder.WriteString("\n")
		}
		builder.WriteString("Tools already used: ")
		builder.WriteString(strings.Join(tools, ", "))
	}
	return builder.String()
}

// pruneToRecent is the critical-pressure floor: keep only the trailing window,
// never opening it on a tool message, and leave one placeholder where the
// dropped span used to be.
func (b *Builder) pruneToRecent(messages []model.Message) []model.Message {
	keep := b.cfg.AggressiveKeepMessages
	if keep < 1 || len(messages) <= keep {
		return messages
	}
	cut := len(messages) - keep
	for cut < len(messages) && messages[cut].Role == model.RoleTool {
		cut++
	}
	if cut <= 0 || cut >= len(messages) {
		return messages
	}
	out := make([]model.Message, 0, len(messages)-cut+2)
	// The fold summary carries the question, the decisions and what was already
	// ruled out. Pruning it away leaves a window of recent reads with no task in
	// it, which is how a run ends up answering the last passage it read.
	if summary, ok := foldSummaryOf(messages[:cut]); ok {
		out = append(out, summary)
	}
	out = append(out, model.Message{
		Role:    model.RoleUser,
		Content: fmt.Sprintf(PrunedPlaceholder, cut),
	})
	return append(out, messages[cut:]...)
}

// foldSummaryOf returns the newest fold summary in a dropped span.
func foldSummaryOf(messages []model.Message) (model.Message, bool) {
	for index := len(messages) - 1; index >= 0; index-- {
		if strings.HasPrefix(messages[index].Content, foldSummaryPrefix) {
			return messages[index], true
		}
	}
	return model.Message{}, false
}

// fitToBudget drops the oldest turns until the estimate fits. It never drops the
// last user message and never leaves a tool message without its assistant call.
func (b *Builder) fitToBudget(messages []model.Message, limit int) []model.Message {
	if b.estimate(messages) <= limit {
		return messages
	}
	start := 0
	for start < len(messages) && b.estimate(messages[start:]) > limit {
		start++
		// Never start on a tool message: it would be an orphan.
		for start < len(messages) && messages[start].Role == model.RoleTool {
			start++
		}
		// Keep the most recent user message in view.
		if start >= len(messages) {
			break
		}
	}
	if start >= len(messages) {
		// Fall back to the last message so the model always sees something.
		return messages[len(messages)-1:]
	}
	return messages[start:]
}

// renderSystemWithinBudget keeps the system section inside its share of the
// window. Lessons go first — they are the cheapest part to drop and the most
// likely to be stale — then the system text itself is clipped.
func (b *Builder) renderSystemWithinBudget(system string, lessons []string) string {
	rendered := renderSystem(system, lessons, b.cfg.MaxLessons)
	budget := b.systemBudget()
	if budget <= 0 || b.tokens(rendered) <= budget {
		return rendered
	}
	keep := len(lessons)
	if keep > b.cfg.MaxLessons {
		keep = b.cfg.MaxLessons
	}
	for count := keep - 1; count >= 0; count-- {
		rendered = renderSystem(system, lessons[:count], b.cfg.MaxLessons)
		if b.tokens(rendered) <= budget {
			return rendered
		}
	}
	return truncateHeadTail(renderSystem(system, nil, 0), budget*4)
}

// truncateHeadTail keeps the head and the tail of an over-budget system section
// and drops the middle. The harness appends its runtime rules and environment
// block at the tail; a head-only clip removed exactly those, so the newest
// rules were always the first to go.
func truncateHeadTail(text string, limitBytes int) string {
	if limitBytes <= 0 || len(text) <= limitBytes {
		return text
	}
	const marker = "\n…[system prompt truncated]…\n"
	if limitBytes <= 128 {
		return truncate(text, limitBytes)
	}
	tailBytes := limitBytes / 4
	headBytes := limitBytes - tailBytes - len(marker)
	if headBytes <= 0 {
		return truncate(text, limitBytes)
	}
	head := truncate(text, headBytes)
	tailStart := len(text) - tailBytes
	for tailStart > 0 && !utf8.RuneStart(text[tailStart]) {
		tailStart--
	}
	return head + marker + text[tailStart:]
}

// systemBudget resolves the token ceiling of the system section.
func (b *Builder) systemBudget() int {
	ratio := b.cfg.SystemBudgetRatio
	if ratio <= 0 || ratio >= 1 {
		ratio = DefaultConfig().SystemBudgetRatio
	}
	budget := int(float64(b.limit()) * ratio)
	if cap := b.cfg.SystemBudgetCapTokens; cap > 0 && budget > cap {
		budget = cap
	}
	return budget
}

func renderSystem(system string, lessons []string, maxLessons int) string {
	builder := strings.Builder{}
	builder.WriteString(strings.TrimSpace(system))
	if len(lessons) == 0 {
		return builder.String()
	}
	if maxLessons > 0 && len(lessons) > maxLessons {
		lessons = lessons[:maxLessons]
	}
	builder.WriteString("\n\n## Relevant experience\n")
	for _, lesson := range lessons {
		line := strings.TrimSpace(lesson)
		if line == "" {
			continue
		}
		builder.WriteString(fmt.Sprintf("- %s\n", line))
	}
	return strings.TrimSpace(builder.String())
}

// limit is the token ceiling for the assembled prompt.
// Window is the model's full context window. The loop reports it to the model
// through context_remaining, so the number the model sees is the same one the
// builder compacts against.
func (b *Builder) Window() int {
	if b.cfg.ContextWindow > 0 {
		return b.cfg.ContextWindow
	}
	return DefaultContextWindow
}

func (b *Builder) limit() int {
	return b.budget.EffectiveInput()
}

func truncate(value string, limit int) string {
	if len(value) <= limit {
		return value
	}
	return cutRunes(value, limit) + "\n[truncated]"
}

// truncateMiddle keeps the head and the tail, like Astra's sanitizer and
// codex's truncate_middle: the beginning carries the shape of the output and the
// end carries the result, while the middle is usually repetition.
func truncateMiddle(value string, limit int) string {
	if len(value) <= limit {
		return value
	}
	if limit <= 0 {
		return ""
	}
	marker := fmt.Sprintf("\n[… truncated %d characters …]\n", len(value)-limit)
	keep := limit - len(marker)
	if keep <= 0 {
		return cutRunes(value, limit) + "\n[truncated]"
	}
	head := keep * 4 / 10
	tail := keep - head
	return cutRunes(value, head) + marker + tailRunes(value, tail)
}

// cutRunes returns at most limit bytes, ending on a rune boundary.
func cutRunes(value string, limit int) string {
	if limit <= 0 {
		return ""
	}
	if len(value) <= limit {
		return value
	}
	cut := limit
	for cut > 0 && (value[cut]&0xC0) == 0x80 {
		cut--
	}
	return value[:cut]
}

// tailRunes returns at most limit bytes from the end, starting on a rune
// boundary.
func tailRunes(value string, limit int) string {
	if limit <= 0 {
		return ""
	}
	if len(value) <= limit {
		return value
	}
	start := len(value) - limit
	for start < len(value) && (value[start]&0xC0) == 0x80 {
		start++
	}
	return value[start:]
}

// tokens counts one string, preferring the injected tokenizer.
func (b *Builder) tokens(text string) (count int) {
	if b.cfg.CountTokens != nil {
		defer func() {
			if recover() != nil || count <= 0 {
				count = len(text)/4 + 1
			}
		}()
		if n := b.cfg.CountTokens(text); n > 0 {
			return n
		}
	}
	return len(text)/4 + 1
}

// estimate is the token estimate for a message list, about four bytes per token
// plus per-message framing when no real tokenizer is injected.
func (b *Builder) estimate(messages []model.Message) int {
	total := 0
	for _, message := range messages {
		total += 4
		total += b.tokens(message.Content)
		for _, call := range message.ToolCalls {
			total += b.tokens(call.Name) + b.tokens(call.Arguments) + 4
		}
	}
	return total
}

// ensureToolOutputs gives every declared tool call a result.
//
// A provider rejects a request whose assistant message carries tool_calls that
// no following tool message answers. The loop writes a result for every call it
// runs, but a run that was cancelled, checkpointed mid-batch, or recovered from
// a panic can leave the transcript one message short — and every later request
// in that run would fail on a 400 instead of on anything the model did.
//
// The message is synthesized rather than dropped: the model needs to learn that
// the call did not return, not that it was never asked.
func ensureToolOutputs(history []model.Message) []model.Message {
	var patched []model.Message
	for index := 0; index < len(history); index++ {
		message := history[index]
		patched = append(patched, message)
		if message.Role != model.RoleAssistant || len(message.ToolCalls) == 0 {
			continue
		}
		answered := make(map[string]struct{}, len(message.ToolCalls))
		next := index + 1
		for next < len(history) && history[next].Role == model.RoleTool {
			answered[history[next].ToolCallID] = struct{}{}
			patched = append(patched, history[next])
			next++
		}
		for _, call := range message.ToolCalls {
			if _, ok := answered[call.ID]; ok {
				continue
			}
			patched = append(patched, model.Message{
				Role:       model.RoleTool,
				ToolCallID: call.ID,
				Name:       call.Name,
				Content:    "[aborted] the call did not return a result",
			})
		}
		index = next - 1
	}
	if patched == nil {
		return history
	}
	return patched
}
