// Package agentctx assembles and compacts the model input for one run.
//
// Compaction happens before each model call and never touches loop state: it
// only decides what the model sees. The strategy is graded by context pressure
// (see budget.go), so an almost-empty window keeps tool output verbatim while a
// nearly full one keeps only the trailing turns.
package agentctx

import (
	"fmt"
	"strings"

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
	// KeepRecentToolResults is how many trailing tool results stay verbatim at
	// normal pressure. Higher tiers keep fewer.
	KeepRecentToolResults int
	// MaxToolResultBytes truncates a single tool result.
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
	// enough here because the harness has a single system section.
	SystemBudgetCapTokens int
}

// DefaultConfig returns production defaults.
func DefaultConfig() Config {
	return Config{
		MaxInputTokens:         96000,
		ContextWindow:          DefaultContextWindow,
		InputBudgetRatio:       DefaultInputBudgetRatio,
		OutputReserveTokens:    DefaultOutputReserveTokens,
		KeepRecentToolResults:  6,
		MaxToolResultBytes:     16 << 10,
		MaxLessons:             5,
		KeepRecentMessages:     8,
		AggressiveKeepMessages: DefaultAggressiveKeepMessages,
		CompactionRatio:        0.7,
		SystemBudgetRatio:      0.10,
		SystemBudgetCapTokens:  2_000,
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
	if cfg.KeepRecentToolResults <= 0 {
		cfg.KeepRecentToolResults = defaults.KeepRecentToolResults
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

// PrintedPlaceholder replaces a dropped span of history at critical pressure.
const PrunedPlaceholder = "[earlier context pruned: %d messages]"

// Build assembles the prompt: system (with lessons), then compacted history.
func (b *Builder) Build(system string, lessons []string, history []model.Message) []model.Message {
	return b.BuildWithTier(system, lessons, history, TierNormal)
}

// BuildWithTier forces at least minTier. A prompt-too-long rejection from the
// provider is the one signal that must override the estimate: the provider just
// proved the prompt does not fit, so the next attempt compacts harder instead of
// failing the run.
func (b *Builder) BuildWithTier(system string, lessons []string, history []model.Message, minTier Tier) []model.Message {
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
	compacted = b.clearOldToolResults(compacted, tier)

	if tier >= TierCompactHistory {
		compacted = b.foldWithSummary(compacted, headroom)
	}
	if tier >= TierAggressivePrune {
		compacted = b.pruneToRecent(compacted)
	}
	compacted = b.fitToBudget(compacted, headroom)

	out := make([]model.Message, 0, len(compacted)+1)
	out = append(out, model.Message{Role: model.RoleSystem, Content: systemContent})
	out = append(out, compacted...)
	return out
}

// clearOldToolResults replaces tool output the model no longer needs and
// truncates the results that stay. Under pressure both the number of verbatim
// results and their per-result budget shrink.
func (b *Builder) clearOldToolResults(messages []model.Message, tier Tier) []model.Message {
	toolIndexes := make([]int, 0, len(messages))
	for index, message := range messages {
		if message.Role == model.RoleTool {
			toolIndexes = append(toolIndexes, index)
		}
	}
	keepCount := keepRecentToolResults(tier, b.cfg.KeepRecentToolResults)
	verbatimFrom := len(toolIndexes) - keepCount
	if verbatimFrom < 0 {
		verbatimFrom = 0
	}
	keep := make(map[int]struct{}, keepCount)
	for _, position := range toolIndexes[verbatimFrom:] {
		keep[position] = struct{}{}
	}

	resultLimit := toolResultBudget(tier, b.cfg.MaxToolResultBytes)
	out := make([]model.Message, len(messages))
	copy(out, messages)
	for index, message := range out {
		if message.Role != model.RoleTool {
			continue
		}
		if _, ok := keep[index]; !ok {
			message.Content = ClearedToolPlaceholder
			out[index] = message
			continue
		}
		if len(message.Content) > resultLimit {
			message.Content = truncateMiddle(message.Content, resultLimit)
			out[index] = message
		}
	}
	return out
}

// foldWithSummary replaces the oldest turns with one summary message once the
// history exceeds the budget. Dropping whole turns (fitToBudget) loses the
// question that started the work; a summary keeps the thread of the task while
// freeing the tokens.
func (b *Builder) foldWithSummary(messages []model.Message, limit int) []model.Message {
	if b.estimate(messages) <= limit {
		return messages
	}
	keep := b.cfg.KeepRecentMessages
	if keep < 1 || keep >= len(messages) {
		return messages
	}

	cut := len(messages) - keep
	for cut < len(messages) && messages[cut].Role == model.RoleTool {
		cut++
	}
	if cut >= len(messages) {
		return messages
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
		return messages
	}

	dropped := messages[:cut]
	summary := b.summarize(dropped)
	if strings.TrimSpace(summary) == "" {
		return messages
	}

	out := make([]model.Message, 0, len(messages)-cut+1)
	out = append(out, model.Message{
		Role:    model.RoleUser,
		Content: fmt.Sprintf("[earlier turns compacted: %d messages]\n%s", len(dropped), summary),
	})
	out = append(out, messages[cut:]...)
	return out
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
	out := make([]model.Message, 0, len(messages)-cut+1)
	out = append(out, model.Message{
		Role:    model.RoleUser,
		Content: fmt.Sprintf(PrunedPlaceholder, cut),
	})
	return append(out, messages[cut:]...)
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
	return truncate(renderSystem(system, nil, 0), budget*4)
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
