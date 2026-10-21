// Package agentctx assembles and compacts the model input for one run.
//
// Compaction happens before each model call and never touches loop state: it
// only decides what the model sees.
package agentctx

import (
	"fmt"
	"strings"

	"ragflow/internal/harness/agent/model"
)

// Config bounds the assembled prompt.
type Config struct {
	// MaxInputTokens is the soft budget used to decide how much history to keep.
	MaxInputTokens int
	// KeepRecentToolResults is how many trailing tool results stay verbatim.
	KeepRecentToolResults int
	// MaxToolResultBytes truncates a single tool result.
	MaxToolResultBytes int
	// MaxLessons caps injected experience entries.
	MaxLessons int
}

// DefaultConfig returns production defaults.
func DefaultConfig() Config {
	return Config{
		MaxInputTokens:        96000,
		KeepRecentToolResults: 4,
		MaxToolResultBytes:    16 << 10,
		MaxLessons:            5,
	}
}

// Builder assembles prompts.
type Builder struct {
	cfg Config
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
	return &Builder{cfg: cfg}
}

// ClearedToolPlaceholder replaces a tool result the model no longer needs.
const ClearedToolPlaceholder = "[previous tool output cleared]"

// Build assembles the prompt: system (with lessons), then compacted history.
func (b *Builder) Build(system string, lessons []string, history []model.Message) []model.Message {
	out := make([]model.Message, 0, len(history)+1)
	out = append(out, model.Message{
		Role:    model.RoleSystem,
		Content: renderSystem(system, lessons, b.cfg.MaxLessons),
	})

	compacted := make([]model.Message, len(history))
	copy(compacted, history)

	// Count trailing tool results that stay verbatim.
	toolIndexes := make([]int, 0, len(compacted))
	for index, message := range compacted {
		if message.Role == model.RoleTool {
			toolIndexes = append(toolIndexes, index)
		}
	}
	verbatimFrom := len(toolIndexes) - b.cfg.KeepRecentToolResults
	if verbatimFrom < 0 {
		verbatimFrom = 0
	}
	keep := make(map[int]struct{}, b.cfg.KeepRecentToolResults)
	for _, position := range toolIndexes[verbatimFrom:] {
		keep[position] = struct{}{}
	}

	for index, message := range compacted {
		if message.Role == model.RoleTool {
			if _, ok := keep[index]; !ok {
				message.Content = ClearedToolPlaceholder
			} else if len(message.Content) > b.cfg.MaxToolResultBytes {
				message.Content = truncate(message.Content, b.cfg.MaxToolResultBytes)
			}
		}
		compacted[index] = message
	}

	trimmed := b.fitToBudget(compacted)
	out = append(out, trimmed...)
	return out
}

// fitToBudget drops the oldest turns until the estimate fits. It never drops the
// last user message and never leaves a tool message without its assistant call.
func (b *Builder) fitToBudget(messages []model.Message) []model.Message {
	if estimate(messages) <= b.cfg.MaxInputTokens {
		return messages
	}
	start := 0
	for start < len(messages) && estimate(messages[start:]) > b.cfg.MaxInputTokens {
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

func truncate(value string, limit int) string {
	if len(value) <= limit {
		return value
	}
	return value[:limit] + "\n[truncated]"
}

// estimate is a cheap token estimate (about four bytes per token plus framing).
func estimate(messages []model.Message) int {
	total := 0
	for _, message := range messages {
		total += 4
		total += len(message.Content)/4 + 1
		for _, call := range message.ToolCalls {
			total += len(call.Name)/4 + len(call.Arguments)/4 + 4
		}
	}
	return total
}
