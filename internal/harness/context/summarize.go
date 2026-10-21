package agentctx

import (
	"context"
	"fmt"
	"strings"
	"time"

	"ragflow/internal/harness/agent/model"
	"ragflow/internal/tokenizer"
)

// DefaultCountTokens counts with the RAGFlow cl100k tokenizer.
//
// NumTokensFromString panics when the BPE table has not been downloaded, and a
// missing asset must not take down a run, so the estimate is the fallback. The
// Builder also guards the call; this wrapper keeps the panic out of the caller.
func DefaultCountTokens(text string) (count int) {
	defer func() {
		if recover() != nil || count <= 0 {
			count = len(text)/4 + 1
		}
	}()
	return tokenizer.NumTokensFromString(text)
}

// Summary prompt constants. The instruction set mirrors what the reference
// runtimes keep: goals, decisions, identifiers, conclusions — not raw output.
const (
	summarySystemPrompt = `You are compacting the earlier part of an agent run so it fits back into the context window.
Rewrite it as a terse briefing for the agent that will continue the work.
Keep: the user's goal, constraints, decisions already made, exact identifiers (paths, ids, query strings, numbers), conclusions drawn from tools, and open questions.
Drop: raw tool output, restated boilerplate, and anything the agent can re-derive by calling a tool again.
Answer with the briefing only, no preamble.`

	// summaryTranscriptChars bounds what the summarizer is asked to read.
	summaryTranscriptChars = 60_000
	// summaryTimeout bounds the summarization call; a slow summary must not
	// stall the run it is trying to rescue.
	summaryTimeout = 25 * time.Second
	// DefaultSummaryTokens is the budget for one compaction summary.
	DefaultSummaryTokens = 2_048
)

// NewSummarizer returns a Summarize hook backed by the run's model. On any
// failure it returns an empty string, and the Builder falls back to the
// mechanical summary instead of dropping history silently.
func NewSummarizer(mdl model.Model, maxTokens int) func(dropped []model.Message) string {
	if maxTokens <= 0 {
		maxTokens = DefaultSummaryTokens
	}
	return func(dropped []model.Message) string {
		if mdl == nil || len(dropped) == 0 {
			return ""
		}
		ctx, cancel := context.WithTimeout(context.Background(), summaryTimeout)
		defer cancel()

		response, err := mdl.Complete(ctx, model.Request{
			Messages: []model.Message{
				{Role: model.RoleSystem, Content: summarySystemPrompt},
				{Role: model.RoleUser, Content: renderTranscript(dropped)},
			},
			MaxTokens: maxTokens,
		})
		if err != nil {
			return ""
		}
		return strings.TrimSpace(response.Message.Content)
	}
}

// renderTranscript flattens messages for the summarizer, keeping the tool name
// and call arguments but not the tool output bodies.
func renderTranscript(messages []model.Message) string {
	var builder strings.Builder
	for _, message := range messages {
		switch message.Role {
		case model.RoleUser:
			fmt.Fprintf(&builder, "USER: %s\n", message.Content)
		case model.RoleAssistant:
			if content := strings.TrimSpace(message.Content); content != "" {
				fmt.Fprintf(&builder, "ASSISTANT: %s\n", content)
			}
			for _, call := range message.ToolCalls {
				fmt.Fprintf(&builder, "ASSISTANT CALL: %s(%s)\n", call.Name, truncate(call.Arguments, 600))
			}
		case model.RoleTool:
			fmt.Fprintf(&builder, "TOOL %s: %s\n", message.Name, truncate(message.Content, 400))
		default:
			if content := strings.TrimSpace(message.Content); content != "" {
				fmt.Fprintf(&builder, "%s: %s\n", strings.ToUpper(string(message.Role)), truncate(content, 600))
			}
		}
		if builder.Len() > summaryTranscriptChars {
			builder.WriteString("[… transcript truncated …]\n")
			break
		}
	}
	return builder.String()
}
