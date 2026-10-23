package model

import "strings"

// ModelInfo is what the harness knows about a model beyond its name: the
// capabilities that decide whether a request is even legal.
//
// It is deliberately a small, declarative table rather than a probe. A wrong
// entry can only suppress an optional parameter (thinking on a model that does
// not accept it) or mask a capability we would otherwise offer; it never
// changes a window or a budget, where a wrong number would silently corrupt
// compaction.
type ModelInfo struct {
	// Family is the prefix this entry matches, lowercased.
	Family string
	// SupportsReasoning reports whether the provider accepts the thinking /
	// reasoning-effort parameter. False means the harness drops it instead of
	// sending a field the provider rejects.
	SupportsReasoning bool
	// SupportsTools reports whether the model accepts tool definitions.
	// Unknown models are assumed to support them: every model the harness is
	// wired to is a chat model with tools.
	SupportsTools bool
	// SupportsSearchTool reports a provider-native web search tool. The harness
	// exposes its own web_search, so this is reporting only.
	SupportsSearchTool bool
}

var modelInfoTable = []ModelInfo{
	{Family: "minimax", SupportsReasoning: true, SupportsTools: true},
	{Family: "gpt-5", SupportsReasoning: true, SupportsTools: true, SupportsSearchTool: true},
	{Family: "gpt-4", SupportsReasoning: false, SupportsTools: true},
	{Family: "o3", SupportsReasoning: true, SupportsTools: true},
	{Family: "o4", SupportsReasoning: true, SupportsTools: true},
	{Family: "claude", SupportsReasoning: true, SupportsTools: true},
	{Family: "gemini", SupportsReasoning: true, SupportsTools: true, SupportsSearchTool: true},
	{Family: "qwen", SupportsReasoning: true, SupportsTools: true},
	{Family: "deepseek", SupportsReasoning: true, SupportsTools: true},
	{Family: "glm", SupportsReasoning: true, SupportsTools: true},
	{Family: "kimi", SupportsReasoning: true, SupportsTools: true},
	{Family: "llama", SupportsReasoning: false, SupportsTools: true},
	{Family: "mistral", SupportsReasoning: false, SupportsTools: true},
}

// ModelInfoFor looks a model up by name. An unknown model reports "no
// opinions" (all capabilities true except provider-native search), which keeps
// the harness from stripping a parameter the provider would have accepted.
func ModelInfoFor(name string) ModelInfo {
	lowered := strings.ToLower(strings.TrimSpace(name))
	if lowered != "" {
		for _, info := range modelInfoTable {
			if strings.HasPrefix(lowered, info.Family) {
				return info
			}
		}
	}
	return ModelInfo{
		Family:        lowered,
		SupportsTools: true,
		// Unknown models may or may not support reasoning; assume they do so a
		// configured thinking flag is not silently dropped for a model that
		// was added after this table.
		SupportsReasoning: true,
	}
}

// SupportsThinking reports whether a thinking flag may be sent to this model.
func SupportsThinking(name string) bool { return ModelInfoFor(name).SupportsReasoning }
