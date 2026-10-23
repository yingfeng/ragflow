// Package model is the thin model wire used by the harness loop: one request
// in, one response out, with tool calls carried verbatim.
//
// It holds no provider implementation. Provider talks to the RAGFlow model
// provider (internal/entity/models) through ChatCompleter, so tenant
// resolution, endpoints, API keys and provider quirks stay in one place; Fake
// is the scripted stand-in used by tests.
package model

import "context"

// Role is a chat role.
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// ToolCall is a model-requested tool invocation.
type ToolCall struct {
	ID        string `json:"id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// Message is one chat message.
type Message struct {
	Role       Role       `json:"role"`
	Content    string     `json:"content,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	Name       string     `json:"name,omitempty"`
}

// ToolSpec is the JSON schema offered to the model.
type ToolSpec struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

// Request is one completion.
type Request struct {
	Messages    []Message
	Tools       []ToolSpec
	Temperature float64
	MaxTokens   int
	// ToolChoice names one tool the model must call on this request ("auto"
	// when empty). It backs the required-tool gate: a capability declares the
	// tools a run must use before anything else, and the loop forces the first
	// one until the transcript shows it. The agentic pipeline drives the same
	// mechanism through its retrieval gate (agentic_rag/retrieval_model.go).
	ToolChoice string
}

// Usage is token accounting for one completion.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
	// CachedPromptTokens is the part of the prompt a provider served from its
	// prefix cache. Zero means either no cache or a cache miss; the audit log
	// keeps both cases distinguishable by also recording the prefix hash.
	CachedPromptTokens int `json:"cached_prompt_tokens,omitempty"`
}

// Response is one completion result.
type Response struct {
	Message    Message
	Usage      Usage
	StopReason string
}

// Model completes a request.
type Model interface {
	Complete(ctx context.Context, req Request) (Response, error)
}
