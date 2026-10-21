// Package model is the thin model wire used by the harness loop. It is
// deliberately independent of any agent framework: one request in, one response
// out, with tool calls carried verbatim.
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
}

// Usage is token accounting for one completion.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
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
