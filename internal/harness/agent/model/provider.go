package model

import (
	"context"
	"encoding/json"
	"fmt"

	"ragflow/internal/common"
	providermodel "ragflow/internal/entity/models"
)

// ChatCompleter is the RAGFlow model-provider port this package depends on.
//
// The harness never speaks a provider protocol itself: tenant resolution, API
// keys, endpoints, retries and provider quirks all belong to
// internal/entity/models, and *providermodel.ChatModel (built by
// service.ModelFactory) satisfies this interface directly.
type ChatCompleter interface {
	ChatWithMessages(ctx context.Context, messages []providermodel.Message, config *providermodel.ChatConfig, usage *common.ModelUsage) (*providermodel.ChatResponse, error)
}

// The provider's chat model must stay the only implementation: if this
// assertion breaks, the harness has started to depend on something other than
// the RAGFlow model provider and calls would have to be wired twice.
var _ ChatCompleter = (*providermodel.ChatModel)(nil)

// ProviderConfig pins the sampling defaults for one deployment.
type ProviderConfig struct {
	Temperature *float64
	MaxTokens   *int
	Thinking    *bool
}

// Provider adapts a RAGFlow chat model to the harness wire.
type Provider struct {
	chat   ChatCompleter
	config ProviderConfig
}

// NewProvider wraps a RAGFlow chat model.
func NewProvider(chat ChatCompleter, config ProviderConfig) *Provider {
	return &Provider{chat: chat, config: config}
}

// Complete implements Model.
func (p *Provider) Complete(ctx context.Context, req Request) (Response, error) {
	if p == nil || p.chat == nil {
		return Response{}, fmt.Errorf("model provider: chat model is required")
	}

	messages := make([]providermodel.Message, 0, len(req.Messages))
	for _, message := range req.Messages {
		messages = append(messages, toProviderMessage(message))
	}

	config := &providermodel.ChatConfig{
		Temperature: firstFloat(p.config.Temperature, req.Temperature),
		MaxTokens:   firstInt(p.config.MaxTokens, req.MaxTokens),
		Thinking:    p.config.Thinking,
	}
	if tools := toProviderTools(req.Tools); len(tools) > 0 {
		config.Tools = tools
		auto := "auto"
		config.ToolChoice = &auto
	}

	usage := &common.ModelUsage{}
	response, err := p.chat.ChatWithMessages(ctx, messages, config, usage)
	if err != nil {
		return Response{}, err
	}
	if response == nil {
		return Response{}, fmt.Errorf("model provider: empty response")
	}
	return toHarnessResponse(response), nil
}

func toProviderMessage(message Message) providermodel.Message {
	out := providermodel.Message{
		Role:       string(message.Role),
		Content:    message.Content,
		ToolCallID: message.ToolCallID,
	}
	if message.Name != "" {
		out.Name = message.Name
	}
	if len(message.ToolCalls) > 0 {
		calls := make([]map[string]interface{}, 0, len(message.ToolCalls))
		for _, call := range message.ToolCalls {
			calls = append(calls, map[string]interface{}{
				"id":   call.ID,
				"type": "function",
				"function": map[string]interface{}{
					"name":      call.Name,
					"arguments": call.Arguments,
				},
			})
		}
		out.ToolCalls = calls
	}
	return out
}

// toProviderTools renders the harness tool specs as the OpenAI-style entries the
// drivers forward verbatim.
func toProviderTools(specs []ToolSpec) []map[string]any {
	if len(specs) == 0 {
		return nil
	}
	out := make([]map[string]any, 0, len(specs))
	for _, spec := range specs {
		function := map[string]any{
			"name":        spec.Name,
			"description": spec.Description,
		}
		if spec.Parameters != nil {
			function["parameters"] = spec.Parameters
		}
		out = append(out, map[string]any{"type": "function", "function": function})
	}
	return out
}

func toHarnessResponse(response *providermodel.ChatResponse) Response {
	out := Response{Message: Message{Role: RoleAssistant}}
	if response.Answer != nil {
		out.Message.Content = *response.Answer
	}
	for _, call := range response.ToolCalls {
		if parsed, ok := toHarnessToolCall(call); ok {
			out.Message.ToolCalls = append(out.Message.ToolCalls, parsed)
		}
	}
	if response.Usage != nil {
		out.Usage = Usage{
			PromptTokens:     response.Usage.PromptTokens,
			CompletionTokens: response.Usage.CompletionTokens,
			TotalTokens:      response.Usage.TotalTokens,
		}
	}
	out.StopReason = "stop"
	if len(out.Message.ToolCalls) > 0 {
		out.StopReason = "tool_calls"
	}
	return out
}

// toHarnessToolCall reads one provider tool call. The arguments are kept as the
// raw JSON string the model produced; a provider that hands back a decoded
// object instead is re-encoded so callers always see the same shape.
func toHarnessToolCall(raw map[string]interface{}) (ToolCall, bool) {
	call := ToolCall{}
	if id, ok := raw["id"].(string); ok {
		call.ID = id
	}
	function, ok := raw["function"].(map[string]interface{})
	if !ok {
		return ToolCall{}, false
	}
	if name, ok := function["name"].(string); ok {
		call.Name = name
	}
	switch arguments := function["arguments"].(type) {
	case string:
		call.Arguments = arguments
	case nil:
		call.Arguments = "{}"
	default:
		encoded, err := json.Marshal(arguments)
		if err != nil {
			call.Arguments = "{}"
		} else {
			call.Arguments = string(encoded)
		}
	}
	return call, call.Name != ""
}

func firstFloat(preferred *float64, fallback float64) *float64 {
	if preferred != nil {
		return preferred
	}
	if fallback == 0 {
		return nil
	}
	return &fallback
}

func firstInt(preferred *int, fallback int) *int {
	if preferred != nil {
		return preferred
	}
	if fallback == 0 {
		return nil
	}
	return &fallback
}
