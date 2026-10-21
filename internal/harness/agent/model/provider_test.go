package model

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"

	"ragflow/internal/common"
	providermodel "ragflow/internal/entity/models"
)

type fakeCompleter struct {
	mu       sync.Mutex
	messages []providermodel.Message
	config   *providermodel.ChatConfig
	usage    *common.ModelUsage
	response *providermodel.ChatResponse
	err      error
}

func (f *fakeCompleter) ChatWithMessages(_ context.Context, messages []providermodel.Message, config *providermodel.ChatConfig, usage *common.ModelUsage) (*providermodel.ChatResponse, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.messages = messages
	f.config = config
	f.usage = usage
	return f.response, f.err
}

func TestProviderMapsRequestOntoRAGFlowChatModel(t *testing.T) {
	answer := "final answer"
	completer := &fakeCompleter{response: &providermodel.ChatResponse{Answer: &answer}}

	provider := NewProvider(completer, ProviderConfig{})
	_, err := provider.Complete(context.Background(), Request{
		Messages: []Message{
			{Role: RoleSystem, Content: "be brief"},
			{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "call-1", Name: "search", Arguments: `{"q":"x"}`}}},
			{Role: RoleTool, Content: "chunk", ToolCallID: "call-1", Name: "search"},
		},
		Tools:       []ToolSpec{{Name: "search", Description: "Search.", Parameters: map[string]any{"type": "object"}}},
		Temperature: 0.3,
		MaxTokens:   512,
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}

	if len(completer.messages) != 3 {
		t.Fatalf("messages = %d, want 3", len(completer.messages))
	}
	if completer.messages[0].Role != "system" || completer.messages[0].Content != "be brief" {
		t.Errorf("system message = %#v", completer.messages[0])
	}
	if len(completer.messages[1].ToolCalls) != 1 {
		t.Fatalf("assistant tool calls = %#v", completer.messages[1].ToolCalls)
	}
	call := completer.messages[1].ToolCalls[0]
	if call["id"] != "call-1" || call["type"] != "function" {
		t.Errorf("tool call envelope = %#v", call)
	}
	function, _ := call["function"].(map[string]interface{})
	if function["name"] != "search" || function["arguments"] != `{"q":"x"}` {
		t.Errorf("tool call function = %#v", function)
	}
	if completer.messages[2].ToolCallID != "call-1" || completer.messages[2].Name != "search" {
		t.Errorf("tool result = %#v", completer.messages[2])
	}

	tools, ok := completer.config.Tools.([]map[string]any)
	if !ok || len(tools) != 1 {
		t.Fatalf("config tools = %#v, want one OpenAI-style tool", completer.config.Tools)
	}
	toolFunction, _ := tools[0]["function"].(map[string]any)
	if toolFunction["name"] != "search" || toolFunction["description"] != "Search." {
		t.Errorf("tool function = %#v", toolFunction)
	}
	if completer.config.ToolChoice == nil || *completer.config.ToolChoice != "auto" {
		t.Errorf("tool choice = %#v, want auto", completer.config.ToolChoice)
	}
	if completer.config.Temperature == nil || *completer.config.Temperature != 0.3 {
		t.Errorf("temperature = %#v, want 0.3", completer.config.Temperature)
	}
	if completer.config.MaxTokens == nil || *completer.config.MaxTokens != 512 {
		t.Errorf("max tokens = %#v, want 512", completer.config.MaxTokens)
	}
	if completer.usage == nil {
		t.Error("usage was not passed to the model provider")
	}
}

func TestProviderLeavesToolChoiceUnsetWithoutTools(t *testing.T) {
	completer := &fakeCompleter{response: &providermodel.ChatResponse{}}
	provider := NewProvider(completer, ProviderConfig{})

	if _, err := provider.Complete(context.Background(), Request{Messages: []Message{{Role: RoleUser, Content: "hi"}}}); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if completer.config.Tools != nil {
		t.Errorf("tools = %#v, want nil", completer.config.Tools)
	}
	if completer.config.ToolChoice != nil {
		t.Errorf("tool choice = %#v, want nil", completer.config.ToolChoice)
	}
}

func TestProviderPrefersConfiguredDefaults(t *testing.T) {
	temperature, maxTokens := 0.1, 128
	completer := &fakeCompleter{response: &providermodel.ChatResponse{}}
	provider := NewProvider(completer, ProviderConfig{Temperature: &temperature, MaxTokens: &maxTokens})

	if _, err := provider.Complete(context.Background(), Request{Messages: []Message{{Role: RoleUser, Content: "hi"}}, Temperature: 0.9, MaxTokens: 4096}); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if *completer.config.Temperature != 0.1 || *completer.config.MaxTokens != 128 {
		t.Errorf("config = %#v, want deployment defaults", completer.config)
	}
}

func TestProviderMapsToolCallsAndUsage(t *testing.T) {
	answer := ""
	completer := &fakeCompleter{response: &providermodel.ChatResponse{
		Answer: &answer,
		ToolCalls: []map[string]interface{}{
			{
				"id":   "call-9",
				"type": "function",
				"function": map[string]interface{}{
					"name":      "lookup",
					"arguments": "{\"q\":\"ragflow\"}",
				},
			},
			{
				"id": "call-10",
				"function": map[string]interface{}{
					"name":      "count",
					"arguments": map[string]interface{}{"limit": float64(2)},
				},
			},
			{"id": "call-11"},
		},
		Usage: &providermodel.TokenUsage{PromptTokens: 11, CompletionTokens: 7, TotalTokens: 18},
	}}
	provider := NewProvider(completer, ProviderConfig{})

	response, err := provider.Complete(context.Background(), Request{Messages: []Message{{Role: RoleUser, Content: "go"}}})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if len(response.Message.ToolCalls) != 2 {
		t.Fatalf("tool calls = %#v, want 2 (the malformed entry is dropped)", response.Message.ToolCalls)
	}
	if response.Message.ToolCalls[0].Name != "lookup" || response.Message.ToolCalls[0].Arguments != `{"q":"ragflow"}` {
		t.Errorf("first tool call = %#v", response.Message.ToolCalls[0])
	}
	if response.Message.ToolCalls[1].Arguments != `{"limit":2}` {
		t.Errorf("structured arguments = %q, want JSON", response.Message.ToolCalls[1].Arguments)
	}
	if response.StopReason != "tool_calls" {
		t.Errorf("stop reason = %q, want tool_calls", response.StopReason)
	}
	if response.Usage.PromptTokens != 11 || response.Usage.TotalTokens != 18 {
		t.Errorf("usage = %#v", response.Usage)
	}
}

func TestProviderDefaultsNilArgumentsToEmptyObject(t *testing.T) {
	completer := &fakeCompleter{response: &providermodel.ChatResponse{
		ToolCalls: []map[string]interface{}{{
			"id":       "call-1",
			"function": map[string]interface{}{"name": "no_args"},
		}},
	}}
	provider := NewProvider(completer, ProviderConfig{})

	response, err := provider.Complete(context.Background(), Request{Messages: []Message{{Role: RoleUser, Content: "go"}}})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if got := response.Message.ToolCalls[0].Arguments; got != "{}" {
		t.Errorf("arguments = %q, want {}", got)
	}
}

func TestProviderPropagatesErrors(t *testing.T) {
	completer := &fakeCompleter{err: errors.New("provider refused")}
	provider := NewProvider(completer, ProviderConfig{})

	_, err := provider.Complete(context.Background(), Request{Messages: []Message{{Role: RoleUser, Content: "go"}}})
	if err == nil || !strings.Contains(err.Error(), "provider refused") {
		t.Fatalf("error = %v, want the provider error", err)
	}
}

func TestProviderRequiresChatModel(t *testing.T) {
	_, err := NewProvider(nil, ProviderConfig{}).Complete(context.Background(), Request{})
	if err == nil || !strings.Contains(err.Error(), "chat model is required") {
		t.Fatalf("error = %v, want chat model required", err)
	}
}
