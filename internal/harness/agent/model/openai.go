package model

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// OpenAIConfig configures the OpenAI-compatible client.
type OpenAIConfig struct {
	BaseURL     string
	APIKey      string
	Model       string
	HTTPClient  *http.Client
	Temperature float64
	MaxTokens   int
	Timeout     time.Duration
}

// OpenAIClient talks to any OpenAI-compatible /chat/completions endpoint.
type OpenAIClient struct {
	cfg  OpenAIConfig
	http *http.Client
}

// NewOpenAI builds a client. BaseURL without a scheme is rejected by the first
// call rather than silently defaulting to a vendor host.
func NewOpenAI(cfg OpenAIConfig) *OpenAIClient {
	client := cfg.HTTPClient
	if client == nil {
		timeout := cfg.Timeout
		if timeout <= 0 {
			timeout = 120 * time.Second
		}
		client = &http.Client{Timeout: timeout}
	}
	return &OpenAIClient{cfg: cfg, http: client}
}

type openAIRequest struct {
	Model       string          `json:"model"`
	Messages    []openAIMessage `json:"messages"`
	Tools       []openAITool    `json:"tools,omitempty"`
	Temperature float64         `json:"temperature,omitempty"`
	MaxTokens   int             `json:"max_tokens,omitempty"`
}

type openAIMessage struct {
	Role       string         `json:"role"`
	Content    string         `json:"content,omitempty"`
	ToolCalls  []openAIToolCt `json:"tool_calls,omitempty"`
	ToolCallID string         `json:"tool_call_id,omitempty"`
	Name       string         `json:"name,omitempty"`
}

type openAIToolCt struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function openAIFunctionCt `json:"function"`
}

type openAIFunctionCt struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type openAITool struct {
	Type     string             `json:"type"`
	Function openAIToolFunction `json:"function"`
}

type openAIToolFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

type openAIResponse struct {
	Choices []struct {
		Message      openAIMessage `json:"message"`
		FinishReason string        `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
	Error *struct {
		Message string `json:"message"`
		Type    string `json:"type"`
	} `json:"error"`
}

// Complete implements Model.
func (c *OpenAIClient) Complete(ctx context.Context, req Request) (Response, error) {
	if strings.TrimSpace(c.cfg.BaseURL) == "" {
		return Response{}, fmt.Errorf("model: base url is required")
	}
	if strings.TrimSpace(c.cfg.Model) == "" {
		return Response{}, fmt.Errorf("model: model name is required")
	}

	payload := openAIRequest{
		Model:       c.cfg.Model,
		Temperature: c.cfg.Temperature,
		MaxTokens:   c.cfg.MaxTokens,
	}
	if req.Temperature > 0 {
		payload.Temperature = req.Temperature
	}
	if req.MaxTokens > 0 {
		payload.MaxTokens = req.MaxTokens
	}
	for _, msg := range req.Messages {
		item := openAIMessage{
			Role:       string(msg.Role),
			Content:    msg.Content,
			ToolCallID: msg.ToolCallID,
			Name:       msg.Name,
		}
		for _, call := range msg.ToolCalls {
			item.ToolCalls = append(item.ToolCalls, openAIToolCt{
				ID:       call.ID,
				Type:     "function",
				Function: openAIFunctionCt{Name: call.Name, Arguments: call.Arguments},
			})
		}
		payload.Messages = append(payload.Messages, item)
	}
	for _, spec := range req.Tools {
		payload.Tools = append(payload.Tools, openAITool{
			Type: "function",
			Function: openAIToolFunction{
				Name:        spec.Name,
				Description: spec.Description,
				Parameters:  spec.Parameters,
			},
		})
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return Response{}, fmt.Errorf("model: encode request: %w", err)
	}

	url := strings.TrimRight(c.cfg.BaseURL, "/") + "/chat/completions"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return Response{}, fmt.Errorf("model: build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if c.cfg.APIKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.cfg.APIKey)
	}

	resp, err := c.http.Do(httpReq)
	if err != nil {
		return Response{}, fmt.Errorf("model: request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	raw, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if err != nil {
		return Response{}, fmt.Errorf("model: read response: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return Response{}, fmt.Errorf("model: http %d: %s", resp.StatusCode, truncate(string(raw), 512))
	}

	var parsed openAIResponse
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return Response{}, fmt.Errorf("model: decode response: %w", err)
	}
	if parsed.Error != nil {
		return Response{}, fmt.Errorf("model: %s: %s", parsed.Error.Type, parsed.Error.Message)
	}
	if len(parsed.Choices) == 0 {
		return Response{}, fmt.Errorf("model: response has no choices")
	}

	choice := parsed.Choices[0]
	out := Response{
		Message: Message{
			Role:       RoleAssistant,
			Content:    choice.Message.Content,
			ToolCallID: choice.Message.ToolCallID,
		},
		Usage: Usage{
			PromptTokens:     parsed.Usage.PromptTokens,
			CompletionTokens: parsed.Usage.CompletionTokens,
			TotalTokens:      parsed.Usage.TotalTokens,
		},
		StopReason: choice.FinishReason,
	}
	for _, call := range choice.Message.ToolCalls {
		out.Message.ToolCalls = append(out.Message.ToolCalls, ToolCall{
			ID:        call.ID,
			Name:      call.Function.Name,
			Arguments: call.Function.Arguments,
		})
	}
	return out, nil
}

func truncate(value string, limit int) string {
	if len(value) <= limit {
		return value
	}
	return value[:limit] + "..."
}
