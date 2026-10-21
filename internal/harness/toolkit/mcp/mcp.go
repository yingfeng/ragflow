// Package mcp introduces external MCP tools. Tools are declared by the server,
// not by the agent: introducing the capability is the only configuration step.
package mcp

import (
	"context"
	"fmt"
	"strings"
	"time"

	"ragflow/internal/harness/agent/model"
	"ragflow/internal/harness/toolkit"
)

// MaxResultBytes bounds one MCP tool result. An MCP server can return a whole
// document; the transcript cannot afford it.
const MaxResultBytes = 100 << 10

// callWithRetry retries one transient MCP failure. MCP servers are external
// processes: a restart between two calls is normal, not a tool error.
func callWithRetry(ctx context.Context, client Client, tool, arguments string) (string, error) {
	var lastErr error
	for attempt := 1; attempt <= 2; attempt++ {
		content, err := client.Call(ctx, tool, arguments)
		if err == nil {
			return content, nil
		}
		lastErr = err
		message := strings.ToLower(err.Error())
		if !strings.Contains(message, "timeout") && !strings.Contains(message, "connection") && !strings.Contains(message, "temporarily") {
			return "", err
		}
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		case <-time.After(time.Duration(attempt) * 300 * time.Millisecond):
		}
	}
	return "", lastErr
}

// clipResult keeps the head and the tail of a large result.
func clipResult(content string) string {
	if len(content) <= MaxResultBytes {
		return content
	}
	marker := fmt.Sprintf("\n[… %d characters omitted; call the tool with a narrower request for the rest …]\n", len(content)-MaxResultBytes)
	keep := MaxResultBytes - len(marker)
	if keep <= 0 {
		return content[:MaxResultBytes]
	}
	head := keep * 4 / 10
	return content[:head] + marker + content[len(content)-(keep-head):]
}

// Client is an MCP session.
type Client interface {
	// Tools returns the server's tool list. It must be safe to call once at
	// registration time.
	Tools() []model.ToolSpec
	// Call invokes a tool by name and returns its textual result.
	Call(ctx context.Context, tool, arguments string) (string, error)
}

// Capability exposes one MCP server's tools.
type Capability struct {
	name   string
	client Client
}

// New builds the capability. The name identifies the server in traces.
func New(name string, client Client) *Capability {
	return &Capability{name: name, client: client}
}

// Name implements toolkit.Capability.
func (c *Capability) Name() string { return "mcp:" + c.name }

// Tools implements toolkit.Capability.
func (c *Capability) Tools() []model.ToolSpec {
	if c.client == nil {
		return nil
	}
	return c.client.Tools()
}

// Invoke implements toolkit.Capability.
func (c *Capability) Invoke(ctx context.Context, call toolkit.Call) (toolkit.Result, error) {
	if c.client == nil {
		return toolkit.Result{}, fmt.Errorf("mcp server %q is not connected", c.name)
	}
	content, err := callWithRetry(ctx, c.client, call.Tool, call.Arguments)
	if err != nil {
		return toolkit.Result{}, err
	}
	return toolkit.Result{Content: clipResult(content)}, nil
}
