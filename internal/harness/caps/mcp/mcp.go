// Package mcp introduces external MCP tools. Tools are declared by the server,
// not by the agent: introducing the capability is the only configuration step.
package mcp

import (
	"context"
	"fmt"

	"ragflow/internal/harness/agent/model"
	"ragflow/internal/harness/caps"
)

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

// Name implements caps.Capability.
func (c *Capability) Name() string { return "mcp:" + c.name }

// Tools implements caps.Capability.
func (c *Capability) Tools() []model.ToolSpec {
	if c.client == nil {
		return nil
	}
	return c.client.Tools()
}

// Invoke implements caps.Capability.
func (c *Capability) Invoke(ctx context.Context, call caps.Call) (caps.Result, error) {
	if c.client == nil {
		return caps.Result{}, fmt.Errorf("mcp server %q is not connected", c.name)
	}
	content, err := c.client.Call(ctx, call.Tool, call.Arguments)
	if err != nil {
		return caps.Result{}, err
	}
	return caps.Result{Content: content}, nil
}
