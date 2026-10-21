// Package subagent lets a run delegate to an independent child run. The child has
// its own session and budget; only its final output crosses the boundary.
package subagent

import (
	"context"
	"fmt"
	"strings"

	"ragflow/internal/harness/agent/model"
	"ragflow/internal/harness/caps"
)

// Request is one delegation.
type Request struct {
	AgentID string
	Prompt  string
	Scope   caps.Scope
}

// Outcome is the child's result.
type Outcome struct {
	RunID  string
	Output string
}

// Spawner is implemented by the run service.
type Spawner interface {
	Spawn(ctx context.Context, req Request) (Outcome, error)
}

// Capability exposes the delegation tool.
type Capability struct {
	spawner Spawner
}

// New builds the capability.
func New(spawner Spawner) *Capability {
	return &Capability{spawner: spawner}
}

// Name implements caps.Capability.
func (*Capability) Name() string { return "subagent" }

// Tools implements caps.Capability.
func (*Capability) Tools() []model.ToolSpec {
	return []model.ToolSpec{{
		Name:        "subagent",
		Description: "Delegate a self-contained task to a child agent and wait for its answer.",
		Parameters: caps.ObjectSchema(map[string]any{
			"prompt":   map[string]any{"type": "string", "description": "Self-contained task statement."},
			"agent_id": map[string]any{"type": "string", "description": "Optional agent definition to run."},
		}, "prompt"),
	}}
}

type spawnArgs struct {
	Prompt  string `json:"prompt"`
	AgentID string `json:"agent_id"`
}

// Invoke implements caps.Capability.
func (c *Capability) Invoke(ctx context.Context, call caps.Call) (caps.Result, error) {
	if call.Tool != "subagent" {
		return caps.Result{}, fmt.Errorf("subagent: unknown tool %q", call.Tool)
	}
	if c.spawner == nil {
		return caps.Result{}, fmt.Errorf("subagents are not configured for this deployment")
	}
	args, err := caps.DecodeArgs[spawnArgs](call.Arguments)
	if err != nil {
		return caps.Result{}, err
	}
	if strings.TrimSpace(args.Prompt) == "" {
		return caps.Result{}, fmt.Errorf("prompt is required")
	}
	outcome, err := c.spawner.Spawn(ctx, Request{
		AgentID: args.AgentID,
		Prompt:  args.Prompt,
		Scope:   caps.ScopeFrom(ctx),
	})
	if err != nil {
		return caps.Result{}, err
	}
	return caps.Result{
		Content: outcome.Output,
		Meta:    map[string]any{"child_run_id": outcome.RunID},
	}, nil
}
