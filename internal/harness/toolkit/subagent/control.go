package subagent

import (
	"context"
	"fmt"
	"strings"
	"time"

	"ragflow/internal/harness/agent/model"
	"ragflow/internal/harness/toolkit"
)

// defaultStatusWait bounds one status call that asks to wait. A tool call that
// blocks for minutes stalls the whole batch and hides the fact that the child is
// making no progress.
const defaultStatusWait = 30 * time.Second

// maxStatusWait is the ceiling a model may request.
const maxStatusWait = 120 * time.Second

// statusPollInterval is how often a waiting status call re-reads the children.
const statusPollInterval = 500 * time.Millisecond

func (c *Capability) controlTools() []model.ToolSpec {
	return []model.ToolSpec{
		{
			Name: StartName,
			Description: fmt.Sprintf("Start a child agent without waiting for it and return its run id. Use %s to collect the answer. A child never outlives its parent run.",
				StatusName),
			Parameters: toolkit.ObjectSchema(map[string]any{
				"prompt": map[string]any{"type": "string", "description": "Self-contained task statement."},
			}, "prompt"),
		},
		{
			Name:        StatusName,
			Description: "Report child runs. With no ids it lists this run's children; with ids it reports those. Set wait_seconds to block until they finish.",
			Parameters: toolkit.ObjectSchema(map[string]any{
				"run_ids": map[string]any{
					"type":        "array",
					"items":       map[string]any{"type": "string"},
					"description": "Child run ids. Empty lists this run's children.",
				},
				"wait_seconds": map[string]any{
					"type":        "integer",
					"description": fmt.Sprintf("Wait for a terminal state, at most %d seconds.", int(maxStatusWait.Seconds())),
				},
			}),
		},
		{
			Name:        StopName,
			Description: "Cancel one child run. It reports whether the child was running on this worker.",
			Parameters: toolkit.ObjectSchema(map[string]any{
				"run_id": map[string]any{"type": "string", "description": "Child run id."},
			}, "run_id"),
		},
		{
			Name:        MessageName,
			Description: "Send one message to a running child. The child sees it at its next round; it does not restart the child.",
			Parameters: toolkit.ObjectSchema(map[string]any{
				"run_id":  map[string]any{"type": "string", "description": "Child run id."},
				"message": map[string]any{"type": "string", "description": "Message to deliver."},
			}, "run_id", "message"),
		},
	}
}

type startArgs struct {
	Prompt string `json:"prompt"`
}

type statusArgs struct {
	RunIDs      []string `json:"run_ids"`
	WaitSeconds int      `json:"wait_seconds"`
}

type stopArgs struct {
	RunID string `json:"run_id"`
}

type messageArgs struct {
	RunID   string `json:"run_id"`
	Message string `json:"message"`
}

func (c *Capability) control(ctx context.Context, call toolkit.Call) (toolkit.Result, error) {
	if c.controller == nil {
		return toolkit.Result{}, fmt.Errorf("subagent: the control plane is not configured for this deployment")
	}
	if err := c.checkDepth(ctx); err != nil {
		return toolkit.Result{}, err
	}

	switch call.Tool {
	case StartName:
		return c.start(ctx, call)
	case StatusName:
		return c.status(ctx, call)
	case StopName:
		return c.stop(ctx, call)
	case MessageName:
		return c.message(ctx, call)
	}
	return toolkit.Result{}, fmt.Errorf("subagent: unknown control tool %q", call.Tool)
}

func (c *Capability) start(ctx context.Context, call toolkit.Call) (toolkit.Result, error) {
	args, err := toolkit.DecodeArgs[startArgs](call.Arguments)
	if err != nil {
		return toolkit.Result{}, err
	}
	if strings.TrimSpace(args.Prompt) == "" {
		return toolkit.Result{}, fmt.Errorf("prompt is required")
	}
	runID, err := c.controller.Start(ctx, Request{
		Prompt: args.Prompt,
		Scope:  toolkit.ScopeFrom(ctx),
		Depth:  DepthFrom(ctx) + 1,
	})
	if err != nil {
		return toolkit.Result{}, err
	}
	return toolkit.Result{Content: fmt.Sprintf("<subagent_started run_id=%q />", runID)}, nil
}

func (c *Capability) status(ctx context.Context, call toolkit.Call) (toolkit.Result, error) {
	args, err := toolkit.DecodeArgs[statusArgs](call.Arguments)
	if err != nil {
		return toolkit.Result{}, err
	}
	wait := time.Duration(args.WaitSeconds) * time.Second
	if wait < 0 {
		wait = 0
	}
	if wait > maxStatusWait {
		wait = maxStatusWait
	}
	deadline := time.Now().Add(wait)

	for {
		children, err := c.controller.List(ctx, args.RunIDs)
		if err != nil {
			return toolkit.Result{}, err
		}
		if time.Now().After(deadline) || allDone(children) {
			return toolkit.Result{Content: renderChildren(children)}, nil
		}
		select {
		case <-ctx.Done():
			return toolkit.Result{Content: renderChildren(children) + "\n[run cancelled while waiting]"}, nil
		case <-time.After(statusPollInterval):
		}
	}
}

func allDone(children []ChildState) bool {
	for _, child := range children {
		if !child.Done() {
			return false
		}
	}
	return len(children) > 0
}

func renderChildren(children []ChildState) string {
	if len(children) == 0 {
		return "<subagent_children count=\"0\" />"
	}
	var builder strings.Builder
	fmt.Fprintf(&builder, "<subagent_children count=\"%d\">", len(children))
	for _, child := range children {
		fmt.Fprintf(&builder, "\n<child run_id=%q status=%q>", child.RunID, child.Status)
		if child.Error != "" {
			builder.WriteString("error: " + clipResult(child.Error))
		} else {
			builder.WriteString(clipResult(child.Output))
		}
		builder.WriteString("</child>")
	}
	builder.WriteString("\n</subagent_children>")
	return builder.String()
}

func (c *Capability) stop(ctx context.Context, call toolkit.Call) (toolkit.Result, error) {
	args, err := toolkit.DecodeArgs[stopArgs](call.Arguments)
	if err != nil {
		return toolkit.Result{}, err
	}
	if strings.TrimSpace(args.RunID) == "" {
		return toolkit.Result{}, fmt.Errorf("run_id is required")
	}
	if c.controller.Stop(ctx, args.RunID) {
		return toolkit.Result{Content: fmt.Sprintf("<subagent_stopped run_id=%q />", args.RunID)}, nil
	}
	return toolkit.Result{Content: fmt.Sprintf("<subagent_stop run_id=%q status=\"not_running\" />", args.RunID)}, nil
}

func (c *Capability) message(ctx context.Context, call toolkit.Call) (toolkit.Result, error) {
	args, err := toolkit.DecodeArgs[messageArgs](call.Arguments)
	if err != nil {
		return toolkit.Result{}, err
	}
	if strings.TrimSpace(args.RunID) == "" || strings.TrimSpace(args.Message) == "" {
		return toolkit.Result{}, fmt.Errorf("run_id and message are required")
	}
	if c.controller.Message(ctx, args.RunID, args.Message) {
		return toolkit.Result{Content: fmt.Sprintf("<subagent_message run_id=%q status=\"delivered\" />", args.RunID)}, nil
	}
	return toolkit.Result{Content: fmt.Sprintf("<subagent_message run_id=%q status=\"not_running\" />", args.RunID)}, nil
}
