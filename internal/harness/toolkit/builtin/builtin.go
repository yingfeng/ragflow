// Package builtin exposes the generic tools every agent gets regardless of its
// data reach: think, todo_write and run_javascript.
//
// They are the agentic pipeline's own tool implementations, instantiated here
// and invoked directly, so their prompts and output shapes stay identical.
package builtin

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	einotool "github.com/cloudwego/eino/components/tool"

	"ragflow/internal/agentic_rag"
	"ragflow/internal/harness/agent/model"
	"ragflow/internal/harness/toolkit"
)

// toolFactory builds one generic tool. The generic tools ignore data reach.
type toolFactory func() einotool.InvokableTool

var genericTools = map[string]toolFactory{
	"think": func() einotool.InvokableTool {
		return agentic_rag.NewThinkTool()
	},
	"todo_write": func() einotool.InvokableTool {
		return agentic_rag.NewTodoWriteTool()
	},
	"run_javascript": func() einotool.InvokableTool {
		return agentic_rag.NewRunJavascriptTool()
	},
}

// Capability exposes the generic tools.
type Capability struct {
	factories map[string]toolFactory
	specs     []model.ToolSpec
}

// New describes the generic tools.
func New(ctx context.Context) (*Capability, error) {
	names := make([]string, 0, len(genericTools))
	for name := range genericTools {
		names = append(names, name)
	}
	sort.Strings(names)

	specs := make([]model.ToolSpec, 0, len(names))
	for _, name := range names {
		info, err := genericTools[name]().Info(ctx)
		if err != nil {
			return nil, fmt.Errorf("caps/builtin: describe %s: %w", name, err)
		}
		spec := model.ToolSpec{Name: name, Description: info.Desc}
		if info.ParamsOneOf != nil {
			// ParamsOneOf keeps its fields unexported, so marshalling it directly
			// yields "{}" and the model would receive a tool with no arguments
			// declared at all. ToJSONSchema is the only correct conversion.
			declared, err := info.ParamsOneOf.ToJSONSchema()
			if err != nil {
				return nil, fmt.Errorf("caps/builtin: schema %s: %w", name, err)
			}
			if declared != nil {
				raw, err := json.Marshal(declared)
				if err != nil {
					return nil, fmt.Errorf("caps/builtin: schema %s: %w", name, err)
				}
				params := map[string]any{}
				if err := json.Unmarshal(raw, &params); err != nil {
					return nil, fmt.Errorf("caps/builtin: decode schema %s: %w", name, err)
				}
				spec.Parameters = params
			}
		}
		specs = append(specs, spec)
	}
	return &Capability{factories: genericTools, specs: specs}, nil
}

// Name implements toolkit.Capability.
func (*Capability) Name() string { return "builtin" }

// Tools implements toolkit.Capability.
func (c *Capability) Tools() []model.ToolSpec {
	out := make([]model.ToolSpec, len(c.specs))
	copy(out, c.specs)
	return out
}

// Invoke implements toolkit.Capability.
func (c *Capability) Invoke(ctx context.Context, call toolkit.Call) (toolkit.Result, error) {
	factory, ok := c.factories[call.Tool]
	if !ok {
		return toolkit.Result{}, fmt.Errorf("builtin: unknown tool %q", call.Tool)
	}
	arguments := call.Arguments
	if strings.TrimSpace(arguments) == "" {
		arguments = "{}"
	}
	content, err := factory().InvokableRun(ctx, arguments)
	if err != nil {
		return toolkit.Result{}, fmt.Errorf("%s: %w", call.Tool, err)
	}
	result := toolkit.Result{Content: content}
	// A plan belongs in the run state, not only in this tool's echo: the loop
	// renders it into every later prompt so compaction cannot take it away.
	if call.Tool == todoWriteTool {
		if plan, ok := planFromArguments(arguments); ok {
			result.Meta = map[string]any{toolkit.MetaPlan: plan}
		}
	}
	return result, nil
}

// todoWriteTool is the planning tool the generic tool set exposes. Its plan is
// published through Result.Meta so the loop can carry it across rounds.
const todoWriteTool = "todo_write"

// planFromArguments decodes the step list a todo_write call declared.
func planFromArguments(arguments string) ([]toolkit.PlanStep, bool) {
	var decoded struct {
		Steps []toolkit.PlanStep `json:"steps"`
	}
	if err := json.Unmarshal([]byte(arguments), &decoded); err != nil || len(decoded.Steps) == 0 {
		return nil, false
	}
	return decoded.Steps, true
}
