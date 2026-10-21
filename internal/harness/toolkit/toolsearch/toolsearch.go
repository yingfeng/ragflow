// Package toolsearch exposes tool_search: the model asks for tools by intent and
// receives their calling contract instead of carrying every schema in the system
// prompt.
//
// Selecting a tool also enables it for subsequent turns, so the model never has
// to search twice for the same tool. Discovery is not authorization: a tool still
// runs through its own capability, which enforces the run scope.
package toolsearch

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"ragflow/internal/harness/agent/model"
	"ragflow/internal/harness/toolkit"
)

// ToolName is the search tool's name.
const ToolName = "tool_search"

// Capability exposes tool_search.
type Capability struct {
	registry *toolkit.Registry
}

// New builds the capability over a registry.
func New(registry *toolkit.Registry) *Capability {
	return &Capability{registry: registry}
}

// Name implements toolkit.Capability.
func (*Capability) Name() string { return "toolsearch" }

// Tools implements toolkit.Capability.
func (*Capability) Tools() []model.ToolSpec {
	return []model.ToolSpec{{
		Name: ToolName,
		Description: "Find tools by intent. Returns each matching tool's name, description and parameter schema. " +
			"Selected tools become available for your next turns. Use this instead of guessing a tool name.",
		Parameters: toolkit.ObjectSchema(map[string]any{
			"query": map[string]any{
				"type":        "string",
				"description": "What you want to do, e.g. \"search the corpus by exact phrase\" or \"run a shell command\".",
			},
			"select": map[string]any{
				"type":        "array",
				"items":       map[string]any{"type": "string"},
				"description": "Exact tool names to enable without searching.",
			},
			"limit": map[string]any{
				"type":        "integer",
				"description": "Maximum results to return (default 8).",
			},
		}, "query"),
	}}
}

type searchArgs struct {
	Query  string   `json:"query"`
	Select []string `json:"select"`
	Limit  int      `json:"limit"`
}

// Invoke implements toolkit.Capability.
func (c *Capability) Invoke(ctx context.Context, call toolkit.Call) (toolkit.Result, error) {
	if call.Tool != ToolName {
		return toolkit.Result{}, fmt.Errorf("toolsearch: unknown tool %q", call.Tool)
	}
	args, err := toolkit.DecodeArgs[searchArgs](call.Arguments)
	if err != nil {
		return toolkit.Result{}, err
	}
	if c.registry == nil {
		return toolkit.Result{}, fmt.Errorf("toolsearch: registry is not configured")
	}

	var specs []model.ToolSpec
	if len(args.Select) > 0 {
		specs = c.registry.Enable(args.Select)
	} else {
		if strings.TrimSpace(args.Query) == "" {
			return toolkit.Result{}, fmt.Errorf("query or select is required")
		}
		specs = c.registry.Search(args.Query, args.Limit)
	}
	if len(specs) == 0 {
		return toolkit.Result{Content: "<tool_search_result count=\"0\">No tool matched. Try different words.</tool_search_result>"}, nil
	}

	for _, spec := range specs {
		c.registry.Enable([]string{spec.Name})
	}

	payload, err := json.MarshalIndent(specs, "", "  ")
	if err != nil {
		return toolkit.Result{}, fmt.Errorf("toolsearch: encode: %w", err)
	}
	return toolkit.Result{
		Content: fmt.Sprintf("<tool_search_result count=\"%d\">\n%s\n</tool_search_result>", len(specs), string(payload)),
		Meta:    map[string]any{"enabled": len(specs)},
	}, nil
}
