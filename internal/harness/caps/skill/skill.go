// Package skill exposes RAGFlow skills with progressive disclosure: the tool
// list carries only name and description, the body is loaded on demand.
package skill

import (
	"context"
	"fmt"
	"strings"

	"ragflow/internal/harness/agent/model"
	"ragflow/internal/harness/caps"
)

// Skill is one installed skill.
type Skill struct {
	Name        string
	Description string
	Body        string
	Resources   []string
}

// Loader is implemented over the platform skills infrastructure.
type Loader interface {
	List(ctx context.Context) ([]Skill, error)
	Load(ctx context.Context, name string) (Skill, error)
}

// Capability exposes skill tools.
type Capability struct {
	loader Loader
}

// New builds the capability.
func New(loader Loader) *Capability {
	return &Capability{loader: loader}
}

// Name implements caps.Capability.
func (*Capability) Name() string { return "skill" }

// Tools implements caps.Capability.
func (*Capability) Tools() []model.ToolSpec {
	return []model.ToolSpec{
		{
			Name:        "skill_list",
			Description: "List installed skills with one-line descriptions.",
			Parameters:  caps.ObjectSchema(map[string]any{}),
		},
		{
			Name:        "skill_load",
			Description: "Load a skill's instructions. Call before using its scripts or resources.",
			Parameters:  caps.ObjectSchema(map[string]any{"name": map[string]any{"type": "string"}}, "name"),
		},
	}
}

type loadArgs struct {
	Name string `json:"name"`
}

// Invoke implements caps.Capability.
func (c *Capability) Invoke(ctx context.Context, call caps.Call) (caps.Result, error) {
	if c.loader == nil {
		return caps.Result{}, fmt.Errorf("skills are not configured for this deployment")
	}
	switch call.Tool {
	case "skill_list":
		skills, err := c.loader.List(ctx)
		if err != nil {
			return caps.Result{}, err
		}
		if len(skills) == 0 {
			return caps.Result{Content: "no skills installed"}, nil
		}
		var builder strings.Builder
		for _, item := range skills {
			builder.WriteString(fmt.Sprintf("- %s: %s\n", item.Name, item.Description))
		}
		return caps.Result{Content: strings.TrimSpace(builder.String())}, nil
	case "skill_load":
		args, err := caps.DecodeArgs[loadArgs](call.Arguments)
		if err != nil {
			return caps.Result{}, err
		}
		if strings.TrimSpace(args.Name) == "" {
			return caps.Result{}, fmt.Errorf("name is required")
		}
		item, err := c.loader.Load(ctx, args.Name)
		if err != nil {
			return caps.Result{}, err
		}
		return caps.Result{Content: render(item)}, nil
	default:
		return caps.Result{}, fmt.Errorf("skill: unknown tool %q", call.Tool)
	}
}

func render(item Skill) string {
	var builder strings.Builder
	builder.WriteString("# " + item.Name + "\n")
	if item.Description != "" {
		builder.WriteString(item.Description + "\n")
	}
	builder.WriteString("\n" + item.Body)
	if len(item.Resources) > 0 {
		builder.WriteString("\n\nResources:\n")
		for _, resource := range item.Resources {
			builder.WriteString("- " + resource + "\n")
		}
	}
	return strings.TrimSpace(builder.String())
}
