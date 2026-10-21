// Package skill exposes RAGFlow skills with progressive disclosure: the tool
// list carries only name and description, the body is loaded on demand.
package skill

import (
	"context"
	"fmt"
	"strings"

	"ragflow/internal/harness/agent/model"
	"ragflow/internal/harness/toolkit"
)

// Budgets follow the reference runtimes, which all cap what the skill catalog
// and one skill body may cost:
//
//   - Astra: DEFAULT_METADATA_BUDGET=10k tokens, MAX_CHARS_PER_SKILL=20k.
//   - codex: metadata ≤2% of the window, single skill prompt ≤8 KB.
//   - pi: name/description in the system prompt, body read on demand.
const (
	// MetadataBudgetBytes bounds the whole skill catalog in one tool result
	// (about 10k tokens at four bytes per token).
	MetadataBudgetBytes = 40 << 10
	// MaxBodyChars bounds one skill body; the dropped part is announced rather
	// than silently lost.
	MaxBodyChars = 20_000
	// MaxNameLength and MaxDescriptionLength keep one entry from crowding out
	// the rest of the catalog.
	MaxNameLength        = 64
	MaxDescriptionLength = 1024
)

// Skill is one installed skill.
type Skill struct {
	Name        string
	Description string
	// Version and WhenToUse come from the manifest and are shown to the model:
	// they are the cheapest way to stop it loading a skill it does not need.
	Version   string
	WhenToUse string
	// AllowedTools narrows the tool surface while the skill is active. Empty
	// means "no opinion", which is not the same as "deny all".
	AllowedTools []string
	// MaxTokens is the skill's own body budget, when the manifest declares one.
	MaxTokens int
	// DisableModelInvocation hides the skill from the model-driven catalog; it
	// can still be loaded by an explicit user action.
	DisableModelInvocation bool

	Body      string
	Resources []string
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

// Name implements toolkit.Capability.
func (*Capability) Name() string { return "skill" }

// ToolRisk implements toolkit.RiskClassifier: reading a skill changes nothing.
func (*Capability) ToolRisk(string) toolkit.Risk { return toolkit.RiskReadOnly }

// Tools implements toolkit.Capability.
func (*Capability) Tools() []model.ToolSpec {
	return []model.ToolSpec{
		{
			Name:        "skill_list",
			Description: "List installed skills with one-line descriptions.",
			Parameters:  toolkit.ObjectSchema(map[string]any{}),
		},
		{
			Name:        "skill_load",
			Description: "Load a skill's instructions. Call before using its scripts or resources.",
			Parameters:  toolkit.ObjectSchema(map[string]any{"name": map[string]any{"type": "string"}}, "name"),
		},
	}
}

type loadArgs struct {
	Name string `json:"name"`
}

// Invoke implements toolkit.Capability.
func (c *Capability) Invoke(ctx context.Context, call toolkit.Call) (toolkit.Result, error) {
	if c.loader == nil {
		return toolkit.Result{}, fmt.Errorf("skills are not configured for this deployment")
	}
	switch call.Tool {
	case "skill_list":
		skills, err := c.loader.List(ctx)
		if err != nil {
			return toolkit.Result{}, err
		}
		return toolkit.Result{Content: renderCatalog(skills)}, nil
	case "skill_load":
		args, err := toolkit.DecodeArgs[loadArgs](call.Arguments)
		if err != nil {
			return toolkit.Result{}, err
		}
		if strings.TrimSpace(args.Name) == "" {
			return toolkit.Result{}, fmt.Errorf("name is required")
		}
		item, err := c.loader.Load(ctx, args.Name)
		if err != nil {
			return toolkit.Result{}, err
		}
		// Loading a skill that declares allowed_tools narrows the rest of the
		// run: the declaration is only worth writing if it is enforced, and the
		// registry refuses anything outside it from here on.
		if len(item.AllowedTools) > 0 {
			if allowlist := toolkit.AllowlistFrom(ctx); allowlist != nil {
				allowlist.Set(item.AllowedTools)
			}
		}
		return toolkit.Result{Content: render(item)}, nil
	default:
		return toolkit.Result{}, fmt.Errorf("skill: unknown tool %q", call.Tool)
	}
}

// renderCatalog renders the metadata list under one budget. Entries the manifest
// hides from the model are skipped; a catalog that would exceed the budget stops
// early and says how many entries it withheld, because a silently short list
// reads like "there is nothing else".
func renderCatalog(skills []Skill) string {
	visible := make([]Skill, 0, len(skills))
	for _, item := range skills {
		if item.DisableModelInvocation {
			continue
		}
		visible = append(visible, item)
	}
	if len(visible) == 0 {
		return "no skills installed"
	}

	var builder strings.Builder
	written := 0
	for _, item := range visible {
		name := clipRunes(item.Name, MaxNameLength)
		description := clipRunes(item.Description, MaxDescriptionLength)
		line := fmt.Sprintf("- %s: %s\n", name, description)
		if item.WhenToUse != "" {
			line = fmt.Sprintf("- %s: %s (use when: %s)\n", name, description, clipRunes(item.WhenToUse, 200))
		}
		if builder.Len()+len(line) > MetadataBudgetBytes {
			break
		}
		builder.WriteString(line)
		written++
	}
	if written < len(visible) {
		builder.WriteString(fmt.Sprintf("\n[%d more skill(s) omitted: catalog budget reached]\n", len(visible)-written))
	}
	return strings.TrimSpace(builder.String())
}

func render(item Skill) string {
	var builder strings.Builder
	builder.WriteString("# " + item.Name + "\n")
	if item.Version != "" {
		builder.WriteString("version: " + item.Version + "\n")
	}
	if item.Description != "" {
		builder.WriteString(item.Description + "\n")
	}
	if item.WhenToUse != "" {
		builder.WriteString("use when: " + item.WhenToUse + "\n")
	}
	if len(item.AllowedTools) > 0 {
		builder.WriteString("allowed tools: " + strings.Join(item.AllowedTools, ", ") + "\n")
	}
	budget := MaxBodyChars
	if item.MaxTokens > 0 {
		if declared := item.MaxTokens * 4; declared < budget {
			budget = declared
		}
	}
	builder.WriteString("\n" + clipMiddle(item.Body, budget))
	if len(item.Resources) > 0 {
		builder.WriteString("\n\nResources:\n")
		for _, resource := range item.Resources {
			builder.WriteString("- " + resource + "\n")
		}
	}
	return strings.TrimSpace(builder.String())
}

func clipRunes(value string, limit int) string {
	runes := []rune(value)
	if limit <= 0 || len(runes) <= limit {
		return value
	}
	return string(runes[:limit]) + "…"
}

// clipMiddle keeps the opening instructions and the closing caveats of a skill
// body, which is where the constraints usually live.
func clipMiddle(value string, limit int) string {
	if limit <= 0 || len(value) <= limit {
		return value
	}
	marker := fmt.Sprintf("\n[… %d characters omitted …]\n", len(value)-limit)
	keep := limit - len(marker)
	if keep <= 0 {
		return value[:limit]
	}
	head := keep * 6 / 10
	tail := keep - head
	return value[:head] + marker + value[len(value)-tail:]
}
