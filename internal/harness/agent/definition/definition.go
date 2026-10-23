// Package definition holds the prompt-first agent definition and its immutable
// versions. A definition never carries a dataset scope: data reach is injected
// per run (see toolkit.Scope).
package definition

import (
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"

	"ragflow/internal/harness/toolkit"
)

// ApprovalMode controls whether high-risk tools pause the loop for approval.
type ApprovalMode string

const (
	ApprovalAuto ApprovalMode = "auto"
	ApprovalAsk  ApprovalMode = "ask"
)

// Writable surfaces for RSI. Anything outside this list is rejected.
const (
	SurfacePromptAddendum = "prompt.addendum"
	SurfaceToolPolicy     = "tool.policy"
	SurfaceSkillSelection = "skill.selection"
	SurfaceRAGParams      = "rag.params"
)

// WritableSurfaces is the closed set RSI may edit.
var WritableSurfaces = []string{
	SurfacePromptAddendum,
	SurfaceToolPolicy,
	SurfaceSkillSelection,
	SurfaceRAGParams,
}

// CapRef introduces a capability. Introducing it makes its tools available; it
// is not an orchestration instruction.
type CapRef struct {
	Kind   string         `json:"kind"`
	Name   string         `json:"name"`
	Config map[string]any `json:"config,omitempty"`
}

// PolicyConfig bounds one run.
type PolicyConfig struct {
	MaxRounds    int          `json:"max_rounds,omitempty"`
	MaxTokens    int          `json:"max_tokens,omitempty"`
	MaxToolCalls int          `json:"max_tool_calls,omitempty"`
	MaxWallSec   int          `json:"max_wall_sec,omitempty"`
	MaxCostUSD   float64      `json:"max_cost_usd,omitempty"`
	Approval     ApprovalMode `json:"approval,omitempty"`
	// Permissions are operator-authored tool-call rules. They are part of the
	// definition so what an agent may run is reviewable with its prompt.
	Permissions []toolkit.PermissionRule `json:"permissions,omitempty"`
}

// RSIConfig opts an agent into self-improvement and bounds it.
type RSIConfig struct {
	Enabled       bool     `json:"enabled"`
	Surfaces      []string `json:"surfaces,omitempty"`
	UseFeedback   bool     `json:"use_feedback,omitempty"`
	MinConfidence float64  `json:"min_confidence,omitempty"`
	MaxLessons    int      `json:"max_lessons,omitempty"`
}

// Definition is what a user writes.
// SamplingConfig pins how the run's model samples. Without it the provider
// default applies, and the default changes with the model: a comparison
// between two models is then also a comparison between two temperatures, and
// no run is reproducible across a model swap. agentic_rag pins 0.5 for exactly
// this reason (conf/agentic_rag.yaml).
type SamplingConfig struct {
	Temperature *float64 `json:"temperature,omitempty"`
	TopP        *float64 `json:"top_p,omitempty"`
	// Thinking enables the model's reasoning output. It is sampling, not
	// identity: the same model is chatty or silent depending on this switch,
	// and a run that cannot see its own reasoning panel cannot be followed.
	Thinking *bool `json:"thinking,omitempty"`
}

type Definition struct {
	ID     string       `json:"id"`
	Name   string       `json:"name"`
	Prompt string       `json:"prompt"`
	Caps   []CapRef     `json:"caps,omitempty"`
	Policy PolicyConfig `json:"policy,omitempty"`
	RSI    RSIConfig    `json:"rsi,omitempty"`
	// Sampling pins the model's sampling behaviour for this agent. Nil keeps
	// the provider default, which is a moving target — see SamplingConfig.
	Sampling *SamplingConfig `json:"sampling,omitempty"`

	// Addendum is written only by RSI (prompt.addendum surface).
	Addendum string `json:"addendum,omitempty"`

	// Skills selects skills from the caller's skillspace.
	//
	// The built-in skills ship with the binary and always apply; this list
	// selects from the user's own space, which is a catalog shared by every
	// agent of the tenant. Naming none keeps the agent on the built-in set —
	// loading a whole skillspace into every agent's prompt would spend context
	// on procedures the agent was not asked to follow.
	Skills []string `json:"skills,omitempty"`
}

var (
	// ErrEmptyPrompt rejects a definition without instructions.
	ErrEmptyPrompt = errors.New("definition: prompt is required")
	// ErrEmptyName rejects a definition without a name.
	ErrEmptyName = errors.New("definition: name is required")
	// ErrUnknownSurface rejects an RSI surface outside the writable set.
	ErrUnknownSurface = errors.New("definition: unknown rsi surface")
)

// Validate checks the definition before a version is created.
func (d *Definition) Validate() error {
	if strings.TrimSpace(d.Name) == "" {
		return ErrEmptyName
	}
	if strings.TrimSpace(d.Prompt) == "" {
		return ErrEmptyPrompt
	}
	switch d.Policy.Approval {
	case "", ApprovalAuto, ApprovalAsk:
	default:
		return fmt.Errorf("definition: unknown approval mode %q", d.Policy.Approval)
	}
	for _, surface := range d.RSI.Surfaces {
		if !IsWritableSurface(surface) {
			return fmt.Errorf("%w: %s", ErrUnknownSurface, surface)
		}
	}
	if d.Policy.MaxRounds < 0 || d.Policy.MaxTokens < 0 || d.Policy.MaxToolCalls < 0 ||
		d.Policy.MaxWallSec < 0 || d.Policy.MaxCostUSD < 0 {
		return errors.New("definition: policy limits must not be negative")
	}
	if d.Sampling != nil {
		if d.Sampling.Temperature != nil && (*d.Sampling.Temperature < 0 || *d.Sampling.Temperature > 2) {
			return fmt.Errorf("definition: sampling temperature %v outside [0, 2]", *d.Sampling.Temperature)
		}
		if d.Sampling.TopP != nil && (*d.Sampling.TopP <= 0 || *d.Sampling.TopP > 1) {
			return fmt.Errorf("definition: sampling top_p %v outside (0, 1]", *d.Sampling.TopP)
		}
	}
	if _, err := toolkit.NewRuleSet(d.Policy.Permissions); err != nil {
		return fmt.Errorf("definition: %w", err)
	}
	if d.RSI.MinConfidence < 0 || d.RSI.MinConfidence > 1 {
		return errors.New("definition: rsi min_confidence must be within [0,1]")
	}
	return nil
}

// IsWritableSurface reports whether RSI may edit the surface.
func IsWritableSurface(surface string) bool {
	for _, allowed := range WritableSurfaces {
		if allowed == surface {
			return true
		}
	}
	return false
}

// EffectivePrompt joins the user prompt and the RSI addendum.
func (d *Definition) EffectivePrompt() string {
	if strings.TrimSpace(d.Addendum) == "" {
		return d.Prompt
	}
	return d.Prompt + "\n\n" + d.Addendum
}

// Version is an immutable snapshot. Runs bind to one version.
type Version struct {
	ID         string     `json:"id"`
	AgentID    string     `json:"agent_id"`
	ParentID   string     `json:"parent_id,omitempty"`
	Definition Definition `json:"definition"`
	CreatedAt  time.Time  `json:"created_at"`
}

// NewVersion validates and freezes a definition.
func NewVersion(agentID, parentID string, def Definition) (*Version, error) {
	if err := def.Validate(); err != nil {
		return nil, err
	}
	if def.ID == "" {
		def.ID = uuid.NewString()
	}
	return &Version{
		ID:         uuid.NewString(),
		AgentID:    agentID,
		ParentID:   parentID,
		Definition: def,
		CreatedAt:  time.Now().UTC(),
	}, nil
}
