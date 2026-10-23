// Package toolkit defines the capability layer: a capability contributes tools
// and executes them. Data reach is injected per call through Scope, never stored
// on the agent definition.
package toolkit

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"sync"

	"ragflow/internal/harness/agent/model"
)

// Scope is the run's reach. It is injected by the caller.
type Scope struct {
	TenantID  string
	Datasets  []string
	Docs      []string
	Principal string
}

type scopeKey struct{}

// WithScope binds the run scope to a context.
func WithScope(ctx context.Context, scope Scope) context.Context {
	return context.WithValue(ctx, scopeKey{}, scope)
}

// ScopeFrom reads the run scope from a context.
func ScopeFrom(ctx context.Context) Scope {
	scope, _ := ctx.Value(scopeKey{}).(Scope)
	return scope
}

// HasScope reports whether a scope was bound to the context.
func HasScope(ctx context.Context) bool {
	_, ok := ctx.Value(scopeKey{}).(Scope)
	return ok
}

// Evidence is one citation-bearing fact produced by a tool.
type Evidence struct {
	Source  string  `json:"source"`
	Ref     string  `json:"ref"`
	Snippet string  `json:"snippet,omitempty"`
	Score   float64 `json:"score,omitempty"`
}

// MetaPlan is the Result.Meta key a tool uses to publish a structured plan.
//
// The runtime lifts a published plan out of the transcript and renders it into
// every later prompt, so a plan keeps its shape when the conversation is
// compacted and the model does not have to re-read its own tool output to know
// what is left. Any tool may publish one; nothing here is specific to the
// planning tool's name.
const MetaPlan = "plan"

// PlanStep is one step of a published plan.
type PlanStep struct {
	ID          string `json:"id,omitempty"`
	Description string `json:"description"`
	Status      string `json:"status,omitempty"`
}

// Call is one tool invocation.
type Call struct {
	ID        string
	Tool      string
	Arguments string
}

// Result is one tool outcome.
type Result struct {
	Content string
	IsError bool
	// Inline keeps the content verbatim in the transcript instead of letting the
	// loop spill it behind a handle. A tool that already returns a bounded window
	// of a larger payload sets this: spilling its output again would replace the
	// window with a handle to itself and the model could never read the text.
	Inline   bool
	Evidence []Evidence
	Meta     map[string]any
}

// Capability contributes tools and executes them.
type Capability interface {
	Name() string
	Tools() []model.ToolSpec
	Invoke(ctx context.Context, call Call) (Result, error)
}

// RequiredToolList is an optional Capability extension. It names the tools a
// run must call before it may do anything else with them; the loop turns them
// into a forced tool_choice until the transcript shows one. Retrieval is the
// use case: an agent whose whole job is evidence must start by gathering some,
// and a round of pure planning or chatter is a wasted round.
type RequiredToolList interface {
	RequiredTools() []string
}

// Guide is an optional Capability extension. It contributes instructions that
// must be in force from the first round, rather than discovered on demand.
//
// The distinction from a tool description is what the text is for. A schema
// says how to call one tool; nothing in it says which leg to try first, when a
// snippet is enough, or that one routing call replaces several blind searches.
// The distinction from a skill body is when it is read: a skill is loaded by
// the model when it recognises the situation, which cannot work for a rule the
// model does not yet know it needs. Policy therefore comes from this interface.
//
// The content is still data, not code: the skill capability implements this by
// returning the bodies of the skills that declared auto_load, so an operator
// changes the policy by editing a file.
type Guide interface {
	// SystemNote renders the capability's standing instructions. Empty means
	// the capability adds nothing.
	SystemNote(ctx context.Context) string
}

// Registry routes tool calls to the owning capability.
//
// A capability can be registered active or deferred. Deferred tools stay in the
// catalog — searchable and callable — but are not part of the tool list sent to
// the model until something enables them, which is what keeps a large tool
// surface from costing prompt tokens every turn.
type Registry struct {
	mu      sync.RWMutex
	caps    []Capability
	owner   map[string]Capability
	enabled map[string]bool
}

// NewRegistry builds an empty registry.
func NewRegistry() *Registry {
	return &Registry{
		owner:   make(map[string]Capability),
		enabled: make(map[string]bool),
	}
}

// Register adds a capability. Duplicate tool names are rejected: silently
// shadowing one capability's tool with another's is not debuggable at runtime.
func (r *Registry) Register(capability Capability) error {
	if capability == nil {
		return fmt.Errorf("caps: nil capability")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, spec := range capability.Tools() {
		if spec.Name == "" {
			return fmt.Errorf("caps: capability %q declares an unnamed tool", capability.Name())
		}
		if existing, ok := r.owner[spec.Name]; ok {
			return fmt.Errorf("caps: tool %q already owned by %q", spec.Name, existing.Name())
		}
	}
	for _, spec := range capability.Tools() {
		r.owner[spec.Name] = capability
		r.enabled[spec.Name] = true
	}
	r.caps = append(r.caps, capability)
	return nil
}

// RegisterDeferred adds a capability whose tools are searchable but not offered
// to the model until Enable names them.
func (r *Registry) RegisterDeferred(capability Capability) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if capability == nil {
		return fmt.Errorf("caps: nil capability")
	}
	for _, spec := range capability.Tools() {
		if spec.Name == "" {
			return fmt.Errorf("caps: capability %q declares an unnamed tool", capability.Name())
		}
		if existing, ok := r.owner[spec.Name]; ok {
			return fmt.Errorf("caps: tool %q already owned by %q", spec.Name, existing.Name())
		}
	}
	for _, spec := range capability.Tools() {
		r.owner[spec.Name] = capability
		r.enabled[spec.Name] = false
	}
	r.caps = append(r.caps, capability)
	return nil
}

// Has reports whether a tool name is known to the registry.
func (r *Registry) Has(tool string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	_, ok := r.owner[tool]
	return ok
}

// Enable activates tools by name and returns their specs, so the caller can
// hand the calling contract to the model in the same turn.
func (r *Registry) Enable(names []string) []model.ToolSpec {
	r.mu.Lock()
	defer r.mu.Unlock()

	out := make([]model.ToolSpec, 0, len(names))
	for _, name := range names {
		if _, ok := r.owner[name]; !ok {
			continue
		}
		r.enabled[name] = true
		for _, capability := range r.caps {
			for _, spec := range capability.Tools() {
				if spec.Name == name {
					out = append(out, spec)
				}
			}
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// Search ranks catalog tools against a free-text query. A name hit outweighs a
// description hit; ties are broken by name so results are stable.
func (r *Registry) Search(query string, limit int) []model.ToolSpec {
	if limit <= 0 {
		limit = 8
	}
	terms := strings.Fields(strings.ToLower(query))

	type scored struct {
		spec  model.ToolSpec
		score int
	}
	r.mu.RLock()
	candidates := make([]scored, 0, len(r.owner))
	for _, capability := range r.caps {
		for _, spec := range capability.Tools() {
			score := 0
			lowerName := strings.ToLower(spec.Name)
			lowerDesc := strings.ToLower(spec.Description)
			for _, term := range terms {
				if strings.Contains(lowerName, term) {
					score += 3
				}
				if strings.Contains(lowerDesc, term) {
					score++
				}
			}
			if score == 0 && len(terms) > 0 {
				continue
			}
			candidates = append(candidates, scored{spec: spec, score: score})
		}
	}
	r.mu.RUnlock()

	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].score != candidates[j].score {
			return candidates[i].score > candidates[j].score
		}
		return candidates[i].spec.Name < candidates[j].spec.Name
	})
	if len(candidates) > limit {
		candidates = candidates[:limit]
	}
	out := make([]model.ToolSpec, 0, len(candidates))
	for _, candidate := range candidates {
		out = append(out, candidate.spec)
	}
	return out
}

// Names returns the registered capability names.
func (r *Registry) Names() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]string, 0, len(r.caps))
	for _, capability := range r.caps {
		out = append(out, capability.Name())
	}
	sort.Strings(out)
	return out
}

// RequiredTools collects the required-tool declarations of every registered
// capability, in registration order, deduplicated. Empty means no gate applies.
func (r *Registry) RequiredTools() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	seen := make(map[string]struct{})
	var out []string
	for _, capability := range r.caps {
		list, ok := capability.(RequiredToolList)
		if !ok {
			continue
		}
		for _, name := range list.RequiredTools() {
			if _, exists := seen[name]; exists {
				continue
			}
			if _, owned := r.owner[name]; !owned {
				continue
			}
			seen[name] = struct{}{}
			out = append(out, name)
		}
	}
	return out
}

// SystemNotes collects the standing instructions of every registered capability
// that declares them, in registration order.
//
// Registration order, not name order, is what a reader expects: a capability's
// note follows the capability it belongs to. The result is part of the system
// prompt, so it is assembled once per run and stays byte-stable for the run — a
// note that reshuffles between rounds breaks the provider's prefix cache for no
// gain.
func (r *Registry) SystemNotes(ctx context.Context) string {
	if r == nil {
		return ""
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	parts := make([]string, 0, len(r.caps))
	for _, capability := range r.caps {
		guide, ok := capability.(Guide)
		if !ok {
			continue
		}
		if text := strings.TrimSpace(guide.SystemNote(ctx)); text != "" {
			parts = append(parts, text)
		}
	}
	return strings.Join(parts, "\n\n")
}

// Without returns a registry that keeps every capability except the ones owning
// any of the named tools. The capabilities themselves are shared, not copied,
// so the child registry costs one slice and adds no state of its own. A run that
// hands its registry to a child uses this to drop the tools that spawned it,
// which is what keeps delegation from recursing without a bound.
func (r *Registry) Without(tools ...string) *Registry {
	if r == nil {
		return NewRegistry()
	}
	blocked := make(map[string]struct{}, len(tools))
	for _, tool := range tools {
		blocked[tool] = struct{}{}
	}

	child := NewRegistry()
	r.mu.RLock()
	defer r.mu.RUnlock()
	for _, capability := range r.caps {
		specs := capability.Tools()
		excluded := false
		for _, spec := range specs {
			if _, ok := blocked[spec.Name]; ok {
				excluded = true
				break
			}
		}
		if excluded {
			continue
		}
		for _, spec := range specs {
			child.owner[spec.Name] = capability
			child.enabled[spec.Name] = r.enabled[spec.Name]
		}
		child.caps = append(child.caps, capability)
	}
	return child
}

// ToolSpecs returns the tools currently offered to the model, ordered by name.
// Deferred tools are excluded until Enable names them.
func (r *Registry) ToolSpecs() []model.ToolSpec {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]model.ToolSpec, 0, len(r.owner))
	for _, capability := range r.caps {
		for _, spec := range capability.Tools() {
			if r.enabled[spec.Name] {
				out = append(out, spec)
			}
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// SchemaBudgetTokens is Astra's DEFAULT_TOOL_SCHEMA_BUDGET_TOKENS. Tool schemas
// compete with the transcript for the same window; a large tool set must not
// silently consume it.
const SchemaBudgetTokens = 800

// OnlyReadOnly returns a registry view whose capabilities have no mutating
// tools. It is how a read-only delegation profile is enforced: a capability with
// a single writer is dropped whole, because offering half of one is a capability
// the caller did not ask for.
func (r *Registry) OnlyReadOnly() *Registry {
	if r == nil {
		return NewRegistry()
	}
	child := NewRegistry()
	r.mu.RLock()
	defer r.mu.RUnlock()
	for _, capability := range r.caps {
		readOnly := true
		for _, spec := range capability.Tools() {
			if riskOrMutating(spec.Name) != RiskReadOnly {
				readOnly = false
				break
			}
		}
		if !readOnly {
			continue
		}
		for _, spec := range capability.Tools() {
			child.owner[spec.Name] = capability
			child.enabled[spec.Name] = r.enabled[spec.Name]
		}
		child.caps = append(child.caps, capability)
	}
	return child
}

// ToolSpecsWithinBudget returns the offered tools in name order until the
// estimated cost of their schemas reaches budgetTokens. Excluded tools stay
// registered and reachable through tool_search — they are dropped from the
// request, not from the deployment.
//
// A zero or negative budget means no limit.
func (r *Registry) ToolSpecsWithinBudget(budgetTokens int) []model.ToolSpec {
	specs := r.ToolSpecs()
	if budgetTokens <= 0 {
		return specs
	}
	out := make([]model.ToolSpec, 0, len(specs))
	used := 0
	for _, spec := range specs {
		cost := estimateSchemaTokens(spec)
		if used+cost > budgetTokens && len(out) > 0 {
			continue
		}
		out = append(out, spec)
		used += cost
	}
	return out
}

// estimateSchemaTokens approximates the prompt cost of one tool schema. It is
// intentionally pessimistic: the JSON envelope and field names cost tokens even
// when the description is short.
func estimateSchemaTokens(spec model.ToolSpec) int {
	total := len(spec.Name) + len(spec.Description) + 16
	if spec.Parameters != nil {
		if payload, err := json.Marshal(spec.Parameters); err == nil {
			total += len(payload)
		}
	}
	return total/4 + 1
}

// Invoke routes one call. An unknown tool is a tool-level error, not a run
// failure: the model can recover by choosing another tool.
func (r *Registry) Invoke(ctx context.Context, call Call) (Result, error) {
	r.mu.RLock()
	capability, ok := r.owner[call.Tool]
	r.mu.RUnlock()
	if !ok {
		return ErrorResult(fmt.Errorf("unknown tool %q", call.Tool)), nil
	}
	// A skill may narrow the surface. Admission is checked separately from
	// projection because the model can name a tool it was never offered.
	if allowlist := AllowlistFrom(ctx); !allowlist.Allows(call.Tool) {
		return ErrorResult(fmt.Errorf("tool %q is not available while a skill restricts the tool surface", call.Tool)), nil
	}
	result, err := capability.Invoke(ctx, call)
	if err != nil {
		return ErrorResult(err), nil
	}
	return result, nil
}

// ErrorResult renders an error as a tool result the model can read.
func ErrorResult(err error) Result {
	return Result{Content: "<tool_error>" + err.Error() + "</tool_error>", IsError: true}
}

// WithinScope returns an error when requested ids are not all inside bound.
// An empty bound means the scope was not constrained upstream.
func WithinScope(bound, requested []string) error {
	if len(requested) == 0 || len(bound) == 0 {
		return nil
	}
	allowed := make(map[string]struct{}, len(bound))
	for _, id := range bound {
		allowed[id] = struct{}{}
	}
	for _, id := range requested {
		if _, ok := allowed[id]; !ok {
			return fmt.Errorf("id %q is outside the run scope", id)
		}
	}
	return nil
}
