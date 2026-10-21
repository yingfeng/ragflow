// Package caps defines the capability layer: a capability contributes tools and
// executes them. Data reach is injected per call through Scope, never stored on
// the agent definition.
package caps

import (
	"context"
	"fmt"
	"sort"
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

// Call is one tool invocation.
type Call struct {
	ID        string
	Tool      string
	Arguments string
}

// Result is one tool outcome.
type Result struct {
	Content  string
	IsError  bool
	Evidence []Evidence
	Meta     map[string]any
}

// Capability contributes tools and executes them.
type Capability interface {
	Name() string
	Tools() []model.ToolSpec
	Invoke(ctx context.Context, call Call) (Result, error)
}

// Registry routes tool calls to the owning capability.
type Registry struct {
	mu    sync.RWMutex
	caps  []Capability
	owner map[string]Capability
}

// NewRegistry builds an empty registry.
func NewRegistry() *Registry {
	return &Registry{owner: make(map[string]Capability)}
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
	}
	r.caps = append(r.caps, capability)
	return nil
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

// ToolSpecs returns every offered tool, ordered by name.
func (r *Registry) ToolSpecs() []model.ToolSpec {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]model.ToolSpec, 0, len(r.owner))
	for _, capability := range r.caps {
		out = append(out, capability.Tools()...)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
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
