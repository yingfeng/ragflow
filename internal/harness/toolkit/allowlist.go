package toolkit

import (
	"context"
	"sync"

	"ragflow/internal/harness/agent/model"
)

// skillToolsStayAvailable keeps the skill surface reachable while a skill
// narrows the tools: without them the model could neither switch skills nor
// discover what it is missing, and the restriction would look like a broken
// deployment rather than a scope.
var skillToolsStayAvailable = map[string]struct{}{
	"skill_list": {},
	"skill_load": {},
}

// Allowlist narrows the tool surface while a skill is active. A run installs one
// holder in its context; skill_load narrows it, and both projection
// (ToolSpecsWithinBudget) and admission (Invoke) read the same holder. It is a
// pointer in the context precisely so a capability can narrow it without
// returning a new context.
type Allowlist struct {
	mu     sync.RWMutex
	active bool
	names  map[string]struct{}
}

// NewAllowlist builds an unrestricted allowlist.
func NewAllowlist() *Allowlist { return &Allowlist{} }

// Set activates the list. The skill tools stay available so the model can leave
// the skill; everything outside the list is refused.
func (a *Allowlist) Set(names []string) {
	if a == nil {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	a.names = make(map[string]struct{}, len(names)+len(skillToolsStayAvailable))
	for _, name := range names {
		if name == "" {
			continue
		}
		a.names[name] = struct{}{}
	}
	for name := range skillToolsStayAvailable {
		a.names[name] = struct{}{}
	}
	a.active = true
}

// Clear lifts the restriction.
func (a *Allowlist) Clear() {
	if a == nil {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	a.active = false
	a.names = nil
}

// Active reports whether a skill narrowed the surface.
func (a *Allowlist) Active() bool {
	if a == nil {
		return false
	}
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.active
}

// Allows reports whether a tool may run. A nil or inactive list allows
// everything, which is the case for an agent that never loads a skill.
func (a *Allowlist) Allows(tool string) bool {
	if a == nil {
		return true
	}
	a.mu.RLock()
	defer a.mu.RUnlock()
	if !a.active {
		return true
	}
	_, ok := a.names[tool]
	return ok
}

type allowlistKey struct{}

// WithAllowlist installs a fresh holder on the context when none is present, so
// every node and tool call of one run shares the same restriction.
func WithAllowlist(ctx context.Context) context.Context {
	if AllowlistFrom(ctx) != nil {
		return ctx
	}
	return context.WithValue(ctx, allowlistKey{}, NewAllowlist())
}

// AllowlistFrom reads the holder installed by WithAllowlist.
func AllowlistFrom(ctx context.Context) *Allowlist {
	list, _ := ctx.Value(allowlistKey{}).(*Allowlist)
	return list
}

// RestrictSpecs drops the schemas a skill made unreachable. Sending a schema the
// model may not call is a trap: it invites a call that is then refused.
func RestrictSpecs(specs []model.ToolSpec, allowlist *Allowlist) []model.ToolSpec {
	if allowlist == nil || !allowlist.Active() {
		return specs
	}
	kept := make([]model.ToolSpec, 0, len(specs))
	for _, spec := range specs {
		if allowlist.Allows(spec.Name) {
			kept = append(kept, spec)
		}
	}
	return kept
}
