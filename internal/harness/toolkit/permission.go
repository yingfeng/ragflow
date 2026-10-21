package toolkit

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"path"
	"strings"
	"sync"
)

// RuleAction is what a matching rule does with a call.
type RuleAction string

const (
	// ActionAllow runs the call without asking.
	ActionAllow RuleAction = "allow"
	// ActionAsk defers to the next approver, which is normally the human gate.
	ActionAsk RuleAction = "ask"
	// ActionDeny refuses the call. A deny is final: no later rule and no
	// standing grant can reopen it.
	ActionDeny RuleAction = "deny"
)

// MatchKind is how a rule matches a call.
type MatchKind string

const (
	// MatchExact compares the tool name exactly.
	MatchExact MatchKind = "exact"
	// MatchPrefix compares the tool name prefix, which covers a capability's
	// family such as fs_*.
	MatchPrefix MatchKind = "prefix"
	// MatchPathPrefix compares the paths named in the arguments against a
	// directory prefix. A call with no path argument never matches.
	MatchPathPrefix MatchKind = "path_prefix"
)

// PermissionRule is one operator-authored statement about a tool call. Rules are
// data, not code: an agent definition carries them, so what an agent may do is
// reviewable without reading its prompt.
type PermissionRule struct {
	Match   MatchKind  `json:"match"`
	Pattern string     `json:"pattern"`
	Action  RuleAction `json:"action"`
	Reason  string     `json:"reason,omitempty"`
}

func (r PermissionRule) validate() error {
	switch r.Match {
	case MatchExact, MatchPrefix, MatchPathPrefix:
	default:
		return fmt.Errorf("toolkit: unknown permission match %q", r.Match)
	}
	switch r.Action {
	case ActionAllow, ActionAsk, ActionDeny:
	default:
		return fmt.Errorf("toolkit: unknown permission action %q", r.Action)
	}
	if strings.TrimSpace(r.Pattern) == "" {
		return fmt.Errorf("toolkit: permission rule needs a pattern")
	}
	return nil
}

func (r PermissionRule) matches(call Call) bool {
	switch r.Match {
	case MatchExact:
		return call.Tool == r.Pattern
	case MatchPrefix:
		return strings.HasPrefix(call.Tool, r.Pattern)
	case MatchPathPrefix:
		for _, candidate := range argumentPaths(call.Arguments) {
			if underPath(candidate, r.Pattern) {
				return true
			}
		}
	}
	return false
}

// RuleSet evaluates an ordered rule list. Deny outranks ask, which outranks
// allow, so the order rules are written in never turns a denial into a hole.
type RuleSet struct {
	rules []PermissionRule
}

// NewRuleSet validates the rules and freezes them.
func NewRuleSet(rules []PermissionRule) (*RuleSet, error) {
	validated := make([]PermissionRule, 0, len(rules))
	for _, rule := range rules {
		if err := rule.validate(); err != nil {
			return nil, err
		}
		validated = append(validated, rule)
	}
	return &RuleSet{rules: validated}, nil
}

// Rules returns the frozen rules.
func (s *RuleSet) Rules() []PermissionRule {
	if s == nil {
		return nil
	}
	return append([]PermissionRule(nil), s.rules...)
}

// Decide reports the strongest matching action. The third result is false when
// no rule matched, which leaves the call to the next approver.
func (s *RuleSet) Decide(call Call) (RuleAction, string, bool) {
	if s == nil {
		return "", "", false
	}
	var (
		action  RuleAction
		reason  string
		matched bool
	)
	for _, rule := range s.rules {
		if !rule.matches(call) {
			continue
		}
		if !matched || actionRank(rule.Action) > actionRank(action) {
			action = rule.Action
			reason = rule.Reason
			matched = true
		}
		if action == ActionDeny {
			break
		}
	}
	return action, reason, matched
}

func actionRank(action RuleAction) int {
	switch action {
	case ActionDeny:
		return 3
	case ActionAsk:
		return 2
	case ActionAllow:
		return 1
	}
	return 0
}

// argumentPaths collects the path-like strings a call carries. It walks the JSON
// instead of naming fields, because every capability spells its directory
// argument differently.
func argumentPaths(raw string) []string {
	if strings.TrimSpace(raw) == "" {
		return nil
	}
	var decoded any
	if err := json.Unmarshal([]byte(raw), &decoded); err != nil {
		return nil
	}
	paths := make([]string, 0, 4)
	collectPaths(decoded, &paths)
	return paths
}

func collectPaths(value any, out *[]string) {
	switch typed := value.(type) {
	case string:
		if looksLikePath(typed) {
			*out = append(*out, typed)
		}
	case []any:
		for _, item := range typed {
			collectPaths(item, out)
		}
	case map[string]any:
		for _, item := range typed {
			collectPaths(item, out)
		}
	}
}

func looksLikePath(value string) bool {
	switch {
	case value == "":
		return false
	case strings.HasPrefix(value, "/"),
		strings.HasPrefix(value, "./"),
		strings.HasPrefix(value, "../"),
		strings.HasPrefix(value, "~"):
		return true
	}
	return false
}

// underPath reports whether candidate sits inside prefix. Both sides are cleaned
// so ".." cannot walk out of the prefix a rule named.
func underPath(candidate, prefix string) bool {
	cleanedCandidate := path.Clean(candidate)
	cleanedPrefix := path.Clean(prefix)
	if cleanedCandidate == cleanedPrefix {
		return true
	}
	return strings.HasPrefix(cleanedCandidate, strings.TrimSuffix(cleanedPrefix, "/")+"/")
}

// Grant is a standing permission for one tool and one argument set. It is keyed
// by the argument hash so "allow this exact command" cannot be reused for a
// different one.
type Grant struct {
	TenantID string `json:"tenant_id"`
	Tool     string `json:"tool"`
	ArgsHash string `json:"args_hash"`
}

// GrantStore remembers standing permissions.
type GrantStore interface {
	Granted(ctx context.Context, grant Grant) (bool, error)
	Remember(ctx context.Context, grant Grant) error
}

// MemoryGrantStore keeps grants for the lifetime of the process. It is the
// fallback when no database is configured, and the store tests use it.
type MemoryGrantStore struct {
	mu     sync.RWMutex
	grants map[string]struct{}
}

// NewMemoryGrantStore builds an empty grant store.
func NewMemoryGrantStore() *MemoryGrantStore {
	return &MemoryGrantStore{grants: make(map[string]struct{})}
}

func grantKey(grant Grant) string {
	return grant.TenantID + "|" + grant.Tool + "|" + grant.ArgsHash
}

// Granted reports whether the grant is known.
func (s *MemoryGrantStore) Granted(_ context.Context, grant Grant) (bool, error) {
	if s == nil {
		return false, nil
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	_, ok := s.grants[grantKey(grant)]
	return ok, nil
}

// Remember records a grant.
func (s *MemoryGrantStore) Remember(_ context.Context, grant Grant) error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.grants[grantKey(grant)] = struct{}{}
	return nil
}

// ArgumentsHash fingerprints a call for grants and idempotency. The tool name is
// part of the hash so the same arguments against a different tool never share a
// decision.
func ArgumentsHash(tool, arguments string) string {
	sum := sha256.Sum256([]byte(tool + "\x00" + arguments))
	return hex.EncodeToString(sum[:])
}

// PermissionApprover evaluates rules, then standing grants, then the wrapped
// approver. Order matters: a deny rule must beat a standing grant, otherwise an
// operator who once allowed a command could never revoke it.
type PermissionApprover struct {
	Rules  *RuleSet
	Grants GrantStore
	// TenantID keys the grants. Empty disables grant lookup.
	TenantID string
	// Next is the approval channel, normally an ApprovalGate over the broker.
	Next Approver
}

// Approve implements Approver.
func (p *PermissionApprover) Approve(ctx context.Context, call Call, risk Risk) ApprovalDecision {
	if p == nil {
		return ApprovalDecision{Allow: true}
	}

	// Rules first: a deny is final even if a grant exists.
	if p.Rules != nil {
		action, reason, _ := p.Rules.Decide(call)
		switch action {
		case ActionDeny:
			if reason == "" {
				reason = fmt.Sprintf("tool %q is denied by policy", call.Tool)
			}
			return ApprovalDecision{Reason: reason}
		case ActionAllow:
			return ApprovalDecision{Allow: true, Reason: reason}
		case ActionAsk:
			// Fall through to the approval channel.
		}
	}

	if p.Grants != nil && p.TenantID != "" {
		grant := Grant{TenantID: p.TenantID, Tool: call.Tool, ArgsHash: ArgumentsHash(call.Tool, call.Arguments)}
		if ok, err := p.Grants.Granted(ctx, grant); err == nil && ok {
			return ApprovalDecision{Allow: true, Reason: "standing grant"}
		}
	}

	if p.Next == nil {
		return ApprovalDecision{Allow: true}
	}
	return p.Next.Approve(ctx, call, risk)
}

// GrantFor builds the grant an "allow always" decision records.
func GrantFor(tenantID string, call Call) Grant {
	return Grant{
		TenantID: tenantID,
		Tool:     call.Tool,
		ArgsHash: ArgumentsHash(call.Tool, call.Arguments),
	}
}
