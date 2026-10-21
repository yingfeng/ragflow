package toolkit

import (
	"context"
	"testing"
)

func TestRuleSetDenyOutranksAllowAndAsk(t *testing.T) {
	rules, err := NewRuleSet([]PermissionRule{
		{Match: MatchExact, Pattern: "exec_shell", Action: ActionAllow},
		{Match: MatchPrefix, Pattern: "exec_", Action: ActionDeny, Reason: "shell is disabled"},
		{Match: MatchExact, Pattern: "exec_shell", Action: ActionAsk},
	})
	if err != nil {
		t.Fatalf("rules: %v", err)
	}

	action, reason, matched := rules.Decide(Call{Tool: "exec_shell"})
	if !matched {
		t.Fatal("expected a match")
	}
	if action != ActionDeny {
		t.Fatalf("action = %s, want %s", action, ActionDeny)
	}
	if reason != "shell is disabled" {
		t.Fatalf("reason = %q", reason)
	}
}

func TestRuleSetMatchesPathsUnderPrefix(t *testing.T) {
	rules, err := NewRuleSet([]PermissionRule{
		{Match: MatchPathPrefix, Pattern: "/srv/data", Action: ActionAllow},
	})
	if err != nil {
		t.Fatalf("rules: %v", err)
	}

	cases := map[string]bool{
		`{"path":"/srv/data/a.txt"}`:         true,
		`{"path":"/srv/data"}`:               true,
		`{"path":"/srv/data/../etc/passwd"}`: false,
		`{"path":"/srv/database/a.txt"}`:     false,
		`{}`:                                 false,
	}
	for arguments, want := range cases {
		_, _, matched := rules.Decide(Call{Tool: "fs_read", Arguments: arguments})
		if matched != want {
			t.Errorf("arguments %s matched = %v, want %v", arguments, matched, want)
		}
	}
}

func TestRuleSetRejectsUnknownRule(t *testing.T) {
	if _, err := NewRuleSet([]PermissionRule{{Match: "glob", Pattern: "*", Action: ActionAllow}}); err == nil {
		t.Fatal("expected an error for an unknown match kind")
	}
	if _, err := NewRuleSet([]PermissionRule{{Match: MatchExact, Pattern: "x", Action: "maybe"}}); err == nil {
		t.Fatal("expected an error for an unknown action")
	}
}

func TestPermissionApproverOrder(t *testing.T) {
	grants := NewMemoryGrantStore()
	call := Call{ID: "c1", Tool: "exec_shell", Arguments: `{"cmd":"rm -rf /tmp/x"}`}
	if err := grants.Remember(context.Background(), GrantFor("tenant-1", call)); err != nil {
		t.Fatalf("remember: %v", err)
	}

	// A deny rule must beat a standing grant, otherwise a granted command could
	// never be revoked.
	deny, err := NewRuleSet([]PermissionRule{{Match: MatchExact, Pattern: "exec_shell", Action: ActionDeny}})
	if err != nil {
		t.Fatalf("rules: %v", err)
	}
	approver := &PermissionApprover{Rules: deny, Grants: grants, TenantID: "tenant-1", Next: allowAll{}}
	if decision := approver.Approve(context.Background(), call, RiskDangerous); decision.Allow {
		t.Fatal("deny rule lost to a standing grant")
	}

	// Without the deny rule the grant answers, and the next approver is not asked.
	counting := &countingApprover{}
	approver = &PermissionApprover{Rules: nil, Grants: grants, TenantID: "tenant-1", Next: counting}
	decision := approver.Approve(context.Background(), call, RiskDangerous)
	if !decision.Allow {
		t.Fatalf("grant was ignored: %s", decision.Reason)
	}
	if counting.calls != 0 {
		t.Fatalf("next approver called %d times, want 0", counting.calls)
	}

	// A different argument set is a different grant.
	other := Call{ID: "c2", Tool: "exec_shell", Arguments: `{"cmd":"ls"}`}
	approver.Approve(context.Background(), other, RiskDangerous)
	if counting.calls != 1 {
		t.Fatalf("next approver called %d times for an unlisted command, want 1", counting.calls)
	}
}

func TestPermissionApproverAskDefersAndAllowShortCircuits(t *testing.T) {
	next := &countingApprover{}
	rules, err := NewRuleSet([]PermissionRule{
		{Match: MatchExact, Pattern: "exec_shell", Action: ActionAsk},
		{Match: MatchExact, Pattern: "fs_ls", Action: ActionAllow},
	})
	if err != nil {
		t.Fatalf("rules: %v", err)
	}
	approver := &PermissionApprover{Rules: rules, Next: next}

	approver.Approve(context.Background(), Call{Tool: "exec_shell"}, RiskDangerous)
	if next.calls != 1 {
		t.Fatalf("ask did not defer: next called %d times", next.calls)
	}

	decision := approver.Approve(context.Background(), Call{Tool: "fs_ls"}, RiskReadOnly)
	if !decision.Allow || next.calls != 1 {
		t.Fatalf("allow rule consulted the next approver (calls=%d)", next.calls)
	}
}

type allowAll struct{}

func (allowAll) Approve(context.Context, Call, Risk) ApprovalDecision {
	return ApprovalDecision{Allow: true}
}

type countingApprover struct{ calls int }

func (c *countingApprover) Approve(context.Context, Call, Risk) ApprovalDecision {
	c.calls++
	return ApprovalDecision{Allow: true}
}
