package toolkit

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// ApprovalDecision is the verdict on one tool call.
type ApprovalDecision struct {
	Allow  bool
	Reason string
}

// Approver decides whether one call may run. It is consulted after the risk is
// known and before execution, which is the only place where "dangerous" can
// become "ask a human" without teaching the model about risk classes.
type Approver interface {
	Approve(ctx context.Context, call Call, risk Risk) ApprovalDecision
}

// ApprovalGate puts an Approver in front of the calls at or above a risk
// threshold and lets everything below it through.
//
// A gate with a nil Approver denies the gated calls: it is the safe default for
// a deployment that configured approvals but not an approver, and it makes the
// failure visible in the tool result instead of running a shell command.
type ApprovalGate struct {
	Approver  Approver
	Threshold Risk
}

// NewApprovalGate builds a gate. Zero threshold gates mutating and dangerous
// calls; use RiskDangerous to gate only calls that leave the process.
func NewApprovalGate(approver Approver, threshold Risk) *ApprovalGate {
	return &ApprovalGate{Approver: approver, Threshold: threshold}
}

// Approve implements Approver: calls below the threshold pass, the rest are
// delegated. Without an approver the gated calls are denied, which is the safe
// failure for a deployment that configured approvals but no human channel.
func (g *ApprovalGate) Approve(ctx context.Context, call Call, risk Risk) ApprovalDecision {
	if g == nil || risk < g.Threshold {
		return ApprovalDecision{Allow: true}
	}
	if g.Approver == nil {
		return ApprovalDecision{Reason: fmt.Sprintf("tool %q requires approval and no approver is configured", call.Tool)}
	}
	decision := g.Approver.Approve(ctx, call, risk)
	if !decision.Allow && decision.Reason == "" {
		decision.Reason = "rejected by approver"
	}
	return decision
}

// Allow is the (allowed, reason) form of Approve.
func (g *ApprovalGate) Allow(ctx context.Context, call Call, risk Risk) (bool, string) {
	decision := g.Approve(ctx, call, risk)
	return decision.Allow, decision.Reason
}

// RiskOfCall is a convenience for gates that need the registry's classification.
func RiskOfCall(registry *Registry, call Call) Risk {
	return registry.RiskOf(call.Tool)
}

// ApprovalBroker is a rendezvous between a running tool batch and an operator.
// The waiting side holds the batch; the operator resolves it over the API.
//
// It is process-local on purpose: a pending approval cannot outlive the worker.
// When it dies, the run's lease expires and crash recovery resumes the run,
// which asks again — presuming approval would run a dangerous tool nobody saw.
type ApprovalBroker struct {
	mu      sync.Mutex
	pending map[string]*pendingApproval
	timeout time.Duration
}

// pendingApproval is one waiting request. The call and its tenant are kept so an
// operator's "allow always" can be turned into a standing grant without asking
// the running batch to resend anything.
type pendingApproval struct {
	decision chan ApprovalDecision
	call     Call
	tenantID string
}

// PendingApproval is what an operator needs to answer a request.
type PendingApproval struct {
	Call     Call
	TenantID string
}

// DefaultApprovalTimeout bounds how long a batch waits for a human.
const DefaultApprovalTimeout = 5 * time.Minute

// NewApprovalBroker builds a broker. Zero timeout uses DefaultApprovalTimeout.
func NewApprovalBroker(timeout time.Duration) *ApprovalBroker {
	if timeout <= 0 {
		timeout = DefaultApprovalTimeout
	}
	return &ApprovalBroker{pending: make(map[string]*pendingApproval), timeout: timeout}
}

type runIDKey struct{}

// WithRunID binds the current run to the context, so an approval request can be
// addressed by the API.
func WithRunID(ctx context.Context, runID string) context.Context {
	if runID == "" {
		return ctx
	}
	return context.WithValue(ctx, runIDKey{}, runID)
}

// RunIDFrom reads the run bound to the context.
func RunIDFrom(ctx context.Context) string {
	runID, _ := ctx.Value(runIDKey{}).(string)
	return runID
}

// Approve implements Approver. It blocks until an operator resolves the request
// or the timeout expires, and denies on timeout.
func (b *ApprovalBroker) Approve(ctx context.Context, call Call, risk Risk) ApprovalDecision {
	requestID := RunIDFrom(ctx) + ":" + call.ID
	decision := make(chan ApprovalDecision, 1)

	b.mu.Lock()
	b.pending[requestID] = &pendingApproval{
		decision: decision,
		call:     call,
		tenantID: ScopeFrom(ctx).TenantID,
	}
	b.mu.Unlock()
	defer func() {
		b.mu.Lock()
		delete(b.pending, requestID)
		b.mu.Unlock()
	}()

	timer := time.NewTimer(b.timeout)
	defer timer.Stop()
	select {
	case resolution := <-decision:
		return resolution
	case <-ctx.Done():
		return ApprovalDecision{Reason: "run cancelled while waiting for approval"}
	case <-timer.C:
		return ApprovalDecision{Reason: "approval timed out"}
	}
}

// Resolve settles one pending request. It reports whether the request was
// waiting; a false result means it already timed out or never existed.
func (b *ApprovalBroker) Resolve(runID, requestID string, allow bool, reason string) bool {
	b.mu.Lock()
	entry, ok := b.pending[runID+":"+requestID]
	b.mu.Unlock()
	if !ok {
		return false
	}
	select {
	case entry.decision <- ApprovalDecision{Allow: allow, Reason: reason}:
		return true
	default:
		return false
	}
}

// Pending returns the call waiting on one request, so an "allow always" answer
// can be recorded as a standing grant for exactly that call.
func (b *ApprovalBroker) Pending(runID, requestID string) (PendingApproval, bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	entry, ok := b.pending[runID+":"+requestID]
	if !ok {
		return PendingApproval{}, false
	}
	return PendingApproval{Call: entry.call, TenantID: entry.tenantID}, true
}
