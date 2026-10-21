// Package gate decides whether an evaluated proposal may reach the prompt.
//
// The gate is deliberately conservative: a lesson costs prompt tokens on every
// future run, so it has to earn its place with repeated evidence and a measured
// improvement that comes without regressions.
package gate

import (
	"fmt"

	"ragflow/internal/harness/rsi/evaluate"
	"ragflow/internal/harness/rsi/propose"
)

// Policy is the admission policy.
type Policy struct {
	// MinEvidence is how many independent bad runs must share the cause.
	MinEvidence int
	// MinImprovement is the required pass-rate delta (0.1 = ten points).
	MinImprovement float64
	// MinPassRate is an absolute floor the candidate must reach.
	MinPassRate float64
}

// DefaultPolicy is the RSI P1 starting point.
func DefaultPolicy() Policy {
	return Policy{MinEvidence: 2, MinImprovement: 0.1, MinPassRate: 0.5}
}

// Decision is an admission verdict with its justification.
type Decision struct {
	Admit  bool
	Reason string
}

// Admit evaluates one proposal.
func (p Policy) Admit(proposal propose.Proposal, evidence int, baseline, candidate evaluate.Score) Decision {
	if evidence < p.MinEvidence {
		return Decision{Reason: fmt.Sprintf("evidence %d < required %d", evidence, p.MinEvidence)}
	}
	if candidate.Cases == 0 {
		return Decision{Reason: "no replay cases"}
	}
	if candidate.PassRate() < p.MinPassRate {
		return Decision{Reason: fmt.Sprintf("pass rate %.2f < floor %.2f", candidate.PassRate(), p.MinPassRate)}
	}
	delta := candidate.PassRate() - baseline.PassRate()
	if delta < p.MinImprovement {
		return Decision{Reason: fmt.Sprintf("improvement %.2f < required %.2f", delta, p.MinImprovement)}
	}
	return Decision{Admit: true, Reason: fmt.Sprintf("improvement %.2f over %d cases", delta, candidate.Cases)}
}
