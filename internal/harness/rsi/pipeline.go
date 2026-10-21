// Package rsi runs the self-improvement cycle: signals → attribution →
// proposals → replay evaluation → gate → lesson library.
//
// The cycle never mutates an agent definition. It only produces lessons, and a
// lesson reaches a prompt after the gate admitted it, so a bad cycle costs one
// rejected proposal rather than a permanently degraded agent.
package rsi

import (
	"context"
	"fmt"

	"ragflow/internal/harness/rsi/attribute"
	"ragflow/internal/harness/rsi/evaluate"
	"ragflow/internal/harness/rsi/gate"
	"ragflow/internal/harness/rsi/library"
	"ragflow/internal/harness/rsi/propose"
	"ragflow/internal/harness/rsi/signal"
)

// Signal is one learning input.
type Signal = signal.Signal

// Config wires a cycle.
type Config struct {
	// Policy is the admission policy. Zero value uses the default.
	Policy gate.Policy
	// Library stores admitted lessons. Nil keeps the cycle read-only.
	Library library.Store
	// Runner replays cases. Without it nothing can be admitted, because a
	// proposal is never promoted on plausibility alone.
	Runner evaluate.Runner
	// Cases are the replay cases.
	Cases []evaluate.Case
	// AgentID scopes the stored lessons.
	AgentID string
}

// Result is the outcome for one proposal.
type Result struct {
	Cause     attribute.Cause
	Proposal  propose.Proposal
	Evidence  int
	Baseline  evaluate.Score
	Candidate evaluate.Score
	Decision  gate.Decision
	Lesson    *library.Lesson
}

// Cycle executes the pipeline.
type Cycle struct{ cfg Config }

// New builds a cycle.
func New(cfg Config) *Cycle {
	if cfg.Policy == (gate.Policy{}) {
		cfg.Policy = gate.DefaultPolicy()
	}
	return &Cycle{cfg: cfg}
}

// Learn processes the signals and returns one result per considered proposal.
func (c *Cycle) Learn(ctx context.Context, signals []signal.Signal) ([]Result, error) {
	if ctx == nil {
		ctx = context.Background()
	}

	learnable := signal.Collect(signals)
	if len(learnable) == 0 {
		return nil, nil
	}

	grouped := make(map[attribute.Cause][]signal.Signal, len(learnable))
	for _, item := range learnable {
		cause := attribute.Classify(item)
		grouped[cause] = append(grouped[cause], item)
	}

	results := make([]Result, 0, len(grouped))
	for cause, group := range grouped {
		for _, proposal := range propose.FromCause(cause, group[0]) {
			result := Result{Cause: cause, Proposal: proposal, Evidence: len(group)}

			decision := c.cfg.Policy.Admit(proposal, result.Evidence, evaluate.Score{}, evaluate.Score{})
			if c.cfg.Runner != nil && len(c.cfg.Cases) > 0 {
				baseline, candidate, err := evaluate.Evaluate(ctx, c.cfg.Runner, proposal, c.cfg.Cases)
				if err != nil {
					return nil, fmt.Errorf("rsi: evaluate %s: %w", proposal.Describe(), err)
				}
				result.Baseline, result.Candidate = baseline, candidate
				decision = c.cfg.Policy.Admit(proposal, result.Evidence, baseline, candidate)
			}
			result.Decision = decision

			if decision.Admit && c.cfg.Library != nil {
				lesson := library.FromProposal(proposal, result.Evidence)
				if err := c.cfg.Library.Save(ctx, lesson); err != nil {
					return nil, fmt.Errorf("rsi: save lesson %q: %w", lesson.Title, err)
				}
				result.Lesson = &lesson
			}
			results = append(results, result)
		}
	}
	return results, nil
}
