// Package evaluate replays a proposal against recorded cases before it is
// allowed into the prompt. A lesson that cannot be shown to help is not
// promoted, however plausible it sounds.
package evaluate

import (
	"context"
	"fmt"
	"strings"

	"ragflow/internal/harness/rsi/propose"
)

// Case is one replayable input.
type Case struct {
	RunID    string
	Question string
	// MustMention lists strings the answer has to contain.
	MustMention []string
	// MustCite requires at least one chunk reference.
	MustCite bool
}

// Score is the outcome of a replay.
type Score struct {
	Cases  int
	Passed int
}

// PassRate is the fraction of passing cases.
func (s Score) PassRate() float64 {
	if s.Cases == 0 {
		return 0
	}
	return float64(s.Passed) / float64(s.Cases)
}

// Improved reports whether the candidate is strictly better than the baseline.
func (s Score) Improved(baseline Score) bool {
	return s.PassRate() > baseline.PassRate()
}

// Runner executes one case under the candidate lesson.
type Runner interface {
	// Run executes the question with the given lesson applied, returning the
	// answer and whether it carried citations.
	Run(ctx context.Context, lesson string, question string) (answer string, cited bool, err error)
}

// Evaluate replays every case twice: once without the lesson (baseline) and
// once with it.
func Evaluate(ctx context.Context, runner Runner, proposal propose.Proposal, cases []Case) (baseline, candidate Score, err error) {
	if runner == nil {
		return Score{}, Score{}, fmt.Errorf("evaluate: runner is required")
	}
	if len(cases) == 0 {
		return Score{}, Score{}, fmt.Errorf("evaluate: no cases to replay")
	}

	lesson := proposal.Title + ". " + proposal.Body
	for _, item := range cases {
		baseAnswer, baseCited, err := runner.Run(ctx, "", item.Question)
		if err != nil {
			return Score{}, Score{}, fmt.Errorf("evaluate: baseline %s: %w", item.RunID, err)
		}
		baseline.Cases++
		if passes(item, baseAnswer, baseCited) {
			baseline.Passed++
		}

		candAnswer, candCited, err := runner.Run(ctx, lesson, item.Question)
		if err != nil {
			return Score{}, Score{}, fmt.Errorf("evaluate: candidate %s: %w", item.RunID, err)
		}
		candidate.Cases++
		if passes(item, candAnswer, candCited) {
			candidate.Passed++
		}
	}
	return baseline, candidate, nil
}

func passes(item Case, answer string, cited bool) bool {
	if item.MustCite && !cited {
		return false
	}
	for _, needle := range item.MustMention {
		if needle == "" {
			continue
		}
		if !strings.Contains(answer, needle) {
			return false
		}
	}
	return true
}
