// Package attribute assigns a cause to a learning signal. Attribution is a
// closed classification, not free text: an unclassified failure cannot be
// evaluated, and an unevaluated lesson must not reach the prompt.
package attribute

import (
	"strings"

	"ragflow/internal/harness/rsi/signal"
)

// Cause is the classified reason of a bad run.
type Cause string

const (
	// CauseRetrievalShortage: the loop searched but never gathered evidence.
	CauseRetrievalShortage Cause = "retrieval_shortage"
	// CauseToolMisuse: tools failed or were called with unusable arguments.
	CauseToolMisuse Cause = "tool_misuse"
	// CauseCitationMissing: an answer was produced without supporting evidence.
	CauseCitationMissing Cause = "citation_missing"
	// CauseBudgetExhausted: the run hit its round or token budget.
	CauseBudgetExhausted Cause = "budget_exhausted"
	// CauseEmptyAnswer: the model ended its turn with nothing.
	CauseEmptyAnswer Cause = "empty_answer"
	// CauseUnknown: no rule matched.
	CauseUnknown Cause = "unknown"
)

// Classify maps a signal onto one cause. The order matters: a run that was
// blocked for citations is a citation problem even if it also used few tools.
func Classify(s signal.Signal) Cause {
	reason := strings.ToLower(s.Reason)

	switch {
	case strings.Contains(reason, "citation"):
		return CauseCitationMissing
	case strings.Contains(reason, "budget"), strings.Contains(reason, "round"):
		return CauseBudgetExhausted
	case strings.Contains(reason, "no final text"), strings.Contains(reason, "empty"):
		return CauseEmptyAnswer
	case strings.Contains(reason, "unknown tool"), strings.Contains(reason, "scope"), strings.Contains(reason, "arguments"):
		return CauseToolMisuse
	}

	switch {
	case s.EvidenceCount == 0 && s.Failed():
		return CauseRetrievalShortage
	case strings.TrimSpace(s.FinalText) != "" && s.Citations == 0:
		return CauseCitationMissing
	case strings.TrimSpace(s.FinalText) == "":
		return CauseEmptyAnswer
	}
	return CauseUnknown
}
