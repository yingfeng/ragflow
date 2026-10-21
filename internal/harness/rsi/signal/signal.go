// Package signal collects the raw evidence a self-improvement cycle starts
// from: how a run ended, why it was blocked, and what a human said about it.
package signal

import "strings"

// Outcome is the terminal shape of a run, mirrored here so this package stays
// independent of the loop implementation.
type Outcome struct {
	RunID         string
	TenantID      string
	AgentID       string
	Status        string // done | blocked | cancelled | failed | paused
	Reason        string
	FinalText     string
	Citations     int
	EvidenceCount int
	Rounds        int
}

// Feedback is one human signal attached to a run.
type Feedback struct {
	Rating  int
	Comment string
	Labels  []string
}

// Signal is one self-improvement input.
type Signal struct {
	Outcome
	Feedback Feedback
}

// Failed reports whether the run ended without a usable answer.
func (s Signal) Failed() bool {
	if s.Status == "failed" || s.Status == "blocked" || s.Status == "cancelled" {
		return true
	}
	return strings.TrimSpace(s.FinalText) == ""
}

// Negative reports whether a human judged the run badly.
func (s Signal) Negative() bool {
	if s.Feedback.Rating > 0 && s.Feedback.Rating <= 2 {
		return true
	}
	for _, label := range s.Feedback.Labels {
		switch strings.ToLower(strings.TrimSpace(label)) {
		case "wrong", "incomplete", "unsupported", "hallucination":
			return true
		}
	}
	return false
}

// WorthLearning reports whether the signal should enter the RSI pipeline.
func (s Signal) WorthLearning() bool {
	return s.Failed() || s.Negative()
}

// Collect keeps the signals that are worth learning from.
func Collect(signals []Signal) []Signal {
	out := make([]Signal, 0, len(signals))
	for _, item := range signals {
		if item.WorthLearning() {
			out = append(out, item)
		}
	}
	return out
}
