// Package library is the lesson store's harness-facing view: what to inject into
// a prompt and how a promoted proposal becomes an entry.
package library

import (
	"context"
	"fmt"
	"strings"

	"ragflow/internal/harness/rsi/propose"
)

// Lesson is one promoted experience entry.
type Lesson struct {
	ID            string
	Kind          string
	Title         string
	Body          string
	EvidenceCount int
}

// Store reads and writes lessons.
type Store interface {
	Save(ctx context.Context, lesson Lesson) error
	Active(ctx context.Context, tenantID, agentID string, limit int) ([]Lesson, error)
}

// FromProposal builds a lesson from an admitted proposal.
func FromProposal(proposal propose.Proposal, evidence int) Lesson {
	return Lesson{
		Kind:          proposal.Kind,
		Title:         proposal.Title,
		Body:          proposal.Body,
		EvidenceCount: evidence,
	}
}

// PromptLines renders the lessons as bullet lines for the system prompt.
//
// Only the title and body are rendered: the internal kind, id and evidence count
// are bookkeeping and would read as noise to the model.
func PromptLines(lessons []Lesson) []string {
	out := make([]string, 0, len(lessons))
	seen := make(map[string]struct{}, len(lessons))
	for _, lesson := range lessons {
		key := strings.TrimSpace(lesson.Title)
		if key == "" {
			continue
		}
		if _, dup := seen[key]; dup {
			continue
		}
		seen[key] = struct{}{}
		out = append(out, fmt.Sprintf("- %s: %s", strings.TrimSpace(lesson.Title), strings.TrimSpace(lesson.Body)))
	}
	return out
}
