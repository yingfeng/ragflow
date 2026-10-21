package rsi

import (
	"context"
	"testing"

	"ragflow/internal/harness/rsi/attribute"
	"ragflow/internal/harness/rsi/evaluate"
	"ragflow/internal/harness/rsi/library"
	"ragflow/internal/harness/rsi/signal"
)

// improvingRunner answers better once a lesson is applied.
type improvingRunner struct{}

func (improvingRunner) Run(_ context.Context, lesson, _ string) (string, bool, error) {
	if lesson == "" {
		return "unclear", false, nil
	}
	return "the answer is X", true, nil
}

// flatRunner ignores the lesson, so nothing can be promoted.
type flatRunner struct{}

func (flatRunner) Run(_ context.Context, lesson, _ string) (string, bool, error) {
	return "the answer is X", true, nil
}

type memoryLibrary struct{ lessons []library.Lesson }

func (m *memoryLibrary) Save(_ context.Context, lesson library.Lesson) error {
	m.lessons = append(m.lessons, lesson)
	return nil
}

func (m *memoryLibrary) Active(context.Context, string, string, int) ([]library.Lesson, error) {
	return m.lessons, nil
}

func failedSignals(count int) []signal.Signal {
	out := make([]signal.Signal, 0, count)
	for i := 0; i < count; i++ {
		out = append(out, signal.Signal{
			Outcome: signal.Outcome{
				RunID:     "run",
				TenantID:  "tenant",
				Status:    "blocked",
				Reason:    "no evidence gathered",
				Rounds:    3,
				FinalText: "",
			},
		})
	}
	return out
}

func TestCycleAdmitsMeasuredImprovement(t *testing.T) {
	store := &memoryLibrary{}
	cycle := New(Config{
		Library: store,
		Runner:  improvingRunner{},
		Cases:   []evaluate.Case{{RunID: "c1", Question: "q", MustMention: []string{"X"}, MustCite: true}},
	})

	results, err := cycle.Learn(context.Background(), failedSignals(2))
	if err != nil {
		t.Fatalf("Learn: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("results = %d, want 1", len(results))
	}
	result := results[0]
	if result.Cause != attribute.CauseRetrievalShortage {
		t.Fatalf("cause = %q, want retrieval_shortage", result.Cause)
	}
	if !result.Decision.Admit {
		t.Fatalf("proposal rejected: %s", result.Decision.Reason)
	}
	if result.Lesson == nil || len(store.lessons) != 1 {
		t.Fatalf("lesson not stored: %+v", result.Lesson)
	}
}

func TestCycleRejectsWithoutEvidence(t *testing.T) {
	store := &memoryLibrary{}
	cycle := New(Config{
		Library: store,
		Runner:  improvingRunner{},
		Cases:   []evaluate.Case{{RunID: "c1", Question: "q", MustMention: []string{"X"}}},
	})

	results, err := cycle.Learn(context.Background(), failedSignals(1))
	if err != nil {
		t.Fatalf("Learn: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("results = %d, want 1", len(results))
	}
	if results[0].Decision.Admit {
		t.Fatal("a single failing run must not promote a lesson")
	}
	if len(store.lessons) != 0 {
		t.Fatalf("lessons = %d, want 0", len(store.lessons))
	}
}

func TestCycleRejectsUnmeasuredImprovement(t *testing.T) {
	store := &memoryLibrary{}
	cycle := New(Config{
		Library: store,
		Runner:  flatRunner{},
		Cases:   []evaluate.Case{{RunID: "c1", Question: "q", MustMention: []string{"X"}}},
	})

	results, err := cycle.Learn(context.Background(), failedSignals(3))
	if err != nil {
		t.Fatalf("Learn: %v", err)
	}
	if len(results) != 1 || results[0].Decision.Admit {
		t.Fatalf("a lesson with no measured gain was admitted: %+v", results)
	}
}

func TestCycleIgnoresHealthyRuns(t *testing.T) {
	cycle := New(Config{})
	results, err := cycle.Learn(context.Background(), []signal.Signal{{
		Outcome: signal.Outcome{RunID: "ok", Status: "done", FinalText: "answer", Citations: 2},
	}})
	if err != nil {
		t.Fatalf("Learn: %v", err)
	}
	if len(results) != 0 {
		t.Fatalf("results = %d, want 0 for a healthy run", len(results))
	}
}
