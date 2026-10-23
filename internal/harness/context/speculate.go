package agentctx

import (
	"strings"
	"sync"

	"ragflow/internal/harness/agent/model"
)

// Speculation prepares the compaction summary before the round that needs it.
//
// Folding history costs one model call, and today that call happens inside the
// model request path: the round that crosses the budget pauses for the length of
// a summarization before it can send anything. The reference runtime runs the
// same work ahead of time in the background and adopts the result at the next
// checkpoint, which is what this type does.
//
// Adoption is by prefix, not by exact version: our loop appends to the history
// between the snapshot and the fold, so requiring the two to be identical would
// discard every candidate. A hint covers messages[:Covered], and history is
// append-only, so those messages are still the same messages when the fold
// happens; foldWithSummary folds to the covered prefix and keeps the rest
// verbatim.
type Speculator struct {
	builder *Builder

	mu       sync.Mutex
	inflight bool
	hint     *FoldHint
	adopted  int
	discards int
	done     chan struct{}
}

// FoldHint is a summary computed ahead of the round that needs it.
type FoldHint struct {
	// Covered is how many leading messages Summary replaces.
	Covered int
	// Summary is the briefing over messages[:Covered].
	Summary string
}

// speculativeGrowth is how much larger the prompt is expected to be by the time
// the next round sends it. Speculation starts when the current transcript is
// within this factor of the budget, which is roughly one round before the fold.
const speculativeGrowth = 1.15

// NewSpeculator returns a speculator for one run. A builder without a model
// summarizer gets a speculator too, but it never starts work: folding is then
// pure CPU and running it in the background saves nothing.
func NewSpeculator(builder *Builder) *Speculator {
	return &Speculator{builder: builder}
}

// Hint returns the prepared summary, if one is ready. The caller may use it for
// one fold; a stale hint that covers too little is ignored by the builder.
func (s *Speculator) Hint() *FoldHint {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.hint
}

// Adopted and Discarded report how the speculation went, for the trace.
func (s *Speculator) Adopted() int {
	if s == nil {
		return 0
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.adopted
}

// MaybeStart speculates when the transcript is close enough to the budget that
// the next round is likely to fold it. It never blocks: the summary is built in
// the background while the run keeps working.
func (s *Speculator) MaybeStart(system string, lessons []string, history []model.Message) {
	if s == nil || s.builder == nil || s.builder.cfg.Summarize == nil {
		return
	}
	cut := s.builder.speculativeCut(system, lessons, history)
	if cut <= 0 {
		return
	}

	s.mu.Lock()
	if s.inflight || (s.hint != nil && s.hint.Covered >= cut) {
		s.mu.Unlock()
		return
	}
	s.inflight = true
	done := make(chan struct{})
	s.done = done
	// Copy under the caller's goroutine: the run appends to its history while
	// this summary is being written.
	span := make([]model.Message, cut)
	copy(span, history[:cut])
	s.mu.Unlock()

	go func() {
		defer close(done)
		summary := s.builder.summarize(span)

		s.mu.Lock()
		defer s.mu.Unlock()
		s.inflight = false
		if trimmed := strings.TrimSpace(summary); trimmed == "" {
			s.discards++
			return
		}
		if s.hint != nil && s.hint.Covered >= cut {
			s.discards++
			return
		}
		s.hint = &FoldHint{Covered: cut, Summary: summary}
	}()
}

// NoteHintAdopted records that a fold consumed the hint, so the trace can tell a
// speculation that paid off from one that was thrown away.
func (s *Speculator) NoteHintAdopted() {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.adopted++
	s.hint = nil
}

// WaitForIdle blocks until no summary is being written. Tests use it; the run
// never does, because speculation must not delay a round.
func (s *Speculator) WaitForIdle() {
	if s == nil {
		return
	}
	for {
		s.mu.Lock()
		inflight := s.inflight
		done := s.done
		s.mu.Unlock()
		if !inflight || done == nil {
			return
		}
		<-done
	}
}

// speculativeCut is the prefix the next fold is expected to drop, or zero when
// no fold is close enough to be worth preparing.
func (b *Builder) speculativeCut(system string, lessons []string, history []model.Message) int {
	limit := b.limit() - b.tokens(b.renderSystemWithinBudget(system, lessons))
	if limit <= 0 {
		return 0
	}
	projected := int(float64(limit) / speculativeGrowth)
	if b.estimate(history) <= projected {
		return 0
	}
	return b.foldCut(history, projected)
}
