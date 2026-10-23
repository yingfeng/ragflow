package subagent

import (
	"context"
	"sync"
	"testing"

	"ragflow/internal/harness/agent/definition"
	"ragflow/internal/harness/agent/loop"
	"ragflow/internal/harness/agent/model"
	"ragflow/internal/harness/events"
	"ragflow/internal/harness/toolkit"
	"ragflow/internal/harness/toolkit/spill"
	capagent "ragflow/internal/harness/toolkit/subagent"
)

// answeringModel ends a child run in one round: the test is about what the child
// inherits, not about how it reasons.
type answeringModel struct {
	mu    sync.Mutex
	calls int
}

func (m *answeringModel) Complete(_ context.Context, _ model.Request) (model.Response, error) {
	m.mu.Lock()
	m.calls++
	m.mu.Unlock()
	return model.Response{
		Message:    model.Message{Role: model.RoleAssistant, Content: "child answer"},
		StopReason: "stop",
	}, nil
}

func TestSpawnIsolatesChildRunState(t *testing.T) {
	mdl := &answeringModel{}
	registry := toolkit.NewRegistry()

	var mu sync.Mutex
	var steps []loop.StepRecord
	deps := loop.Deps{
		Model:        mdl,
		Registry:     registry,
		SystemPrompt: "you are a child test agent",
		Sink: func(record loop.StepRecord) {
			mu.Lock()
			steps = append(steps, record)
			mu.Unlock()
		},
	}
	spawner, err := New(deps, definition.PolicyConfig{}, 0)
	if err != nil {
		t.Fatalf("spawner: %v", err)
	}

	// The parent's context carries everything a child must not inherit.
	parentInbox := loop.NewInbox()
	parentInbox.Push("a message for the parent")
	parentStore := spill.New(2)
	parentHandle := parentStore.Put("parent payload")
	var streamed []string

	parent := context.Background()
	parent = loop.WithInbox(parent, parentInbox)
	parent = spill.WithStore(parent, parentStore)
	parent = loop.WithDeltaSink(parent, func(text string) { streamed = append(streamed, text) })
	parent = loop.WithReasoningSink(parent, func(text string) { streamed = append(streamed, text) })
	parent = loop.WithProgressSink(parent, func(progress loop.Progress) {
		streamed = append(streamed, progress.Tool)
	})

	outcome, err := spawner.Spawn(parent, capagent.Request{Prompt: "child: task", Depth: 1})
	if err != nil {
		t.Fatalf("spawn: %v", err)
	}
	if outcome.Output != "child answer" {
		t.Fatalf("child output = %q", outcome.Output)
	}
	if len(streamed) != 0 {
		t.Fatalf("the child wrote into the parent's stream: %v", streamed)
	}
	if messages := parentInbox.Drain(); len(messages) != 1 || messages[0] != "a message for the parent" {
		t.Fatalf("the child consumed the parent's steer messages: %v", messages)
	}
	if _, ok := parentStore.Get(parentHandle); !ok {
		t.Fatal("the parent's own spill handle was lost")
	}

	// A child's rounds are run steps like its parent's: they go to the sink the
	// spawner was built with, which is what the run service persists.
	mu.Lock()
	recorded := append([]loop.StepRecord(nil), steps...)
	mu.Unlock()
	if len(recorded) == 0 {
		t.Fatal("no child step reached the sink")
	}
	for _, record := range recorded {
		if record.RunID != outcome.RunID {
			t.Fatalf("step %q belongs to run %q, want the child run %q", record.Phase, record.RunID, outcome.RunID)
		}
	}
}

func TestSpawnRecordsChildEventsUnderItsOwnTrace(t *testing.T) {
	log := events.NewMemoryEventStore()
	spawner, err := New(loop.Deps{
		Model:        &answeringModel{},
		Registry:     toolkit.NewRegistry(),
		SystemPrompt: "you are a child test agent",
	}, definition.PolicyConfig{}, 0)
	if err != nil {
		t.Fatalf("spawner: %v", err)
	}
	spawner = spawner.WithAudit(log, nil)

	// The parent already has a recorder; the child must not write into it.
	parent := events.ContextWithRecorder(context.Background(),
		events.NewEventRecorder(log, events.WithTraceID("parent-run"), events.WithThreadID("parent-run")))

	outcome, err := spawner.Spawn(parent, capagent.Request{Prompt: "child: task", Depth: 1})
	if err != nil {
		t.Fatalf("spawn: %v", err)
	}

	childEvents, err := log.Range(context.Background(), 0, 0, events.EventFilter{TraceID: outcome.RunID})
	if err != nil {
		t.Fatalf("read child trace: %v", err)
	}
	if len(childEvents) == 0 {
		t.Fatal("the child recorded no events under its own run id")
	}
	parentEvents, err := log.Range(context.Background(), 0, 0, events.EventFilter{TraceID: "parent-run"})
	if err != nil {
		t.Fatalf("read parent trace: %v", err)
	}
	if len(parentEvents) != 0 {
		t.Fatalf("the child wrote %d events into the parent's trace", len(parentEvents))
	}
}
