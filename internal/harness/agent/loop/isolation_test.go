package loop

import (
	"context"
	"testing"

	"ragflow/internal/harness/toolkit/spill"
)

func TestFreshRunContextDetachesParentState(t *testing.T) {
	parentStore := spill.New(4)
	parentHandle := parentStore.Put("parent payload")
	parentInbox := NewInbox()
	parentInbox.Push("a message for the parent")

	parentStream := 0
	parentProgress := 0
	parent := context.Background()
	parent = WithInbox(parent, parentInbox)
	parent = spill.WithStore(parent, parentStore)
	parent = WithDeltaSink(parent, func(string) { parentStream++ })
	parent = WithReasoningSink(parent, func(string) { parentStream++ })
	parent = WithProgressSink(parent, func(Progress) { parentProgress++ })

	child := FreshRunContext(parent)

	if DeltaSinkFromContext(child) != nil {
		t.Error("the child inherited the parent's answer stream")
	}
	if ReasoningSinkFromContext(child) != nil {
		t.Error("the child inherited the parent's reasoning stream")
	}
	if ProgressSinkFromContext(child) != nil {
		t.Error("the child inherited the parent's progress stream")
	}

	childInbox := InboxFrom(child)
	if childInbox == nil || childInbox == parentInbox {
		t.Fatalf("child inbox = %p, parent = %p: the child must own its inbox", childInbox, parentInbox)
	}
	if messages := childInbox.Drain(); len(messages) != 0 {
		t.Errorf("the child drained the parent's steer messages: %v", messages)
	}
	if messages := parentInbox.Drain(); len(messages) != 1 || messages[0] != "a message for the parent" {
		t.Errorf("parent inbox = %v, want its own message intact", messages)
	}

	childStore := spill.StoreFrom(child)
	if childStore == nil || childStore == parentStore {
		t.Fatalf("child spill store = %p, parent = %p: the child must own its store", childStore, parentStore)
	}
	if _, ok := childStore.Get(parentHandle); ok {
		t.Error("the child's spill store resolves a handle the parent minted")
	}
}

func TestFreshRunContextKeepsCancellation(t *testing.T) {
	parent, cancel := context.WithCancel(context.Background())
	child := FreshRunContext(parent)
	cancel()

	select {
	case <-child.Done():
	default:
		t.Fatal("cancelling the parent left the child running")
	}
}
