package loop

import (
	"context"
	"strings"
	"sync"
)

// Inbox carries messages injected while a run is executing.
//
// Steering is not cancellation: the run keeps its state, budget and evidence,
// and sees the new message at the start of the next round. The inbox lives
// outside State so nothing has to be serialized into a checkpoint.
type Inbox struct {
	mu    sync.Mutex
	items []string
}

// NewInbox builds an empty inbox.
func NewInbox() *Inbox { return &Inbox{} }

// Push queues a message. Empty messages are ignored.
func (i *Inbox) Push(message string) bool {
	if i == nil {
		return false
	}
	message = strings.TrimSpace(message)
	if message == "" {
		return false
	}
	i.mu.Lock()
	i.items = append(i.items, message)
	i.mu.Unlock()
	return true
}

// Drain removes and returns the queued messages.
func (i *Inbox) Drain() []string {
	if i == nil {
		return nil
	}
	i.mu.Lock()
	defer i.mu.Unlock()
	if len(i.items) == 0 {
		return nil
	}
	out := i.items
	i.items = nil
	return out
}

type inboxKey struct{}

// WithInbox binds an inbox to the run context.
func WithInbox(ctx context.Context, inbox *Inbox) context.Context {
	if inbox == nil {
		return ctx
	}
	return context.WithValue(ctx, inboxKey{}, inbox)
}

// InboxFrom reads the run's inbox.
func InboxFrom(ctx context.Context) *Inbox {
	inbox, _ := ctx.Value(inboxKey{}).(*Inbox)
	return inbox
}
