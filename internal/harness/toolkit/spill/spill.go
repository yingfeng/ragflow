// Package spill keeps oversized tool output out of the context window.
//
// A tool result that exceeds the inline threshold is stored in the run's spill
// store and replaced by a short preview plus a handle. The model can read the
// full text back with read_spill when it actually needs it, so a single verbose
// tool call no longer evicts the rest of the conversation.
package spill

import (
	"context"
	"fmt"
	"strings"
	"sync"

	"ragflow/internal/harness/agent/model"
	"ragflow/internal/harness/toolkit"
)

// DefaultInlineBytes is the threshold above which output is spilled.
const DefaultInlineBytes = 8 << 10

// previewBytes is how much of the spilled text stays inline.
const previewBytes = 2 << 10

// Store holds spilled payloads for one run.
type Store struct {
	mu      sync.Mutex
	items   map[string]string
	seq     int
	maxItem int
}

// New builds a store bounded to maxItems entries, evicting the oldest.
func New(maxItems int) *Store {
	if maxItems <= 0 {
		maxItems = 32
	}
	return &Store{items: make(map[string]string, maxItems), maxItem: maxItems}
}

// Put stores content and returns its handle plus an inline preview.
func (s *Store) Put(tool, content string) (id, preview string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.seq++
	id = fmt.Sprintf("spill-%d", s.seq)
	s.items[id] = content

	// Bounded store: dropping the oldest handle keeps memory flat. A model that
	// needed that output would have read it already.
	if len(s.items) > s.maxItem {
		oldest := fmt.Sprintf("spill-%d", s.seq-s.maxItem)
		delete(s.items, oldest)
	}

	preview = truncateRunes(content, previewBytes)
	return id, preview
}

// Get returns a spilled payload.
func (s *Store) Get(id string) (string, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	content, ok := s.items[id]
	return content, ok
}

type storeKey struct{}

// WithStore binds a run's spill store to the context.
func WithStore(ctx context.Context, store *Store) context.Context {
	if store == nil {
		return ctx
	}
	return context.WithValue(ctx, storeKey{}, store)
}

// StoreFrom reads the run's spill store.
func StoreFrom(ctx context.Context) *Store {
	store, _ := ctx.Value(storeKey{}).(*Store)
	return store
}

// MaybeSpill replaces content with a preview when it exceeds threshold. Without
// a store on the context the content is only truncated, never silently lost.
func MaybeSpill(ctx context.Context, tool, content string, threshold int) string {
	if threshold <= 0 {
		threshold = DefaultInlineBytes
	}
	if len(content) <= threshold {
		return content
	}

	store := StoreFrom(ctx)
	if store == nil {
		return truncateRunes(content, threshold) + "\n[output truncated]"
	}

	id, preview := store.Put(tool, content)
	return fmt.Sprintf("%s\n\n[output truncated: %d bytes total, stored as %s — call read_spill with id %q to read the rest]",
		preview, len(content), id, id)
}

// Capability exposes read_spill.
type Capability struct{}

// NewCapability builds the capability.
func NewCapability() *Capability { return &Capability{} }

// Name implements toolkit.Capability.
func (*Capability) Name() string { return "spill" }

// Tools implements toolkit.Capability.
func (*Capability) Tools() []model.ToolSpec {
	return []model.ToolSpec{{
		Name:        "read_spill",
		Description: "Read the full output of a tool result that was truncated in the conversation. Use the handle shown in the truncation notice.",
		Parameters: toolkit.ObjectSchema(map[string]any{
			"id": map[string]any{"type": "string", "description": "Spill handle, e.g. spill-3."},
		}, "id"),
	}}
}

type readArgs struct {
	ID string `json:"id"`
}

// Invoke implements toolkit.Capability.
func (c *Capability) Invoke(ctx context.Context, call toolkit.Call) (toolkit.Result, error) {
	if call.Tool != "read_spill" {
		return toolkit.Result{}, fmt.Errorf("spill: unknown tool %q", call.Tool)
	}
	args, err := toolkit.DecodeArgs[readArgs](call.Arguments)
	if err != nil {
		return toolkit.Result{}, err
	}
	store := StoreFrom(ctx)
	if store == nil {
		return toolkit.Result{}, fmt.Errorf("no spilled output in this run")
	}
	content, ok := store.Get(strings.TrimSpace(args.ID))
	if !ok {
		return toolkit.Result{}, fmt.Errorf("spill handle %q not found (it may have expired)", args.ID)
	}
	return toolkit.Result{Content: content}, nil
}

func truncateRunes(value string, limit int) string {
	if len(value) <= limit {
		return value
	}
	// Cut on a rune boundary so the preview is valid UTF-8.
	cut := limit
	for cut > 0 && (value[cut]&0xC0) == 0x80 {
		cut--
	}
	return value[:cut]
}
