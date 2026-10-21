// Package spill keeps oversized tool output out of the context window.
//
// A tool result that exceeds the inline threshold is stored in the run's spill
// store and replaced by a preview plus a handle. The model can read the full
// text back with read_spill when it actually needs it, so a single verbose
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

// The inline preview takes this share of the threshold. It is derived from the
// threshold rather than fixed so a budget the caller already tightened stays
// respected.
const (
	previewNum   = 2
	previewDenom = 3
)

// ReadWindowBytes bounds one read_spill response. Reading a spilled payload is
// a windowed read, not a dump: a 200 KB payload returned whole would evict the
// transcript it exists to protect.
const ReadWindowBytes = 16 << 10

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
	return &Store{items: make(map[string]string), maxItem: maxItems}
}

// Put stores content and returns its handle.
func (s *Store) Put(content string) (id string) {
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

	return id
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
func MaybeSpill(ctx context.Context, content string, threshold int) string {
	if threshold <= 0 {
		threshold = DefaultInlineBytes
	}
	if len(content) <= threshold {
		return content
	}

	previewLimit := threshold * previewNum / previewDenom
	if previewLimit <= 0 || previewLimit > threshold {
		previewLimit = threshold
	}

	store := StoreFrom(ctx)
	if store == nil {
		return truncateRunes(content, previewLimit) + "\n[output truncated]"
	}

	id := store.Put(content)
	return fmt.Sprintf("%s\n\n[output truncated: %d bytes total, stored as %s — call read_spill with id %q (offset=0, limit=%d) to read it in windows]",
		truncateRunes(content, previewLimit), len(content), id, id, ReadWindowBytes)
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
		Name: "read_spill",
		Description: "Read a tool result that was truncated in the conversation, by handle. " +
			"Returns one window (default 16 KB) and reports the next offset; pass offset to continue. " +
			"Read the parts you need instead of the whole payload.",
		Parameters: toolkit.ObjectSchema(map[string]any{
			"id":     map[string]any{"type": "string", "description": "Spill handle, e.g. spill-3."},
			"offset": map[string]any{"type": "integer", "description": "Byte offset to start at (default 0). Use the next offset reported by the previous read."},
			"limit":  map[string]any{"type": "integer", "description": "Bytes to return, at most 16384 (default 16384)."},
		}, "id"),
	}}
}

type readArgs struct {
	ID     string `json:"id"`
	Offset int    `json:"offset,omitempty"`
	Limit  int    `json:"limit,omitempty"`
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
	window, next := windowOf(content, args.Offset, args.Limit)
	total := len(content)
	if next < total {
		window += fmt.Sprintf("\n\n[window ends at byte %d of %d — continue with offset=%d]", next, total, next)
	}
	// The window is already bounded; spilling it again would hand back a handle
	// to the payload the model is trying to read.
	return toolkit.Result{Content: window, Inline: true}, nil
}

// windowOf returns the readable slice starting at offset and the offset that
// follows it. Both ends are moved onto rune boundaries so a window never splits
// a multi-byte character.
func windowOf(content string, offset, limit int) (string, int) {
	total := len(content)
	if offset < 0 {
		offset = 0
	}
	if offset > total {
		offset = total
	}
	for offset > 0 && offset < total && content[offset]&0xC0 == 0x80 {
		offset++
	}
	if limit <= 0 || limit > ReadWindowBytes {
		limit = ReadWindowBytes
	}
	end := offset + limit
	if end > total {
		end = total
	}
	for end > offset && end < total && content[end]&0xC0 == 0x80 {
		end--
	}
	return content[offset:end], end
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
