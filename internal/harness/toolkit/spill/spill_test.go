package spill

import (
	"context"
	"strconv"
	"strings"
	"testing"

	"ragflow/internal/harness/toolkit"
)

func ctxWithStore() (context.Context, *Store) {
	store := New(4)
	return WithStore(context.Background(), store), store
}

func TestMaybeSpillKeepsHandleAndPreview(t *testing.T) {
	ctx, _ := ctxWithStore()
	content := strings.Repeat("a", DefaultInlineBytes*2)

	out := MaybeSpill(ctx, content, DefaultInlineBytes)
	if !strings.Contains(out, "read_spill") {
		t.Fatalf("spilled output carries no handle: %q", out[:80])
	}
	preview := strings.SplitN(out, "\n\n", 2)[0]
	if len(preview) > DefaultInlineBytes {
		t.Fatalf("preview = %d bytes, want at most the threshold", len(preview))
	}
	if len(preview) < DefaultInlineBytes/2 {
		t.Errorf("preview = %d bytes, want a usable share of the threshold", len(preview))
	}
}

// Reading a spilled payload must return text, not a handle to itself: the loop
// spills oversized results again, so an unbounded read would hand the model a
// new handle every time and it could never reach the content.
func TestReadSpillReturnsBoundedInlineWindow(t *testing.T) {
	ctx, store := ctxWithStore()
	payload := strings.Repeat("关羽斩将", ReadWindowBytes)
	id := store.Put(payload)

	capability := NewCapability()
	first, err := capability.Invoke(ctx, toolkit.Call{Tool: "read_spill", Arguments: `{"id":"` + id + `"}`})
	if err != nil {
		t.Fatalf("read_spill: %v", err)
	}
	if !first.Inline {
		t.Error("read_spill result must stay inline, otherwise the loop spills it again")
	}
	if len(first.Content) > ReadWindowBytes+256 {
		t.Errorf("window = %d bytes, want a bounded window plus the continuation notice", len(first.Content))
	}
	if !strings.Contains(first.Content, "offset=") {
		t.Error("an incomplete window must report the next offset")
	}

	// The continuation reaches the end, which is where the notice stops.
	second, err := capability.Invoke(ctx, toolkit.Call{Tool: "read_spill", Arguments: `{"id":"` + id + `","offset":` + strconv.Itoa(ReadWindowBytes) + `}`})
	if err != nil {
		t.Fatalf("read_spill offset: %v", err)
	}
	if second.Content == first.Content {
		t.Error("offset was ignored: both windows are identical")
	}
}

func TestReadSpillUnknownHandle(t *testing.T) {
	ctx, _ := ctxWithStore()
	_, err := NewCapability().Invoke(ctx, toolkit.Call{Tool: "read_spill", Arguments: `{"id":"spill-99"}`})
	if err == nil {
		t.Fatal("expected an error for an unknown handle")
	}
}
