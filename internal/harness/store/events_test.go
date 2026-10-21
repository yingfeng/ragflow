package store

import (
	"context"
	"testing"

	"github.com/glebarez/sqlite"
	"gorm.io/gorm"

	"ragflow/internal/entity"
	"ragflow/internal/harness/events"
)

func newTestEventStore(t *testing.T) *EventStore {
	t.Helper()
	db, err := gorm.Open(sqlite.Open("file:"+t.Name()+"?mode=memory&cache=shared"), &gorm.Config{})
	if err != nil {
		t.Fatalf("open sqlite: %v", err)
	}
	if err := db.AutoMigrate(&entity.HarnessAgentEvent{}); err != nil {
		t.Fatalf("migrate: %v", err)
	}
	return NewEventStore(db)
}

func recordRun(t *testing.T, store *EventStore, recorder *events.EventRecorder) {
	t.Helper()
	ctx := context.Background()
	recorder.RecordToolCall(ctx, "fs_ls", map[string]any{"path": "/"}, "[]", 3, 0, "")
	recorder.RecordVerdict(ctx, "continue", "evidence pending", 1, 0)
	if got := recorder.Failures(); got != 0 {
		t.Fatalf("recorder failed %d times, want 0", got)
	}
}

func TestEventStoreRoundTripKeepsLayerAndCausalChain(t *testing.T) {
	store := newTestEventStore(t)
	recorder := events.NewEventRecorder(store, events.WithTraceID("run-1"), events.WithThreadID("run-1"))
	recordRun(t, store, recorder)

	items, err := store.Range(context.Background(), 0, 0, events.EventFilter{TraceID: "run-1"})
	if err != nil {
		t.Fatalf("range: %v", err)
	}
	if len(items) != 3 {
		t.Fatalf("expected 3 events, got %d", len(items))
	}
	if items[0].Layer != events.LayerAudit {
		t.Errorf("tool event layer = %s, want %s", items[0].Layer, events.LayerAudit)
	}
	if items[2].Layer != events.LayerTrace {
		t.Errorf("verdict layer = %s, want %s", items[2].Layer, events.LayerTrace)
	}
	if items[1].ParentID != items[0].ID {
		t.Errorf("tool result parent = %s, want %s", items[1].ParentID, items[0].ID)
	}
	if len(items[1].CausedBy) != 1 || items[1].CausedBy[0] != items[0].ID {
		t.Errorf("tool result caused by %v, want [%s]", items[1].CausedBy, items[0].ID)
	}
	if len(items[0].Metadata) != 1 || items[0].Metadata["tool"] != "fs_ls" {
		t.Errorf("metadata = %v, want tool=fs_ls", items[0].Metadata)
	}
	if items[1].Payload == nil {
		t.Error("tool result lost its payload")
	}
}

func TestEventStoreFiltersByLayerAndTrace(t *testing.T) {
	store := newTestEventStore(t)
	ctx := context.Background()

	first := events.NewEventRecorder(store, events.WithTraceID("run-a"))
	recordRun(t, store, first)
	second := events.NewEventRecorder(store, events.WithTraceID("run-b"))
	recordRun(t, store, second)

	auditOnly, err := store.Range(ctx, 0, 0, events.EventFilter{TraceID: "run-a", Layers: []events.Layer{events.LayerAudit}})
	if err != nil {
		t.Fatalf("range by layer: %v", err)
	}
	if len(auditOnly) != 2 {
		t.Fatalf("expected the tool start/result pair, got %d", len(auditOnly))
	}
	for _, item := range auditOnly {
		if item.Layer != events.LayerAudit {
			t.Errorf("unexpected layer %s", item.Layer)
		}
	}

	verdicts, err := store.Range(ctx, 0, 0, events.EventFilter{TraceID: "run-b", Types: []events.EventType{events.EventVerdict}})
	if err != nil {
		t.Fatalf("range by type: %v", err)
	}
	if len(verdicts) != 1 {
		t.Fatalf("expected one verdict, got %d", len(verdicts))
	}
}

func TestEventStoreGetSeekAndLength(t *testing.T) {
	store := newTestEventStore(t)
	ctx := context.Background()
	recorder := events.NewEventRecorder(store, events.WithTraceID("run-c"))
	recordRun(t, store, recorder)

	total, err := store.Length(ctx)
	if err != nil {
		t.Fatalf("length: %v", err)
	}
	if total != 3 {
		t.Fatalf("length = %d, want 3", total)
	}

	iterator, err := store.Seek(ctx, 2)
	if err != nil {
		t.Fatalf("seek: %v", err)
	}
	seen := 0
	for {
		if _, ok := iterator.Next(ctx); !ok {
			break
		}
		seen++
	}
	if err := iterator.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	if seen != 2 {
		t.Fatalf("seek(2) returned %d events, want 2", seen)
	}

	items, err := store.Range(ctx, 0, 0, events.EventFilter{TraceID: "run-c"})
	if err != nil {
		t.Fatalf("range: %v", err)
	}
	found, err := store.Get(ctx, items[0].ID)
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	if found == nil || found.Type != items[0].Type {
		t.Fatalf("get returned %v, want event %s", found, items[0].ID)
	}

	missing, err := store.Get(ctx, "evt-missing")
	if err != nil {
		t.Fatalf("get missing: %v", err)
	}
	if missing != nil {
		t.Fatalf("get missing returned %v, want nil", missing)
	}
}
