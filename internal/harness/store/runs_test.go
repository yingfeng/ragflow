package store

import (
	"context"
	"testing"
	"time"

	"github.com/glebarez/sqlite"
	"gorm.io/gorm"

	"ragflow/internal/entity"
)

func newTestStore(t *testing.T) *RunStore {
	t.Helper()
	db, err := gorm.Open(sqlite.Open("file:"+t.Name()+"?mode=memory&cache=shared"), &gorm.Config{})
	if err != nil {
		t.Fatalf("open sqlite: %v", err)
	}
	if err := db.AutoMigrate(&entity.HarnessAgentRun{}, &entity.HarnessAgentStep{}); err != nil {
		t.Fatalf("migrate: %v", err)
	}
	return NewRunStore(db)
}

func createRun(t *testing.T, store *RunStore, id, status string, leaseUntil *time.Time) {
	t.Helper()
	record := &entity.HarnessAgentRun{
		ID:         id,
		AgentID:    "agent-1",
		TenantID:   "tenant-1",
		Status:     status,
		LeaseOwner: "worker-a",
		LeaseUntil: leaseUntil,
	}
	if err := store.CreateRun(context.Background(), record); err != nil {
		t.Fatalf("create run %s: %v", id, err)
	}
}

func TestListRecoverableReturnsRunsWithoutLiveLease(t *testing.T) {
	store := newTestStore(t)
	expired := time.Now().Add(-time.Minute)
	future := time.Now().Add(time.Minute)

	createRun(t, store, "crashed", "running", &expired)
	createRun(t, store, "healthy", "running", &future)
	createRun(t, store, "finished", "done", nil)

	runs, err := store.ListRecoverable(context.Background(), 10)
	if err != nil {
		t.Fatalf("ListRecoverable: %v", err)
	}
	if len(runs) != 1 || runs[0].ID != "crashed" {
		t.Fatalf("recoverable = %+v, want only the crashed run", runs)
	}
}

func TestReclaimLeaseMakesRunClaimable(t *testing.T) {
	store := newTestStore(t)
	expired := time.Now().Add(-time.Minute)
	createRun(t, store, "crashed", "running", &expired)

	if err := store.ReclaimLease(context.Background(), "crashed"); err != nil {
		t.Fatalf("ReclaimLease: %v", err)
	}
	if ok, err := store.AcquireLease(context.Background(), "crashed", "worker-b", time.Now().Add(time.Minute)); err != nil || !ok {
		t.Fatalf("claim after reclaim = %v, %v", ok, err)
	}
}

func TestMarkFailedClosesRun(t *testing.T) {
	store := newTestStore(t)
	expired := time.Now().Add(-time.Minute)
	createRun(t, store, "doomed", "running", &expired)

	if err := store.MarkFailed(context.Background(), "doomed", "no checkpoint"); err != nil {
		t.Fatalf("MarkFailed: %v", err)
	}

	var record entity.HarnessAgentRun
	if err := store.db.First(&record, "id = ?", "doomed").Error; err != nil {
		t.Fatalf("load: %v", err)
	}
	if record.Status != "failed" || record.Error != "no checkpoint" {
		t.Fatalf("record = %+v, want failed with the reason", record)
	}
	if record.LeaseOwner != "" || record.LeaseUntil != nil {
		t.Fatalf("lease was not cleared: %+v", record)
	}
}

func TestFinishRunPersistsCost(t *testing.T) {
	store := newTestStore(t)
	createRun(t, store, "priced", "running", nil)

	if err := store.FinishRun(context.Background(), "priced", "done", "answer", "", 100, 50, 0.0123); err != nil {
		t.Fatalf("FinishRun: %v", err)
	}

	var record entity.HarnessAgentRun
	if err := store.db.First(&record, "id = ?", "priced").Error; err != nil {
		t.Fatalf("load: %v", err)
	}
	if record.TokensIn != 100 || record.TokensOut != 50 {
		t.Fatalf("tokens = %d/%d", record.TokensIn, record.TokensOut)
	}
	if record.Cost != 0.0123 {
		t.Fatalf("cost = %v, want 0.0123", record.Cost)
	}
}
