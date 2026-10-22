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

	reclaimed, err := store.ReclaimLease(context.Background(), "crashed")
	if err != nil || !reclaimed {
		t.Fatalf("ReclaimLease = %v, %v", reclaimed, err)
	}
	if ok, _, err := store.AcquireLease(context.Background(), "crashed", "worker-b", time.Now().Add(time.Minute)); err != nil || !ok {
		t.Fatalf("claim after reclaim = %v, %v", ok, err)
	}
}

// Two recovering workers must not both take the same run. A live lease is not
// reclaimable, so the second worker's attempt reports "not reclaimed" instead of
// wiping the lease the first worker just acquired.
func TestReclaimLeaseRefusesLiveLease(t *testing.T) {
	store := newTestStore(t)
	live := time.Now().Add(time.Minute)
	createRun(t, store, "healthy", "running", &live)

	reclaimed, err := store.ReclaimLease(context.Background(), "healthy")
	if err != nil {
		t.Fatalf("ReclaimLease: %v", err)
	}
	if reclaimed {
		t.Fatal("a live lease was reclaimed: two workers can now run the same run")
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

	applied, err := store.FinishRun(context.Background(), "priced", 0, "done", "answer", "", 100, 50, 0.0123)
	if err != nil || !applied {
		t.Fatalf("FinishRun = %v, %v", applied, err)
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

// A run is written once. A late terminal write must not resurrect a run that
// recovery or an operator already closed.
func TestFinishRunDoesNotReopenClosedRun(t *testing.T) {
	store := newTestStore(t)
	createRun(t, store, "closed", "done", nil)

	applied, err := store.FinishRun(context.Background(), "closed", 0, "finished", "late", "", 0, 0, 0)
	if err != nil {
		t.Fatalf("FinishRun: %v", err)
	}
	if applied {
		t.Fatal("a closed run was written again")
	}
}

// A terminal write carrying a stale lease generation must be rejected: the run
// was recovered by another worker, and the old one must not overwrite it.
func TestFinishRunFencesStaleGeneration(t *testing.T) {
	store := newTestStore(t)
	createRun(t, store, "handed-over", "running", nil)

	// Worker A acquires from a different holder, then the lease expires and
	// worker B takes the run. (createRun seeds the row with owner "worker-a",
	// so acquiring as "worker-a" would read as a renewal and keep the number.)
	// Worker A's lease is already expired — that is what lets B take the run.
	_, genA, err := store.AcquireLease(context.Background(), "handed-over", "worker-c", time.Now().Add(-time.Second))
	if err != nil {
		t.Fatalf("acquire A: %v", err)
	}
	if genA != 1 {
		t.Fatalf("first handover must bump the generation, got %d", genA)
	}
	_, genB, err := store.AcquireLease(context.Background(), "handed-over", "worker-d", time.Now().Add(time.Minute))
	if err != nil {
		t.Fatalf("acquire B: %v", err)
	}
	if genB <= genA {
		t.Fatalf("generation did not advance on handover: %d -> %d", genA, genB)
	}

	// Worker A wakes from its long pause and writes its terminal state.
	staleApplied, err := store.FinishRun(context.Background(), "handed-over", genA, "failed", "old output", "", 1, 1, 0)
	if err != nil {
		t.Fatalf("FinishRun stale: %v", err)
	}
	if staleApplied {
		t.Fatal("a fenced worker's terminal write was accepted")
	}

	// Worker B's terminal write still goes through.
	applied, err := store.FinishRun(context.Background(), "handed-over", genB, "finished", "new output", "", 2, 2, 0)
	if err != nil || !applied {
		t.Fatalf("FinishRun current = %v, %v", applied, err)
	}
	var record entity.HarnessAgentRun
	if err := store.db.First(&record, "id = ?", "handed-over").Error; err != nil {
		t.Fatalf("load: %v", err)
	}
	if record.Output != "new output" {
		t.Fatalf("output = %q, want the successor's record", record.Output)
	}
}

// Renewal by the same owner must not fence itself: the heartbeat keeps the run
// alive without invalidating its own terminal write.
func TestAcquireLeaseRenewalKeepsGeneration(t *testing.T) {
	store := newTestStore(t)
	createRun(t, store, "steady", "running", nil)

	_, first, err := store.AcquireLease(context.Background(), "steady", "worker-b", time.Now().Add(time.Second))
	if err != nil {
		t.Fatalf("acquire: %v", err)
	}
	_, second, err := store.AcquireLease(context.Background(), "steady", "worker-b", time.Now().Add(time.Minute))
	if err != nil {
		t.Fatalf("renew: %v", err)
	}
	if first != second {
		t.Fatalf("renewal bumped the generation: %d -> %d", first, second)
	}
}

// A terminal run must never be leased again: recovery would re-execute work
// that already happened and overwrite the record of an answer already sent.
func TestAcquireLeaseRefusesTerminalRun(t *testing.T) {
	store := newTestStore(t)
	createRun(t, store, "answered", "done", nil)

	if ok, generation, err := store.AcquireLease(context.Background(), "answered", "worker-b", time.Now().Add(time.Minute)); err != nil {
		t.Fatalf("AcquireLease: %v", err)
	} else if ok {
		t.Fatalf("a finished run was leased again (generation %d)", generation)
	}
}

// A paused run is not terminal: the operator resumed it, so it must be
// claimable.
func TestAcquireLeaseTakesPausedRun(t *testing.T) {
	store := newTestStore(t)
	createRun(t, store, "paused", "paused", nil)

	if ok, _, err := store.AcquireLease(context.Background(), "paused", "worker-b", time.Now().Add(time.Minute)); err != nil || !ok {
		t.Fatalf("AcquireLease on a paused run = %v, %v", ok, err)
	}
}

// The heartbeat renews under the generation it entered with. Once the run moved
// on, the renewal must report it instead of taking the lease back: a
// re-acquisition would bump the generation and fence this worker's own write.
func TestRenewLeaseReportsLostOwnership(t *testing.T) {
	store := newTestStore(t)
	createRun(t, store, "moving", "running", nil)

	_, held, err := store.AcquireLease(context.Background(), "moving", "worker-b", time.Now().Add(time.Minute))
	if err != nil {
		t.Fatalf("acquire: %v", err)
	}
	renewed, err := store.RenewLease(context.Background(), "moving", "worker-b", held, time.Now().Add(time.Minute))
	if err != nil || !renewed {
		t.Fatalf("renew by the holder = %v, %v", renewed, err)
	}
	if again, _, err := store.AcquireLease(context.Background(), "moving", "worker-c", time.Now().Add(time.Minute)); err != nil {
		t.Fatalf("acquire while live: %v", err)
	} else if again {
		t.Fatal("a live lease was taken by a second worker")
	}

	// The heartbeat stops and the lease lapses while worker-b is paused.
	if renewed, err := store.RenewLease(context.Background(), "moving", "worker-b", held, time.Now().Add(-time.Second)); err != nil || !renewed {
		t.Fatalf("lapsing renew = %v, %v", renewed, err)
	}
	if _, successor, err := store.AcquireLease(context.Background(), "moving", "worker-c", time.Now().Add(time.Minute)); err != nil || successor <= held {
		t.Fatalf("handover = %d, %v (held %d)", successor, err, held)
	}

	renewed, err = store.RenewLease(context.Background(), "moving", "worker-b", held, time.Now().Add(time.Minute))
	if err != nil {
		t.Fatalf("renew after handover: %v", err)
	}
	if renewed {
		t.Fatal("a worker renewed a lease it no longer held")
	}

	// And it must not have disturbed the successor's lease.
	if ok, _, err := store.AcquireLease(context.Background(), "moving", "worker-d", time.Now().Add(time.Minute)); err != nil {
		t.Fatalf("third acquire: %v", err)
	} else if ok {
		t.Fatal("a stale renewal freed the successor's lease")
	}
}

// A run that never held a lease is not "reclaimed": it is claimed. Reporting a
// reclaim would tell the caller it cleared a row it never touched.
func TestReclaimLeaseRefusesUnleasedRun(t *testing.T) {
	store := newTestStore(t)
	record := &entity.HarnessAgentRun{ID: "never-leased", AgentID: "a", TenantID: "t", Status: "pending"}
	if err := store.CreateRun(context.Background(), record); err != nil {
		t.Fatalf("create: %v", err)
	}
	reclaimed, err := store.ReclaimLease(context.Background(), "never-leased")
	if err != nil {
		t.Fatalf("ReclaimLease: %v", err)
	}
	if reclaimed {
		t.Fatal("an unleased run reported a reclaim")
	}
}

// Recovery must not fail a run another worker is executing: that worker would
// then lose its own terminal write to a row that says "failed".
func TestMarkFailedRefusesLiveLease(t *testing.T) {
	store := newTestStore(t)
	createRun(t, store, "working", "running", ptrTime(time.Now().Add(time.Minute)))

	if err := store.MarkFailed(context.Background(), "working", "no checkpoint"); err != nil {
		t.Fatalf("MarkFailed: %v", err)
	}

	var record entity.HarnessAgentRun
	if err := store.db.First(&record, "id = ?", "working").Error; err != nil {
		t.Fatalf("load: %v", err)
	}
	if record.Status != "running" {
		t.Fatalf("status = %q, want the live run untouched", record.Status)
	}
	if record.LeaseOwner != "worker-a" {
		t.Fatalf("lease owner = %q, want the live holder kept", record.LeaseOwner)
	}
}

// A parent blocks on its child, so the child holds no lease. Recovery that
// picked it up would run the same delegation twice.
func TestListRecoverableSkipsChildRuns(t *testing.T) {
	store := newTestStore(t)
	child := &entity.HarnessAgentRun{
		ID:          "child",
		AgentID:     "a",
		TenantID:    "t",
		ParentRunID: "parent",
		Status:      "running",
	}
	if err := store.CreateRun(context.Background(), child); err != nil {
		t.Fatalf("create child: %v", err)
	}

	runs, err := store.ListRecoverable(context.Background(), 10)
	if err != nil {
		t.Fatalf("ListRecoverable: %v", err)
	}
	for _, item := range runs {
		if item.ID == "child" {
			t.Fatal("a running child run was offered to recovery")
		}
	}
}

func ptrTime(value time.Time) *time.Time { return &value }

// Only one replica may hold the sweep ticket at a time; the holder renews, a
// second replica is told "not leader" and skips.
func TestLeaderLeaseSingleOwner(t *testing.T) {
	db, err := gorm.Open(sqlite.Open("file:leader-leader?mode=memory&cache=shared"), &gorm.Config{})
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	if err := db.AutoMigrate(&entity.HarnessLeaderLease{}); err != nil {
		t.Fatalf("migrate: %v", err)
	}
	s := &RunStore{db: db}

	first, err := s.TryAcquireLeaderLease(context.Background(), "run-recovery", "replica-1", time.Now().Add(time.Minute))
	if err != nil || !first {
		t.Fatalf("first acquire = %v, %v", first, err)
	}
	second, err := s.TryAcquireLeaderLease(context.Background(), "run-recovery", "replica-2", time.Now().Add(time.Minute))
	if err != nil {
		t.Fatalf("second acquire: %v", err)
	}
	if second {
		t.Fatal("a second replica became the sweep leader while the lease is live")
	}
	renewed, err := s.TryAcquireLeaderLease(context.Background(), "run-recovery", "replica-1", time.Now().Add(2*time.Minute))
	if err != nil || !renewed {
		t.Fatalf("renewal by the leader = %v, %v", renewed, err)
	}
}
