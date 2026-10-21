package store

import (
	"context"
	"testing"

	"github.com/glebarez/sqlite"
	"gorm.io/gorm"

	"ragflow/internal/entity"
	"ragflow/internal/harness/toolkit"
)

func newTestGrantStore(t *testing.T) *GrantStore {
	t.Helper()
	db, err := gorm.Open(sqlite.Open("file:"+t.Name()+"?mode=memory&cache=shared"), &gorm.Config{})
	if err != nil {
		t.Fatalf("open sqlite: %v", err)
	}
	if err := db.AutoMigrate(&entity.HarnessAgentPermissionGrant{}); err != nil {
		t.Fatalf("migrate: %v", err)
	}
	return NewGrantStore(db)
}

func TestGrantStoreRemembersAndRevokes(t *testing.T) {
	store := newTestGrantStore(t)
	ctx := context.Background()
	grant := toolkit.GrantFor("tenant-1", toolkit.Call{Tool: "exec_shell", Arguments: `{"cmd":"df -h"}`})

	if ok, err := store.Granted(ctx, grant); err != nil || ok {
		t.Fatalf("grant present before it was recorded (ok=%v err=%v)", ok, err)
	}
	if err := store.Remember(ctx, grant); err != nil {
		t.Fatalf("remember: %v", err)
	}
	if ok, err := store.Granted(ctx, grant); err != nil || !ok {
		t.Fatalf("grant missing after it was recorded (ok=%v err=%v)", ok, err)
	}

	// A different command must not ride on the same grant.
	other := toolkit.GrantFor("tenant-1", toolkit.Call{Tool: "exec_shell", Arguments: `{"cmd":"rm -rf /"}`})
	if ok, _ := store.Granted(ctx, other); ok {
		t.Fatal("a different command matched the grant")
	}
	// Neither must another tenant.
	foreign := toolkit.GrantFor("tenant-2", toolkit.Call{Tool: "exec_shell", Arguments: `{"cmd":"df -h"}`})
	if ok, _ := store.Granted(ctx, foreign); ok {
		t.Fatal("a different tenant matched the grant")
	}

	// Recording twice is a no-op, not an error.
	if err := store.Remember(ctx, grant); err != nil {
		t.Fatalf("repeat remember: %v", err)
	}

	removed, err := store.Revoke(ctx, grant)
	if err != nil {
		t.Fatalf("revoke: %v", err)
	}
	if !removed {
		t.Fatal("revoke reported no row")
	}
	if ok, _ := store.Granted(ctx, grant); ok {
		t.Fatal("grant survived revocation")
	}
}

func TestGrantStoreRejectsIncompleteGrant(t *testing.T) {
	store := newTestGrantStore(t)
	if err := store.Remember(context.Background(), toolkit.Grant{TenantID: "tenant-1"}); err == nil {
		t.Fatal("expected an error for a grant without tool or args hash")
	}
	// An incomplete grant never matches, so an unscoped grant cannot leak.
	if ok, err := store.Granted(context.Background(), toolkit.Grant{TenantID: "tenant-1"}); err != nil || ok {
		t.Fatalf("incomplete grant matched (ok=%v err=%v)", ok, err)
	}
}
