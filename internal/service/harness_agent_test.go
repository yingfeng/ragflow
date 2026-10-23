package service

import (
	"errors"
	"testing"

	"github.com/glebarez/sqlite"
	"gorm.io/gorm"

	"ragflow/internal/dao"
	"ragflow/internal/entity"
	"ragflow/internal/harness/agent/definition"
)

// pushHarnessAgentDB swaps in a migrated in-memory database. The service builds
// its DAO from the package handle, so it has to be installed before the service
// is constructed.
func pushHarnessAgentDB(t *testing.T) {
	t.Helper()
	db, err := gorm.Open(sqlite.Open(":memory:"), &gorm.Config{TranslateError: true})
	if err != nil {
		t.Fatalf("open sqlite: %v", err)
	}
	if err := db.AutoMigrate(&entity.HarnessAgent{}, &entity.HarnessAgentVersion{}); err != nil {
		t.Fatalf("migrate: %v", err)
	}
	original := dao.DB
	dao.DB = db
	t.Cleanup(func() { dao.DB = original })
}

func newTestAgent(t *testing.T, svc *HarnessAgentService, name string) *ResolvedAgent {
	t.Helper()
	created, err := svc.Create(t.Context(), "tenant-1", "user-1", definition.Definition{
		Name:   name,
		Prompt: "you are a test agent",
		Policy: definition.PolicyConfig{MaxRounds: 8},
	})
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	return created
}

// An edit is a new version, and the runtime resolves the new version on its next
// turn: that is what makes the UI's save button take effect without a restart.
func TestHarnessAgentUpdateAppendsVersion(t *testing.T) {
	pushHarnessAgentDB(t)
	svc := NewHarnessAgentService()
	created := newTestAgent(t, svc, "thread-agent")

	updated, err := svc.Update(t.Context(), "tenant-1", created.AgentID, "user-1", definition.Definition{
		Prompt: "you are a test agent",
		Policy: definition.PolicyConfig{MaxRounds: 0},
	})
	if err != nil {
		t.Fatalf("update: %v", err)
	}
	if updated.Version != 2 {
		t.Fatalf("version = %d, want 2", updated.Version)
	}
	if updated.Name != "thread-agent" {
		t.Fatalf("name = %q, want the name kept when the body omits it", updated.Name)
	}

	// What the next run executes.
	resolved, err := svc.Resolve(t.Context(), "tenant-1", created.AgentID, "")
	if err != nil {
		t.Fatalf("resolve: %v", err)
	}
	if resolved.Definition.Policy.MaxRounds != 0 {
		t.Fatalf("max_rounds = %d, want the edited 0 (unlimited)", resolved.Definition.Policy.MaxRounds)
	}
	if resolved.Version != 2 {
		t.Fatalf("resolved version = %d, want the newest", resolved.Version)
	}

	// What the editor shows.
	detail, err := svc.Detail(t.Context(), "tenant-1", created.AgentID)
	if err != nil {
		t.Fatalf("detail: %v", err)
	}
	if detail.Version != 2 || detail.Definition.Prompt != "you are a test agent" {
		t.Fatalf("detail = v%d %q", detail.Version, detail.Definition.Prompt)
	}
	if detail.Definition.ID != created.AgentID {
		t.Fatalf("detail id = %q, want the agent id", detail.Definition.ID)
	}
}

// Saving the form without touching anything must not mint a version: a history of
// identical versions hides the edit somebody actually made.
func TestHarnessAgentUpdateRefusesAnUnchangedDefinition(t *testing.T) {
	pushHarnessAgentDB(t)
	svc := NewHarnessAgentService()
	created := newTestAgent(t, svc, "thread-agent")

	_, err := svc.Update(t.Context(), "tenant-1", created.AgentID, "user-1", definition.Definition{
		Name:   "thread-agent",
		Prompt: "you are a test agent",
		Policy: definition.PolicyConfig{MaxRounds: 8},
	})
	if !errors.Is(err, ErrNoChange) {
		t.Fatalf("err = %v, want ErrNoChange", err)
	}
}

// Names are unique per tenant; renaming onto a taken one is refused rather than
// silently suffixed, because the caller asked for a specific name.
func TestHarnessAgentUpdateRefusesATakenName(t *testing.T) {
	pushHarnessAgentDB(t)
	svc := NewHarnessAgentService()
	first := newTestAgent(t, svc, "first")
	newTestAgent(t, svc, "second")

	_, err := svc.Update(t.Context(), "tenant-1", first.AgentID, "user-1", definition.Definition{
		Name:   "second",
		Prompt: "you are a test agent",
	})
	if !errors.Is(err, ErrAgentNameTaken) {
		t.Fatalf("err = %v, want ErrAgentNameTaken", err)
	}
}

// A foreign tenant's agent id is indistinguishable from a missing one.
func TestHarnessAgentUpdateScopesByTenant(t *testing.T) {
	pushHarnessAgentDB(t)
	svc := NewHarnessAgentService()
	created := newTestAgent(t, svc, "thread-agent")

	_, err := svc.Update(t.Context(), "tenant-2", created.AgentID, "user-2", definition.Definition{
		Prompt: "hijacked",
	})
	if !errors.Is(err, gorm.ErrRecordNotFound) {
		t.Fatalf("err = %v, want not found", err)
	}
	if _, err := svc.Detail(t.Context(), "tenant-2", created.AgentID); !errors.Is(err, gorm.ErrRecordNotFound) {
		t.Fatalf("detail err = %v, want not found", err)
	}
}

// A definition without instructions is rejected before it becomes a version.
func TestHarnessAgentUpdateValidatesTheBody(t *testing.T) {
	pushHarnessAgentDB(t)
	svc := NewHarnessAgentService()
	created := newTestAgent(t, svc, "thread-agent")

	_, err := svc.Update(t.Context(), "tenant-1", created.AgentID, "user-1", definition.Definition{
		Name: "thread-agent",
	})
	if !errors.Is(err, definition.ErrEmptyPrompt) {
		t.Fatalf("err = %v, want ErrEmptyPrompt", err)
	}
}
