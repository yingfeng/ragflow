package service

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/glebarez/sqlite"
	"gorm.io/gorm"

	"ragflow/internal/dao"
	"ragflow/internal/entity"
	"ragflow/internal/harness/agent/definition"
)

func setupChatAgentServiceTestDB(t *testing.T) *gorm.DB {
	t.Helper()

	db, err := gorm.Open(sqlite.Open(":memory:"), &gorm.Config{TranslateError: true})
	if err != nil {
		t.Fatalf("failed to open sqlite: %v", err)
	}
	if err := db.AutoMigrate(
		&entity.Tenant{},
		&entity.Chat{},
		&entity.HarnessAgent{},
		&entity.HarnessAgentVersion{},
	); err != nil {
		t.Fatalf("failed to migrate test schema: %v", err)
	}

	origDB := dao.DB
	dao.DB = db
	t.Cleanup(func() { dao.DB = origDB })

	tenant := &entity.Tenant{ID: "tenant-1"}
	if err := db.Create(tenant).Error; err != nil {
		t.Fatalf("failed to seed tenant: %v", err)
	}
	return db
}

func TestBuildCreateAgentDefinition(t *testing.T) {
	if def, err := buildCreateAgentDefinition("chat", nil); err != nil || def != nil {
		t.Fatalf("absent agent = (%v, %v), want (nil, nil)", def, err)
	}
	// An empty object is still a request for an agent: presence is the intent.
	if def, err := buildCreateAgentDefinition("chat", map[string]interface{}{}); err != nil || def == nil {
		t.Fatalf("empty agent = (%v, %v), want a default definition", def, err)
	}

	// Name and prompt fall back to the thread's name and the built-in prompt: the
	// UI sends only the agent-specific overrides.
	def, err := buildCreateAgentDefinition("my thread", map[string]interface{}{
		"caps": []interface{}{map[string]interface{}{"kind": "skill", "name": "agentic-rag"}},
	})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	if def.Name != "my thread" {
		t.Fatalf("name = %q", def.Name)
	}
	if def.Prompt != definition.DefaultPrompt {
		t.Fatal("prompt did not default to the built-in agent prompt")
	}
	if len(def.Caps) != 1 || def.Caps[0].Kind != "skill" || def.Caps[0].Name != "agentic-rag" {
		t.Fatalf("caps = %+v", def.Caps)
	}

	if _, err := buildCreateAgentDefinition("chat", map[string]interface{}{"policy": map[string]interface{}{"max_rounds": -1}}); err == nil {
		t.Fatal("a negative budget was accepted")
	}
	if _, err := buildCreateAgentDefinition("chat", "not-an-object"); err == nil {
		t.Fatal("a non-object agent body was accepted")
	}
}

func TestCreateChatWithAgentCreatesThreadAndAgent(t *testing.T) {
	db := setupChatAgentServiceTestDB(t)
	svc := NewChatService()

	result, code, err := svc.Create(context.Background(), "tenant-1", map[string]interface{}{
		"name":  "research buddy",
		"agent": map[string]interface{}{"policy": map[string]interface{}{"approval": "ask"}},
	})
	if err != nil {
		t.Fatalf("create: %v (code %v)", err, code)
	}

	agentID, _ := result["agent_id"].(string)
	if agentID == "" {
		t.Fatal("response carries no agent_id")
	}

	var chat entity.Chat
	if err := db.Where("id = ?", result["id"]).Take(&chat).Error; err != nil {
		t.Fatalf("load chat: %v", err)
	}
	if chat.AgentID != agentID {
		t.Fatalf("chat agent_id = %q, want %q", chat.AgentID, agentID)
	}

	var agent entity.HarnessAgent
	if err := db.Where("id = ?", agentID).Take(&agent).Error; err != nil {
		t.Fatalf("load agent: %v", err)
	}
	if agent.Name != "research buddy" {
		t.Fatalf("agent name = %q", agent.Name)
	}

	var version entity.HarnessAgentVersion
	if err := db.Where("agent_id = ?", agentID).Take(&version).Error; err != nil {
		t.Fatalf("load version: %v", err)
	}
	var def definition.Definition
	if err := json.Unmarshal([]byte(version.Definition), &def); err != nil {
		t.Fatalf("decode definition: %v", err)
	}
	if def.Prompt != definition.DefaultPrompt {
		t.Fatal("stored definition lost the default prompt")
	}
	if def.Policy.Approval != definition.ApprovalAsk {
		t.Fatalf("approval mode = %q", def.Policy.Approval)
	}
}

func TestCreatePlainChatLeavesNoAgent(t *testing.T) {
	db := setupChatAgentServiceTestDB(t)
	svc := NewChatService()

	result, _, err := svc.Create(context.Background(), "tenant-1", map[string]interface{}{"name": "plain"})
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	if agentID, _ := result["agent_id"].(string); agentID != "" {
		t.Fatalf("plain chat got agent_id %q", agentID)
	}

	var count int64
	if err := db.Model(&entity.HarnessAgent{}).Count(&count).Error; err != nil {
		t.Fatalf("count agents: %v", err)
	}
	if count != 0 {
		t.Fatalf("plain chat created %d agent row(s)", count)
	}
}

func TestCreateChatWithInvalidAgentCreatesNothing(t *testing.T) {
	db := setupChatAgentServiceTestDB(t)
	svc := NewChatService()

	_, _, err := svc.Create(context.Background(), "tenant-1", map[string]interface{}{
		"name":  "bad agent",
		"agent": map[string]interface{}{"policy": map[string]interface{}{"approval": "whenever"}},
	})
	if err == nil {
		t.Fatal("invalid approval mode was accepted")
	}
	if !strings.Contains(err.Error(), "approval") {
		t.Fatalf("error = %v", err)
	}

	var chats, agents int64
	if err := db.Model(&entity.Chat{}).Count(&chats).Error; err != nil {
		t.Fatalf("count chats: %v", err)
	}
	if err := db.Model(&entity.HarnessAgent{}).Count(&agents).Error; err != nil {
		t.Fatalf("count agents: %v", err)
	}
	if chats != 0 || agents != 0 {
		t.Fatalf("failed create left %d chat(s) and %d agent(s)", chats, agents)
	}
}

func TestCreateChatWithAgentSuffixesTakenName(t *testing.T) {
	db := setupChatAgentServiceTestDB(t)
	if err := db.Create(&entity.HarnessAgent{
		ID: "agent-existing", TenantID: "tenant-1", Name: "research buddy", Status: "active",
	}).Error; err != nil {
		t.Fatalf("seed agent: %v", err)
	}
	svc := NewChatService()

	result, _, err := svc.Create(context.Background(), "tenant-1", map[string]interface{}{
		"name":  "research buddy",
		"agent": map[string]interface{}{},
	})
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	agentID, _ := result["agent_id"].(string)
	var agent entity.HarnessAgent
	if err := db.Where("id = ?", agentID).Take(&agent).Error; err != nil {
		t.Fatalf("load agent: %v", err)
	}
	if agent.Name == "research buddy" {
		t.Fatal("agent reused a name already held by another agent")
	}
	if !strings.HasPrefix(agent.Name, "research buddy (") {
		t.Fatalf("agent name = %q", agent.Name)
	}
}
