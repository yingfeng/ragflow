package service

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/google/uuid"
	"gorm.io/gorm"

	"ragflow/internal/dao"
	"ragflow/internal/entity"
	"ragflow/internal/harness/agent/definition"
)

// HarnessAgentService loads agent definitions for the harness runtime. The
// runtime only needs the immutable version body plus its id; everything else
// (tenant, run scope) is passed per run.
type HarnessAgentService struct {
	agents *dao.HarnessAgentDAO
}

// NewHarnessAgentService builds the service.
func NewHarnessAgentService() *HarnessAgentService {
	return &HarnessAgentService{agents: dao.NewHarnessAgentDAO()}
}

// ResolvedAgent is one version of one agent.
type ResolvedAgent struct {
	AgentID    string
	VersionID  string
	Version    int
	Definition definition.Definition
}

// Resolve returns the definition a run should execute.
func (s *HarnessAgentService) Resolve(ctx context.Context, tenantID, agentID, versionID string) (*ResolvedAgent, error) {
	agent, err := s.agents.Get(ctx, tenantID, agentID)
	if err != nil {
		return nil, err
	}
	if versionID == "" {
		versionID = agent.CurrentVersionID
	}
	version, err := s.agents.GetVersion(ctx, agentID, versionID)
	if err != nil {
		return nil, err
	}
	var parsed definition.Definition
	if err := json.Unmarshal([]byte(version.Definition), &parsed); err != nil {
		return nil, fmt.Errorf("service: decode agent %s version %d: %w", agentID, version.Version, err)
	}
	if parsed.ID == "" {
		parsed.ID = agentID
	}
	if parsed.Name == "" {
		parsed.Name = agent.Name
	}
	return &ResolvedAgent{
		AgentID:    agentID,
		VersionID:  version.ID,
		Version:    version.Version,
		Definition: parsed,
	}, nil
}

// Create stores a new agent and its first version.
func (s *HarnessAgentService) Create(ctx context.Context, tenantID, createdBy string, def definition.Definition) (*ResolvedAgent, error) {
	return s.CreateTx(ctx, dao.DB, tenantID, createdBy, def)
}

// CreateTx stores a new agent inside the caller's transaction. A chat thread and
// the agent that runs it are created together: a run that has a thread but no
// agent (or the reverse) is unrecoverable from the UI's point of view.
func (s *HarnessAgentService) CreateTx(ctx context.Context, db *gorm.DB, tenantID, createdBy string, def definition.Definition) (*ResolvedAgent, error) {
	if err := def.Validate(); err != nil {
		return nil, err
	}
	agentID := uuid.NewString()
	versionID := uuid.NewString()

	// An agent created from a chat thread is named after the thread. Agent names
	// are unique per tenant while chat names are unique per tenant as well, so a
	// standalone agent may already hold this name; suffix it instead of failing
	// the chat creation the user actually asked for.
	def.Name = s.uniqueName(ctx, db, tenantID, def.Name, agentID)

	body, err := json.Marshal(def)
	if err != nil {
		return nil, fmt.Errorf("service: encode agent definition: %w", err)
	}
	agent := &entity.HarnessAgent{
		ID:               agentID,
		TenantID:         tenantID,
		Name:             def.Name,
		Description:      "",
		CurrentVersionID: versionID,
		Status:           "active",
		CreatedBy:        createdBy,
	}
	version := &entity.HarnessAgentVersion{
		ID:         versionID,
		AgentID:    agentID,
		Version:    1,
		Definition: string(body),
		CreatedBy:  createdBy,
	}
	if err := s.agents.CreateWithVersion(ctx, db, agent, version); err != nil {
		return nil, err
	}
	def.ID = agentID
	return &ResolvedAgent{AgentID: agentID, VersionID: versionID, Version: 1, Definition: def}, nil
}

// uniqueName returns name, or a suffixed variant when the tenant already uses it.
func (s *HarnessAgentService) uniqueName(ctx context.Context, db *gorm.DB, tenantID, name, agentID string) string {
	taken, err := s.agents.ExistsByName(ctx, db, tenantID, name)
	if err != nil || !taken {
		return name
	}
	suffix := agentID
	if len(suffix) > 6 {
		suffix = suffix[:6]
	}
	return name + " (" + suffix + ")"
}
