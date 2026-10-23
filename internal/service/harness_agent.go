package service

import (
	"context"
	"encoding/json"
	"errors"
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

// ErrAgentNameTaken reports a rename onto a name the tenant already uses. Agent
// names are unique per tenant, and a second agent with the same name makes every
// list ambiguous, so the edit is refused instead of silently suffixed.
var ErrAgentNameTaken = errors.New("service: agent name is already used")

// ErrNoChange reports an edit that would store the definition already in place.
var ErrNoChange = errors.New("service: agent definition is unchanged")

// AgentDetail is one agent as the editing API presents it: the identity a client
// addresses, plus the definition it may change.
type AgentDetail struct {
	AgentID    string                `json:"agent_id"`
	VersionID  string                `json:"version_id"`
	Version    int                   `json:"version"`
	Name       string                `json:"name"`
	Definition definition.Definition `json:"definition"`
}

// Detail returns the agent's current definition for editing. It reads the same
// version a run would execute, so what an editor shows is what the agent does.
func (s *HarnessAgentService) Detail(ctx context.Context, tenantID, agentID string) (*AgentDetail, error) {
	agent, err := s.agents.Get(ctx, tenantID, agentID)
	if err != nil {
		return nil, err
	}
	version, err := s.agents.GetVersion(ctx, agentID, agent.CurrentVersionID)
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
	return &AgentDetail{
		AgentID:    agentID,
		VersionID:  version.ID,
		Version:    version.Version,
		Name:       agent.Name,
		Definition: parsed,
	}, nil
}

// Update stores the edit as a new version and points the agent at it.
//
// Versions are immutable, so a run that already executed keeps the definition it
// ran with: editing an agent changes what happens next, never what already
// happened. An edit that changes nothing is refused rather than stored, since a
// history of identical versions hides the one version somebody actually meant.
func (s *HarnessAgentService) Update(ctx context.Context, tenantID, agentID, updatedBy string, def definition.Definition) (*AgentDetail, error) {
	agent, err := s.agents.Get(ctx, tenantID, agentID)
	if err != nil {
		return nil, err
	}
	current, err := s.agents.GetVersion(ctx, agentID, agent.CurrentVersionID)
	if err != nil {
		return nil, err
	}
	if def.Name == "" {
		def.Name = agent.Name
	}
	if err := def.Validate(); err != nil {
		return nil, err
	}
	if def.Name != agent.Name {
		taken, err := s.agents.ExistsByName(ctx, dao.DB, tenantID, def.Name)
		if err != nil {
			return nil, err
		}
		if taken {
			return nil, ErrAgentNameTaken
		}
	}
	// The id is the row's, not the body's: a stored version keeps the definition as
	// the client wrote it, which is also what makes the no-change check below
	// compare like with like.
	body, err := json.Marshal(def)
	if err != nil {
		return nil, fmt.Errorf("service: encode agent definition: %w", err)
	}
	if string(body) == current.Definition {
		return nil, ErrNoChange
	}
	version, err := s.agents.AppendVersion(ctx, agentID, current.Version+1, string(body), updatedBy, def.Name)
	if err != nil {
		return nil, err
	}
	def.ID = agentID
	return &AgentDetail{
		AgentID:    agentID,
		VersionID:  version.ID,
		Version:    version.Version,
		Name:       def.Name,
		Definition: def,
	}, nil
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
