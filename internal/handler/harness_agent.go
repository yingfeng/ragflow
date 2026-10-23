package handler

import (
	"context"
	"errors"

	"github.com/gin-gonic/gin"
	"gorm.io/gorm"

	"ragflow/internal/common"
	"ragflow/internal/harness/agent/definition"
	"ragflow/internal/service"
)

// harnessAgentService is the consumer-side interface for HarnessAgentHandler.
//
// An agent definition is only ever read and replaced: a run captures the version
// it executed, so an edit is a new version rather than a mutation of the old one.
type harnessAgentService interface {
	Detail(ctx context.Context, tenantID, agentID string) (*service.AgentDetail, error)
	Update(ctx context.Context, tenantID, agentID, updatedBy string, def definition.Definition) (*service.AgentDetail, error)
}

// HarnessAgentHandler edits the agent that runs a chat thread.
type HarnessAgentHandler struct {
	agents harnessAgentService
}

// NewHarnessAgentHandler builds the handler.
func NewHarnessAgentHandler(agents harnessAgentService) *HarnessAgentHandler {
	return &HarnessAgentHandler{agents: agents}
}

// Register mounts the agent editing routes. They sit under the harness group, so
// the tenant is already bound by the caller-tenant middleware.
func (h *HarnessAgentHandler) Register(group *gin.RouterGroup) {
	group.GET("/agents/:agent_id", h.getAgent)
	group.PUT("/agents/:agent_id", h.updateAgent)
}

// getAgent returns the definition the agent currently executes.
func (h *HarnessAgentHandler) getAgent(c *gin.Context) {
	user, code, message := GetUser(c)
	if code != common.CodeSuccess {
		common.ErrorWithCode(c, code, message)
		return
	}
	agentID := c.Param("agent_id")
	if agentID == "" {
		common.ResponseWithCodeData(c, common.CodeParamError, nil, "agent_id is required")
		return
	}
	detail, err := h.agents.Detail(c.Request.Context(), user.ID, agentID)
	if err != nil {
		h.fail(c, err)
		return
	}
	common.ResponseWithCodeData(c, common.CodeSuccess, detail, "")
}

// updateAgent replaces the definition with the body and answers with the version
// it stored, so a client can show which version is now live.
func (h *HarnessAgentHandler) updateAgent(c *gin.Context) {
	user, code, message := GetUser(c)
	if code != common.CodeSuccess {
		common.ErrorWithCode(c, code, message)
		return
	}
	agentID := c.Param("agent_id")
	if agentID == "" {
		common.ResponseWithCodeData(c, common.CodeParamError, nil, "agent_id is required")
		return
	}
	var body definition.Definition
	if err := c.ShouldBindJSON(&body); err != nil {
		common.ResponseWithCodeData(c, common.CodeParamError, nil, "invalid agent definition: "+err.Error())
		return
	}
	detail, err := h.agents.Update(c.Request.Context(), user.ID, agentID, user.ID, body)
	if err != nil {
		h.fail(c, err)
		return
	}
	common.ResponseWithCodeData(c, common.CodeSuccess, detail, "")
}

// fail maps the service's errors onto the codes a client can act on. Validation
// answers 400 because the body is fixable; a missing agent answers 404 because a
// retry with the same id cannot succeed.
func (h *HarnessAgentHandler) fail(c *gin.Context, err error) {
	switch {
	case errors.Is(err, gorm.ErrRecordNotFound):
		common.ResponseWithCodeData(c, common.CodeNotFound, nil, "agent not found")
	case errors.Is(err, service.ErrAgentNameTaken):
		common.ResponseWithCodeData(c, common.CodeDataError, nil, "agent name is already used")
	case errors.Is(err, service.ErrNoChange):
		common.ResponseWithCodeData(c, common.CodeDataError, nil, "agent definition is unchanged")
	case errors.Is(err, definition.ErrEmptyPrompt), errors.Is(err, definition.ErrEmptyName):
		common.ResponseWithCodeData(c, common.CodeDataError, nil, err.Error())
	default:
		common.ResponseWithCodeData(c, common.CodeServerError, nil, err.Error())
	}
}
