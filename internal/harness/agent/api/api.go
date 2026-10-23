// Package api exposes the harness over HTTP: start a run, stream its events,
// cancel it and read its audit trail.
//
// The handler talks to an interface, not to *run.Service, so a host can plug a
// different executor (or a test double) without this package importing the
// store or the model layer.
package api

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"

	"ragflow/internal/entity"
	"ragflow/internal/harness/agent/loop"
	"ragflow/internal/harness/agent/run"
	"ragflow/internal/harness/events"
	"ragflow/internal/harness/toolkit"
)

// EventQuery selects one slice of a run's audit log. Layers are the C0–C5 data
// classes, so an operator can read the control facts without pulling the trace.
type EventQuery struct {
	Layers []string
	Types  []string
	Limit  int
}

// ApprovalRef is one pending tool approval as the API presents it. RequestID
// addresses the request on its own, so a client that reloaded and lost the event
// stream can still settle it.
type ApprovalRef struct {
	RequestID string `json:"request_id"`
	RunID     string `json:"run_id"`
	Tool      string `json:"tool"`
	Args      string `json:"args"`
	Risk      string `json:"risk"`
	TenantID  string `json:"tenant_id"`
	// Diff is what the call will change, for a write: a reviewer approves a
	// diff, not a JSON argument blob. Empty for calls with no preview.
	Diff string `json:"diff,omitempty"`
}

// Runs is the executor the handler drives.
type Runs interface {
	Invoke(ctx context.Context, req run.Request) (loop.Outcome, error)
	Subscribe(runID string) chan run.Event
	Unsubscribe(runID string, events chan run.Event)
	Cancel(runID string) bool
	Steer(runID, message string) bool
	Steps(ctx context.Context, runID string) ([]entity.HarnessAgentStep, error)
	// Resume continues a paused run from its checkpoint.
	Resume(ctx context.Context, req run.Request) (loop.Outcome, error)
	// ResolveApproval settles one pending tool approval. It reports whether the
	// request was still waiting. remember turns an allow into a standing grant
	// for that exact tool-and-arguments pair.
	ResolveApproval(runID, requestID string, allow bool, reason string, remember bool) bool
	// PendingApprovals lists one tenant's approvals still waiting in this
	// process, so a client that missed the announcement can still show them.
	PendingApprovals(tenantID string) []ApprovalRef
	// ResolveApprovalKey settles one request by the id PendingApprovals returned.
	// A request belonging to another tenant is not answerable here.
	ResolveApprovalKey(requestID, tenantID string, allow bool, reason string, remember bool) bool
	// PendingInteractions lists one tenant's questions still waiting for a
	// human answer, so a client that missed the announcement can still show them.
	PendingInteractions(tenantID string) []toolkit.InteractRef
	// ResolveInteractionKey answers one question by the id the interaction list
	// handed out. A question another tenant's run asked is not answerable here.
	ResolveInteractionKey(requestID, tenantID, answer string) bool
	// Events reads the run's append-only audit log.
	Events(ctx context.Context, runID string, query EventQuery) ([]*events.Event, error)
}

// Handler serves the harness HTTP surface.
type Handler struct {
	runs Runs
}

// New builds a handler.
func New(runs Runs) *Handler { return &Handler{runs: runs} }

// Register mounts the routes on a group.
func (h *Handler) Register(group *gin.RouterGroup) {
	group.POST("/agents/:agent_id/runs", h.createRun)
	group.GET("/runs/:run_id/stream", h.streamRun)
	group.POST("/runs/:run_id/cancel", h.cancelRun)
	group.POST("/runs/:run_id/steer", h.steerRun)
	group.GET("/runs/:run_id/steps", h.listSteps)
	group.GET("/runs/:run_id/events", h.listEvents)
	group.POST("/runs/:run_id/resume", h.resumeRun)
	group.POST("/runs/:run_id/approvals/:request_id", h.resolveApproval)
	group.GET("/approvals", h.listApprovals)
	group.POST("/approvals/:request_id", h.resolveApprovalKey)
	group.GET("/interactions", h.listInteractions)
	group.POST("/interactions/:request_id", h.resolveInteraction)
}

// interactionAnswer is the accepted body of an interaction answer.
type interactionAnswer struct {
	Answer string `json:"answer"`
}

// listInteractions returns the questions waiting for this tenant's users. The
// same strict tenant filter as approvals applies: one tenant must not read
// another tenant's questions.
func (h *Handler) listInteractions(c *gin.Context) {
	tenantID := c.GetString("tenant_id")
	if tenantID == "" {
		c.JSON(http.StatusForbidden, gin.H{"code": 403, "message": "tenant is required"})
		return
	}
	items := h.runs.PendingInteractions(tenantID)
	if items == nil {
		items = []toolkit.InteractRef{}
	}
	c.JSON(http.StatusOK, gin.H{"code": 0, "data": items})
}

// resolveInteraction answers one waiting question by its request id.
func (h *Handler) resolveInteraction(c *gin.Context) {
	tenantID := c.GetString("tenant_id")
	if tenantID == "" {
		c.JSON(http.StatusForbidden, gin.H{"code": 403, "message": "tenant is required"})
		return
	}
	var body interactionAnswer
	if err := c.ShouldBindJSON(&body); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"code": 400, "message": err.Error()})
		return
	}
	if strings.TrimSpace(body.Answer) == "" {
		c.JSON(http.StatusBadRequest, gin.H{"code": 400, "message": "answer is required"})
		return
	}
	if !h.runs.ResolveInteractionKey(c.Param("request_id"), tenantID, body.Answer) {
		c.JSON(http.StatusNotFound, gin.H{"code": 404, "message": "question is not waiting"})
		return
	}
	c.JSON(http.StatusOK, gin.H{"code": 0})
}

// listApprovals returns the approvals waiting in this process. The tenant of an
// authenticated caller narrows the list; an operator endpoint that listed every
// tenant's pending dangerous calls would leak what other tenants are doing.
func (h *Handler) listApprovals(c *gin.Context) {
	tenantID := c.GetString("tenant_id")
	if tenantID == "" {
		c.JSON(http.StatusForbidden, gin.H{"code": 403, "message": "tenant is required"})
		return
	}
	items := h.runs.PendingApprovals(tenantID)
	if items == nil {
		items = []ApprovalRef{}
	}
	c.JSON(http.StatusOK, gin.H{"code": 0, "data": items})
}

// resolveApprovalKey settles one request by the id the approval list handed out.
// The run that owns it is found from the id, so a client does not have to track
// which run announced it.
func (h *Handler) resolveApprovalKey(c *gin.Context) {
	tenantID := c.GetString("tenant_id")
	if tenantID == "" {
		c.JSON(http.StatusForbidden, gin.H{"code": 403, "message": "tenant is required"})
		return
	}
	var body approvalRequest
	if err := c.ShouldBindJSON(&body); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"code": 400, "message": err.Error()})
		return
	}
	if !h.runs.ResolveApprovalKey(c.Param("request_id"), tenantID, body.Allow, body.Reason, body.Remember) {
		c.JSON(http.StatusNotFound, gin.H{"code": 404, "message": "approval request is not pending"})
		return
	}
	c.JSON(http.StatusOK, gin.H{"code": 0})
}

// resumeRun continues a paused run. The request body is optional and only
// carries the tenant when the caller is not authenticated.
func (h *Handler) resumeRun(c *gin.Context) {
	runID := c.Param("run_id")
	var body struct {
		TenantID string `json:"tenant_id"`
	}
	_ = c.ShouldBindJSON(&body)
	tenantID := body.TenantID
	if tenantID == "" {
		tenantID = c.GetString("tenant_id")
	}
	outcome, err := h.runs.Resume(c.Request.Context(), run.Request{
		RunID:    runID,
		TenantID: tenantID,
		UserID:   c.GetString("user_id"),
	})
	if err != nil {
		c.JSON(http.StatusConflict, gin.H{"code": 409, "message": err.Error()})
		return
	}
	c.JSON(http.StatusOK, gin.H{"code": 0, "data": outcome})
}

// approvalRequest is the operator's verdict on one pending tool call. Remember
// stores the answer as a standing grant for the same tool and arguments.
type approvalRequest struct {
	Allow    bool   `json:"allow"`
	Reason   string `json:"reason"`
	Remember bool   `json:"remember"`
}

// resolveApproval answers a pending approval request.
func (h *Handler) resolveApproval(c *gin.Context) {
	var body approvalRequest
	if err := c.ShouldBindJSON(&body); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"code": 400, "message": err.Error()})
		return
	}
	if !h.runs.ResolveApproval(c.Param("run_id"), c.Param("request_id"), body.Allow, body.Reason, body.Remember) {
		c.JSON(http.StatusNotFound, gin.H{"code": 404, "message": "approval request is not pending"})
		return
	}
	c.JSON(http.StatusOK, gin.H{"code": 0})
}

// createRunRequest is the accepted body.
type createRunRequest struct {
	TenantID string `json:"tenant_id"`
	// ModelRef names the model for this run. Empty falls back to the tenant's
	// default model, so a deployment needs no process-wide model setting.
	ModelRef    string   `json:"model_ref"`
	AgentName   string   `json:"agent_name"`
	UserMessage string   `json:"user_message"`
	DatasetIDs  []string `json:"dataset_ids"`
	Docs        []string `json:"doc_ids"`
}

// createRun starts a run in the background and returns its id immediately: a run
// outlives the request that created it.
func (h *Handler) createRun(c *gin.Context) {
	var body createRunRequest
	if err := c.ShouldBindJSON(&body); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"code": 400, "message": err.Error()})
		return
	}
	if body.UserMessage == "" {
		c.JSON(http.StatusBadRequest, gin.H{"code": 400, "message": "user_message is required"})
		return
	}

	runID := uuid.NewString()
	tenantID := body.TenantID
	if tenantID == "" {
		tenantID = c.GetString("tenant_id")
	}

	request := run.Request{
		RunID:       runID,
		AgentID:     c.Param("agent_id"),
		TenantID:    tenantID,
		UserID:      c.GetString("user_id"),
		ModelRef:    body.ModelRef,
		AgentName:   body.AgentName,
		UserMessage: body.UserMessage,
		Trigger:     "api",
		Scope: toolkit.Scope{
			TenantID: tenantID,
			Datasets: body.DatasetIDs,
			Docs:     body.Docs,
		},
	}

	go func() {
		// The run owns its lifetime; cancelling the HTTP request must not kill it.
		_, _ = h.runs.Invoke(context.Background(), request)
	}()

	c.JSON(http.StatusOK, gin.H{"code": 0, "data": gin.H{"run_id": runID}})
}

// streamRun relays the run's events as server-sent events.
func (h *Handler) streamRun(c *gin.Context) {
	runID := c.Param("run_id")
	events := h.runs.Subscribe(runID)
	defer h.runs.Unsubscribe(runID, events)

	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Status(http.StatusOK)

	flusher, canFlush := c.Writer.(http.Flusher)
	if canFlush {
		flusher.Flush()
	}

	for {
		select {
		case <-c.Request.Context().Done():
			return
		case event, ok := <-events:
			if !ok {
				return
			}
			payload, err := json.Marshal(event)
			if err != nil {
				continue
			}
			if _, err := fmt.Fprintf(c.Writer, "event: %s\ndata: %s\n\n", event.Type, payload); err != nil {
				return
			}
			if canFlush {
				flusher.Flush()
			}
		}
	}
}

// cancelRun stops a running run.
func (h *Handler) cancelRun(c *gin.Context) {
	if !h.runs.Cancel(c.Param("run_id")) {
		c.JSON(http.StatusNotFound, gin.H{"code": 404, "message": "run is not executing here"})
		return
	}
	c.JSON(http.StatusOK, gin.H{"code": 0})
}

// steerRequest is the accepted body of a steer call.
type steerRequest struct {
	Message string `json:"message"`
}

// steerRun injects a message into a running run.
func (h *Handler) steerRun(c *gin.Context) {
	var body steerRequest
	if err := c.ShouldBindJSON(&body); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"code": 400, "message": err.Error()})
		return
	}
	if body.Message == "" {
		c.JSON(http.StatusBadRequest, gin.H{"code": 400, "message": "message is required"})
		return
	}
	if !h.runs.Steer(c.Param("run_id"), body.Message) {
		c.JSON(http.StatusNotFound, gin.H{"code": 404, "message": "run is not executing here"})
		return
	}
	c.JSON(http.StatusOK, gin.H{"code": 0})
}

// listSteps returns the run's audit trail.
func (h *Handler) listSteps(c *gin.Context) {
	steps, err := h.runs.Steps(c.Request.Context(), c.Param("run_id"))
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"code": 500, "message": err.Error()})
		return
	}
	c.JSON(http.StatusOK, gin.H{"code": 0, "data": steps})
}

// listEvents returns the event log of one run. The layer and type filters are
// repeatable: ?layer=C2&layer=C0 reads the durable facts plus the control state,
// which is what an audit normally wants.
func (h *Handler) listEvents(c *gin.Context) {
	query := EventQuery{
		Layers: c.QueryArray("layer"),
		Types:  c.QueryArray("type"),
		Limit:  defaultEventLimit,
	}
	if raw := c.Query("limit"); raw != "" {
		parsed, err := strconv.Atoi(raw)
		if err != nil || parsed <= 0 {
			c.JSON(http.StatusBadRequest, gin.H{"code": 400, "message": "limit must be a positive integer"})
			return
		}
		query.Limit = parsed
	}
	items, err := h.runs.Events(c.Request.Context(), c.Param("run_id"), query)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"code": 500, "message": err.Error()})
		return
	}
	c.JSON(http.StatusOK, gin.H{"code": 0, "data": items})
}

// defaultEventLimit caps one audit read.
const defaultEventLimit = 500
