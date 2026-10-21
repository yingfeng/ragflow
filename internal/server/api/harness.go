package api

import (
	"context"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"

	"ragflow/internal/dao"
	"ragflow/internal/entity"
	"ragflow/internal/harness"
	harnessapi "ragflow/internal/harness/agent/api"
	"ragflow/internal/harness/agent/definition"
	"ragflow/internal/harness/agent/loop"
	"ragflow/internal/harness/agent/run"
	"ragflow/internal/harness/events"
	"ragflow/internal/harness/toolkit"
	"ragflow/internal/router"
	"ragflow/internal/service"
)

// harnessManager owns the runtimes of the runs executing on this worker.
//
// A runtime is built per run: the tool surface depends on the run's scope (which
// datasets it may read), and sharing one runtime would leak one tenant's
// datasets into another tenant's tool list. The stores, grants, breakers and the
// approval broker come from the process-wide state, so an approval raised by an
// API run can be answered next to one raised by a chat turn.
type harnessManager struct {
	settings harness.Settings
	proc     *service.HarnessProcess

	mu       sync.Mutex
	runtimes map[string]*harness.Runtime
}

func newHarnessManager() *harnessManager {
	return &harnessManager{
		settings: harness.LoadSettings(),
		proc:     service.SharedHarnessProcess(),
		runtimes: make(map[string]*harness.Runtime),
	}
}

// buildRuntime creates the runtime for one request. The agent definition, when
// provided, supplies the prompt and the policy; the scope comes from the
// request, never from the definition.
func (m *harnessManager) buildRuntime(ctx context.Context, req run.Request) (*harness.Runtime, error) {
	policy := definition.PolicyConfig{}
	prompt := ""
	versionID := req.AgentVersionID
	if req.AgentID != "" {
		resolved, err := m.proc.Agents.Resolve(ctx, req.TenantID, req.AgentID, req.AgentVersionID)
		if err != nil {
			return nil, err
		}
		policy = resolved.Definition.Policy
		prompt = resolved.Definition.Prompt
		if resolved.Definition.Addendum != "" {
			prompt += "\n\n" + resolved.Definition.Addendum
		}
		versionID = resolved.VersionID
	}
	req.AgentVersionID = versionID

	// "auto" is the agent's decision to run dangerous tools unattended. Leaving
	// the broker out of the runtime expresses it: the approval gate is built from
	// it, so no gate exists. Explicit deny rules are still applied either way.
	approvals := m.proc.Approvals
	if policy.Approval == definition.ApprovalAuto {
		approvals = nil
	}

	return harness.NewRuntime(ctx, harness.RuntimeConfig{
		Store:          m.proc.Store,
		Checkpointer:   m.proc.Checkpointer(),
		Resolver:       m.proc.Resolver,
		ModelRef:       req.ModelRef,
		Scope:          req.Scope,
		ActorUserID:    req.UserID,
		AgentID:        req.AgentID,
		AgentVersionID: versionID,
		AgentName:      req.AgentName,
		SystemPrompt:   prompt,
		PolicyConfig:   policy,
		MaxRounds:      policy.MaxRounds,
		SkillsRoot:     m.settings.SkillsRoot,
		ExecRoot:       m.settings.ExecRoot,
		Approvals:      approvals,
		Grants:         m.proc.Grants,
		Breakers:       m.proc.Breakers,
		EventLog:       m.proc.Events,
		EventLayers:    service.EventLayers(),
	})
}

func (m *harnessManager) track(runID string, runtime *harness.Runtime) func() {
	m.mu.Lock()
	m.runtimes[runID] = runtime
	m.mu.Unlock()
	return func() {
		m.mu.Lock()
		delete(m.runtimes, runID)
		m.mu.Unlock()
	}
}

// Invoke builds the runtime, registers it for streaming and runs the request.
func (m *harnessManager) Invoke(ctx context.Context, req run.Request) (loop.Outcome, error) {
	runtime, err := m.buildRuntime(ctx, req)
	if err != nil {
		return loop.Outcome{}, err
	}
	defer m.track(req.RunID, runtime)()
	return runtime.Service.Invoke(ctx, req)
}

func (m *harnessManager) runtimeOf(runID string) *harness.Runtime {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.runtimes[runID]
}

// Subscribe relays a run's events. The stream request can arrive before the
// goroutine that built the runtime, so a short wait is part of the contract
// rather than an error: giving up immediately would look like an empty run.
func (m *harnessManager) Subscribe(runID string) chan run.Event {
	deadline := time.Now().Add(3 * time.Second)
	for {
		if runtime := m.runtimeOf(runID); runtime != nil {
			return runtime.Service.Subscribe(runID)
		}
		if time.Now().After(deadline) {
			closed := make(chan run.Event)
			close(closed)
			return closed
		}
		time.Sleep(20 * time.Millisecond)
	}
}

func (m *harnessManager) Unsubscribe(runID string, events chan run.Event) {
	if runtime := m.runtimeOf(runID); runtime != nil {
		runtime.Service.Unsubscribe(runID, events)
		return
	}
	close(events)
}

func (m *harnessManager) Cancel(runID string) bool {
	if runtime := m.runtimeOf(runID); runtime != nil {
		return runtime.Service.Cancel(runID)
	}
	return false
}

func (m *harnessManager) Steer(runID, message string) bool {
	if runtime := m.runtimeOf(runID); runtime != nil {
		return runtime.Service.Steer(runID, message)
	}
	return false
}

func (m *harnessManager) Steps(ctx context.Context, runID string) ([]entity.HarnessAgentStep, error) {
	return m.proc.Store.ListSteps(ctx, runID, 0)
}

// Events reads one run's audit log. The run id is the trace id, so a child run is
// read as its own trace rather than folded into the parent's.
func (m *harnessManager) Events(ctx context.Context, runID string, query harnessapi.EventQuery) ([]*events.Event, error) {
	if m.proc.Events == nil {
		return nil, nil
	}
	filter := events.EventFilter{TraceID: runID, Limit: query.Limit}
	for _, layer := range query.Layers {
		filter.Layers = append(filter.Layers, events.Layer(layer))
	}
	for _, typ := range query.Types {
		filter.Types = append(filter.Types, events.EventType(typ))
	}
	return m.proc.Events.Range(ctx, 0, 0, filter)
}

func (m *harnessManager) Resume(ctx context.Context, req run.Request) (loop.Outcome, error) {
	runtime, err := m.buildRuntime(ctx, req)
	if err != nil {
		return loop.Outcome{}, err
	}
	defer m.track(req.RunID, runtime)()
	return runtime.Service.Resume(ctx, req)
}

func (m *harnessManager) ResolveApproval(runID, requestID string, allow bool, reason string, remember bool) bool {
	pending, hasPending := m.proc.Approvals.Pending(runID, requestID)
	if !m.proc.Approvals.Resolve(runID, requestID, allow, reason) {
		return false
	}
	if remember && allow && hasPending && pending.TenantID != "" {
		_ = m.proc.Grants.Remember(context.Background(), toolkit.GrantFor(pending.TenantID, pending.Call))
	}
	return true
}

// PendingApprovals lists one tenant's waiting calls. The filter is strict: a
// broker shared by the whole process must never hand one tenant's pending
// dangerous call to another tenant's client.
func (m *harnessManager) PendingApprovals(tenantID string) []harnessapi.ApprovalRef {
	refs := m.proc.Approvals.List()
	out := make([]harnessapi.ApprovalRef, 0, len(refs))
	for _, ref := range refs {
		if ref.TenantID != tenantID {
			continue
		}
		runID, _, _ := strings.Cut(ref.Key, ":")
		out = append(out, harnessapi.ApprovalRef{
			RequestID: ref.Key,
			RunID:     runID,
			Tool:      ref.Call.Tool,
			Args:      ref.Call.Arguments,
			Risk:      ref.Risk.String(),
			TenantID:  ref.TenantID,
		})
	}
	return out
}

func (m *harnessManager) ResolveApprovalKey(requestID, tenantID string, allow bool, reason string, remember bool) bool {
	var pending toolkit.PendingRef
	found := false
	for _, ref := range m.proc.Approvals.List() {
		if ref.Key != requestID {
			continue
		}
		if ref.TenantID != tenantID {
			return false
		}
		pending, found = ref, true
		break
	}
	if !found || !m.proc.Approvals.ResolveKey(requestID, allow, reason) {
		return false
	}
	if remember && allow && pending.TenantID != "" {
		_ = m.proc.Grants.Remember(context.Background(), toolkit.GrantFor(pending.TenantID, pending.Call))
	}
	return true
}

// recoverAll resumes the runs a previous worker left behind.
func (m *harnessManager) recoverAll(ctx context.Context) (int, error) {
	runs, err := m.proc.Store.ListRecoverable(ctx, 50)
	if err != nil {
		return 0, err
	}
	recovered := 0
	for _, item := range runs {
		if _, err := m.Resume(ctx, run.Request{
			RunID:    item.ID,
			TenantID: item.TenantID,
			UserID:   item.UserID,
			AgentID:  item.AgentID,
			Scope:    toolkit.Scope{TenantID: item.TenantID},
		}); err != nil {
			_ = m.proc.Store.MarkFailed(ctx, item.ID, "recovery failed: "+err.Error())
			continue
		}
		recovered++
	}
	return recovered, nil
}

// registerHarness mounts the harness API.
//
// It is mounted unconditionally: which agents run, and on which model, is a
// per-agent and per-thread decision stored in the database, so there is no
// process switch that can make an existing agent stop working. A deployment with
// no agents has nothing to run and every endpoint answers empty.
func registerHarness(routerInstance *router.Router) {
	manager := newHarnessManager()
	handler := harnessapi.New(manager)
	routerInstance.SetHarness(func(group *gin.RouterGroup) {
		group.Use(bindCallerTenant())
		handler.Register(group)
	})
	go func() {
		_, _ = manager.recoverAll(context.Background())
	}()
}

// tenantOfUser resolves the tenant a user acts as. The chat services resolve it
// the same way, from the user's own tenant row.
var tenantOfUser = func(ctx context.Context, userID string) (string, error) {
	tenant, err := dao.NewTenantDAO().GetByID(ctx, dao.GetDB(), userID)
	if err != nil {
		return "", err
	}
	return tenant.ID, nil
}

// bindCallerTenant fills the tenant a harness request acts as.
//
// Auth puts tenant_id in the context only where the build resolves it from the
// token; the OSS build resolves nothing there, and every service derives the
// tenant from the user instead. A harness route that scopes by tenant would then
// see an empty one and reject callers it had just authenticated, so fill it here
// rather than assume the enterprise binding.
func bindCallerTenant() gin.HandlerFunc {
	return func(c *gin.Context) {
		if c.GetString("tenant_id") == "" {
			userID := c.GetString("user_id")
			if userID != "" {
				if tenantID, err := tenantOfUser(c.Request.Context(), userID); err == nil && tenantID != "" {
					c.Set("tenant_id", tenantID)
				}
			}
		}
		c.Next()
	}
}
