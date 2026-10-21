package api

import (
	"context"
	"os"
	"sync"
	"time"

	"ragflow/internal/dao"
	"ragflow/internal/engine/kvrocks"
	"ragflow/internal/entity"
	"ragflow/internal/harness"
	harnessapi "ragflow/internal/harness/agent/api"
	"ragflow/internal/harness/agent/definition"
	"ragflow/internal/harness/agent/loop"
	"ragflow/internal/harness/agent/model"
	"ragflow/internal/harness/agent/run"
	"ragflow/internal/harness/events"
	"ragflow/internal/harness/graph/checkpoint"
	"ragflow/internal/harness/store"
	"ragflow/internal/harness/toolkit"
	"ragflow/internal/harness/toolkit/exec"
	"ragflow/internal/router"
	"ragflow/internal/service"
)

// harnessSettings is the deployment surface of the agent harness. Every field
// has a safe default: enabled=false means the routes are not mounted at all.
type harnessSettings struct {
	Enabled       bool
	ModelRef      string
	SkillsRoot    string
	ExecRoot      string
	ExecSandbox   string
	ExecUnsandbox bool
}

func loadHarnessSettings() harnessSettings {
	return harnessSettings{
		Enabled:       os.Getenv("HARNESS_ENABLED") == "true",
		ModelRef:      os.Getenv("HARNESS_MODEL_REF"),
		SkillsRoot:    os.Getenv("HARNESS_SKILLS_ROOT"),
		ExecRoot:      os.Getenv("HARNESS_EXEC_ROOT"),
		ExecSandbox:   os.Getenv("HARNESS_EXEC_SANDBOX"),
		ExecUnsandbox: os.Getenv("HARNESS_EXEC_UNSANDBOXED") == "true",
	}
}

// harnessResolver adapts service.ModelFactory to the harness port. It lives here
// rather than in the harness so the harness never imports business services.
type harnessResolver struct {
	factory *service.ModelFactory
}

func (r harnessResolver) Resolve(ctx context.Context, tenantID, modelRef string) (model.ChatCompleter, error) {
	return r.factory.NewChatModel(ctx, service.ModelAccess{TenantID: tenantID}, modelRef)
}

// harnessManager owns the runtimes of the runs executing on this worker.
//
// A runtime is built per run: the tool surface depends on the run's scope (which
// datasets it may read), and sharing one runtime would leak one tenant's
// datasets into another tenant's tool list.
type harnessManager struct {
	settings  harnessSettings
	store     *store.RunStore
	eventLog  *store.EventStore
	grants    toolkit.GrantStore
	breakers  *model.BreakerRegistry
	checker   checkpoint.BaseCheckpointer
	resolver  harness.ModelResolver
	agents    *service.HarnessAgentService
	approvals *toolkit.ApprovalBroker

	mu       sync.Mutex
	runtimes map[string]*harness.Runtime
}

func newHarnessManager() (*harnessManager, error) {
	manager := &harnessManager{
		settings: loadHarnessSettings(),
		store:    store.NewRunStore(dao.GetDB()),
		eventLog: store.NewEventStore(dao.GetDB()),
		grants:   store.NewGrantStore(dao.GetDB()),
		// One registry per process: a provider outage discovered by one run must
		// fail the other runs fast instead of each burning its own retries.
		breakers:  model.NewBreakerRegistry(5, 30*time.Second),
		resolver:  harnessResolver{factory: service.NewModelFactory()},
		agents:    service.NewHarnessAgentService(),
		approvals: toolkit.NewApprovalBroker(0),
		runtimes:  make(map[string]*harness.Runtime),
	}
	if kvrocks.IsEnabled() {
		if client := kvrocks.Get(); client != nil {
			if raw := client.GetClient(); raw != nil {
				manager.checker = store.NewCheckpointer(raw, 0)
			}
		}
	}
	return manager, nil
}

// buildRuntime creates the runtime for one request. The agent definition, when
// provided, supplies the prompt and the policy; the scope comes from the
// request, never from the definition.
func (m *harnessManager) buildRuntime(ctx context.Context, req run.Request) (*harness.Runtime, error) {
	policy := definition.PolicyConfig{}
	prompt := ""
	versionID := req.AgentVersionID
	if req.AgentID != "" {
		resolved, err := m.agents.Resolve(ctx, req.TenantID, req.AgentID, req.AgentVersionID)
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

	return harness.NewRuntime(ctx, harness.RuntimeConfig{
		Store:                m.store,
		Checkpointer:         m.checker,
		Resolver:             m.resolver,
		ModelRef:             m.settings.ModelRef,
		Scope:                req.Scope,
		AgentID:              req.AgentID,
		AgentVersionID:       versionID,
		AgentName:            req.AgentName,
		SystemPrompt:         prompt,
		PolicyConfig:         policy,
		MaxRounds:            policy.MaxRounds,
		SkillsRoot:           m.settings.SkillsRoot,
		ExecRoot:             m.settings.ExecRoot,
		ExecSandbox:          exec.SandboxMode(m.settings.ExecSandbox),
		AllowUnsandboxedExec: m.settings.ExecUnsandbox,
		Approvals:            m.approvals,
		Grants:               m.grants,
		Breakers:             m.breakers,
		EventLog:             m.eventLog,
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
	return m.store.ListSteps(ctx, runID, 0)
}

// Events reads one run's audit log. The run id is the trace id, so a child run is
// read as its own trace rather than folded into the parent's.
func (m *harnessManager) Events(ctx context.Context, runID string, query harnessapi.EventQuery) ([]*events.Event, error) {
	if m.eventLog == nil {
		return nil, nil
	}
	filter := events.EventFilter{TraceID: runID, Limit: query.Limit}
	for _, layer := range query.Layers {
		filter.Layers = append(filter.Layers, events.Layer(layer))
	}
	for _, typ := range query.Types {
		filter.Types = append(filter.Types, events.EventType(typ))
	}
	return m.eventLog.Range(ctx, 0, 0, filter)
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
	pending, hasPending := m.approvals.Pending(runID, requestID)
	if !m.approvals.Resolve(runID, requestID, allow, reason) {
		return false
	}
	if remember && allow && hasPending && pending.TenantID != "" {
		_ = m.grants.Remember(context.Background(), toolkit.GrantFor(pending.TenantID, pending.Call))
	}
	return true
}

// recoverAll resumes the runs a previous worker left behind.
func (m *harnessManager) recoverAll(ctx context.Context) (int, error) {
	runs, err := m.store.ListRecoverable(ctx, 50)
	if err != nil {
		return 0, err
	}
	recovered := 0
	for _, item := range runs {
		if _, err := m.Resume(ctx, run.Request{
			RunID:    item.ID,
			TenantID: item.TenantID,
			AgentID:  item.AgentID,
			Scope:    toolkit.Scope{TenantID: item.TenantID},
		}); err != nil {
			_ = m.store.MarkFailed(ctx, item.ID, "recovery failed: "+err.Error())
			continue
		}
		recovered++
	}
	return recovered, nil
}

// registerHarness mounts the harness API when the deployment enables it.
func registerHarness(routerInstance *router.Router) {
	settings := loadHarnessSettings()
	if !settings.Enabled || settings.ModelRef == "" {
		return
	}
	manager, err := newHarnessManager()
	if err != nil {
		return
	}
	handler := harnessapi.New(manager)
	routerInstance.SetHarness(handler.Register)
	go func() {
		_, _ = manager.recoverAll(context.Background())
	}()
}
