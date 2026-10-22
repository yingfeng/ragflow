package harness

import (
	"context"
	"fmt"
	"time"

	"ragflow/internal/agentic_rag"
	"ragflow/internal/cli/filesystem"
	"ragflow/internal/harness/agent/definition"
	"ragflow/internal/harness/agent/loop"
	"ragflow/internal/harness/agent/model"
	"ragflow/internal/harness/agent/run"
	agentsubagent "ragflow/internal/harness/agent/subagent"
	agentctx "ragflow/internal/harness/context"
	"ragflow/internal/harness/events"
	"ragflow/internal/harness/graph/checkpoint"
	"ragflow/internal/harness/store"
	"ragflow/internal/harness/toolkit"
	"ragflow/internal/harness/toolkit/builtin"
	"ragflow/internal/harness/toolkit/exec"
	"ragflow/internal/harness/toolkit/fs"
	"ragflow/internal/harness/toolkit/mcp"
	"ragflow/internal/harness/toolkit/rag"
	"ragflow/internal/harness/toolkit/skill"
	"ragflow/internal/harness/toolkit/subagent"
)

// ModelResolver resolves one tenant's chat model for one actor. The service
// layer implements it over service.ModelFactory, which keeps tenant resolution
// and API keys out of the harness while still making the RAGFlow model provider
// the only way a harness run talks to a model.
//
// userID is the actor the run acts for, not a convenience: model access (which
// provider, whose key) resolves per user, and a tenant-only lookup silently
// falls back to a provider the tenant may have no credit on.
type ModelResolver interface {
	Resolve(ctx context.Context, tenantID, userID, modelRef string) (model.ChatCompleter, error)
}

// RuntimeConfig describes one run's wiring. Everything that varies per tenant or
// per run (scope, model, filesystem, skills root) lives here; the harness holds
// no global state, so a runtime is cheap to build and safe to keep per request.
type RuntimeConfig struct {
	// Store persists runs and steps. Nil disables persistence, which also
	// disables leases and crash recovery.
	Store *store.RunStore
	// Checkpointer stores graph state. Nil keeps checkpoints in memory, so a
	// pod restart loses paused runs.
	Checkpointer checkpoint.BaseCheckpointer
	// Resolver and ModelRef select the chat model for this run.
	Resolver ModelResolver
	ModelRef string
	// Model overrides Resolver when the host already holds the model to use —
	// a failover chain, for instance, which the host can build from refs the
	// harness never sees. Nil means resolve ModelRef through Resolver.
	Model model.ChatCompleter
	// Scope is the run's data reach (tenant + datasets).
	Scope toolkit.Scope
	// ActorUserID is the user the run acts for. Model access resolves per user,
	// so leaving it empty can select a different provider than the caller's.
	ActorUserID string
	// Agent identity, written to the run record.
	AgentID        string
	AgentVersionID string
	AgentName      string
	// SystemPrompt is the agent definition's instruction block.
	SystemPrompt string
	// PolicyConfig and MaxRounds bound the loop.
	PolicyConfig definition.PolicyConfig
	MaxRounds    int
	// MaxOutputTokens caps one completion.
	MaxOutputTokens int
	// Filesystem backs fs_* tools. Nil leaves them unregistered: a read tool
	// over a missing engine would fail on every call.
	Filesystem *filesystem.Engine
	// SkillsRoot is the directory holding <name>/SKILL.md. Empty disables skills.
	SkillsRoot string
	// ExecRoot is the shell workspace. Empty leaves exec_shell unregistered:
	// there is no sandbox to run it in, and registering it anyway would expose
	// the host shell.
	ExecRoot string
	// ExecSandbox selects the isolation mechanism. Empty auto-detects; "none"
	// additionally requires AllowUnsandboxedExec.
	ExecSandbox exec.SandboxMode
	// AllowUnsandboxedExec accepts the host shell when no sandbox works. It is a
	// development escape hatch: a production deployment must not set it.
	AllowUnsandboxedExec bool
	// WebSearch enables web_search inside the retrieval capability.
	WebSearch agentic_rag.WebSearchFunc
	// MCPClient adds one MCP server's tools.
	MCPClient mcp.Client
	// Approver gates tool calls (see toolkit.ApprovalGate).
	Approver toolkit.Approver
	// Approvals is the operator rendezvous. When set, dangerous calls wait for a
	// verdict instead of being rejected outright.
	Approvals *toolkit.ApprovalBroker
	// Grants remembers "allow always" answers. Nil keeps no standing permission,
	// which means every dangerous call asks again.
	Grants toolkit.GrantStore
	// Lessons supplies experience entries for the system prompt.
	Lessons func(ctx context.Context) []string
	// Breakers shares provider health across runs in this process. Nil disables
	// the cross-run circuit breaker, leaving only per-call retries.
	Breakers *model.BreakerRegistry
	// EventLog receives the fine-grained audit log. Nil leaves only the run and
	// step tables, which record what happened but not why.
	EventLog events.EventLog
	// EventLayers selects the data classes written to EventLog. Nil drops
	// diagnostics (raw model chunks).
	EventLayers events.LayerFilter
	// Prices enable cost accounting.
	PromptCostPerToken     float64
	CompletionCostPerToken float64
	// Concurrency bounds parallel read-only tool calls.
	Concurrency int
}

// Runtime is a fully wired harness for one scope.
type Runtime struct {
	Service  *run.Service
	Registry *toolkit.Registry
	// SelfDelegation is false when the sub-agent capability could not be wired.
	SelfDelegation bool
	// Notes are the capabilities that degraded at wiring time. The caller logs
	// them: a missing sandbox or a stripped delegation tool changes what the
	// agent can do, and silence would look like a working deployment.
	Notes []string
}

// NewRuntime wires model, tools, loop and run service. It fails fast on a
// missing model; every other capability degrades by omission and is reported in
// the startup log by the caller.
func NewRuntime(ctx context.Context, cfg RuntimeConfig) (*Runtime, error) {
	chat := cfg.Model
	if chat == nil {
		if cfg.Resolver == nil {
			return nil, fmt.Errorf("harness: a model resolver is required")
		}
		resolved, err := cfg.Resolver.Resolve(ctx, cfg.Scope.TenantID, cfg.ActorUserID, cfg.ModelRef)
		if err != nil {
			return nil, fmt.Errorf("harness: resolve model %q: %w", cfg.ModelRef, err)
		}
		chat = resolved
	}
	chatModel := NewModel(chat, ModelProviderConfig{})

	registry := toolkit.NewRegistry()
	notes, err := registerCapabilities(ctx, registry, cfg)
	if err != nil {
		return nil, err
	}

	builder := agentctx.New(agentctx.Config{
		CountTokens: agentctx.DefaultCountTokens,
		Summarize:   agentctx.NewSummarizer(chatModel, 0),
	})

	approver := cfg.Approver
	if approver == nil && cfg.Approvals != nil {
		approver = toolkit.NewApprovalGate(cfg.Approvals, toolkit.RiskDangerous)
	}
	// Rules and standing grants sit in front of the approval channel: a deny must
	// never reach a human, and a granted call must never wake one.
	rules, err := toolkit.NewRuleSet(cfg.PolicyConfig.Permissions)
	if err != nil {
		return nil, fmt.Errorf("harness: permissions: %w", err)
	}
	if len(rules.Rules()) > 0 || (cfg.Grants != nil && cfg.Scope.TenantID != "") {
		approver = &toolkit.PermissionApprover{
			Rules:    rules,
			Grants:   cfg.Grants,
			TenantID: cfg.Scope.TenantID,
			Next:     approver,
		}
	}

	// The environment block is assembled per run, after the operator's prompt and
	// the runtime rules: it is the only part of the system prompt that changes on
	// its own, and a model that does not know the date will date-stamp facts on
	// its own.
	systemPrompt := definition.EffectiveSystemPrompt(cfg.SystemPrompt) + "\n\n" +
		definition.EnvironmentSection(time.Now(), cfg.ExecRoot)

	deps := loop.Deps{
		Model:                  chatModel,
		Registry:               registry,
		Builder:                builder,
		ModelName:              cfg.ModelRef,
		Breaker:                cfg.Breakers.For(cfg.ModelRef),
		SystemPrompt:           systemPrompt,
		Lessons:                cfg.Lessons,
		Concurrency:            cfg.Concurrency,
		MaxOutputTokens:        cfg.MaxOutputTokens,
		Scope:                  cfg.Scope,
		PromptCostPerToken:     cfg.PromptCostPerToken,
		CompletionCostPerToken: cfg.CompletionCostPerToken,
		Approver:               approver,
	}

	// Delegation is registered after the deps exist, because the spawner needs
	// the very loop it will run children on.
	selfDelegation := false
	if spawner, err := agentsubagent.New(deps, cfg.PolicyConfig, 0); err == nil {
		// The store is what makes a child inspectable: without it the control
		// tools cannot answer, so the spawner reports the plane as unavailable
		// and only the blocking delegation tools are registered.
		spawner = spawner.WithStore(cfg.Store)
		if err := registry.Register(subagent.New(spawner)); err == nil {
			selfDelegation = true
		}
	}

	service, err := run.New(run.Config{
		Loop: loop.Config{
			Deps:         deps,
			PolicyConfig: cfg.PolicyConfig,
			MaxRounds:    cfg.MaxRounds,
			Checkpointer: cfg.Checkpointer,
		},
		Store:       cfg.Store,
		Approvals:   cfg.Approvals,
		Grants:      cfg.Grants,
		Events:      cfg.EventLog,
		EventLayers: cfg.EventLayers,
	})
	if err != nil {
		return nil, err
	}
	if !selfDelegation {
		notes = append(notes, "delegation tools are unavailable: the sub-agent spawner could not be built")
	}
	return &Runtime{Service: service, Registry: registry, SelfDelegation: selfDelegation, Notes: notes}, nil
}

// registerCapabilities installs the tools this deployment actually has. A
// capability is registered only when its dependency exists, so the model never
// sees a tool that would fail on every call. It returns the degradations it took,
// which the caller surfaces: a missing sandbox changes what the agent can do.
func registerCapabilities(ctx context.Context, registry *toolkit.Registry, cfg RuntimeConfig) ([]string, error) {
	notes := make([]string, 0, 2)

	builtinCapability, err := builtin.New(ctx)
	if err != nil {
		return nil, fmt.Errorf("harness: builtin tools: %w", err)
	}
	if err := registry.Register(builtinCapability); err != nil {
		return nil, err
	}

	retrieval, err := rag.New(ctx, rag.Config{
		TenantID:   cfg.Scope.TenantID,
		DatasetIDs: cfg.Scope.Datasets,
		WebSearch:  cfg.WebSearch,
	})
	if err != nil {
		return nil, fmt.Errorf("harness: retrieval tools: %w", err)
	}
	if err := registry.Register(retrieval); err != nil {
		return nil, err
	}

	if cfg.Filesystem != nil {
		if err := registry.Register(fs.New(fs.FromEngine(cfg.Filesystem), fs.NewMemoryWritable())); err != nil {
			return nil, err
		}
	}
	if cfg.SkillsRoot != "" {
		if err := registry.Register(skill.New(&skill.FSLoader{Root: cfg.SkillsRoot})); err != nil {
			return nil, err
		}
	}
	if cfg.ExecRoot != "" {
		// Shell access is only offered inside a sandbox. When none works the tool
		// is left out rather than run on the host.
		runner, err := exec.NewRunner(exec.RunnerConfig{
			Root:             cfg.ExecRoot,
			Mode:             cfg.ExecSandbox,
			AllowUnsandboxed: cfg.AllowUnsandboxedExec,
		})
		if err != nil {
			notes = append(notes, "exec_shell is unavailable: "+err.Error())
		} else if err := registry.Register(exec.New(runner)); err != nil {
			return nil, err
		}
	}
	if cfg.MCPClient != nil {
		if err := registry.Register(mcp.New("default", cfg.MCPClient)); err != nil {
			return nil, err
		}
	}
	return notes, nil
}
