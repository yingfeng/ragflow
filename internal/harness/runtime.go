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
	// Sampling pins the model's sampling behaviour. Nil keeps the provider
	// default, which moves with the model — see definition.SamplingConfig.
	Sampling *definition.SamplingConfig
	// PolicyConfig and MaxRounds bound the loop.
	PolicyConfig definition.PolicyConfig
	MaxRounds    int
	// MaxOutputTokens caps one completion.
	MaxOutputTokens int
	// Filesystem backs fs_* tools. Nil leaves them unregistered: a read tool
	// over a missing engine would fail on every call.
	Filesystem *filesystem.Engine
	// SkillsRoot is a directory holding <name>/SKILL.md, layered over the
	// built-in skills. Empty leaves only the built-in set.
	SkillsRoot string
	// SkillLoader supplies skills from the platform's skillspace. It layers over
	// the built-in set, so an installed skill with a bundled name replaces it
	// instead of appearing twice.
	SkillLoader skill.Loader
	// EnabledSkills is the agent's selection from SkillLoader. Empty means the
	// agent runs with the built-in skills only: a tenant's skillspace is a
	// catalog, and putting all of it in one agent's prompt is not a default.
	EnabledSkills []string
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
	// Sampling is pinned from the definition: the provider default moves with
	// the model, and an unpinned run is not reproducible across a model swap
	// (the agentic pipeline pins 0.5 for the same reason).
	providerCfg := model.ProviderConfig{}
	if cfg.Sampling != nil {
		providerCfg.Temperature = cfg.Sampling.Temperature
		providerCfg.TopP = cfg.Sampling.TopP
		providerCfg.Thinking = cfg.Sampling.Thinking
	}
	chatModel := NewModel(chat, providerCfg)

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

	// Standing instructions follow the runtime rules. They come from auto_load
	// skills, so the policy is a file an operator edits, not a constant; the
	// registry only decides which capabilities get to contribute one.
	//
	// The environment block is assembled last: it is the only part of the system
	// prompt that changes on its own, and everything before it has to stay
	// byte-stable for the provider's prefix cache.
	systemPrompt := definition.EffectiveSystemPrompt(cfg.SystemPrompt)
	if note := registry.SystemNotes(ctx); note != "" {
		systemPrompt += "\n\n" + note
	}
	systemPrompt += "\n\n" + definition.EnvironmentSection(time.Now(), cfg.ExecRoot)

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

	// Delegation is registered after the service, so the children inherit the
	// service's step sink: a child's rounds belong in the run tables and the
	// stream next to its parent's, and a spawner built from the bare deps would
	// drop every one of them. The registry is read per round, so registering the
	// tools after the graph is compiled still offers them to the parent, and the
	// spawner's snapshot of the registry (taken before this registration) is what
	// keeps children from delegating again.
	childDeps := deps
	childDeps.Sink = service.StepSink()
	selfDelegation := false
	if spawner, spawnErr := agentsubagent.New(childDeps, cfg.PolicyConfig, 0); spawnErr == nil {
		// The store is what makes a child inspectable: without it the control
		// tools cannot answer, so the spawner reports the plane as unavailable
		// and only the blocking delegation tools are registered. The audit log
		// makes the child's own trace readable under its run id.
		spawner = spawner.WithStore(cfg.Store).
			WithAudit(cfg.EventLog, cfg.EventLayers).
			WithAgentIdentity(cfg.AgentID)
		if registerErr := registry.Register(subagent.New(spawner)); registerErr == nil {
			selfDelegation = true
		}
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
	// Skills are always registered: the built-in set (the retrieval policy that
	// every turn runs under) ships with the binary, so a deployment without a
	// skills directory still has policy. Installed skills layer over it, and the
	// agent's selection narrows only that installed layer.
	if err := registry.Register(skill.New(skillLayers(cfg))); err != nil {
		return nil, err
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

// skillLayers composes the skill sources in precedence order: the bundled set is
// the floor, the tenant directory overrides it, and the platform skillspace
// contributes only what the agent selected.
func skillLayers(cfg RuntimeConfig) skill.Loader {
	layers := []skill.Loader{skill.Bundled()}
	if cfg.SkillsRoot != "" {
		layers = append(layers, skill.FSLoader{Root: cfg.SkillsRoot})
	}
	if cfg.SkillLoader != nil {
		layers = append(layers, skill.SelectedLoader{Inner: cfg.SkillLoader, Names: cfg.EnabledSkills})
	}
	return skill.MergedLoader{Layers: layers}
}
