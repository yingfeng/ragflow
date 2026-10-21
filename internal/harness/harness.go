// Package harness is the entry point for the embedded agent runtime.
//
// The runtime itself is deliberately split by concern: graph is the execution
// engine, agent/definition owns the immutable agent configuration, agent/loop is
// the fixed reasoning graph, caps supplies tools, policy decides when a run
// stops, context assembles prompts, rsi learns from finished runs and store
// persists state. This package only re-exports the pieces a host needs so
// callers do not have to depend on the internal layout.
package harness

import (
	"context"
	"fmt"

	"ragflow/internal/harness/agent/definition"
	"ragflow/internal/harness/agent/loop"
	"ragflow/internal/harness/agent/model"
	"ragflow/internal/harness/agent/run"
	"ragflow/internal/harness/agent/subagent"
	"ragflow/internal/harness/policy"
	"ragflow/internal/harness/rsi"
	"ragflow/internal/harness/store"
	"ragflow/internal/harness/toolkit"
	"ragflow/internal/harness/toolkit/builtin"
	"ragflow/internal/harness/toolkit/rag"
	capsubagent "ragflow/internal/harness/toolkit/subagent"
)

// Configuration types.
type (
	// Definition is an agent's immutable configuration.
	Definition = definition.Definition
	// PolicyConfig bounds one run.
	PolicyConfig = definition.PolicyConfig
	// Scope is the run's data reach.
	Scope = toolkit.Scope
	// Capability contributes tools.
	Capability = toolkit.Capability
	// Registry routes tool calls.
	Registry = toolkit.Registry
	// Evidence is one citation-bearing fact.
	Evidence = toolkit.Evidence
)

// Runtime types.
type (
	// LoopConfig configures the built-in reasoning graph.
	LoopConfig = loop.Config
	// LoopDeps are the reasoning graph's dependencies.
	LoopDeps = loop.Deps
	// LoopInput starts a run.
	LoopInput = loop.Input
	// Outcome is a finished run.
	Outcome = loop.Outcome
	// StepRecord is one auditable step.
	StepRecord = loop.StepRecord
	// Service executes runs and owns their leases.
	Service = run.Service
	// ServiceConfig configures the run service.
	ServiceConfig = run.Config
	// Request starts a run.
	Request = run.Request
	// Event is one streamed runtime update.
	Event = run.Event
	// RunStore persists runs, steps, feedback and lessons.
	RunStore = store.RunStore
	// Checkpointer stores graph checkpoints.
	Checkpointer = store.Checkpointer
)

// Builders.

// BuildLoop compiles the reasoning graph.
func BuildLoop(cfg loop.Config) (*loop.Runner, error) { return loop.Build(cfg) }

// NewService builds the run service.
func NewService(cfg run.Config) (*run.Service, error) { return run.New(cfg) }

// NewRAGCapability exposes the retrieval tools for a tenant and dataset scope.
func NewRAGCapability(ctx context.Context, tenantID string, datasetIDs []string) (*rag.Capability, error) {
	return rag.New(ctx, rag.Config{TenantID: tenantID, DatasetIDs: datasetIDs})
}

// NewBuiltinCapability exposes the generic tools (think, todo_write,
// run_javascript).
func NewBuiltinCapability(ctx context.Context) (*builtin.Capability, error) {
	return builtin.New(ctx)
}

// NewModel adapts a RAGFlow chat model for the loop. The chat model comes from
// service.ModelFactory, so tenant resolution, API configuration and provider
// behaviour stay in one place; the harness only translates the wire types.
func NewModel(chat model.ChatCompleter, config model.ProviderConfig) model.Model {
	return model.NewProvider(chat, config)
}

// ChatCompleter is the model-provider port NewModel accepts.
type ChatCompleter = model.ChatCompleter

// ModelProviderConfig configures the adapter built by NewModel.
type ModelProviderConfig = model.ProviderConfig

// NewSubagentCapability builds the delegation tools over deps. deps.Model and
// deps.Registry are what the children inherit; the children lose both
// delegation tools, so one run cannot fork without a bound.
func NewSubagentCapability(deps loop.Deps, policyConfig definition.PolicyConfig, maxRounds int) (*capsubagent.Capability, error) {
	spawner, err := subagent.New(deps, policyConfig, maxRounds)
	if err != nil {
		return nil, err
	}
	return capsubagent.New(spawner), nil
}

// RegisterSubagentTools registers the delegation tools on registry. A host that
// wants a run to be able to delegate calls this once, next to the other
// capability registrations.
func RegisterSubagentTools(registry *toolkit.Registry, deps loop.Deps, policyConfig definition.PolicyConfig, maxRounds int) error {
	if registry == nil {
		return fmt.Errorf("harness: a registry is required to register the sub-agent tools")
	}
	capability, err := NewSubagentCapability(deps, policyConfig, maxRounds)
	if err != nil {
		return err
	}
	return registry.Register(capability)
}

// Policy primitives.
type (
	// Verdict is a policy decision.
	Verdict = policy.Verdict
	// Snapshot is what policies decide on.
	Snapshot = policy.Snapshot
)

// RSI cycle.

// Learn runs one self-improvement cycle over the given signals.
func Learn(cfg rsi.Config, signals []rsi.Signal) ([]rsi.Result, error) {
	cycle := rsi.New(cfg)
	return cycle.Learn(nil, signals)
}
