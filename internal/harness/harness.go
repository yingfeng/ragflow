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

	"ragflow/internal/harness/agent/definition"
	"ragflow/internal/harness/agent/loop"
	"ragflow/internal/harness/agent/model"
	"ragflow/internal/harness/agent/run"
	"ragflow/internal/harness/policy"
	"ragflow/internal/harness/rsi"
	"ragflow/internal/harness/store"
	"ragflow/internal/harness/toolkit"
	"ragflow/internal/harness/toolkit/builtin"
	"ragflow/internal/harness/toolkit/rag"
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
