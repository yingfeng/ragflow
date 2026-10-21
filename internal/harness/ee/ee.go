// Package ee declares the six extension points the enterprise build replaces.
//
// The open-source build ships no implementation: every field of Extensions is
// nil and the runtime must behave correctly without it. An extension adds
// governance around the runtime (audit sinks, identity, redaction, spend caps,
// replay, SIEM export); it never changes loop semantics, which is what keeps an
// agent portable between builds.
package ee

import (
	"context"
	"sync/atomic"

	"ragflow/internal/harness/toolkit"
)

// StoreExtension replaces the run/step persistence backend.
type StoreExtension interface {
	SaveRun(ctx context.Context, runID string, payload []byte) error
	SaveStep(ctx context.Context, runID string, seq int, payload []byte) error
}

// AuthExtension resolves the principal behind a run.
type AuthExtension interface {
	Principal(ctx context.Context, tenantID, userID string) (toolkit.Scope, error)
}

// PIIExtension redacts sensitive spans before text leaves the process.
type PIIExtension interface {
	Redact(ctx context.Context, text string) (string, error)
}

// BudgetExtension enforces tenant-level spend; the loop budget is per run.
type BudgetExtension interface {
	Allow(ctx context.Context, tenantID string, estimatedTokens int) (bool, string, error)
}

// ReplayExtension replays a recorded run for evaluation.
type ReplayExtension interface {
	Replay(ctx context.Context, runID string) ([]byte, error)
}

// SIEMExtension exports audit records to an external sink.
type SIEMExtension interface {
	Emit(ctx context.Context, event map[string]any) error
}

// Extensions is the injected set. Every field may be nil.
type Extensions struct {
	Store  StoreExtension
	Auth   AuthExtension
	PII    PIIExtension
	Budget BudgetExtension
	Replay ReplayExtension
	SIEM   SIEMExtension
}

var current atomic.Value

// Set installs the extensions. It is meant to be called once at process start.
func Set(extensions Extensions) { current.Store(extensions) }

// Get returns the installed extensions, or an empty set.
func Get() Extensions {
	value := current.Load()
	if value == nil {
		return Extensions{}
	}
	extensions, _ := value.(Extensions)
	return extensions
}

// Redact runs the PII extension when present.
func Redact(ctx context.Context, text string) (string, error) {
	if extension := Get().PII; extension != nil {
		return extension.Redact(ctx, text)
	}
	return text, nil
}

// Allow checks the tenant budget when an extension is present.
func Allow(ctx context.Context, tenantID string, estimatedTokens int) (bool, string, error) {
	if extension := Get().Budget; extension != nil {
		return extension.Allow(ctx, tenantID, estimatedTokens)
	}
	return true, "", nil
}

// Emit exports one audit event when a SIEM extension is present.
func Emit(ctx context.Context, event map[string]any) error {
	if extension := Get().SIEM; extension != nil {
		return extension.Emit(ctx, event)
	}
	return nil
}
