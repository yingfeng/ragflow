package service

import (
	"sync"
	"time"

	"ragflow/internal/dao"
	"ragflow/internal/engine/kvrocks"
	"ragflow/internal/harness"
	harnessmodel "ragflow/internal/harness/agent/model"
	"ragflow/internal/harness/events"
	"ragflow/internal/harness/graph/checkpoint"
	"ragflow/internal/harness/store"
	"ragflow/internal/harness/toolkit"
)

// HarnessProcess is the state every harness run in this process shares: the run
// and event stores, the standing grants, the provider breakers, the approval
// broker, the agent definitions and the model resolver.
//
// It is process-wide because sharing is the point. A breaker that were per turn
// would never see a second failure; a broker that were per turn could not be
// answered by an operator holding a different request; and two brokers would
// silently split the pending approvals of one deployment in half.
type HarnessProcess struct {
	Store      *store.RunStore
	Events     *store.EventStore
	Grants     toolkit.GrantStore
	Breakers   *harnessmodel.BreakerRegistry
	Approvals  *toolkit.ApprovalBroker
	Agents     *HarnessAgentService
	Resolver   harness.ModelResolver
	Checkpoint checkpoint.BaseCheckpointer
}

var harnessProcess struct {
	once sync.Once
	proc *HarnessProcess
}

// SharedHarnessProcess returns the process-wide harness state, building it on
// first use. Both entry points use it: the chat pipeline runs a turn with it,
// and the harness API builds its runtimes from it, so a chat turn's approval can
// be answered through the harness endpoint and vice versa.
func SharedHarnessProcess() *HarnessProcess {
	harnessProcess.once.Do(func() {
		db := dao.GetDB()
		proc := &HarnessProcess{
			Store:    store.NewRunStore(db),
			Events:   store.NewEventStore(db),
			Grants:   store.NewGrantStore(db),
			Breakers: harnessmodel.NewBreakerRegistry(5, 30*time.Second),
			// One broker per process: a pending approval is answered by whoever
			// holds the API, not by the goroutine that happens to own the run.
			Approvals: toolkit.NewApprovalBroker(0),
			Agents:    NewHarnessAgentService(),
			Resolver:  NewHarnessModelResolver(),
		}
		harnessProcess.proc = proc
	})
	return harnessProcess.proc
}

// Checkpointer returns the run checkpointer, or nil when no durable store is
// configured (checkpoints then live in the runtime's memory). It is resolved
// lazily rather than in SharedHarnessProcess because kvrocks is initialised by
// the server bootstrap, which may run after the first shared access.
func (p *HarnessProcess) Checkpointer() checkpoint.BaseCheckpointer {
	if p == nil {
		return nil
	}
	if p.Checkpoint != nil {
		return p.Checkpoint
	}
	if !kvrocks.IsEnabled() {
		return nil
	}
	client := kvrocks.Get()
	if client == nil {
		return nil
	}
	raw := client.GetClient()
	if raw == nil {
		return nil
	}
	p.Checkpoint = store.NewCheckpointer(raw, 0)
	return p.Checkpoint
}

// EventLayers is the audit depth every run writes: control and audit facts plus
// the transcript, without the per-chunk diagnostics that would dominate the
// table. It is one value for both entry points so a chat turn and an API run
// produce the same audit trail.
func EventLayers() events.LayerFilter { return events.DefaultLayerFilter }
