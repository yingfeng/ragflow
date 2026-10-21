package subagent

import (
	"context"
	"fmt"

	"github.com/google/uuid"

	"ragflow/internal/harness/agent/loop"
	"ragflow/internal/harness/toolkit"
	"ragflow/internal/harness/toolkit/subagent"
)

// ControlPlaneEnabled reports whether this spawner can serve the control tools.
// Listing and inspecting a child reads the run store, so without one the
// control tools would always fail and are therefore not offered.
func (s *Spawner) ControlPlaneEnabled() bool { return s != nil && s.store != nil }

// childHandle is one child that is running on this worker. A child that lives in
// another process has no handle: List still reports it from the store, but Stop
// and Message can only reach a child this worker owns.
type childHandle struct {
	cancel context.CancelFunc
	inbox  *loop.Inbox
}

// Start opens a child run and returns immediately.
//
// The child is bound to the parent's context rather than detached from it: when
// the parent run is cancelled or finishes, its children are cancelled too. A
// child that outlived its parent would keep spending budget on a question nobody
// can read the answer to.
func (s *Spawner) Start(ctx context.Context, req subagent.Request) (string, error) {
	if s == nil {
		return "", fmt.Errorf("subagent: spawner is not configured")
	}
	runner, err := s.runnerFor(s.childRounds(ctx))
	if err != nil {
		return "", err
	}

	runID := uuid.NewString()
	childCtx, cancel := context.WithCancel(ctx)
	inbox := loop.NewInbox()
	childCtx = loop.WithInbox(childCtx, inbox)
	childCtx = subagent.WithDepth(childCtx, req.Depth)
	if remaining, ok := toolkit.RemainingRounds(ctx); ok {
		childCtx = toolkit.WithRemainingRounds(childCtx, remaining)
	}

	s.openChild(ctx, runID, toolkit.RunIDFrom(ctx), req)

	s.mu.Lock()
	if s.children == nil {
		s.children = make(map[string]*childHandle)
	}
	s.children[runID] = &childHandle{cancel: cancel, inbox: inbox}
	s.mu.Unlock()

	go func() {
		defer func() {
			s.mu.Lock()
			delete(s.children, runID)
			s.mu.Unlock()
			cancel()
		}()
		outcome, runErr := runner.Invoke(childCtx, &loop.Input{
			RunID:       runID,
			AgentName:   req.AgentID,
			UserMessage: req.Prompt,
			Scope:       req.Scope,
		}, nil)
		// The parent may already be gone, but the child's terminal row still has
		// to land: a child stuck in "running" is what crash recovery mistakes for
		// a live run.
		s.closeChild(context.WithoutCancel(ctx), runID, outcome, runErr)
	}()

	return runID, nil
}

// List reports child runs. Without ids it lists the children of the calling run.
func (s *Spawner) List(ctx context.Context, runIDs []string) ([]subagent.ChildState, error) {
	if s == nil || s.store == nil {
		return nil, fmt.Errorf("subagent: inspecting children requires a run store")
	}

	records := make([]childRecord, 0, len(runIDs))
	if len(runIDs) == 0 {
		children, err := s.store.ListChildren(ctx, toolkit.RunIDFrom(ctx))
		if err != nil {
			return nil, err
		}
		for _, child := range children {
			records = append(records, childRecord{child.ID, child.Status, child.Output, child.Error})
		}
	} else {
		for _, runID := range runIDs {
			record, err := s.store.GetRun(ctx, runID)
			if err != nil {
				return nil, err
			}
			if record == nil {
				records = append(records, childRecord{runID: runID, status: "unknown", errMessage: "run not found"})
				continue
			}
			records = append(records, childRecord{record.ID, record.Status, record.Output, record.Error})
		}
	}

	states := make([]subagent.ChildState, 0, len(records))
	for _, record := range records {
		states = append(states, subagent.ChildState{
			RunID:  record.runID,
			Status: record.status,
			Output: record.output,
			Error:  record.errMessage,
		})
	}
	return states, nil
}

// Stop cancels a child owned by this worker.
func (s *Spawner) Stop(_ context.Context, runID string) bool {
	if s == nil {
		return false
	}
	s.mu.Lock()
	handle, ok := s.children[runID]
	s.mu.Unlock()
	if !ok {
		return false
	}
	handle.cancel()
	return true
}

// Message delivers one message to a child owned by this worker.
func (s *Spawner) Message(_ context.Context, runID, message string) bool {
	if s == nil {
		return false
	}
	s.mu.Lock()
	handle, ok := s.children[runID]
	s.mu.Unlock()
	if !ok {
		return false
	}
	return handle.inbox.Push(message)
}

// childRounds is the round budget a child gets: half of what the parent has left,
// rounded up, so the parent can still answer after collecting the result.
func (s *Spawner) childRounds(ctx context.Context) int {
	maxRounds := s.maxRounds
	if remaining, ok := toolkit.RemainingRounds(ctx); ok {
		slice := (remaining + 1) / 2
		if slice < 1 {
			slice = 1
		}
		if slice < maxRounds {
			maxRounds = slice
		}
	}
	return maxRounds
}

// childRecord keeps the four columns the control plane reports.
type childRecord struct {
	runID      string
	status     string
	output     string
	errMessage string
}
