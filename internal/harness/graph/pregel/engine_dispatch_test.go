package pregel

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"

	"ragflow/internal/harness/graph/channels"
	"ragflow/internal/harness/graph/constants"
	"ragflow/internal/harness/graph/graph"
	"ragflow/internal/harness/graph/types"
)

// TestEngine_BSPDispatchesEverySuccessorOfParallelStep guards the BSP frontier:
// when one superstep completes several nodes, the next superstep must dispatch
// the successors of all of them, not only the last one merged.
func TestEngine_BSPDispatchesEverySuccessorOfParallelStep(t *testing.T) {
	var mu sync.Mutex
	ran := make(map[string]bool)
	record := func(name string) types.NodeFunc {
		return func(_ context.Context, state any) (any, error) {
			mu.Lock()
			ran[name] = true
			mu.Unlock()

			m, _ := state.(map[string]any)
			if m == nil {
				m = map[string]any{}
			}
			m[name] = true
			return m, nil
		}
	}

	sg := graph.NewStateGraph(map[string]any{"root": false})
	sg.AddChannel("root", channels.NewLastValue(false))
	for _, name := range []string{"root", "left", "right", "left_leaf", "right_leaf"} {
		sg.AddNode(name, record(name))
	}
	for _, edge := range [][2]string{
		{constants.Start, "root"},
		{"root", "left"},
		{"root", "right"},
		{"left", "left_leaf"},
		{"right", "right_leaf"},
		{"left_leaf", constants.End},
		{"right_leaf", constants.End},
	} {
		if err := sg.AddEdge(edge[0], edge[1]); err != nil {
			t.Fatalf("add edge %s→%s: %v", edge[0], edge[1], err)
		}
	}

	engine := NewEngine(sg, WithRecursionLimit(20))
	if _, err := engine.RunSync(t.Context(), map[string]any{"root": true}); err != nil {
		t.Fatalf("RunSync: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	for _, name := range []string{"root", "left", "right", "left_leaf", "right_leaf"} {
		if !ran[name] {
			t.Errorf("node %q was never executed", name)
		}
	}
}

// TestEngine_ConditionalEdgeErrorFailsRun guards against a failing router being
// swallowed: a branch must not silently disappear and let the run finish.
func TestEngine_ConditionalEdgeErrorFailsRun(t *testing.T) {
	routeErr := errors.New("routing boom")

	sg := graph.NewStateGraph(map[string]any{"value": ""})
	sg.AddChannel("value", channels.NewLastValue(""))
	for _, name := range []string{"decide", "yes", "no"} {
		sg.AddNode(name, func(_ context.Context, state any) (any, error) {
			return state, nil
		})
	}
	if err := sg.AddEdge(constants.Start, "decide"); err != nil {
		t.Fatal(err)
	}
	if err := sg.AddConditionalEdges("decide", func(_ context.Context, _ any) (any, error) {
		return nil, routeErr
	}, map[string]string{"yes": "yes", "no": "no"}); err != nil {
		t.Fatal(err)
	}
	if err := sg.AddEdge("yes", constants.End); err != nil {
		t.Fatal(err)
	}
	if err := sg.AddEdge("no", constants.End); err != nil {
		t.Fatal(err)
	}

	engine := NewEngine(sg, WithRecursionLimit(10))
	_, err := engine.RunSync(t.Context(), map[string]any{"value": "x"})
	if err == nil || !strings.Contains(err.Error(), routeErr.Error()) {
		t.Fatalf("expected the routing error to surface, got %v", err)
	}
}

// TestEngine_ConditionalEdgeUnmappedResultFailsRun covers the other silent
// drop: a router returning a value missing from its mapping.
func TestEngine_ConditionalEdgeUnmappedResultFailsRun(t *testing.T) {
	sg := graph.NewStateGraph(map[string]any{"value": ""})
	sg.AddChannel("value", channels.NewLastValue(""))
	for _, name := range []string{"decide", "yes"} {
		sg.AddNode(name, func(_ context.Context, state any) (any, error) {
			return state, nil
		})
	}
	if err := sg.AddEdge(constants.Start, "decide"); err != nil {
		t.Fatal(err)
	}
	if err := sg.AddConditionalEdges("decide", func(_ context.Context, _ any) (any, error) {
		return "nowhere", nil
	}, map[string]string{"yes": "yes"}); err != nil {
		t.Fatal(err)
	}
	if err := sg.AddEdge("yes", constants.End); err != nil {
		t.Fatal(err)
	}

	engine := NewEngine(sg, WithRecursionLimit(10))
	_, err := engine.RunSync(t.Context(), map[string]any{"value": "x"})
	if err == nil || !strings.Contains(err.Error(), "no mapped target") {
		t.Fatalf("expected an unmapped-target error, got %v", err)
	}
}
