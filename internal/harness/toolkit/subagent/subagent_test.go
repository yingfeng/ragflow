package subagent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"ragflow/internal/harness/toolkit"
)

type fakeSpawner struct {
	mu       sync.Mutex
	inFlight int
	peak     int
	delay    time.Duration
	failures map[string]error
}

func (f *fakeSpawner) Spawn(_ context.Context, req Request) (Outcome, error) {
	f.mu.Lock()
	f.inFlight++
	if f.inFlight > f.peak {
		f.peak = f.inFlight
	}
	f.mu.Unlock()

	if f.delay > 0 {
		time.Sleep(f.delay)
	}

	f.mu.Lock()
	f.inFlight--
	f.mu.Unlock()

	if err, ok := f.failures[req.Prompt]; ok {
		return Outcome{}, err
	}
	return Outcome{RunID: "run-" + req.Prompt, Output: "answer:" + req.Prompt}, nil
}

func (f *fakeSpawner) peakParallel() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.peak
}

func call(t *testing.T, capability *Capability, tool string, arguments string) toolkit.Result {
	t.Helper()
	result, err := capability.Invoke(context.Background(), toolkit.Call{Tool: tool, Arguments: arguments})
	if err != nil {
		t.Fatalf("Invoke(%s): %v", tool, err)
	}
	return result
}

func TestToolsExposeBothForms(t *testing.T) {
	names := map[string]bool{}
	for _, spec := range New(&fakeSpawner{}).Tools() {
		names[spec.Name] = true
	}
	for _, want := range []string{"subagent", "spawn_subagents"} {
		if !names[want] {
			t.Errorf("missing tool %q, have %v", want, names)
		}
	}
}

func TestSpawnManyRunsTasksConcurrently(t *testing.T) {
	spawner := &fakeSpawner{delay: 30 * time.Millisecond}
	capability := New(spawner)

	result := call(t, capability, "spawn_subagents", `{"tasks":["a","b","c","d"]}`)

	if !strings.Contains(result.Content, `<subagent_results count="4">`) {
		t.Fatalf("unexpected header: %s", result.Content)
	}
	for _, task := range []string{"a", "b", "c", "d"} {
		if !strings.Contains(result.Content, "answer:"+task) {
			t.Errorf("missing answer for %q: %s", task, result.Content)
		}
	}
	if peak := spawner.peakParallel(); peak < 2 {
		t.Errorf("peak parallel children = %d, want >= 2", peak)
	}
}

func TestSpawnManyPreservesInputOrder(t *testing.T) {
	spawner := &fakeSpawner{delay: 10 * time.Millisecond}
	capability := New(spawner)

	result := call(t, capability, "spawn_subagents", `{"tasks":["first","second","third"]}`)

	first := strings.Index(result.Content, `index="1" status="ok">answer:first`)
	second := strings.Index(result.Content, `index="2" status="ok">answer:second`)
	third := strings.Index(result.Content, `index="3" status="ok">answer:third`)
	if first < 0 || second < 0 || third < 0 {
		t.Fatalf("results not matched by index: %s", result.Content)
	}
	if !(first < second && second < third) {
		t.Errorf("results out of order: %s", result.Content)
	}
}

func TestSpawnManyReportsPerTaskFailure(t *testing.T) {
	spawner := &fakeSpawner{failures: map[string]error{"b": errors.New("child timed out")}}
	capability := New(spawner)

	result := call(t, capability, "spawn_subagents", `{"tasks":["a","b","c"]}`)

	if !strings.Contains(result.Content, `index="1" status="ok">answer:a`) {
		t.Errorf("first task should succeed: %s", result.Content)
	}
	if !strings.Contains(result.Content, `index="2" status="error">child timed out`) {
		t.Errorf("second task should report failure: %s", result.Content)
	}
	if !strings.Contains(result.Content, `index="3" status="ok">answer:c`) {
		t.Errorf("third task should survive sibling failure: %s", result.Content)
	}
}

func TestSpawnManyRejectsTooManyTasks(t *testing.T) {
	capability := New(&fakeSpawner{})

	// One task past the admission cap. The cap is 50, so the overflow only shows
	// up when the model tries to fan out wider than the run can absorb.
	tasks := make([]string, MaxParallelChildren+1)
	for index := range tasks {
		tasks[index] = fmt.Sprintf("task-%d", index)
	}
	payload, err := json.Marshal(map[string]any{"tasks": tasks})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	_, err = capability.Invoke(context.Background(), toolkit.Call{
		Tool:      "spawn_subagents",
		Arguments: string(payload),
	})
	if err == nil {
		t.Fatalf("expected error for %d tasks", len(tasks))
	}
	if !strings.Contains(err.Error(), fmt.Sprintf("at most %d", MaxParallelChildren)) {
		t.Errorf("error should state the limit: %v", err)
	}
}

func TestSpawnManyRejectsBlankTasks(t *testing.T) {
	capability := New(&fakeSpawner{})

	_, err := capability.Invoke(context.Background(), toolkit.Call{
		Tool:      "spawn_subagents",
		Arguments: `{"tasks":["  ",""]}`,
	})
	if err == nil || !strings.Contains(err.Error(), "at least one task") {
		t.Fatalf("expected empty-task error, got %v", err)
	}
}

func TestInvokeRejectsUnknownTool(t *testing.T) {
	_, err := New(&fakeSpawner{}).Invoke(context.Background(), toolkit.Call{Tool: "spawn_army"})
	if err == nil || !strings.Contains(err.Error(), `unknown tool "spawn_army"`) {
		t.Fatalf("expected unknown-tool error, got %v", err)
	}
}

func TestSpawnManyWithoutSpawner(t *testing.T) {
	_, err := New(nil).Invoke(context.Background(), toolkit.Call{
		Tool:      "spawn_subagents",
		Arguments: `{"tasks":["a"]}`,
	})
	if err == nil || !strings.Contains(err.Error(), "not configured") {
		t.Fatalf("expected not-configured error, got %v", err)
	}
}

func TestDelegationRefusedPastDepthLimit(t *testing.T) {
	capability := New(&fakeSpawner{})

	ctx := WithDepth(context.Background(), DefaultMaxDepth)
	_, err := capability.Invoke(ctx, toolkit.Call{
		Tool:      SpawnManyName,
		Arguments: `{"tasks":["a"]}`,
	})
	if err == nil || !strings.Contains(err.Error(), "depth limit reached") {
		t.Fatalf("expected a depth-limit error, got %v", err)
	}
}

func TestDelegationAllowedBelowDepthLimit(t *testing.T) {
	capability := New(&fakeSpawner{})

	ctx := WithDepth(context.Background(), DefaultMaxDepth-1)
	result, err := capability.Invoke(ctx, toolkit.Call{
		Tool:      SpawnManyName,
		Arguments: `{"tasks":["a"]}`,
	})
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if !strings.Contains(result.Content, "answer:a") {
		t.Fatalf("unexpected content: %s", result.Content)
	}
}

func TestDepthLimitClampedToAbsoluteCeiling(t *testing.T) {
	if got := NewWithLimits(&fakeSpawner{}, 99).maxDepth; got != AbsoluteMaxDepth {
		t.Fatalf("maxDepth = %d, want %d", got, AbsoluteMaxDepth)
	}
	if got := NewWithLimits(&fakeSpawner{}, 0).maxDepth; got != DefaultMaxDepth {
		t.Fatalf("maxDepth = %d, want %d", got, DefaultMaxDepth)
	}
}

type verboseSpawner struct{}

func (verboseSpawner) Spawn(context.Context, Request) (Outcome, error) {
	return Outcome{Output: "HEAD-" + strings.Repeat("z", 20<<10) + "-TAIL"}, nil
}

func TestSpawnManyClipsChildAnswers(t *testing.T) {
	capability := New(verboseSpawner{})

	result := call(t, capability, SpawnManyName, `{"tasks":["big"]}`)

	if len(result.Content) > maxChildResultBytes+2048 {
		t.Fatalf("content = %d bytes, want the child answer clipped", len(result.Content))
	}
	if !strings.Contains(result.Content, "HEAD-") || !strings.Contains(result.Content, "-TAIL") {
		t.Error("clipping dropped the head or the tail of the child answer")
	}
	if !strings.Contains(result.Content, "truncated") {
		t.Error("clipping was not announced")
	}
}

func TestSpawnOneReportsDepth(t *testing.T) {
	capability := New(&fakeSpawner{})

	result := call(t, capability, ToolName, `{"prompt":"a"}`)

	if got, ok := result.Meta["depth"].(int); !ok || got != 1 {
		t.Fatalf("depth meta = %#v, want 1", result.Meta["depth"])
	}
}

func TestSpawnManyStopsOnCancelledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	spawner := &fakeSpawner{}
	capability := New(spawner)

	result, err := capability.Invoke(ctx, toolkit.Call{
		Tool:      "spawn_subagents",
		Arguments: `{"tasks":["a","b"]}`,
	})
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if strings.Count(result.Content, `status="error"`) != 2 {
		t.Errorf("expected both tasks to fail on cancelled ctx: %s", result.Content)
	}
}
