package store

import (
	"context"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/redis/go-redis/v9"

	"ragflow/internal/harness/graph/constants"
)

func newTestCheckpointer(t *testing.T) (*Checkpointer, *miniredis.Miniredis) {
	t.Helper()
	server := miniredis.RunT(t)
	client := redis.NewClient(&redis.Options{Addr: server.Addr()})
	t.Cleanup(func() { _ = client.Close() })
	return NewCheckpointer(client, time.Hour), server
}

func threadConfig(runID string) map[string]interface{} {
	return map[string]interface{}{constants.ConfigKeyThreadID: runID}
}

func TestCheckpointerMissingReturnsNil(t *testing.T) {
	checkpointer, _ := newTestCheckpointer(t)

	got, err := checkpointer.Get(context.Background(), threadConfig("absent"))
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if got != nil {
		t.Fatalf("Get = %v, want nil for a missing checkpoint", got)
	}
}

func TestCheckpointerLatestAndHistory(t *testing.T) {
	checkpointer, _ := newTestCheckpointer(t)
	ctx := context.Background()
	config := threadConfig("run-1")

	first := map[string]interface{}{"step": float64(1), "node": "prepare"}
	second := map[string]interface{}{"step": float64(2), "node": "think"}
	for _, checkpoint := range []map[string]interface{}{first, second} {
		if err := checkpointer.Put(ctx, config, checkpoint); err != nil {
			t.Fatalf("Put: %v", err)
		}
	}

	latest, err := checkpointer.Get(ctx, config)
	if err != nil {
		t.Fatalf("Get latest: %v", err)
	}
	if latest["node"] != "think" {
		t.Fatalf("latest node = %v, want think", latest["node"])
	}

	history, err := checkpointer.List(ctx, config, 10)
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(history) != 2 {
		t.Fatalf("history len = %d, want 2", len(history))
	}
	if history[0]["node"] != "prepare" || history[1]["node"] != "think" {
		t.Fatalf("history order = %v, want prepare then think", history)
	}
}

func TestCheckpointerTTLIsApplied(t *testing.T) {
	checkpointer, server := newTestCheckpointer(t)
	ctx := context.Background()
	config := threadConfig("run-ttl")

	if err := checkpointer.Put(ctx, config, map[string]interface{}{"step": float64(1)}); err != nil {
		t.Fatalf("Put: %v", err)
	}

	// Drive the clock past the TTL: an interrupted run must stop being resumable.
	server.FastForward(DefaultCheckpointTTL + time.Minute)

	got, err := checkpointer.Get(ctx, config)
	if err != nil {
		t.Fatalf("Get after TTL: %v", err)
	}
	if got != nil {
		t.Fatalf("checkpoint survived its TTL: %v", got)
	}
}

func TestCheckpointerRequiresThreadID(t *testing.T) {
	checkpointer, _ := newTestCheckpointer(t)

	if _, err := checkpointer.Get(context.Background(), map[string]interface{}{}); err == nil {
		t.Fatal("Get accepted a config without thread_id")
	}
}
