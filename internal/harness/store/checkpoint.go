// Package store persists harness runtime state: graph checkpoints in Kvrocks and
// run/step/feedback/lesson records in the relational store.
package store

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"time"

	"github.com/redis/go-redis/v9"

	"ragflow/internal/harness/graph/constants"
)

// DefaultCheckpointTTL is how long an interrupted run stays resumable.
const DefaultCheckpointTTL = 24 * time.Hour

const checkpointPrefix = "harness:cp:"

// Checkpointer implements checkpoint.BaseCheckpointer on Kvrocks (Redis
// protocol). Every Put also refreshes a per-thread history index so a run can be
// replayed, not only resumed from its latest state.
type Checkpointer struct {
	client *redis.Client
	ttl    time.Duration
}

// NewCheckpointer builds a checkpointer. A non-positive ttl falls back to the
// default so a run never becomes silently unresumable.
func NewCheckpointer(client *redis.Client, ttl time.Duration) *Checkpointer {
	if ttl <= 0 {
		ttl = DefaultCheckpointTTL
	}
	return &Checkpointer{client: client, ttl: ttl}
}

// Get returns the checkpoint for the configured thread, or the explicitly
// requested checkpoint id. A missing checkpoint returns (nil, nil).
func (c *Checkpointer) Get(ctx context.Context, config map[string]interface{}) (map[string]interface{}, error) {
	threadID, checkpointID, err := checkpointIDs(config)
	if err != nil {
		return nil, err
	}
	key := latestKey(threadID)
	if checkpointID != "" {
		key = historyKey(threadID, checkpointID)
	}

	raw, err := c.client.Get(ctx, key).Bytes()
	if errors.Is(err, redis.Nil) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("checkpoint: read %s: %w", threadID, err)
	}
	out := map[string]interface{}{}
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, fmt.Errorf("checkpoint: decode %s: %w", threadID, err)
	}
	return out, nil
}

// Put stores the checkpoint as both the thread's latest state and one history
// entry.
func (c *Checkpointer) Put(ctx context.Context, config map[string]interface{}, checkpoint map[string]interface{}) error {
	threadID, _, err := checkpointIDs(config)
	if err != nil {
		return err
	}
	payload, err := json.Marshal(checkpoint)
	if err != nil {
		return fmt.Errorf("checkpoint: encode %s: %w", threadID, err)
	}

	seq, err := c.client.Incr(ctx, seqKey(threadID)).Result()
	if err != nil {
		return fmt.Errorf("checkpoint: sequence %s: %w", threadID, err)
	}

	pipe := c.client.TxPipeline()
	pipe.Set(ctx, latestKey(threadID), payload, c.ttl)
	pipe.Set(ctx, historyKey(threadID, strconv.FormatInt(seq, 10)), payload, c.ttl)
	pipe.Expire(ctx, seqKey(threadID), c.ttl)
	if _, err := pipe.Exec(ctx); err != nil {
		return fmt.Errorf("checkpoint: write %s: %w", threadID, err)
	}
	return nil
}

// List returns the most recent checkpoints of the configured thread, oldest
// first.
func (c *Checkpointer) List(ctx context.Context, config map[string]interface{}, limit int) ([]map[string]interface{}, error) {
	threadID, _, err := checkpointIDs(config)
	if err != nil {
		return nil, err
	}
	if limit <= 0 {
		limit = 20
	}

	total, err := c.client.Get(ctx, seqKey(threadID)).Int64()
	if errors.Is(err, redis.Nil) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("checkpoint: sequence %s: %w", threadID, err)
	}

	start := total - int64(limit) + 1
	if start < 1 {
		start = 1
	}
	keys := make([]string, 0, limit)
	for seq := start; seq <= total; seq++ {
		keys = append(keys, historyKey(threadID, strconv.FormatInt(seq, 10)))
	}
	if len(keys) == 0 {
		return nil, nil
	}

	values, err := c.client.MGet(ctx, keys...).Result()
	if err != nil {
		return nil, fmt.Errorf("checkpoint: list %s: %w", threadID, err)
	}
	out := make([]map[string]interface{}, 0, len(values))
	for _, value := range values {
		raw, ok := value.(string)
		if !ok || raw == "" {
			continue
		}
		entry := map[string]interface{}{}
		if err := json.Unmarshal([]byte(raw), &entry); err != nil {
			return nil, fmt.Errorf("checkpoint: decode history %s: %w", threadID, err)
		}
		out = append(out, entry)
	}
	return out, nil
}

func checkpointIDs(config map[string]interface{}) (threadID, checkpointID string, err error) {
	if config != nil {
		threadID, _ = config[constants.ConfigKeyThreadID].(string)
		checkpointID, _ = config["checkpoint_id"].(string)
	}
	if threadID == "" {
		return "", "", fmt.Errorf("checkpoint: %s is required", constants.ConfigKeyThreadID)
	}
	return threadID, checkpointID, nil
}

func latestKey(threadID string) string { return checkpointPrefix + threadID + ":latest" }

func historyKey(threadID, checkpointID string) string {
	return checkpointPrefix + threadID + ":" + checkpointID
}

func seqKey(threadID string) string { return checkpointPrefix + threadID + ":seq" }
