package loop

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"ragflow/internal/harness/agent/model"
)

func TestErrorClassification(t *testing.T) {
	cases := []struct {
		err     error
		retry   bool
		tooLong bool
	}{
		{errors.New("429 rate limit exceeded"), true, false},
		{errors.New("503 Service Unavailable"), true, false},
		{errors.New("context deadline exceeded"), true, false},
		{errors.New("invalid api key"), false, false},
		{errors.New("prompt is too long: 210000 tokens"), false, true},
		{errors.New("maximum context length is 128000"), false, true},
		{nil, false, false},
	}
	for _, testCase := range cases {
		if got := isRetryableModelError(testCase.err); got != testCase.retry {
			t.Errorf("isRetryableModelError(%v) = %v, want %v", testCase.err, got, testCase.retry)
		}
		if got := isPromptTooLongError(testCase.err); got != testCase.tooLong {
			t.Errorf("isPromptTooLongError(%v) = %v, want %v", testCase.err, got, testCase.tooLong)
		}
	}
}

func TestModelBackoffGrowsAndCaps(t *testing.T) {
	if got := modelBackoff(1); got != modelRetryBase {
		t.Errorf("backoff(1) = %v, want %v", got, modelRetryBase)
	}
	if got := modelBackoff(2); got != 2*modelRetryBase {
		t.Errorf("backoff(2) = %v, want %v", got, 2*modelRetryBase)
	}
	if got := modelBackoff(20); got != modelRetryCeiling {
		t.Errorf("backoff(20) = %v, want the ceiling", got)
	}
}

type flakyModel struct {
	mu       sync.Mutex
	attempts int
	failures int
	err      error
}

func (m *flakyModel) Complete(context.Context, model.Request) (model.Response, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.attempts++
	if m.attempts <= m.failures {
		return model.Response{}, m.err
	}
	return model.Response{Message: model.Message{Role: model.RoleAssistant, Content: "ok"}}, nil
}

func (m *flakyModel) count() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.attempts
}

func TestCallModelRetriesTransientFailures(t *testing.T) {
	flaky := &flakyModel{failures: 2, err: errors.New("503 Service Unavailable")}

	response, err := callModel(context.Background(), flaky, model.Request{}, nil)
	if err != nil {
		t.Fatalf("callModel: %v", err)
	}
	if response.Message.Content != "ok" {
		t.Fatalf("content = %q", response.Message.Content)
	}
	if flaky.count() != 3 {
		t.Fatalf("attempts = %d, want 3", flaky.count())
	}
}

func TestCallModelStopsAfterAttempts(t *testing.T) {
	flaky := &flakyModel{failures: 10, err: errors.New("rate limit")}

	if _, err := callModel(context.Background(), flaky, model.Request{}, nil); err == nil {
		t.Fatal("expected the failure to surface")
	}
	if flaky.count() != modelAttempts {
		t.Fatalf("attempts = %d, want %d", flaky.count(), modelAttempts)
	}
}

func TestCallModelDoesNotRetryPermanentErrors(t *testing.T) {
	flaky := &flakyModel{failures: 10, err: errors.New("invalid api key")}

	if _, err := callModel(context.Background(), flaky, model.Request{}, nil); err == nil {
		t.Fatal("expected an error")
	}
	if flaky.count() != 1 {
		t.Fatalf("attempts = %d, want 1", flaky.count())
	}
}

func TestCallModelDoesNotRetryPromptTooLong(t *testing.T) {
	flaky := &flakyModel{failures: 10, err: errors.New("prompt is too long")}

	_, err := callModel(context.Background(), flaky, model.Request{}, nil)
	if err == nil {
		t.Fatal("expected an error")
	}
	if !isPromptTooLongError(err) {
		t.Fatalf("error lost its class: %v", err)
	}
	if flaky.count() != 1 {
		t.Fatalf("attempts = %d, want 1 (rebuilding the prompt is the caller's job)", flaky.count())
	}
}

func TestCallModelHonoursCancellationDuringBackoff(t *testing.T) {
	flaky := &flakyModel{failures: 10, err: errors.New("rate limit")}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()

	if _, err := callModel(ctx, flaky, model.Request{}, nil); err == nil {
		t.Fatal("expected an error")
	}
	if flaky.count() > modelAttempts {
		t.Fatalf("attempts = %d, want at most %d", flaky.count(), modelAttempts)
	}
}

func TestCompactionTierForBoost(t *testing.T) {
	if got := compactionTierFor(0); got != 0 {
		t.Errorf("boost 0 = %v, want normal", got)
	}
	if got := compactionTierFor(1); got != 2 {
		t.Errorf("boost 1 = %v, want compact history", got)
	}
	if got := compactionTierFor(9); got != 3 {
		t.Errorf("boost 9 = %v, want aggressive prune", got)
	}
}

// A provider that states how long to wait must not be retried earlier: the
// attempt would be spent proving the wait was real.
func TestModelRetryDelayPrefersProviderHint(t *testing.T) {
	err := errors.New("SILICONFLOW API error: 429 Too Many Requests, retry-after: 7, body: ...")
	if got := modelRetryDelay(err, 1); got != 7*time.Second {
		t.Fatalf("delay = %v, want the provider's 7s", got)
	}
	// A wait beyond the ceiling is clamped: one provider must not stall a run.
	err = errors.New("429 retry-after: 3600")
	if got := modelRetryDelay(err, 1); got != modelRetryAfterCeiling {
		t.Fatalf("delay = %v, want the provider ceiling %v", got, modelRetryAfterCeiling)
	}
	// No hint: the local curve still applies.
	if got := modelRetryDelay(errors.New("429 rate limited"), 2); got != modelBackoff(2) {
		t.Fatalf("delay = %v, want the local backoff %v", got, modelBackoff(2))
	}
}
