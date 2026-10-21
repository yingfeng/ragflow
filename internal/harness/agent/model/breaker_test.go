package model

import (
	"errors"
	"testing"
	"time"
)

func TestBreakerOpensAfterThresholdAndRecovers(t *testing.T) {
	now := time.Unix(1000, 0)
	breaker := NewBreaker(2, 30*time.Second)

	for attempt := 0; attempt < 2; attempt++ {
		if _, ok := breaker.Allow(now); !ok {
			t.Fatalf("breaker opened after %d failure(s)", attempt)
		}
		breaker.Failure(now)
	}

	remaining, ok := breaker.Allow(now)
	if ok {
		t.Fatal("breaker stayed closed at the threshold")
	}
	if remaining != 30*time.Second {
		t.Fatalf("remaining = %s, want 30s", remaining)
	}

	// Still cooling down a moment later.
	if _, ok := breaker.Allow(now.Add(10 * time.Second)); ok {
		t.Fatal("breaker closed before the cooldown elapsed")
	}

	// Half-open after the cooldown: one probe is allowed.
	if _, ok := breaker.Allow(now.Add(31 * time.Second)); !ok {
		t.Fatal("breaker did not allow a probe after the cooldown")
	}

	// A success closes it for good.
	breaker.Success()
	open, failures, remaining := breaker.State(now.Add(32 * time.Second))
	if open || failures != 0 || remaining != 0 {
		t.Fatalf("state after success = open=%v failures=%d remaining=%s", open, failures, remaining)
	}
}

func TestBreakerFailureResetsOnSuccess(t *testing.T) {
	now := time.Unix(2000, 0)
	breaker := NewBreaker(3, time.Second)

	breaker.Failure(now)
	breaker.Failure(now)
	breaker.Success()
	breaker.Failure(now)

	if _, ok := breaker.Allow(now); !ok {
		t.Fatal("breaker opened although the successes reset the count")
	}
}

func TestNilBreakerAlwaysAllows(t *testing.T) {
	var breaker *Breaker
	if _, ok := breaker.Allow(time.Now()); !ok {
		t.Fatal("a nil breaker blocked a call")
	}
	breaker.Success()
	breaker.Failure(time.Now())
}

func TestBreakerRegistrySharesStateByKey(t *testing.T) {
	now := time.Unix(3000, 0)
	registry := NewBreakerRegistry(1, time.Minute)

	registry.For("deepseek").Failure(now)

	// A second run must see the first run's discovery: that sharing is what
	// protects the deployment rather than one run.
	if _, ok := registry.For("deepseek").Allow(now); ok {
		t.Fatal("a second run did not see the open breaker")
	}
	if _, ok := registry.For("openai").Allow(now); !ok {
		t.Fatal("an unrelated provider was blocked")
	}

	states := registry.Snapshot(now)
	if !states["deepseek"].Open {
		t.Fatalf("snapshot = %+v", states)
	}
	if states["openai"].Open {
		t.Fatalf("openai reported open: %+v", states["openai"])
	}
}

func TestErrProviderCoolingDownMessage(t *testing.T) {
	err := &ErrProviderCoolingDown{Key: "deepseek", Remaining: 5 * time.Second}
	if !errors.Is(err, err) {
		t.Fatal("error is not comparable to itself")
	}
	if got := err.Error(); got == "" {
		t.Fatal("empty error message")
	}
}
