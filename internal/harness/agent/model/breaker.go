package model

import (
	"fmt"
	"sync"
	"time"
)

// ErrProviderCoolingDown is returned instead of calling a provider whose breaker
// is open. The message names the remaining wait so a caller can retry later.
type ErrProviderCoolingDown struct {
	Key       string
	Remaining time.Duration
}

func (e *ErrProviderCoolingDown) Error() string {
	return fmt.Sprintf("model: provider %q is cooling down for %s", e.Key, e.Remaining.Round(time.Second))
}

// Breaker stops hammering a provider that keeps failing. Retry policy inside one
// call already absorbs a blip; the breaker is what protects *other* runs from a
// provider that is down, by failing fast for a cooldown instead of each run
// burning its own three attempts.
type Breaker struct {
	mu          sync.Mutex
	threshold   int
	cooldown    time.Duration
	consecutive int
	openUntil   time.Time
}

// NewBreaker builds a breaker that opens after threshold consecutive failures and
// stays open for cooldown.
func NewBreaker(threshold int, cooldown time.Duration) *Breaker {
	if threshold <= 0 {
		threshold = 5
	}
	if cooldown <= 0 {
		cooldown = 30 * time.Second
	}
	return &Breaker{threshold: threshold, cooldown: cooldown}
}

// Allow reports whether a call may proceed. A nil breaker always allows, so a
// caller never needs its own nil check.
func (b *Breaker) Allow(now time.Time) (time.Duration, bool) {
	if b == nil {
		return 0, true
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.openUntil.IsZero() {
		return 0, true
	}
	if now.Before(b.openUntil) {
		return b.openUntil.Sub(now), false
	}
	// The cooldown elapsed: let one call through and start counting again.
	b.openUntil = time.Time{}
	b.consecutive = 0
	return 0, true
}

// Success closes the breaker.
func (b *Breaker) Success() {
	if b == nil {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	b.consecutive = 0
	b.openUntil = time.Time{}
}

// Failure records one exhausted call and opens the breaker at the threshold.
func (b *Breaker) Failure(now time.Time) {
	if b == nil {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	b.consecutive++
	if b.consecutive >= b.threshold {
		b.openUntil = now.Add(b.cooldown)
	}
}

// State reports the breaker for a health endpoint.
func (b *Breaker) State(now time.Time) (open bool, failures int, remaining time.Duration) {
	if b == nil {
		return false, 0, 0
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.openUntil.IsZero() || !now.Before(b.openUntil) {
		return false, b.consecutive, 0
	}
	return true, b.consecutive, b.openUntil.Sub(now)
}

// BreakerRegistry keeps one breaker per provider key, shared by every run in the
// process. That sharing is the point: a per-run breaker would only protect the
// run that discovered the outage.
type BreakerRegistry struct {
	mu        sync.Mutex
	breakers  map[string]*Breaker
	threshold int
	cooldown  time.Duration
}

// NewBreakerRegistry builds a registry whose breakers use the given policy.
func NewBreakerRegistry(threshold int, cooldown time.Duration) *BreakerRegistry {
	if threshold <= 0 {
		threshold = 5
	}
	if cooldown <= 0 {
		cooldown = 30 * time.Second
	}
	return &BreakerRegistry{breakers: make(map[string]*Breaker), threshold: threshold, cooldown: cooldown}
}

// For returns the breaker for one key, creating it on first use.
func (r *BreakerRegistry) For(key string) *Breaker {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if breaker, ok := r.breakers[key]; ok {
		return breaker
	}
	breaker := NewBreaker(r.threshold, r.cooldown)
	r.breakers[key] = breaker
	return breaker
}

// Snapshot reports the state of every known breaker.
func (r *BreakerRegistry) Snapshot(now time.Time) map[string]BreakerState {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	breakers := make(map[string]*Breaker, len(r.breakers))
	for key, breaker := range r.breakers {
		breakers[key] = breaker
	}
	r.mu.Unlock()

	states := make(map[string]BreakerState, len(breakers))
	for key, breaker := range breakers {
		open, failures, remaining := breaker.State(now)
		states[key] = BreakerState{Open: open, Failures: failures, Remaining: remaining}
	}
	return states
}

// BreakerState is one provider's health.
type BreakerState struct {
	Open      bool
	Failures  int
	Remaining time.Duration
}
