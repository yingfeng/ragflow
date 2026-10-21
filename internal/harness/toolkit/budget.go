package toolkit

import "context"

// Remaining-round hints travel on the context, not in Scope: they are not data
// reach, and a child must be able to tighten (never widen) its own slice.
//
// The parent run publishes what is left of its budget before a tool batch; a
// spawner reads it so a child cannot outlive the parent and leave the parent
// with no turns to converge. Astra applies the same rule to child deadlines.
type remainingRoundsKey struct{}

// WithRemainingRounds records how many rounds the current run may still take.
func WithRemainingRounds(ctx context.Context, rounds int) context.Context {
	if rounds < 0 {
		rounds = 0
	}
	return context.WithValue(ctx, remainingRoundsKey{}, rounds)
}

// RemainingRounds reports the parent's remaining rounds, false when unset.
func RemainingRounds(ctx context.Context) (int, bool) {
	rounds, ok := ctx.Value(remainingRoundsKey{}).(int)
	return rounds, ok
}
