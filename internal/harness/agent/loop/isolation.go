package loop

import (
	"context"

	"ragflow/internal/harness/toolkit/spill"
)

// FreshRunContext gives a delegated child its own run state.
//
// A child is invoked with its parent's context so cancellation propagates: when
// the parent is stopped, its children stop too. Everything else a run owns has
// to be its own, and the parent's context carries three of them:
//
//   - the inbox that receives steering messages — a shared one let a message the
//     user aimed at the parent be drained into a child's transcript, or the
//     child's own messages be read by the parent;
//   - the spill store that keeps oversized tool output addressable — handles are
//     minted per store, so a shared store lets one run's eviction break the
//     other's handle;
//   - the delta and reasoning sinks — a shared one streamed the child's working
//     text and answer into the parent's answer area and step panel, which is the
//     "draft that a later round overwrote" failure with a second author.
//
// The tool allowlist is not handled here: Invoke installs one holder per run, so
// a skill loaded in the child cannot narrow the parent and vice versa.
//
// The child's progress is deliberately not reported to the parent's stream: the
// parent's own tool call is what the user is watching, and the child's rounds
// stay in the child's run trace where an operator can read them.
func FreshRunContext(ctx context.Context) context.Context {
	ctx = WithInbox(ctx, NewInbox())
	ctx = spill.WithStore(ctx, spill.New(32))
	return withoutStreams(ctx)
}

// withoutStreams detaches the streaming surface from a context.
func withoutStreams(ctx context.Context) context.Context {
	ctx = context.WithValue(ctx, deltaSinkKey{}, DeltaSink(nil))
	ctx = context.WithValue(ctx, reasoningSinkKey{}, ReasoningSink(nil))
	return context.WithValue(ctx, progressSinkKey{}, (func(Progress))(nil))
}
