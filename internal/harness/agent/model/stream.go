package model

import "context"

// StreamSender receives incremental output as the provider produces it. answer is
// visible text, thought is reasoning the UI renders separately. Either argument
// may be nil.
type StreamSender func(answer, thought *string) error

// Streamer is a model that can deliver output incrementally.
//
// Complete stays the contract: a driver that cannot stream simply does not
// implement this, and the loop falls back to one response. Streaming never
// replaces tool calling — the accumulated tool calls still come back on the
// response, because a turn that decides to use a tool produces no answer text to
// stream.
type Streamer interface {
	Stream(ctx context.Context, req Request, sender StreamSender) (Response, error)
}
