package loop

import (
	"context"
	"encoding/json"
	"strings"
	"time"
)

// DeltaSink receives the text of the round that ends the turn, as the provider
// produced it.
type DeltaSink func(text string)

// ReasoningSink receives the text of a round that went on to call a tool: the
// model working out loud before it acts.
//
// That text is not the answer, and a chat UI shows it in its step panel. Letting
// it into the answer area instead is what let one round print a draft the next
// round overwrote.
type ReasoningSink func(text string)

// Progress reports one tool call starting or finishing. It is what a chat UI
// shows as "searching…" while the loop waits on a tool, and it travels through
// the context so it costs nothing when nobody is watching.
type Progress struct {
	Tool       string
	Phase      string
	Summary    string
	IsError    bool
	DurationMS int64
}

// Progress phases.
const (
	ProgressStart = "start"
	ProgressEnd   = "end"
)

// progressSummaryLimit bounds the argument excerpt a progress event carries. The
// point is to show which tool ran on what, not to ship the whole query.
const progressSummaryLimit = 160

type deltaSinkKey struct{}

type reasoningSinkKey struct{}

type progressSinkKey struct{}

// WithDeltaSink installs the text sink for this run.
func WithDeltaSink(ctx context.Context, sink DeltaSink) context.Context {
	if sink == nil {
		return ctx
	}
	return context.WithValue(ctx, deltaSinkKey{}, sink)
}

// DeltaSinkFromContext returns the installed sink, or nil.
func DeltaSinkFromContext(ctx context.Context) DeltaSink {
	sink, _ := ctx.Value(deltaSinkKey{}).(DeltaSink)
	return sink
}

// WithReasoningSink installs the sink for a round's working text.
func WithReasoningSink(ctx context.Context, sink ReasoningSink) context.Context {
	if sink == nil {
		return ctx
	}
	return context.WithValue(ctx, reasoningSinkKey{}, sink)
}

// ReasoningSinkFromContext returns the installed sink, or nil.
func ReasoningSinkFromContext(ctx context.Context) ReasoningSink {
	sink, _ := ctx.Value(reasoningSinkKey{}).(ReasoningSink)
	return sink
}

// flushAnswerText hands held chunks to the answer sink in the order the provider
// produced them.
func flushAnswerText(held []string, sink DeltaSink) {
	if sink == nil {
		return
	}
	for _, text := range held {
		if text == "" {
			continue
		}
		sink(text)
	}
}

// emitReasoning reports one round's working text as a single step: the round is
// the unit a step panel shows, so its chunks are joined rather than streamed.
func emitReasoning(held []string, sink ReasoningSink) {
	if sink == nil || len(held) == 0 {
		return
	}
	text := strings.Join(held, "")
	if strings.TrimSpace(text) == "" {
		return
	}
	sink(text)
}

// WithProgressSink installs the tool-progress sink for this run.
func WithProgressSink(ctx context.Context, sink func(Progress)) context.Context {
	if sink == nil {
		return ctx
	}
	return context.WithValue(ctx, progressSinkKey{}, sink)
}

// ProgressSinkFromContext returns the installed progress sink, or nil.
func ProgressSinkFromContext(ctx context.Context) func(Progress) {
	sink, _ := ctx.Value(progressSinkKey{}).(func(Progress))
	return sink
}

// emitProgress reports one tool lifecycle event when a sink is installed.
func emitProgress(ctx context.Context, progress Progress) {
	if sink := ProgressSinkFromContext(ctx); sink != nil {
		sink(progress)
	}
}

// trackTool reports a tool's start and end around invoke. It returns a function
// the caller defers so a panicking tool still closes its progress event.
//
// The end event takes the failure as a flag rather than an error because a tool
// can fail by returning an error result: the call ran, the model sees a
// <tool_error>, and a progress stream that reported success would disagree with
// the transcript.
func trackTool(ctx context.Context, call toolCallView) func(isError bool) {
	emitProgress(ctx, Progress{
		Tool:    call.tool,
		Phase:   ProgressStart,
		Summary: summarizeArguments(call.arguments),
	})
	started := time.Now()
	return func(isError bool) {
		emitProgress(ctx, Progress{
			Tool:       call.tool,
			Phase:      ProgressEnd,
			Summary:    summarizeArguments(call.arguments),
			IsError:    isError,
			DurationMS: time.Since(started).Milliseconds(),
		})
	}
}

// toolCallView is the part of a pending call progress needs.
type toolCallView struct {
	tool      string
	arguments string
}

// summarizeArguments renders a short single-line excerpt of the call arguments.
func summarizeArguments(arguments string) string {
	trimmed := strings.TrimSpace(arguments)
	if trimmed == "" {
		return ""
	}
	// Decode to compact the JSON the model produced, then fall back to the raw
	// string when it is not JSON at all.
	var decoded any
	if err := json.Unmarshal([]byte(trimmed), &decoded); err == nil {
		if compact, err := json.Marshal(decoded); err == nil {
			trimmed = string(compact)
		}
	}
	trimmed = strings.Join(strings.Fields(trimmed), " ")
	if len(trimmed) <= progressSummaryLimit {
		return trimmed
	}
	return strings.ToValidUTF8(trimmed[:progressSummaryLimit], "") + "…"
}
