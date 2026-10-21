package model

import (
	"context"
	"fmt"
	"sync"
)

// Step is one scripted completion.
type Step func(Request) (Response, error)

// Fake is a scripted Model for tests.
type Fake struct {
	mu       sync.Mutex
	script   []Step
	cursor   int
	requests []Request
}

// NewFake builds a scripted model. Calls beyond the script fail loudly so a
// runaway loop cannot pass a test by accident.
func NewFake(steps ...Step) *Fake {
	return &Fake{script: steps}
}

// Reply scripts an assistant message with no tool calls.
func Reply(text string) Step {
	return func(Request) (Response, error) {
		return Response{
			Message:    Message{Role: RoleAssistant, Content: text},
			StopReason: "stop",
		}, nil
	}
}

// CallTool scripts an assistant message carrying one tool call.
func CallTool(id, name, arguments string) Step {
	return func(Request) (Response, error) {
		return Response{
			Message: Message{
				Role:      RoleAssistant,
				ToolCalls: []ToolCall{{ID: id, Name: name, Arguments: arguments}},
			},
			StopReason: "tool_calls",
		}, nil
	}
}

// Fail scripts a provider error.
func Fail(err error) Step {
	return func(Request) (Response, error) { return Response{}, err }
}

// Complete implements Model.
func (f *Fake) Complete(_ context.Context, req Request) (Response, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.requests = append(f.requests, req)
	if f.cursor >= len(f.script) {
		return Response{}, fmt.Errorf("model fake: script exhausted after %d calls", len(f.script))
	}
	step := f.script[f.cursor]
	f.cursor++
	return step(req)
}

// Requests returns every request the fake received.
func (f *Fake) Requests() []Request {
	f.mu.Lock()
	defer f.mu.Unlock()
	out := make([]Request, len(f.requests))
	copy(out, f.requests)
	return out
}

// Calls returns how many completions were served.
func (f *Fake) Calls() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.cursor
}
