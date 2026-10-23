package loop

import (
	"context"
	"strings"
	"testing"

	"ragflow/internal/harness/agent/model"
	"ragflow/internal/harness/toolkit"
)

// captureModel scripts the model and records the messages each request carried,
// so a test can assert on exactly what the model was shown.
type captureModel struct {
	steps []model.Step
	seen  [][]model.Message
	next  int
}

func (m *captureModel) Complete(_ context.Context, req model.Request) (model.Response, error) {
	m.seen = append(m.seen, append([]model.Message(nil), req.Messages...))
	step := m.steps[m.next]
	m.next++
	return step(req)
}

// lastToolContent returns the tool message the second request carried.
func (m *captureModel) lastToolContent(t *testing.T) string {
	t.Helper()
	if len(m.seen) < 2 {
		t.Fatalf("model was called %d times, want at least 2", len(m.seen))
	}
	messages := m.seen[len(m.seen)-1]
	for index := len(messages) - 1; index >= 0; index-- {
		if messages[index].Role == model.RoleTool {
			return messages[index].Content
		}
	}
	t.Fatalf("no tool message in the request: %+v", messages)
	return ""
}

// TestInlineResultKeepsItsWindow pins the Inline contract: a tool that already
// bounded its own output (a retrieval window) is handed to the model verbatim,
// with no byte ceiling of its own.
//
// Re-spilling it does not shrink the payload, it only moves it behind a handle:
// the preview is a preview of the preview, and the model must spend one
// read_spill round to get back to the text it just asked for. That round comes
// out of the same round budget the searches need, and covering a list is a race
// against exactly that budget. A bounded window needs no second ceiling here:
// the per-result cap in the context builder (MaxToolResultBytes, 64 KiB) is the
// only guard left, by design.
func TestInlineResultKeepsItsWindow(t *testing.T) {
	payload := strings.Repeat("evidence line. ", 4000) // ~60 KiB, past every threshold this loop owns
	mdl := &captureModel{steps: []model.Step{
		model.CallTool("c1", "fake_tool", `{"query":"x"}`),
		model.Reply("done"),
	}}
	capability := &fakeCap{tool: "fake_tool", result: toolkit.Result{Content: payload, Inline: true}}
	runner := buildRunner(t, mdl, capability, Config{Deps: Deps{Policy: noProgressPolicy()}})

	if _, err := runner.Invoke(t.Context(), &Input{RunID: "run-inline", UserMessage: "go"}, nil); err != nil {
		t.Fatalf("invoke: %v", err)
	}

	got := mdl.lastToolContent(t)
	if strings.Contains(got, "output truncated") {
		t.Fatalf("a %d-byte inline window was spilled", len(payload))
	}
	if got != payload {
		t.Fatalf("inline content = %d bytes, want the payload verbatim (%d)", len(got), len(payload))
	}
}

// TestPlainResultStillSpills keeps the other half of the contract in view: a
// result that did NOT bound itself is spilled at the same threshold, so the
// exemption above is about who did the bounding, not about the size.
func TestPlainResultStillSpills(t *testing.T) {
	payload := strings.Repeat("evidence line. ", 1000)
	mdl := &captureModel{steps: []model.Step{
		model.CallTool("c1", "fake_tool", `{"query":"x"}`),
		model.Reply("done"),
	}}
	capability := &fakeCap{tool: "fake_tool", result: toolkit.Result{Content: payload, Inline: false}}
	runner := buildRunner(t, mdl, capability, Config{Deps: Deps{Policy: noProgressPolicy()}})

	if _, err := runner.Invoke(t.Context(), &Input{RunID: "run-plain", UserMessage: "go"}, nil); err != nil {
		t.Fatalf("invoke: %v", err)
	}

	got := mdl.lastToolContent(t)
	if !strings.Contains(got, "output truncated") {
		t.Fatalf("a %d-byte unbounded result stayed inline", len(payload))
	}
	if len(got) >= len(payload) {
		t.Fatalf("spilled content (%d) is not smaller than the payload (%d)", len(got), len(payload))
	}
}
