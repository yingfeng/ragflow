package loop

import (
	"strings"
	"testing"

	"ragflow/internal/harness/agent/model"
	"ragflow/internal/harness/toolkit"
)

// TestAnswerCallCarriesTheEvidencePool pins the reason the pool is re-sent to
// the answer call: compaction drops old tool results first, so a long run's
// prompt no longer contains the passages that proved its earliest findings. A
// 50-round run answered with fewer names than a 6-round one for exactly this
// reason — the evidence was still in memory, just not in the prompt.
func TestAnswerCallCarriesTheEvidencePool(t *testing.T) {
	capability := &fakeCap{tool: "fake_tool", result: toolkit.Result{
		Content: "found the first chapter",
		Evidence: []toolkit.Evidence{
			{Ref: "[ID:2f4f73f8]", Snippet: "程远志措手不及，被云长刀起处，挥为两段"},
		},
	}}
	mdl := &captureModel{steps: []model.Step{
		model.CallTool("c1", "fake_tool", `{"query":"x"}`),
		model.Reply(""), // a terminal round with no answer: one grace round
		model.Reply(""), // grace spent: the answer call takes over
		model.Reply("answer"),
	}}
	runner := buildRunner(t, mdl, capability, Config{Deps: Deps{Policy: noProgressPolicy()}})

	if _, err := runner.Invoke(t.Context(), &Input{RunID: "run-evidence-pool", UserMessage: "go"}, nil); err != nil {
		t.Fatalf("invoke: %v", err)
	}

	if len(mdl.seen) < 4 {
		t.Fatalf("model was called %d times, want a final answer call", len(mdl.seen))
	}
	final := mdl.seen[len(mdl.seen)-1]
	var answerPrompt strings.Builder
	for _, message := range final {
		answerPrompt.WriteString(message.Content)
		answerPrompt.WriteString("\n")
	}
	prompt := answerPrompt.String()
	for _, want := range []string{"Collected evidence", "[ID:2f4f73f8]", "程远志"} {
		if !strings.Contains(prompt, want) {
			t.Fatalf("answer prompt is missing %q:\n%s", want, prompt)
		}
	}
}
