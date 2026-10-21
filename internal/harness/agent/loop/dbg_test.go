package loop

import (
	"context"
	"testing"

	"ragflow/internal/harness/agent/definition"
	"ragflow/internal/harness/agent/model"
	"ragflow/internal/harness/caps"
)

func TestDebugRounds(t *testing.T) {
	capability := &fakeCap{tool: "rag_search", result: caps.Result{Content: "nothing"}}
	mdl := model.NewFake(
		model.CallTool("c1", "rag_search", `{"query":"a"}`),
		model.CallTool("c2", "rag_search", `{"query":"b"}`),
		model.CallTool("c3", "rag_search", `{"query":"c"}`),
		model.CallTool("c4", "rag_search", `{"query":"d"}`),
	)
	cfg := Config{PolicyConfig: definition.PolicyConfig{MaxRounds: 2}}
	cfg.Deps.Sink = func(r StepRecord) {
		t.Logf("phase=%s round=%d decision=%s reason=%s", r.Phase, r.Round, r.Decision, r.Reason)
	}
	runner := buildRunner(t, mdl, capability, cfg)
	_, err := runner.Invoke(context.Background(), &Input{RunID: "dbg", UserMessage: "q"}, nil)
	t.Logf("err=%v", err)
}
