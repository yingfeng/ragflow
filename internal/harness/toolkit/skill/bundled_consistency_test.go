package skill

import (
	"context"
	"os"
	"strings"
	"testing"
)

// legacyRetrievalTemplate is the agentic_rag template the plain `agentic` chat
// path still runs. Its retrieval policy was copied into the bundled retrieval
// skill rather than duplicated in Go, which leaves exactly one failure mode
// worth guarding: the two texts drift and the two chat paths start answering
// under different rules.
//
// This test is the drift detector. It does not require the wording to match —
// the skill is allowed to be tighter — only that every rule the legacy template
// states is still stated somewhere in the skill body.
const legacyRetrievalTemplate = "../../../../conf/agentic_rag.yaml"

// legacyRuleAnchors are phrases that each stand for one retrieval rule. Every
// one of them must survive in both texts: a rule that disappears from the skill
// is a rule the harness stopped enforcing, and one that disappears from the
// template is a rule the legacy path stopped enforcing.
var legacyRuleAnchors = []string{
	"broad alternation",
	"no keyword leg",
	"Claim lines ARE answer material",
	"count completed units",
}

// legacyNonRetrievalTools are in the template's tool list but carry no retrieval
// policy, so the skill does not mention them.
var legacyNonRetrievalTools = map[string]bool{
	"think":          true,
	"todo_write":     true,
	"run_javascript": true,
}

func TestBundledRetrievalSkillMatchesLegacyTemplate(t *testing.T) {
	raw, err := os.ReadFile(legacyRetrievalTemplate)
	if err != nil {
		t.Fatalf("read legacy template: %v", err)
	}
	template := string(raw)

	bundled, err := Bundled().Load(context.Background(), "retrieval")
	if err != nil {
		t.Fatalf("load bundled retrieval skill: %v", err)
	}

	// The skill has to be standing policy, not something the model may discover:
	// a rule it does not know it needs is a rule it will not load.
	if !bundled.AutoLoad {
		t.Fatalf("bundled retrieval skill is not auto_load; its policy would not be in force from round one")
	}

	for _, anchor := range legacyRuleAnchors {
		if !strings.Contains(template, anchor) {
			t.Fatalf("legacy template lost the %q rule; the two policies are drifting", anchor)
		}
		if !strings.Contains(bundled.Body, anchor) {
			t.Fatalf("bundled skill lost the %q rule the legacy template still states", anchor)
		}
	}

	for _, tool := range templateTools(template) {
		if legacyNonRetrievalTools[tool] {
			continue
		}
		if !strings.Contains(bundled.Body, tool) {
			t.Fatalf("template tool %q has no counterpart in the bundled skill", tool)
		}
	}
}

// templateTools extracts the tools list from the legacy template without pulling
// a YAML dependency into this package: the list is a flat sequence of "- name"
// lines under "tools:".
func templateTools(template string) []string {
	var out []string
	inTools := false
	for _, line := range strings.Split(template, "\n") {
		trimmed := strings.TrimSpace(line)
		if trimmed == "tools:" {
			inTools = true
			continue
		}
		if !inTools {
			continue
		}
		if !strings.HasPrefix(trimmed, "- ") {
			break
		}
		out = append(out, strings.TrimSpace(strings.TrimPrefix(trimmed, "- ")))
	}
	return out
}
