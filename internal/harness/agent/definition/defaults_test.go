package definition

import (
	"strings"
	"testing"
)

// The plan must stay optional. The retrieval run that ended without an answer
// had a four-step plan whose last step was "verify no gaps across the whole
// text"; the plan tool answers every call with "All tasks must be completed
// before summarizing or drawing conclusions", and that step can never be closed
// on a 1600-chunk book. The model therefore searched for seventeen rounds and
// never wrote an answer. A runtime rule that prescribes a plan for counting
// questions hands the model that lock.
func TestAgentRulesKeepsThePlanOptional(t *testing.T) {
	if strings.Contains(AgentRules, "write the steps out with the plan tool first") {
		t.Fatal("AgentRules prescribes a plan; the plan tool then forbids answering until every step is closed")
	}
	if !strings.Contains(AgentRules, "The plan tool is optional bookkeeping") {
		t.Fatal("AgentRules must state that the plan is optional bookkeeping")
	}
	for _, want := range []string{
		"must all be completed before summarizing", // the reminder it has to be told to discount
		"the list is not the question",
		"write the answer and say which part",
	} {
		if !strings.Contains(AgentRules, want) {
			t.Fatalf("AgentRules must resolve the plan tool's reminder to answer last: missing %q", want)
		}
	}
}
