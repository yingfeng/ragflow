package policy

import (
	"strings"
	"testing"
)

// decide runs the engine over one snapshot so the tests assert what the loop
// actually consumes, not a single policy in isolation.
func decide(snapshot Snapshot) Verdict {
	return New(Budget{}, Progress{}, Completion{}, Safety{}).Decide(snapshot)
}

func TestProgress_RepeatsPauseOnlyWithoutOpenPlan(t *testing.T) {
	// An enumeration repeats the same query as it walks a list. With open plan
	// steps the run has not claimed to be finished, so repeats alone must not
	// interrupt it — a stall is reported by Progress and bounded by Budget.
	if got := decide(Snapshot{RepeatedCalls: 5, OpenPlanSteps: 2}); got.Action == ActionPause {
		t.Fatalf("open plan paused on repeats: %+v", got)
	}
	if got := decide(Snapshot{RepeatedCalls: 5}); got.Action != ActionPause {
		t.Fatalf("verdict = %+v, want pause for repeats with no plan", got)
	}
}

func TestCompletion_OpenPlanStepDefersFinish(t *testing.T) {
	// One prose-only round is grace for the model to reopen the steps.
	if got := decide(Snapshot{HasFinalText: true, OpenPlanSteps: 1, TerminalRounds: 1}); got.Action != ActionContinue {
		t.Fatalf("verdict = %+v, want continue while the plan has an open step", got)
	}
	// A second prose-only round is not convergence: the answer is due now.
	if got := decide(Snapshot{HasFinalText: true, OpenPlanSteps: 1, TerminalRounds: 2}); got.Action != ActionFinish {
		t.Fatalf("verdict = %+v, want finish after the grace round", got)
	}
	// A closed plan finishes immediately.
	if got := decide(Snapshot{HasFinalText: true, TerminalRounds: 1}); got.Action != ActionFinish {
		t.Fatalf("verdict = %+v, want finish for a closed plan", got)
	}
}

func TestCompletion_NoAnswerGetsOneGraceRound(t *testing.T) {
	// The model ended its turn without an answer: one more round, in the context
	// it already has, beats an immediate summary call.
	if got := decide(Snapshot{ThinkTerminal: true, TerminalRounds: 1}); got.Action != ActionContinue {
		t.Fatalf("verdict = %+v, want continue", got)
	}
	// Second time it is not writing an answer: hand it to the answer call.
	if got := decide(Snapshot{ThinkTerminal: true, TerminalRounds: 2}); got.Action != ActionBlock {
		t.Fatalf("verdict = %+v, want block", got)
	}
}

func TestCompletion_RequireCitationsPauses(t *testing.T) {
	engine := New(Completion{RequireCitations: true})
	if got := engine.Decide(Snapshot{HasFinalText: true, Citations: 0}); got.Action != ActionPause {
		t.Fatalf("verdict = %+v, want pause for an uncited answer", got)
	}
}

// A stall is described to the run, never used to end it: the round that stopped
// adding evidence may be the round that was reading evidence already collected.
func TestProgress_StagnationNudgesInsteadOfStopping(t *testing.T) {
	got := decide(Snapshot{StagnantRounds: 3, Citations: 4})
	if got.Action != ActionNudge {
		t.Fatalf("verdict = %+v, want a nudge", got)
	}
	if got.Advice == "" {
		t.Fatal("nudge carries no advice")
	}
	// An answer that is ready still finishes: the nudge cannot postpone it.
	if got := decide(Snapshot{StagnantRounds: 3, HasFinalText: true}); got.Action != ActionFinish {
		t.Fatalf("verdict = %+v, want finish for a written answer", got)
	}
	// Once the fact has been restated, the answer call takes over from the
	// stalled search instead of the search continuing to spend.
	got = decide(Snapshot{StagnantRounds: 3, Nudges: 2})
	if got.Action != ActionBlock || got.Code != "stalled" {
		t.Fatalf("verdict = %+v, want a stalled hand-off to the answer call", got)
	}
}

func TestBudget_ReportsBudgetLimited(t *testing.T) {
	got := decide(Snapshot{MaxRounds: 5, Round: 5})
	if got.Action != ActionFinish {
		t.Fatalf("verdict = %+v, want finish", got)
	}
	if got.Code != CodeBudgetLimited {
		t.Fatalf("code = %q, want %q: a run out of budget is not a finished run", got.Code, CodeBudgetLimited)
	}
}

// TestProgress_BreaksAPlanThatStopsMoving pins the rule that an open plan is not
// an unbounded licence to continue. A run that keeps retrieving new-but-
// irrelevant passages never accumulates StagnantRounds, so without this rule the
// only stop left is the round budget: three open steps held one run for all 50
// of its rounds, and it answered with less than a six-round run had collected.
func TestProgress_BreaksAPlanThatStopsMoving(t *testing.T) {
	progress := Progress{}

	// The plan is idle AND the pool stopped growing: that is a genuine stall.
	first := progress.Check(Snapshot{OpenPlanSteps: 3, PlanIdleRounds: 7, StagnantRounds: 1})
	if len(first) != 1 || first[0].Action != ActionNudge {
		t.Fatalf("verdicts = %+v, want one nudge for the idle plan", first)
	}

	// Once the fact has been stated as often as it is useful, the run is handed
	// to the answer call instead of being allowed to keep spending rounds.
	stalled := progress.Check(Snapshot{OpenPlanSteps: 3, PlanIdleRounds: 9, StagnantRounds: 1, Nudges: 2})
	if len(stalled) != 1 || stalled[0].Action != ActionBlock || stalled[0].Code != "stalled" {
		t.Fatalf("verdicts = %+v, want a stalled block", stalled)
	}

	// A plan that is still moving is left alone: that is the enumeration case the
	// repeated-call rule deliberately does not interrupt.
	if verdicts := progress.Check(Snapshot{OpenPlanSteps: 2, PlanIdleRounds: 2}); len(verdicts) != 0 {
		t.Fatalf("a moving plan was flagged: %+v", verdicts)
	}
}

// TestProgress_KeepsAPlanIdleOnlyWhileThePoolGrows pins the enumeration case: an
// open step that cannot be ticked until the last item is found is idle by
// construction, so while the evidence pool keeps growing the plan must not be
// judged at all. Blocking it at six rounds ended a run at round ten, and the
// wider eighteen-round limit only moved the cut to round twenty — both while the
// run was still pulling fresh greps and spills every round.
func TestProgress_KeepsAPlanIdleOnlyWhileThePoolGrows(t *testing.T) {
	progress := Progress{}

	// Pool still growing: no plan-idle verdict, however long the step stays open.
	for _, idle := range []int{7, 19, 40} {
		if verdicts := progress.Check(Snapshot{OpenPlanSteps: 3, PlanIdleRounds: idle}); len(verdicts) != 0 {
			t.Fatalf("a productive enumeration was flagged at %d idle rounds: %+v", idle, verdicts)
		}
	}
	// An explicit override still bounds it, for callers that want one.
	bounded := Progress{PlanIdleRoundsWhileProgressing: 12}
	if verdicts := bounded.Check(Snapshot{OpenPlanSteps: 3, PlanIdleRounds: 13}); len(verdicts) != 1 || verdicts[0].Action != ActionNudge {
		t.Fatalf("verdicts = %+v, want the configured bound to apply", verdicts)
	}
	// The same idleness with a flat pool keeps the original fast reaction.
	stagnant := progress.Check(Snapshot{OpenPlanSteps: 3, PlanIdleRounds: 7, StagnantRounds: 3})
	if len(stagnant) != 1 || stagnant[0].Action != ActionNudge {
		t.Fatalf("verdicts = %+v, want the fast nudge once the pool is flat", stagnant)
	}
	// And the nudge then describes the plan, not a search that stopped finding.
	if !strings.Contains(stagnant[0].Advice, "retrieved nothing new") {
		t.Fatalf("advice = %q, want the flat-pool wording", stagnant[0].Advice)
	}
}

// TestProgress_KeepsAnEnumerationAlive: the nudges are spent and the search is
// flat, but the run's own plan still lists open steps. That is an enumeration
// whose last item has not been found, not a run that is finished — handing it
// to the answer call is how a long list comes back short by one name per stall.
func TestProgress_KeepsAnEnumerationAlive(t *testing.T) {
	verdicts := Progress{}.Check(Snapshot{
		Round:          20,
		StagnantRounds: 3,
		OpenPlanSteps:  1,
		// A plan that is merely not moving fast is not the idle case here: the
		// flat pool is what triggered the stall.
		PlanIdleRounds: 1,
		Nudges:         2, // the base allowance is already spent
	})
	if len(verdicts) != 1 || verdicts[0].Action != ActionNudge {
		t.Fatalf("verdicts = %+v, want a nudge, not a hand-off to the answer call", verdicts)
	}
	if !strings.Contains(verdicts[0].Advice, "open step") {
		t.Fatalf("advice = %q, want the open-step wording", verdicts[0].Advice)
	}
}

// TestProgress_EndsAnEnumerationThatNeverClosesItsPlan: the extension is bounded,
// so a plan the model never closes still ends on the policy rather than on hope.
func TestProgress_EndsAnEnumerationThatNeverClosesItsPlan(t *testing.T) {
	verdicts := Progress{}.Check(Snapshot{
		Round:          40,
		StagnantRounds: 3,
		OpenPlanSteps:  1,
		Nudges:         2 + planKeepingNudges,
	})
	if len(verdicts) != 1 || verdicts[0].Action != ActionBlock {
		t.Fatalf("verdicts = %+v, want the bounded hand-off", verdicts)
	}
}

// TestProgress_NoOpenPlanStillHandsOff: without a plan there is no claim that
// anything is left, so the original behaviour stands.
func TestProgress_NoOpenPlanStillHandsOff(t *testing.T) {
	verdicts := Progress{}.Check(Snapshot{StagnantRounds: 3, Nudges: 2})
	if len(verdicts) != 1 || verdicts[0].Action != ActionBlock {
		t.Fatalf("verdicts = %+v, want a block", verdicts)
	}
}
