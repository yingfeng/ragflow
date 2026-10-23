package loop

import (
	"context"
	"fmt"
	"strings"

	"ragflow/internal/harness/agent/model"
)

// reviewChangeLimit bounds one review pass. The reviewer reads diffs, not a
// corpus: past a handful of files the useful question stops being "is each
// change right" and becomes "what was this run trying to do".
const reviewChangeLimit = 8

// reviewSystemPrompt is the whole contract of the review pass. It gets the diff
// and the answer, no tools, and one job: say what is wrong. A reviewer that
// restates the change is noise, and a reviewer that scores it is not trusted.
const reviewSystemPrompt = `You are reviewing the file changes an agent just made, after it answered the user.

You have no tools. Read only what you were given: the answer, and the diff of every file the run wrote. Report only problems that a careful maintainer would fix before shipping:
- a change that contradicts the answer, or an answer that claims something the diff does not show;
- a change that destroys content it did not need to touch, or edits a file the question did not call for;
- a secret, credential or private datum written into a file that will be stored;
- text written in the wrong place (a memory written as a skill, a note stored under a path the loader will not read).

Reply with at most three short bullet points. If nothing is wrong, reply exactly: no issues. Do not summarise the changes and do not grade the answer.`

// reviewChanges runs the review pass over what the run wrote and returns its
// verdict, or "" when the run wrote nothing (or the reviewer failed, which is
// reported as a step rather than failing a run that already answered).
//
// The change list is consumed: a retried or resumed run reviews only what it
// has not reviewed yet.
func reviewChanges(ctx context.Context, state *State, deps Deps) string {
	if deps.Changes == nil {
		return ""
	}
	changes := deps.Changes.Changes()
	if len(changes) == 0 {
		return ""
	}
	deps.Changes.Reset()

	var body strings.Builder
	for index, change := range changes {
		if index >= reviewChangeLimit {
			fmt.Fprintf(&body, "\n[%d further changes not shown]\n", len(changes)-reviewChangeLimit)
			break
		}
		body.WriteString(change.Diff)
		if !strings.HasSuffix(change.Diff, "\n") {
			body.WriteString("\n")
		}
	}

	request := model.Request{
		Messages: []model.Message{
			{Role: model.RoleSystem, Content: reviewSystemPrompt},
			{Role: model.RoleUser, Content: fmt.Sprintf("The answer was:\n%s\n\nThe changes were:\n%s",
				clip(state.FinalText, 2000), body.String())},
		},
	}
	response, err := callModel(ctx, deps.Model, request, deps.Breaker)
	if err != nil {
		record(deps, StepRecord{
			RunID:    state.RunID,
			Phase:    "review",
			Round:    state.Entry.Round,
			Decision: "error",
			Reason:   clip(err.Error(), 200),
		})
		return ""
	}
	review := strings.TrimSpace(response.Message.Content)
	if review == "" || strings.EqualFold(review, "no issues") || strings.EqualFold(review, "no issues.") {
		record(deps, StepRecord{RunID: state.RunID, Phase: "review", Round: state.Entry.Round, Decision: "clean"})
		return ""
	}
	record(deps, StepRecord{
		RunID:    state.RunID,
		Phase:    "review",
		Round:    state.Entry.Round,
		Decision: "flagged",
		Reason:   clip(review, 400),
	})
	return review
}
