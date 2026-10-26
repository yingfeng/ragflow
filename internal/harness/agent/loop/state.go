package loop

import (
	"ragflow/internal/harness/agent/model"
	"ragflow/internal/harness/policy"
	"ragflow/internal/harness/toolkit"
)

// PendingCall is a tool call the model requested and act has not executed yet.
type PendingCall struct {
	ID             string `json:"id"`
	Tool           string `json:"tool"`
	Arguments      string `json:"arguments"`
	IdempotencyKey string `json:"idempotency_key"`
}

// StepError is a tool failure worth surfacing to policies.
type StepError struct {
	Tool    string `json:"tool"`
	Message string `json:"message"`
}

// SearchLog is one retrieval the run already executed, keyed by the query it
// sent rather than by what came back.
//
// It carries no result text: the log exists to stop repeated work, and a list of
// findings here is what makes a model answer from the log instead of finishing
// the search (see renderSearchLog).
type SearchLog struct {
	Tool  string `json:"tool"`
	Query string `json:"query"`
	// Hits is how many passages the call returned; Count is how often the run
	// issued the same query. A count above one is the repetition the log is
	// there to make visible to the model.
	Hits  int `json:"hits,omitempty"`
	Count int `json:"count,omitempty"`
	Round int `json:"round,omitempty"`
}

// EntryMeta is the serializable loop position. Recovery reads this instead of
// inferring position from message content.
type EntryMeta struct {
	Round      int `json:"round"`
	PauseCount int `json:"pause_count"`
}

// Budget is the run's accounting.
type Budget struct {
	MaxRounds    int     `json:"max_rounds"`
	MaxTokens    int     `json:"max_tokens"`
	TokensIn     int     `json:"tokens_in"`
	TokensOut    int     `json:"tokens_out"`
	MaxToolCalls int     `json:"max_tool_calls"`
	ToolCalls    int     `json:"tool_calls"`
	MaxWallMS    int64   `json:"max_wall_ms"`
	ElapsedMS    int64   `json:"elapsed_ms"`
	MaxCostUSD   float64 `json:"max_cost_usd"`
	CostUSD      float64 `json:"cost_usd"`
}

// State is the graph state. It holds data only: model, registry and policies are
// captured by node closures so the state stays serializable.
type State struct {
	RunID     string `json:"run_id"`
	AgentName string `json:"agent_name"`
	StartedMS int64  `json:"started_ms"`

	// History is the conversation. Prompt is the compacted view rebuilt by
	// prepare and consumed by think.
	History []model.Message `json:"history"`
	Prompt  []model.Message `json:"prompt"`

	// Question is this turn's request, unchanged. Compaction drops messages from
	// the front of History, and the request is the first thing a long run loses;
	// prepare re-states it every round so a run that has read forty passages
	// still answers the question it was asked rather than the topic of the last
	// passage it read.
	Question string `json:"question,omitempty"`

	Pending  []PendingCall      `json:"pending"`
	Evidence []toolkit.Evidence `json:"evidence"`
	Errors   []StepError        `json:"errors"`

	// Plan is the model's own step list, published by a tool through
	// toolkit.MetaPlan and rendered into every later prompt. Keeping it out of
	// History is what makes it survive compaction.
	Plan []toolkit.PlanStep `json:"plan,omitempty"`
	// PlanRound is the round the plan last changed in (a step added, closed or
	// reworded). An open plan that stops moving is not progress: it is the model
	// holding the run open with work it is no longer doing, which is how one run
	// spent all 50 of its rounds with three steps open.
	PlanRound int `json:"plan_round,omitempty"`
	// PlanShown is the fingerprint of the plan as last rendered into Prompt.
	// The plan is restated only when it changes: repeating an unchanged step
	// list every round is harness-only noise, and it reads as a reminder that
	// the list is the whole job rather than a record of what is left to do.
	PlanShown string `json:"plan_shown,omitempty"`

	// Searches is the run's own retrieval log: which queries already ran, and
	// how much each returned. It lives on State rather than in History because
	// compaction drops the tool messages it is derived from — and a model that
	// cannot see what it already searched re-runs the same query in new words
	// until the round budget runs out.
	Searches []SearchLog `json:"searches,omitempty"`
	// ReadChunks lists chunk ids the run already retrieved, in first-seen
	// order. Ids only: this is a record of work done, not of findings, because
	// a list of findings is what made a model answer from the list instead of
	// finishing the search.
	ReadChunks []string `json:"read_chunks,omitempty"`

	// RecentEvidence holds the new-evidence count of the last few rounds. A
	// single productive round is not progress: a reworded query returns one more
	// snippet of the same passage, so the pool creeps up every round while
	// nothing new is learned. Judging a window instead of a round is what lets a
	// run that keeps "finding" the same material be called stagnant.
	RecentEvidence []int `json:"recent_evidence,omitempty"`

	// ResumeHandled records that the first act after a restart already treated
	// its inherited batch as possibly-already-dispatched. Later rounds in the
	// same process must execute normally.
	ResumeHandled bool `json:"resume_handled,omitempty"`

	// BudgetNotified records that the run already told the model how much of
	// the budget is gone. One fact, once — nagging spends the budget it reports.
	BudgetNotified bool `json:"budget_notified,omitempty"`

	FinalText string `json:"final_text"`

	Budget  Budget         `json:"budget"`
	Verdict policy.Verdict `json:"verdict"`
	Entry   EntryMeta      `json:"entry"`

	NewEvidence int `json:"new_evidence"`
	// EvidenceBefore is the pool size at the start of the round; NewEvidence is
	// what the round added. EvidenceCalls counts the round's calls that returned
	// evidence units at all, which is how observe tells "searched and found
	// nothing new" from "did not search this round".
	EvidenceBefore int `json:"evidence_before"`
	EvidenceCalls  int `json:"evidence_calls"`
	StagnantRounds int `json:"stagnant_rounds"`
	RepeatedCalls  int `json:"repeated_calls"`
	// Nudges counts the course corrections injected so far. A stalled run is
	// told the facts twice; after that the answer call takes over.
	Nudges          int `json:"nudges"`
	ScopeViolations int `json:"scope_violations"`
	Citations       int `json:"citations"`
	UnresolvedCalls int `json:"unresolved_calls"`

	// SeenOutputs fingerprints every tool output this run has produced, and
	// NewOutputs counts the results of the round that just ran whose content had
	// not appeared before. Together they are the tool-agnostic progress signal:
	// no policy has to know what a "citation" is to tell progress from a loop.
	SeenOutputs map[string]struct{} `json:"seen_outputs"`
	NewOutputs  int                 `json:"new_outputs"`

	SeenCalls map[string]struct{} `json:"seen_calls"`
	// CallRepeats counts each identical call so act can refuse the ones that are
	// clearly a loop rather than progress.
	CallRepeats  map[string]int `json:"call_repeats"`
	BlockedCalls int            `json:"blocked_calls"`

	// PrefixHash identifies the stable prompt prefix of the last round. A change
	// means the provider's prefix cache was rebuilt, which is a cost event worth
	// counting rather than guessing at.
	PrefixHash     string `json:"prefix_hash,omitempty"`
	PrefixRebuilds int    `json:"prefix_rebuilds"`

	Done    bool   `json:"done"`
	Stopped string `json:"stopped,omitempty"`

	// ThinkTerminal records that think ended the turn without tool calls, which
	// distinguishes an empty completion from a mid-flight round.
	ThinkTerminal bool `json:"think_terminal"`
	// TerminalRounds counts consecutive rounds whose think produced no tool
	// call. Completion lets one such round work on an open plan step and no
	// more: a model that only restates its answer is not making progress, and
	// continuing it would spend the budget on prose.
	TerminalRounds int `json:"terminal_rounds"`

	// Steers counts messages injected into the run while it was executing.
	Steers int `json:"steers"`

	// TurnStart is the history index this turn's question sits at. The
	// required-tool gate reads it as the edge of the turn, because prepare
	// appends user messages of its own (the plan, the context note, a search
	// log) and scanning back to the nearest user message would find one of
	// those instead of the question — making every injected note look like a
	// fresh turn that still owes a retrieval call.
	TurnStart int `json:"turn_start"`

	// CompactionBoost is the compaction tier for the retry of a prompt the
	// provider just refused; prepare uses it to force a harder one. It is reset
	// as soon as a call is accepted (see acceptPrompt): a refusal is a fact
	// about one prompt, and keeping the tier raised turned it into a fact about
	// every later round.
	CompactionBoost int `json:"compaction_boost"`
	// PtLErrors counts consecutive prompt-too-long rejections, bounded by
	// ptlAttempts, and is reset by the same accepted call.
	PtLErrors int `json:"ptl_errors"`
	// RetryPrepare asks the graph to rebuild the prompt before calling the model
	// again. It is set by think after a prompt-too-long rejection.
	RetryPrepare bool `json:"retry_prepare"`
	// ContextNotified records that the run was told how full its window is, so
	// the reminder is stated once instead of every round: a restated threshold
	// reads as an instruction to stop early.
	ContextNotified bool `json:"context_notified,omitempty"`
	// Review is the independent reviewer's verdict on the run's file writes.
	// Empty means either nothing was written or the review found no issues.
	Review string `json:"review,omitempty"`
}

// acceptPrompt records that the provider accepted a call, so the compaction
// state that belongs to a refusal is spent.
//
// The tier a refusal raises is what makes the retry smaller. It is not a
// property of the run: a prompt that fits now may not fit twenty rounds later,
// and the round that discovers that will raise the tier again. Keeping it
// raised made every later build compact, which on a retrieval run meant every
// round erased the tool output the model had just read.
func (s *State) acceptPrompt() {
	s.CompactionBoost = 0
	s.PtLErrors = 0
}

// Outcome is what a finished run reports.
type Outcome struct {
	RunID     string
	Text      string
	Rounds    int
	TokensIn  int
	TokensOut int
	// CostUSD is the priced total of this run, zero when the deployment does
	// not configure per-token prices.
	CostUSD     float64
	ToolCalls   int
	Evidence    []toolkit.Evidence
	Errors      []StepError
	Citations   int
	Verdict     policy.Verdict
	Stopped     string
	Interrupted bool
	// PrefixRebuilds counts how often the stable prompt prefix changed, and
	// BlockedCalls how many repeated calls the breaker refused.
	PrefixRebuilds int
	BlockedCalls   int
	// Review is the independent reviewer's verdict on this run's file writes:
	// empty when nothing was written or the review found no issues.
	Review     string
	FinalState *State
}
