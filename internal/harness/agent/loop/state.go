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

	// CompactionBoost is how many prompt-too-long recoveries this run has
	// already spent; prepare uses it to force a harder compaction tier.
	CompactionBoost int `json:"compaction_boost"`
	// PtLErrors counts prompt-too-long rejections, bounded by ptlAttempts.
	PtLErrors int `json:"ptl_errors"`
	// RetryPrepare asks the graph to rebuild the prompt before calling the model
	// again. It is set by think after a prompt-too-long rejection.
	RetryPrepare bool `json:"retry_prepare"`
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
	FinalState     *State
}
