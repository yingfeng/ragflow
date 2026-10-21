package loop

import (
	"ragflow/internal/harness/agent/model"
	"ragflow/internal/harness/caps"
	"ragflow/internal/harness/policy"
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

	Pending  []PendingCall   `json:"pending"`
	Evidence []caps.Evidence `json:"evidence"`
	Errors   []StepError     `json:"errors"`

	FinalText string `json:"final_text"`

	Budget  Budget         `json:"budget"`
	Verdict policy.Verdict `json:"verdict"`
	Entry   EntryMeta      `json:"entry"`

	NewEvidence     int `json:"new_evidence"`
	EvidenceBefore  int `json:"evidence_before"`
	StagnantRounds  int `json:"stagnant_rounds"`
	RepeatedCalls   int `json:"repeated_calls"`
	ScopeViolations int `json:"scope_violations"`
	Citations       int `json:"citations"`
	UnresolvedCalls int `json:"unresolved_calls"`

	SeenCalls map[string]struct{} `json:"seen_calls"`

	Done    bool   `json:"done"`
	Stopped string `json:"stopped,omitempty"`

	// ThinkTerminal records that think ended the turn without tool calls, which
	// distinguishes an empty completion from a mid-flight round.
	ThinkTerminal bool `json:"think_terminal"`
}

// Outcome is what a finished run reports.
type Outcome struct {
	RunID       string
	Text        string
	Rounds      int
	TokensIn    int
	TokensOut   int
	ToolCalls   int
	Evidence    []caps.Evidence
	Errors      []StepError
	Citations   int
	Verdict     policy.Verdict
	Stopped     string
	Interrupted bool
	FinalState  *State
}
