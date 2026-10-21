package toolkit

// Risk classifies a tool call for scheduling. It is runtime metadata, never part
// of the schema sent to the model: asking the model to remember which tools are
// safe to parallelize is a prompt problem with no reliable answer.
//
// The split follows Astra (read-only tools join a concurrent buffer, anything
// with a serial gate becomes its own batch) and codex (parallel callers take a
// read lock, mutating callers take the write lock).
type Risk int

const (
	// RiskReadOnly can run concurrently with any other call: it changes no
	// state the model can observe.
	RiskReadOnly Risk = iota
	// RiskMutating changes state and must not overlap another call.
	RiskMutating
	// RiskDangerous runs outside the process (shell, network write). It is
	// always serial and is the set approvals apply to.
	RiskDangerous
)

// String names the risk level. Audit payloads record the name rather than the
// ordinal so a stored event stays readable without this enum.
func (r Risk) String() string {
	switch r {
	case RiskReadOnly:
		return "read_only"
	case RiskMutating:
		return "mutating"
	case RiskDangerous:
		return "dangerous"
	}
	return "unknown"
}

// RiskClassifier is an optional capability interface. A capability that does not
// implement it is treated as mutating, which schedules every one of its tools
// serially — the safe default, never the clever one.
type RiskClassifier interface {
	// ToolRisk reports the risk of one of this capability's tools.
	ToolRisk(tool string) Risk
}

// builtinRisks covers the tools the harness ships itself. A capability that
// implements RiskClassifier takes precedence, so this table only has to stay
// correct for the tools that do not classify themselves yet.
var builtinRisks = map[string]Risk{
	// retrieval is read-only by construction
	"search_chunks":          RiskReadOnly,
	"search_bm25_chunks":     RiskReadOnly,
	"search_semantic_chunks": RiskReadOnly,
	"search_metadata":        RiskReadOnly,
	"grep_chunks":            RiskReadOnly,
	"list_chunks":            RiskReadOnly,
	"navigate_tree":          RiskReadOnly,
	"navigate_structure":     RiskReadOnly,
	"graph_explore":          RiskReadOnly,
	"graph_doc_outline":      RiskReadOnly,

	// scratch space and catalog lookups
	"think":           RiskReadOnly,
	"todo_write":      RiskReadOnly,
	"read_spill":      RiskReadOnly,
	"tool_search":     RiskReadOnly,
	"skill_list":      RiskReadOnly,
	"skill_load":      RiskReadOnly,
	"web_search":      RiskReadOnly,
	"web_fetch":       RiskReadOnly,
	"fs_ls":           RiskReadOnly,
	"fs_cat":          RiskReadOnly,
	"fs_search":       RiskReadOnly,
	"fs_write":        RiskMutating,
	"fs_edit":         RiskMutating,
	"subagent":        RiskMutating,
	"spawn_subagents": RiskMutating,

	// leaves the process
	"exec_shell":     RiskDangerous,
	"run_javascript": RiskDangerous,
}

// RiskOf reports the registered risk for a tool. A capability's own
// classification wins; the built-in table fills the rest; unknown tools are
// mutating.
func (r *Registry) RiskOf(tool string) Risk {
	if r == nil {
		return riskOrMutating(tool)
	}
	r.mu.RLock()
	capability, ok := r.owner[tool]
	r.mu.RUnlock()
	if !ok || capability == nil {
		return riskOrMutating(tool)
	}
	if classifier, ok := capability.(RiskClassifier); ok {
		return classifier.ToolRisk(tool)
	}
	return riskOrMutating(tool)
}

func riskOrMutating(tool string) Risk {
	if risk, ok := builtinRisks[tool]; ok {
		return risk
	}
	return RiskMutating
}

// Parallelizable reports whether a call may overlap its siblings. A missing
// registry schedules serially: with no capability behind the name, there is
// nothing to promise about the call's side effects.
func (r *Registry) Parallelizable(tool string) bool {
	if r == nil {
		return false
	}
	return r.RiskOf(tool) == RiskReadOnly
}
