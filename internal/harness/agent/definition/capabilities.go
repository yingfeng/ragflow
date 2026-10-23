package definition

import "strings"

// capabilityGroup is one line of the capability list, advertised when the run
// holds any of its tools.
//
// The tool surface is the only fact the runtime has, so deriving the line from
// it is what stops the list from claiming a capability the run cannot reach: an
// agent with no fs tools is not told it can edit files, and an agent whose
// dataset scope is empty is not told it can search a corpus.
type capabilityGroup struct {
	label string
	tools []string
}

// capabilityGroups is layer ② of the system prompt: what the run can do, in a
// fixed order, and nothing about how.
//
// The "how" lives elsewhere on purpose. Tool schemas carry parameters and
// boundaries; the retrieval skill carries the order of work. A capability list
// that explained itself would be duplicating both, every round, in the part of
// the prompt a provider caches — so this block changes when the tool surface
// changes and not when a policy file does.
var capabilityGroups = []capabilityGroup{
	{
		label: "knowledge base retrieval",
		tools: []string{
			"grep_chunks", "list_chunks", "search_bm25_chunks", "search_semantic_chunks",
			"search_chunks", "search_metadata", "navigate_tree", "navigate_structure",
			"graph_explore", "graph_doc_outline",
		},
	},
	{label: "web search", tools: []string{"web_search"}},
	{
		label: "a persistent workspace you can read and edit",
		tools: []string{"fs_cat", "fs_edit", "fs_ls", "fs_search", "fs_write"},
	},
	{
		label: "a sandboxed shell and JavaScript runtime",
		tools: []string{"exec_shell", "run_javascript"},
	},
	{
		label: "delegation to sub-agents",
		tools: []string{"subagent_start", "subagent_status", "subagent_stop", "subagent_message"},
	},
}

// CapabilitySection renders layer ② of the system prompt.
//
// The input is the run's whole tool surface, deferred tools included: a tool the
// model can enable is a capability it holds, and leaving deferred tools out
// would make the list flip mid-run the first time one is loaded. Nothing else
// varies with the input order, so the result is byte-stable for a given surface
// — which is what keeps it inside the prefix a provider caches.
//
// An empty surface (or one whose tools belong to no group) renders nothing: an
// agent with no tools needs no capability list, and inventing one would fill the
// prompt with a description of abilities it does not have.
func CapabilitySection(toolNames []string) string {
	held := make(map[string]struct{}, len(toolNames))
	for _, name := range toolNames {
		held[name] = struct{}{}
	}
	labels := make([]string, 0, len(capabilityGroups))
	for _, group := range capabilityGroups {
		for _, tool := range group.tools {
			if _, ok := held[tool]; ok {
				labels = append(labels, group.label)
				break
			}
		}
	}
	if len(labels) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("## What you can do\nYou have: ")
	b.WriteString(strings.Join(labels, "; "))
	b.WriteString(".")
	return b.String()
}
