package definition

import (
	"strings"
	"testing"
)

// The capability list is layer ② of the system prompt: at most five lines, in a
// fixed order. Growing past that is not a test detail — it turns a summary into
// a tool inventory that costs tokens every round.
func TestCapabilityGroupsStayWithinBudget(t *testing.T) {
	if len(capabilityGroups) > 5 {
		t.Fatalf("%d capability groups; the list is capped at 5", len(capabilityGroups))
	}
	for _, group := range capabilityGroups {
		if strings.TrimSpace(group.label) == "" {
			t.Fatalf("capability group with tools %v has no label", group.tools)
		}
		if len(group.tools) == 0 {
			t.Fatalf("capability group %q advertises nothing", group.label)
		}
	}
}

func TestCapabilitySectionWithoutTools(t *testing.T) {
	if section := CapabilitySection(nil); section != "" {
		t.Fatalf("no tools rendered a section: %q", section)
	}
	if section := CapabilitySection([]string{"unknown_tool"}); section != "" {
		t.Fatalf("a tool in no group rendered a section: %q", section)
	}
}

func TestCapabilitySectionReportsOnlyWhatIsHeld(t *testing.T) {
	section := CapabilitySection([]string{"grep_chunks", "list_chunks"})
	if !strings.Contains(section, "knowledge base retrieval") {
		t.Fatalf("retrieval tools did not advertise retrieval: %q", section)
	}
	for _, absent := range []string{"web search", "workspace", "shell", "sub-agents"} {
		if strings.Contains(section, absent) {
			t.Fatalf("section claims %q without holding its tools: %q", absent, section)
		}
	}
}

func TestCapabilitySectionOrdersGroupsIndependentlyOfInput(t *testing.T) {
	surface := []string{
		"subagent_start", "exec_shell", "fs_cat", "web_search",
		"grep_chunks", "run_javascript", "list_chunks",
	}
	forward := CapabilitySection(surface)
	reversed := make([]string, 0, len(surface))
	for index := len(surface) - 1; index >= 0; index-- {
		reversed = append(reversed, surface[index])
	}
	if backward := CapabilitySection(reversed); backward != forward {
		t.Fatalf("input order changed the section:\nforward  %q\nreversed %q", forward, backward)
	}
	for _, label := range []string{"knowledge base retrieval", "web search", "workspace", "shell", "sub-agents"} {
		if !strings.Contains(forward, label) {
			t.Fatalf("full surface missing %q: %q", label, forward)
		}
	}
}

// The section is inside the cached prefix, so it has to be exactly one stable
// block: a header plus one line. A second line means someone started explaining
// how to use the tools here, which belongs in the schemas and the skill body.
func TestCapabilitySectionIsOneLine(t *testing.T) {
	section := CapabilitySection([]string{"grep_chunks"})
	lines := strings.Split(section, "\n")
	if len(lines) != 2 {
		t.Fatalf("section should be a header plus one line, got %d lines: %q", len(lines), section)
	}
	if lines[0] != "## What you can do" {
		t.Fatalf("unexpected header: %q", lines[0])
	}
}

// A tool that belongs to no group is not advertised, and a missing one must not
// shift the line: the surface changed, so the string has to change too.
func TestCapabilitySectionTracksSurfaceChanges(t *testing.T) {
	without := CapabilitySection([]string{"grep_chunks"})
	with := CapabilitySection([]string{"grep_chunks", "web_search"})
	if without == with {
		t.Fatalf("adding a capability did not change the section: %q", with)
	}
}
