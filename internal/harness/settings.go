package harness

import (
	"os"
	"strings"
)

// Settings are the deployment resources the harness reads from the environment.
//
// There is no on/off switch and no model default here: whether an agent runs is
// a per-agent decision recorded in the database, and which model it uses comes
// from the caller (a chat thread's model, or the run request's model_ref). A
// process-wide flag would make an existing agent silently stop working when a
// deployment variable changed.
//
// What remains is resource location, which the process cannot invent: a skills
// directory either exists on this worker or it does not, and a shell workspace
// is only usable where it exists. A capability whose resource is absent is not
// registered at all (see registerCapabilities), so an empty value degrades the
// tool surface instead of breaking a turn.
type Settings struct {
	// SkillsRoot holds <name>/SKILL.md packages layered over the built-in set.
	// Empty leaves only the built-in skills, which always ship with the binary.
	SkillsRoot string
	// ExecRoot is the shell workspace. Empty leaves exec_shell unregistered.
	// When set, the sandbox is auto-detected: no working isolation means the
	// tool is not registered either, because the host shell is never exposed.
	ExecRoot string
}

// LoadSettings reads the environment.
func LoadSettings() Settings {
	return Settings{
		SkillsRoot: strings.TrimSpace(os.Getenv("HARNESS_SKILLS_ROOT")),
		ExecRoot:   strings.TrimSpace(os.Getenv("HARNESS_EXEC_ROOT")),
	}
}
