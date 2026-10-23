package harness

import (
	"os"
	"strconv"
	"strings"

	"ragflow/internal/harness/agent/loop"
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
	// ModelAttempts and StreamAttempts override the retry budget. Zero keeps
	// the harness defaults (3): a deployment that wants two retries on the main
	// call and none on a stream that may already be on screen says so here.
	ModelAttempts  int
	StreamAttempts int
}

// LoadSettings reads the environment.
func LoadSettings() Settings {
	return Settings{
		SkillsRoot:     strings.TrimSpace(os.Getenv("HARNESS_SKILLS_ROOT")),
		ExecRoot:       strings.TrimSpace(os.Getenv("HARNESS_EXEC_ROOT")),
		ModelAttempts:  envInt("HARNESS_MODEL_ATTEMPTS"),
		StreamAttempts: envInt("HARNESS_STREAM_ATTEMPTS"),
	}
}

// RetryPolicy is the loop's retry budget as this deployment configured it.
func (s Settings) RetryPolicy() loop.RetryPolicy {
	return loop.RetryPolicy{Request: s.ModelAttempts, Stream: s.StreamAttempts}
}

func envInt(name string) int {
	raw := strings.TrimSpace(os.Getenv(name))
	if raw == "" {
		return 0
	}
	value, err := strconv.Atoi(raw)
	if err != nil || value < 0 {
		return 0
	}
	return value
}
