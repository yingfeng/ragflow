package exec

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// SandboxMode names the isolation mechanism a runner uses.
type SandboxMode string

const (
	// SandboxAuto picks the strongest launcher that actually works on this host.
	SandboxAuto SandboxMode = "auto"
	// SandboxBubblewrap runs the command under bubblewrap.
	SandboxBubblewrap SandboxMode = "bwrap"
	// SandboxUnshare runs the command under unshare(1) with new namespaces.
	SandboxUnshare SandboxMode = "unshare"
	// SandboxNone runs the command directly on the host. It is a development
	// escape hatch and must be requested explicitly.
	SandboxNone SandboxMode = "none"
)

// RunnerConfig describes how to build the shell runner.
type RunnerConfig struct {
	// Root bounds the writable workspace.
	Root string
	// MaxOutput caps stdout and stderr.
	MaxOutput int
	// Mode selects the launcher. Empty means SandboxAuto.
	Mode SandboxMode
	// AllowUnsandboxed permits falling back to the host shell. Without it an
	// unavailable sandbox is an error: a deployment must not silently run shell
	// commands on the host because bubblewrap happened to be missing.
	AllowUnsandboxed bool
}

// NewRunner builds the strongest available runner. It returns an error rather
// than a host runner when no sandbox works and the caller did not opt in.
func NewRunner(cfg RunnerConfig) (Runner, error) {
	mode := cfg.Mode
	if mode == "" {
		mode = SandboxAuto
	}
	if cfg.Root == "" {
		return nil, fmt.Errorf("exec: a workspace root is required")
	}
	root, err := filepath.Abs(cfg.Root)
	if err != nil {
		return nil, fmt.Errorf("exec: resolve workspace root: %w", err)
	}
	if info, err := os.Stat(root); err != nil || !info.IsDir() {
		return nil, fmt.Errorf("exec: workspace root %s is not a directory", root)
	}

	if mode == SandboxNone {
		if !cfg.AllowUnsandboxed {
			return nil, fmt.Errorf("exec: SandboxNone requires AllowUnsandboxed: the host shell is not an isolation boundary")
		}
		return NewLocalRunner(root), nil
	}

	candidates := []SandboxMode{SandboxBubblewrap, SandboxUnshare}
	if mode != SandboxAuto {
		candidates = []SandboxMode{mode}
	}
	var failures []string
	for _, candidate := range candidates {
		launcher, err := detectLauncher(candidate, root)
		if err != nil {
			failures = append(failures, fmt.Sprintf("%s: %v", candidate, err))
			continue
		}
		return &SandboxedRunner{launcher: launcher, Root: root, MaxOutput: cfg.MaxOutput}, nil
	}

	if cfg.AllowUnsandboxed {
		return NewLocalRunner(root), nil
	}
	return nil, fmt.Errorf("exec: no working sandbox (%s); set AllowUnsandboxed to accept the host shell",
		strings.Join(failures, "; "))
}

// SandboxedRunner runs commands inside an OS-level sandbox. The workspace is
// mounted read-write and the rest of the filesystem read-only, so a command can
// work on the files it was given but cannot rewrite the host.
type SandboxedRunner struct {
	launcher  *launcher
	Root      string
	MaxOutput int
}

// Describe reports the mechanism in use, for the startup log.
func (r *SandboxedRunner) Describe() string {
	if r == nil || r.launcher == nil {
		return "none"
	}
	return r.launcher.mode.String()
}

// Run implements Runner.
func (r *SandboxedRunner) Run(ctx context.Context, req Request) (Output, error) {
	timeout := req.TimeoutSec
	if timeout <= 0 {
		timeout = 120
	}
	if timeout > 900 {
		timeout = 900
	}
	runCtx, cancel := context.WithTimeout(ctx, time.Duration(timeout)*time.Second)
	defer cancel()

	workdir, err := resolveWorkdir(r.Root, req.Workdir)
	if err != nil {
		return Output{}, err
	}
	argv, err := r.launcher.argv(r.Root, workdir, req.Command)
	if err != nil {
		return Output{}, err
	}

	command := exec.CommandContext(runCtx, argv[0], argv[1:]...)
	// A minimal environment: inheriting the service environment would hand the
	// sandbox every credential the service holds.
	command.Env = sandboxEnv(r.Root)
	command.Dir = r.Root

	stdout := &limitedWriter{buffer: &bytes.Buffer{}, limit: limitOf(r.MaxOutput)}
	stderr := &limitedWriter{buffer: &bytes.Buffer{}, limit: limitOf(r.MaxOutput)}
	command.Stdout = stdout
	command.Stderr = stderr

	runErr := command.Run()
	output := Output{
		Stdout:    stdout.buffer.String(),
		Stderr:    stderr.buffer.String(),
		Truncated: stdout.truncated || stderr.truncated,
	}
	if runErr != nil {
		// A timeout surfaces as a killed child, so the deadline is checked before
		// the exit code: reporting it as a normal non-zero exit would look like
		// the command failed on its own.
		if runCtx.Err() != nil {
			return output, runCtx.Err()
		}
		var exitErr *exec.ExitError
		if errors.As(runErr, &exitErr) {
			output.ExitCode = exitErr.ExitCode()
			return output, nil
		}
		return output, fmt.Errorf("exec: %w", runErr)
	}
	return output, nil
}

// launcher is one detected sandbox binary and the argument shape it needs.
type launcher struct {
	mode SandboxMode
	path string
}

func (l *launcher) argv(root, workdir, command string) ([]string, error) {
	switch l.mode {
	case SandboxBubblewrap:
		return []string{
			l.path,
			"--unshare-all", "--die-with-parent", "--new-session",
			"--ro-bind", "/", "/",
			"--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp",
			"--bind", root, root,
			"--chdir", workdir,
			"--clearenv",
			"--setenv", "PATH", "/usr/local/bin:/usr/bin:/bin",
			"--setenv", "HOME", root,
			"--setenv", "LANG", "C.UTF-8",
			"--", "/bin/sh", "-c", command,
		}, nil
	case SandboxUnshare:
		return []string{
			l.path,
			"--user", "--map-root-user",
			"--mount", "--pid", "--net", "--ipc", "--uts",
			"--fork", "--kill-child",
			"--", "/bin/sh", "-c", "cd " + shellQuote(workdir) + " && exec " + command,
		}, nil
	}
	return nil, fmt.Errorf("exec: unknown sandbox mode %q", l.mode)
}

func (m SandboxMode) String() string { return string(m) }

var (
	launcherCache = struct {
		sync.Mutex
		byRoot map[string]*launcher
	}{byRoot: make(map[string]*launcher)}
)

// detectLauncher finds a working launcher for this workspace. Detection runs a
// real command: a binary on PATH that the kernel refuses to run (the usual case
// inside an unprivileged container) is not a sandbox.
func detectLauncher(mode SandboxMode, root string) (*launcher, error) {
	launcherCache.Lock()
	if cached, ok := launcherCache.byRoot[string(mode)+"|"+root]; ok {
		launcherCache.Unlock()
		if cached == nil {
			return nil, fmt.Errorf("%s is not usable on this host", mode)
		}
		return cached, nil
	}
	launcherCache.Unlock()

	candidate, err := probeLauncher(mode, root)

	launcherCache.Lock()
	if err != nil {
		launcherCache.byRoot[string(mode)+"|"+root] = nil
	} else {
		launcherCache.byRoot[string(mode)+"|"+root] = candidate
	}
	launcherCache.Unlock()

	if err != nil {
		return nil, err
	}
	return candidate, nil
}

func probeLauncher(mode SandboxMode, root string) (*launcher, error) {
	var binary string
	switch mode {
	case SandboxBubblewrap:
		binary = "bwrap"
	case SandboxUnshare:
		binary = "unshare"
	default:
		return nil, fmt.Errorf("unknown sandbox mode %q", mode)
	}
	path, err := exec.LookPath(binary)
	if err != nil {
		return nil, fmt.Errorf("%s not found", binary)
	}
	candidate := &launcher{mode: mode, path: path}

	argv, err := candidate.argv(root, root, "true")
	if err != nil {
		return nil, err
	}
	probeCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	probe := exec.CommandContext(probeCtx, argv[0], argv[1:]...)
	probe.Env = sandboxEnv(root)
	probe.Dir = root
	var stderr bytes.Buffer
	probe.Stderr = &stderr
	if err := probe.Run(); err != nil {
		message := strings.TrimSpace(stderr.String())
		if message == "" {
			message = err.Error()
		}
		return nil, fmt.Errorf("%s probe failed: %s", binary, message)
	}
	return candidate, nil
}

// sandboxEnv is the environment a sandboxed command sees.
func sandboxEnv(root string) []string {
	return []string{
		"PATH=/usr/local/bin:/usr/bin:/bin",
		"HOME=" + root,
		"LANG=C.UTF-8",
	}
}

func resolveWorkdir(root, workdir string) (string, error) {
	if workdir == "" {
		return root, nil
	}
	if filepath.IsAbs(workdir) {
		return "", fmt.Errorf("workdir must be relative to the workspace root")
	}
	joined := filepath.Join(root, workdir)
	cleaned := filepath.Clean(joined)
	if cleaned != root && !strings.HasPrefix(cleaned, root+string(filepath.Separator)) {
		return "", fmt.Errorf("workdir escapes the workspace root")
	}
	return cleaned, nil
}

func limitOf(max int) int {
	if max <= 0 {
		return 64 << 10
	}
	return max
}

// shellQuote single-quotes a path for /bin/sh.
func shellQuote(value string) string {
	return "'" + strings.ReplaceAll(value, "'", `'\''`) + "'"
}
