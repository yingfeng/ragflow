package exec

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// sandboxRunner builds a real sandboxed runner, or skips when this host has no
// working launcher (an unprivileged container usually has the binary but not the
// kernel permission).
func sandboxRunner(t *testing.T) *SandboxedRunner {
	t.Helper()
	root := t.TempDir()
	runner, err := NewRunner(RunnerConfig{Root: root})
	if err != nil {
		t.Skipf("no usable sandbox on this host: %v", err)
	}
	sandboxed, ok := runner.(*SandboxedRunner)
	if !ok {
		t.Fatalf("auto mode returned %T, want a sandboxed runner", runner)
	}
	return sandboxed
}

func TestNewRunnerRefusesHostShellWithoutOptIn(t *testing.T) {
	root := t.TempDir()

	if _, err := NewRunner(RunnerConfig{Root: root, Mode: SandboxNone}); err == nil {
		t.Fatal("SandboxNone was accepted without AllowUnsandboxed")
	}

	runner, err := NewRunner(RunnerConfig{Root: root, Mode: SandboxNone, AllowUnsandboxed: true})
	if err != nil {
		t.Fatalf("explicit host shell was refused: %v", err)
	}
	if _, ok := runner.(*LocalRunner); !ok {
		t.Fatalf("runner = %T, want LocalRunner", runner)
	}
}

func TestNewRunnerRejectsUnknownMode(t *testing.T) {
	if _, err := NewRunner(RunnerConfig{Root: t.TempDir(), Mode: "nsjail"}); err == nil {
		t.Fatal("an unknown sandbox mode was accepted")
	}
}

func TestNewRunnerRequiresWorkspaceDirectory(t *testing.T) {
	if _, err := NewRunner(RunnerConfig{Root: filepath.Join(t.TempDir(), "missing")}); err == nil {
		t.Fatal("a missing workspace root was accepted")
	}
	if _, err := NewRunner(RunnerConfig{}); err == nil {
		t.Fatal("an empty workspace root was accepted")
	}
}

func TestSandboxedRunnerRunsCommand(t *testing.T) {
	runner := sandboxRunner(t)

	output, err := runner.Run(context.Background(), Request{Command: "echo hello"})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if strings.TrimSpace(output.Stdout) != "hello" {
		t.Fatalf("stdout = %q", output.Stdout)
	}
	if output.ExitCode != 0 {
		t.Fatalf("exit code = %d", output.ExitCode)
	}
}

func TestSandboxedRunnerKeepsWritesInsideTheWorkspace(t *testing.T) {
	runner := sandboxRunner(t)

	// The workspace is mounted read-write, so a file the agent creates is real.
	output, err := runner.Run(context.Background(), Request{Command: "touch inside.txt && ls"})
	if err != nil {
		t.Fatalf("run inside workspace: %v", err)
	}
	if !strings.Contains(output.Stdout, "inside.txt") {
		t.Fatalf("workspace write did not land: %q", output.Stdout)
	}
	if _, err := os.Stat(filepath.Join(runner.Root, "inside.txt")); err != nil {
		t.Fatalf("created file is not visible to the host: %v", err)
	}

	// Everything outside it is read-only: the sandbox is the boundary, not the
	// process's good behaviour.
	outside := filepath.Join(os.TempDir(), "harness-sandbox-escape-check")
	_ = os.Remove(outside)
	output, _ = runner.Run(context.Background(), Request{Command: "touch " + outside})
	if _, err := os.Stat(outside); err == nil {
		_ = os.Remove(outside)
		t.Fatal("the sandbox wrote outside the workspace")
	}
}

func TestSandboxedRunnerRejectsEscapingWorkdir(t *testing.T) {
	runner := sandboxRunner(t)

	if _, err := runner.Run(context.Background(), Request{Command: "ls", Workdir: "../.."}); err == nil {
		t.Fatal("a workdir outside the workspace was accepted")
	}
	if _, err := runner.Run(context.Background(), Request{Command: "ls", Workdir: "/etc"}); err == nil {
		t.Fatal("an absolute workdir was accepted")
	}
}

func TestSandboxedRunnerTimesOut(t *testing.T) {
	runner := sandboxRunner(t)

	start := time.Now()
	_, err := runner.Run(context.Background(), Request{Command: "sleep 30", TimeoutSec: 1})
	if err == nil {
		t.Fatal("expected a timeout error")
	}
	if time.Since(start) > 15*time.Second {
		t.Fatalf("timeout took %s", time.Since(start))
	}
}

func TestSandboxedRunnerTruncatesOutput(t *testing.T) {
	runner := sandboxRunner(t)
	runner.MaxOutput = 16

	output, err := runner.Run(context.Background(), Request{Command: "printf 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'"})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if len(output.Stdout) != 16 {
		t.Fatalf("stdout length = %d, want 16", len(output.Stdout))
	}
	if !output.Truncated {
		t.Fatal("truncation was not reported")
	}
	if !strings.Contains(render(output), "output truncated") {
		t.Fatalf("rendered output does not announce the truncation: %q", render(output))
	}
}

func TestLocalRunnerStillConfinesWorkdir(t *testing.T) {
	root := t.TempDir()
	runner := NewLocalRunner(root)

	if _, err := runner.Run(context.Background(), Request{Command: "ls", Workdir: "../.."}); err == nil {
		t.Fatal("LocalRunner accepted an escaping workdir")
	}
	output, err := runner.Run(context.Background(), Request{Command: "pwd"})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if strings.TrimSpace(output.Stdout) != root {
		t.Fatalf("pwd = %q, want %q", strings.TrimSpace(output.Stdout), root)
	}
}
