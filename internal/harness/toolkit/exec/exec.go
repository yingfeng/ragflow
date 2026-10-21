// Package exec exposes shell execution as a capability.
//
// The capability is inert until a Runner is supplied: there is no ambient host
// shell. Production wires a container-backed runner; LocalRunner exists for
// development and is opt-in.
package exec

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"ragflow/internal/harness/agent/model"
	"ragflow/internal/harness/toolkit"
)

// Request is one command.
type Request struct {
	Command    string
	Workdir    string
	TimeoutSec int
}

// Output is one command result.
type Output struct {
	Stdout    string
	Stderr    string
	ExitCode  int
	Truncated bool
}

// Runner executes commands.
type Runner interface {
	Run(ctx context.Context, req Request) (Output, error)
}

// Capability exposes the exec tool.
type Capability struct {
	runner Runner
}

// New builds the capability. A nil runner keeps the tool registered and
// answering "not configured".
func New(runner Runner) *Capability {
	return &Capability{runner: runner}
}

// Name implements toolkit.Capability.
func (*Capability) Name() string { return "exec" }

// Tools implements toolkit.Capability.
func (*Capability) Tools() []model.ToolSpec {
	return []model.ToolSpec{{
		Name:        "exec_shell",
		Description: "Run a shell command in the run workspace. Use for builds, tests and scripts.",
		Parameters: toolkit.ObjectSchema(map[string]any{
			"command":         map[string]any{"type": "string"},
			"workdir":         map[string]any{"type": "string", "description": "Relative to the workspace root."},
			"timeout_seconds": map[string]any{"type": "integer", "description": "Default 120, max 900."},
		}, "command"),
	}}
}

type shellArgs struct {
	Command    string `json:"command"`
	Workdir    string `json:"workdir"`
	TimeoutSec int    `json:"timeout_seconds"`
}

// Invoke implements toolkit.Capability.
func (c *Capability) Invoke(ctx context.Context, call toolkit.Call) (toolkit.Result, error) {
	if call.Tool != "exec_shell" {
		return toolkit.Result{}, fmt.Errorf("exec: unknown tool %q", call.Tool)
	}
	if c.runner == nil {
		return toolkit.Result{}, fmt.Errorf("shell execution is not configured for this deployment")
	}
	args, err := toolkit.DecodeArgs[shellArgs](call.Arguments)
	if err != nil {
		return toolkit.Result{}, err
	}
	if strings.TrimSpace(args.Command) == "" {
		return toolkit.Result{}, fmt.Errorf("command is required")
	}
	out, err := c.runner.Run(ctx, Request{
		Command:    args.Command,
		Workdir:    args.Workdir,
		TimeoutSec: args.TimeoutSec,
	})
	if err != nil && errors.Is(err, context.DeadlineExceeded) {
		return toolkit.Result{Content: "command timed out", IsError: true}, nil
	}
	if err != nil {
		return toolkit.Result{}, err
	}
	return toolkit.Result{
		Content: render(out),
		IsError: out.ExitCode != 0,
		Meta:    map[string]any{"exit_code": out.ExitCode, "truncated": out.Truncated},
	}, nil
}

func render(out Output) string {
	var builder strings.Builder
	builder.WriteString(fmt.Sprintf("exit=%d\n", out.ExitCode))
	if out.Stdout != "" {
		builder.WriteString("--- stdout ---\n")
		builder.WriteString(out.Stdout)
		if !strings.HasSuffix(out.Stdout, "\n") {
			builder.WriteString("\n")
		}
	}
	if out.Stderr != "" {
		builder.WriteString("--- stderr ---\n")
		builder.WriteString(out.Stderr)
		if !strings.HasSuffix(out.Stderr, "\n") {
			builder.WriteString("\n")
		}
	}
	if out.Truncated {
		builder.WriteString("--- output truncated ---\n")
	}
	return strings.TrimSpace(builder.String())
}

// LocalRunner runs commands on the host, confined to a workspace root. It is for
// development only; production wires a container runner.
type LocalRunner struct {
	Root      string
	MaxOutput int
}

// NewLocalRunner builds a host runner. An empty root disables workdir overrides.
func NewLocalRunner(root string) *LocalRunner {
	return &LocalRunner{Root: root, MaxOutput: 64 << 10}
}

// Run implements Runner.
func (r *LocalRunner) Run(ctx context.Context, req Request) (Output, error) {
	timeout := req.TimeoutSec
	if timeout <= 0 {
		timeout = 120
	}
	if timeout > 900 {
		timeout = 900
	}
	runCtx, cancel := context.WithTimeout(ctx, time.Duration(timeout)*time.Second)
	defer cancel()

	command := exec.CommandContext(runCtx, "/bin/sh", "-c", req.Command)
	if r.Root != "" {
		dir, err := r.resolveWorkdir(req.Workdir)
		if err != nil {
			return Output{}, err
		}
		command.Dir = dir
	}

	stdout := &limitedWriter{buffer: &bytes.Buffer{}, limit: r.limit()}
	stderr := &limitedWriter{buffer: &bytes.Buffer{}, limit: r.limit()}
	command.Stdout = stdout
	command.Stderr = stderr

	err := command.Run()
	output := Output{
		Stdout:    stdout.buffer.String(),
		Stderr:    stderr.buffer.String(),
		Truncated: stdout.truncated || stderr.truncated,
	}
	if err != nil {
		var exitErr *exec.ExitError
		if errors.As(err, &exitErr) {
			output.ExitCode = exitErr.ExitCode()
			return output, nil
		}
		if runCtx.Err() != nil {
			return output, runCtx.Err()
		}
		return output, fmt.Errorf("exec: %w", err)
	}
	return output, nil
}

func (r *LocalRunner) limit() int {
	if r.MaxOutput <= 0 {
		return 64 << 10
	}
	return r.MaxOutput
}

func (r *LocalRunner) resolveWorkdir(workdir string) (string, error) {
	if workdir == "" {
		return r.Root, nil
	}
	if filepath.IsAbs(workdir) {
		return "", fmt.Errorf("workdir must be relative to the workspace root")
	}
	joined := filepath.Join(r.Root, workdir)
	rootClean := filepath.Clean(r.Root)
	joinedClean := filepath.Clean(joined)
	if joinedClean != rootClean && !strings.HasPrefix(joinedClean, rootClean+string(filepath.Separator)) {
		return "", fmt.Errorf("workdir escapes the workspace root")
	}
	return joinedClean, nil
}

type limitedWriter struct {
	buffer    *bytes.Buffer
	limit     int
	truncated bool
}

func (w *limitedWriter) Write(p []byte) (int, error) {
	remaining := w.limit - w.buffer.Len()
	if remaining <= 0 {
		w.truncated = true
		return len(p), nil
	}
	if len(p) > remaining {
		w.buffer.Write(p[:remaining])
		w.truncated = true
		return len(p), nil
	}
	return w.buffer.Write(p)
}
