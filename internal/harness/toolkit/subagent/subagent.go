// Package subagent lets a run delegate to an independent child run. The child has
// its own session and budget; only its final output crosses the boundary.
package subagent

import (
	"context"
	"fmt"
	"strings"
	"sync"

	"ragflow/internal/harness/agent/model"
	"ragflow/internal/harness/toolkit"
)

// Tool names. They are exported so a host can strip the delegation tools from
// the registry a child run inherits.
const (
	ToolName      = "subagent"
	SpawnManyName = "spawn_subagents"
	// StartName opens a child without waiting for it.
	StartName = "subagent_start"
	// StatusName lists child runs, or waits for the named ones.
	StatusName = "subagent_status"
	// StopName cancels one child run.
	StopName = "subagent_stop"
	// MessageName injects a message into one child run.
	MessageName = "subagent_message"
)

const (
	// DefaultMaxDepth is how many delegation levels a run may open. Astra's
	// orchestrator profile defaults to 3.
	DefaultMaxDepth = 3
	// AbsoluteMaxDepth is the ceiling no configuration may exceed.
	AbsoluteMaxDepth = 8
	// MaxParallelChildren bounds one fan-out.
	MaxParallelChildren = 50
	// childConcurrency bounds how many children run at once. A batch of 50 is a
	// useful admission limit, but executing 50 model loops simultaneously turns
	// one provider hiccup into 50 retries.
	childConcurrency = 16
	// maxChildResultBytes bounds one child's answer inside the parent's context.
	// The full answer stays in the child run, which is still readable there.
	maxChildResultBytes = 8 << 10
)

// Request is one delegation.
type Request struct {
	AgentID string
	Prompt  string
	Scope   toolkit.Scope
	// Depth is the child's delegation depth: parent depth + 1.
	Depth int
}

// Outcome is the child's result.
type Outcome struct {
	RunID     string
	Output    string
	TokensIn  int
	TokensOut int
	CostUSD   float64
}

// Spawner is implemented by the run service.
type Spawner interface {
	Spawn(ctx context.Context, req Request) (Outcome, error)
}

// ChildState is what the control plane can report about one child run.
type ChildState struct {
	RunID  string
	Status string
	Output string
	Error  string
}

// Done reports whether the child reached a terminal state.
func (c ChildState) Done() bool {
	switch c.Status {
	case "done", "finished", "failed", "cancelled", "paused":
		return true
	}
	return false
}

// Controller is the asynchronous control plane of delegation: start a child
// without blocking, inspect it, stop it, or send it a message. A spawner that
// only implements Spawn still serves the blocking tool; the control tools are
// registered only when the spawner can actually honour them.
type Controller interface {
	// Start opens a child run and returns its id.
	Start(ctx context.Context, req Request) (string, error)
	// List reports the given children, or the current run's children when
	// runIDs is empty.
	List(ctx context.Context, runIDs []string) ([]ChildState, error)
	// Stop cancels one child. It reports whether the child was running here.
	Stop(ctx context.Context, runID string) bool
	// Message delivers one message to a running child.
	Message(ctx context.Context, runID, message string) bool
}

// Capability exposes the delegation tools.
type Capability struct {
	spawner    Spawner
	controller Controller
	maxDepth   int
}

// New builds the capability with the default depth limit. A spawner that also
// implements Controller gets the asynchronous control tools.
func New(spawner Spawner) *Capability {
	return NewWithLimits(spawner, DefaultMaxDepth)
}

// NewWithLimits builds the capability with an explicit depth limit, clamped to
// AbsoluteMaxDepth.
func NewWithLimits(spawner Spawner, maxDepth int) *Capability {
	if maxDepth <= 0 {
		maxDepth = DefaultMaxDepth
	}
	if maxDepth > AbsoluteMaxDepth {
		maxDepth = AbsoluteMaxDepth
	}
	capability := &Capability{spawner: spawner, maxDepth: maxDepth}
	if controller, ok := spawner.(Controller); ok {
		// A controller that cannot serve the tools must not offer them: listing a
		// child needs a run store, and a tool that always fails is worse than no
		// tool at all.
		gate, hasGate := spawner.(interface{ ControlPlaneEnabled() bool })
		if !hasGate || gate.ControlPlaneEnabled() {
			capability.controller = controller
		}
	}
	return capability
}

// Name implements toolkit.Capability.
func (*Capability) Name() string { return "subagent" }

// ToolRisk implements toolkit.RiskClassifier. Spawning a child costs budget and may
// run side effects, so it never overlaps another call.
func (*Capability) ToolRisk(string) toolkit.Risk { return toolkit.RiskMutating }

// Tools implements toolkit.Capability.
func (c *Capability) Tools() []model.ToolSpec {
	specs := []model.ToolSpec{
		{
			Name:        ToolName,
			Description: "Delegate a self-contained task to a child agent and wait for its answer. The child has its own context, so send everything it needs and expect only its final answer.",
			Parameters: toolkit.ObjectSchema(map[string]any{
				"prompt":   map[string]any{"type": "string", "description": "Self-contained task statement."},
				"agent_id": map[string]any{"type": "string", "description": "Optional agent definition to run."},
			}, "prompt"),
		},
		{
			Name:        SpawnManyName,
			Description: fmt.Sprintf("Run independent sub-tasks in parallel, one child per task, and return all answers. Use it when the sub-tasks share no intermediate results. At most %d tasks.", MaxParallelChildren),
			Parameters: toolkit.ObjectSchema(map[string]any{
				"tasks": map[string]any{
					"type":        "array",
					"items":       map[string]any{"type": "string"},
					"description": fmt.Sprintf("Independent task statements, at most %d.", MaxParallelChildren),
				},
			}, "tasks"),
		},
	}
	if c != nil && c.controller != nil {
		specs = append(specs, c.controlTools()...)
	}
	return specs
}

// Invoke implements toolkit.Capability.
func (c *Capability) Invoke(ctx context.Context, call toolkit.Call) (toolkit.Result, error) {
	switch call.Tool {
	case ToolName:
		return c.spawnOne(ctx, call)
	case SpawnManyName:
		return c.spawnMany(ctx, call)
	case StartName, StatusName, StopName, MessageName:
		return c.control(ctx, call)
	default:
		return toolkit.Result{}, fmt.Errorf("subagent: unknown tool %q", call.Tool)
	}
}

type spawnArgs struct {
	Prompt  string `json:"prompt"`
	AgentID string `json:"agent_id"`
}

type spawnManyArgs struct {
	Tasks []string `json:"tasks"`
}

func (c *Capability) spawnMany(ctx context.Context, call toolkit.Call) (toolkit.Result, error) {
	if c.spawner == nil {
		return toolkit.Result{}, fmt.Errorf("subagents are not configured for this deployment")
	}
	if err := c.checkDepth(ctx); err != nil {
		return toolkit.Result{}, err
	}
	args, err := toolkit.DecodeArgs[spawnManyArgs](call.Arguments)
	if err != nil {
		return toolkit.Result{}, err
	}
	tasks := make([]string, 0, len(args.Tasks))
	for _, task := range args.Tasks {
		if strings.TrimSpace(task) != "" {
			tasks = append(tasks, task)
		}
	}
	if len(tasks) == 0 {
		return toolkit.Result{}, fmt.Errorf("tasks must contain at least one task")
	}
	if len(tasks) > MaxParallelChildren {
		return toolkit.Result{}, fmt.Errorf("tasks accepts at most %d entries, got %d", MaxParallelChildren, len(tasks))
	}

	outputs := make([]string, len(tasks))
	failures := make([]error, len(tasks))
	scope := toolkit.ScopeFrom(ctx)
	depth := DepthFrom(ctx) + 1
	// The batch is admitted up to MaxParallelChildren; execution is capped lower
	// so one fan-out cannot monopolise the provider.
	parallel := childConcurrency
	if len(tasks) < parallel {
		parallel = len(tasks)
	}
	semaphore := make(chan struct{}, parallel)
	var wait sync.WaitGroup
	for index, task := range tasks {
		wait.Add(1)
		go func(index int, task string) {
			defer wait.Done()
			// A cancelled run must not start children even when a slot is free:
			// select would pick the send half at random once both are ready.
			if err := ctx.Err(); err != nil {
				failures[index] = err
				return
			}
			select {
			case semaphore <- struct{}{}:
				defer func() { <-semaphore }()
			case <-ctx.Done():
				failures[index] = ctx.Err()
				return
			}
			outcome, err := c.spawner.Spawn(ctx, Request{Prompt: task, Scope: scope, Depth: depth})
			if err != nil {
				failures[index] = err
				return
			}
			outputs[index] = clipResult(outcome.Output)
		}(index, task)
	}
	wait.Wait()

	var builder strings.Builder
	fmt.Fprintf(&builder, "<subagent_results count=\"%d\">", len(tasks))
	for index := range tasks {
		if failures[index] != nil {
			fmt.Fprintf(&builder, "\n<result index=\"%d\" status=\"error\">%s</result>", index+1, failures[index].Error())
			continue
		}
		fmt.Fprintf(&builder, "\n<result index=\"%d\" status=\"ok\">%s</result>", index+1, outputs[index])
	}
	builder.WriteString("\n</subagent_results>")
	return toolkit.Result{Content: builder.String()}, nil
}

func (c *Capability) spawnOne(ctx context.Context, call toolkit.Call) (toolkit.Result, error) {
	if c.spawner == nil {
		return toolkit.Result{}, fmt.Errorf("subagents are not configured for this deployment")
	}
	if err := c.checkDepth(ctx); err != nil {
		return toolkit.Result{}, err
	}
	args, err := toolkit.DecodeArgs[spawnArgs](call.Arguments)
	if err != nil {
		return toolkit.Result{}, err
	}
	if strings.TrimSpace(args.Prompt) == "" {
		return toolkit.Result{}, fmt.Errorf("prompt is required")
	}
	outcome, err := c.spawner.Spawn(ctx, Request{
		AgentID: args.AgentID,
		Prompt:  args.Prompt,
		Scope:   toolkit.ScopeFrom(ctx),
		Depth:   DepthFrom(ctx) + 1,
	})
	if err != nil {
		return toolkit.Result{}, err
	}
	return toolkit.Result{
		Content: clipResult(outcome.Output),
		Meta:    map[string]any{"child_run_id": outcome.RunID, "depth": DepthFrom(ctx) + 1},
	}, nil
}

// checkDepth refuses a delegation that would exceed the limit. The parent's own
// registry is where delegation tools are removed; this is the second gate, for
// hosts that hand a child a registry that still contains them.
func (c *Capability) checkDepth(ctx context.Context) error {
	depth := DepthFrom(ctx)
	limit := c.maxDepth
	if limit <= 0 || limit > AbsoluteMaxDepth {
		limit = DefaultMaxDepth
	}
	if depth >= limit {
		return fmt.Errorf("delegation depth limit reached: depth %d, limit %d", depth, limit)
	}
	return nil
}

type depthKey struct{}

// WithDepth records the current delegation depth on the context.
func WithDepth(ctx context.Context, depth int) context.Context {
	if depth <= 0 {
		return ctx
	}
	return context.WithValue(ctx, depthKey{}, depth)
}

// DepthFrom reports the current delegation depth, zero at the root run.
func DepthFrom(ctx context.Context) int {
	depth, _ := ctx.Value(depthKey{}).(int)
	return depth
}

// clipResult bounds one child answer inside the parent's context, keeping the
// head and the tail so the conclusion survives.
func clipResult(output string) string {
	if len(output) <= maxChildResultBytes {
		return output
	}
	marker := fmt.Sprintf("\n[… child answer truncated, %d characters omitted …]\n", len(output)-maxChildResultBytes)
	keep := maxChildResultBytes - len(marker)
	if keep <= 0 {
		return headBytes(output, maxChildResultBytes)
	}
	head := keep * 4 / 10
	tail := keep - head
	return headBytes(output, head) + marker + tailBytes(output, tail)
}

// headBytes and tailBytes cut on rune boundaries so a UTF-8 answer never becomes
// invalid text on the way into the parent's context.
func headBytes(value string, limit int) string {
	if limit <= 0 {
		return ""
	}
	if len(value) <= limit {
		return value
	}
	cut := limit
	for cut > 0 && (value[cut]&0xC0) == 0x80 {
		cut--
	}
	return value[:cut]
}

func tailBytes(value string, limit int) string {
	if limit <= 0 {
		return ""
	}
	if len(value) <= limit {
		return value
	}
	start := len(value) - limit
	for start < len(value) && (value[start]&0xC0) == 0x80 {
		start++
	}
	return value[start:]
}
