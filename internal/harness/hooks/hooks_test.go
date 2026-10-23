package hooks

import (
	"context"
	"testing"
)

func TestNilRegistryIsNoOp(t *testing.T) {
	var registry *Registry
	if registry.Count() != 0 {
		t.Fatal("nil registry must report zero hooks")
	}
	if m := registry.PreToolUse(context.Background(), ToolContext{Tool: "exec_shell"}); m != nil {
		t.Fatal("nil registry must not mutate")
	}
	registry.Notify(context.Background(), SessionStart, RunContext{})
}

func TestMatchNarrowsByToolAndEvent(t *testing.T) {
	registry := NewRegistry()
	registry.AddFunc(Match{Events: []Event{PreToolUse}, Tools: []string{"exec_shell"}},
		func(context.Context, Event, any) *Mutation { return &Mutation{Veto: true} })

	if m := registry.PreToolUse(context.Background(), ToolContext{Tool: "fs_cat"}); m != nil {
		t.Fatal("a tool outside the match must not be vetoed")
	}
	if m := registry.PreToolUse(context.Background(), ToolContext{Tool: "EXEC_SHELL"}); m == nil || !m.Veto {
		t.Fatal("tool matching must be case-insensitive")
	}
	if m := registry.PostToolUse(context.Background(), ToolResultContext{Tool: "exec_shell"}); m != nil {
		t.Fatal("an event outside the match must not fire")
	}
}

func TestPreToolUseRewritesArguments(t *testing.T) {
	registry := NewRegistry()
	registry.AddFunc(Match{Events: []Event{PreToolUse}},
		func(_ context.Context, event Event, payload any) *Mutation {
			call := payload.(ToolContext)
			if call.Tool == "exec_shell" {
				return &Mutation{Arguments: `{"command":"true"}`}
			}
			return nil
		})

	mutation := registry.PreToolUse(context.Background(), ToolContext{Tool: "exec_shell", Arguments: `{"command":"rm -rf /"}`})
	if mutation == nil || mutation.Arguments != `{"command":"true"}` {
		t.Fatalf("want rewritten arguments, got %+v", mutation)
	}
}

func TestPostToolUseReplacesContent(t *testing.T) {
	registry := NewRegistry()
	registry.AddFunc(Match{Events: []Event{PostToolUse}},
		func(context.Context, Event, any) *Mutation {
			return &Mutation{Content: "[redacted]"}
		})

	mutation := registry.PostToolUse(context.Background(), ToolResultContext{Tool: "fs_cat", Content: "secret"})
	if mutation == nil || mutation.Content != "[redacted]" {
		t.Fatalf("want replaced content, got %+v", mutation)
	}
}

func TestNotifyRejectsUnknownEvent(t *testing.T) {
	registry := NewRegistry()
	seen := ""
	registry.AddFunc(Match{Events: AllEvents()},
		func(_ context.Context, event Event, _ any) *Mutation { seen += string(event); return nil })
	registry.Notify(context.Background(), PreToolUse, RunContext{})
	registry.Notify(context.Background(), UserPromptSubmit, RunContext{Prompt: "hello"})
	if seen != string(UserPromptSubmit) {
		t.Fatalf("Notify must deliver observation events only, saw %q", seen)
	}
}
