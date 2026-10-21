// Package fs wraps RAGFlow's ContextEngine Filesystem (internal/cli/filesystem)
// as an agent capability.
//
// Read semantics come from the existing providers (ls / search / cat). Writes are
// additive and deliberately narrow: only memories/ and skills/ may be written,
// and only through a Writable backend. Dataset content is never a filesystem: it
// is reached through the rag capability.
package fs

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"

	"ragflow/internal/cli/filesystem"
	"ragflow/internal/harness/agent/model"
	"ragflow/internal/harness/caps"
)

// Backend is the read surface of the ContextEngine Filesystem.
type Backend interface {
	List(ctx context.Context, path string, opts *filesystem.ListOptions) (*filesystem.Result, error)
	Search(ctx context.Context, path string, opts *filesystem.SearchOptions) (*filesystem.Result, error)
	Cat(ctx context.Context, path string) ([]byte, error)
}

// Writable is the write surface. It is optional: without it, write/edit tools
// return a clear error instead of silently dropping the change.
type Writable interface {
	// Read returns content and its revision.
	Read(ctx context.Context, path string) ([]byte, string, error)
	// Write stores content. An empty baseRevision only allows creation; a
	// non-empty one is a compare-and-swap and returns ErrConflict on mismatch.
	Write(ctx context.Context, path string, content []byte, baseRevision string) (string, error)
}

// ErrConflict reports a stale base revision.
var ErrConflict = fmt.Errorf("fs: revision conflict")

// ErrWriteDenied reports a write outside memories/ or skills/.
var ErrWriteDenied = fmt.Errorf("fs: writes are limited to memories/ and skills/")

// EngineBackend adapts a filesystem.Engine.
type EngineBackend struct {
	Engine *filesystem.Engine
}

// FromEngine returns a Backend over the ContextEngine Filesystem.
func FromEngine(engine *filesystem.Engine) Backend {
	return &EngineBackend{Engine: engine}
}

// List implements Backend.
func (b *EngineBackend) List(ctx context.Context, path string, opts *filesystem.ListOptions) (*filesystem.Result, error) {
	return b.Engine.List(ctx, path, opts)
}

// Search implements Backend.
func (b *EngineBackend) Search(ctx context.Context, path string, opts *filesystem.SearchOptions) (*filesystem.Result, error) {
	return b.Engine.Search(ctx, path, opts)
}

// Cat implements Backend.
func (b *EngineBackend) Cat(ctx context.Context, path string) ([]byte, error) {
	return b.Engine.Cat(ctx, path)
}

// Capability exposes the fs tools.
type Capability struct {
	backend  Backend
	writable Writable
}

// New builds the capability.
func New(backend Backend, writable Writable) *Capability {
	return &Capability{backend: backend, writable: writable}
}

// Name implements caps.Capability.
func (*Capability) Name() string { return "fs" }

// Tools implements caps.Capability.
func (*Capability) Tools() []model.ToolSpec {
	path := map[string]any{"type": "string", "description": "Virtual path, e.g. memories/<id>/users/profile.md"}
	return []model.ToolSpec{
		{
			Name:        "fs_ls",
			Description: "List a ContextEngine path (datasets / tools / skills / memories).",
			Parameters: caps.ObjectSchema(map[string]any{
				"path":      path,
				"recursive": map[string]any{"type": "boolean"},
				"limit":     map[string]any{"type": "integer"},
			}, "path"),
		},
		{
			Name:        "fs_cat",
			Description: "Read one file by exact path. Use rag_search to locate dataset content; cat is not a search entry.",
			Parameters:  caps.ObjectSchema(map[string]any{"path": path}, "path"),
		},
		{
			Name:        "fs_search",
			Description: "Search within a ContextEngine subtree.",
			Parameters: caps.ObjectSchema(map[string]any{
				"path":      path,
				"query":     map[string]any{"type": "string"},
				"top_k":     map[string]any{"type": "integer"},
				"recursive": map[string]any{"type": "boolean"},
			}, "path", "query"),
		},
		{
			Name:        "fs_write",
			Description: "Create or overwrite a file under memories/ or skills/. Pass base_revision to overwrite an existing file.",
			Parameters: caps.ObjectSchema(map[string]any{
				"path":          path,
				"content":       map[string]any{"type": "string"},
				"base_revision": map[string]any{"type": "string"},
			}, "path", "content"),
		},
		{
			Name:        "fs_edit",
			Description: "Replace one unique occurrence in a file under memories/ or skills/. Fails when old_string is missing or ambiguous.",
			Parameters: caps.ObjectSchema(map[string]any{
				"path":          path,
				"old_string":    map[string]any{"type": "string"},
				"new_string":    map[string]any{"type": "string"},
				"base_revision": map[string]any{"type": "string"},
			}, "path", "old_string", "new_string"),
		},
	}
}

type lsArgs struct {
	Path      string `json:"path"`
	Recursive bool   `json:"recursive"`
	Limit     int    `json:"limit"`
}

type catArgs struct {
	Path string `json:"path"`
}

type searchArgs struct {
	Path      string `json:"path"`
	Query     string `json:"query"`
	TopK      int    `json:"top_k"`
	Recursive bool   `json:"recursive"`
}

type writeArgs struct {
	Path         string `json:"path"`
	Content      string `json:"content"`
	BaseRevision string `json:"base_revision"`
}

type editArgs struct {
	Path         string `json:"path"`
	OldString    string `json:"old_string"`
	NewString    string `json:"new_string"`
	BaseRevision string `json:"base_revision"`
}

// Invoke implements caps.Capability.
func (c *Capability) Invoke(ctx context.Context, call caps.Call) (caps.Result, error) {
	switch call.Tool {
	case "fs_ls":
		return c.list(ctx, call.Arguments)
	case "fs_cat":
		return c.cat(ctx, call.Arguments)
	case "fs_search":
		return c.search(ctx, call.Arguments)
	case "fs_write":
		return c.write(ctx, call.Arguments)
	case "fs_edit":
		return c.edit(ctx, call.Arguments)
	default:
		return caps.Result{}, fmt.Errorf("fs: unknown tool %q", call.Tool)
	}
}

func (c *Capability) list(ctx context.Context, raw string) (caps.Result, error) {
	args, err := caps.DecodeArgs[lsArgs](raw)
	if err != nil {
		return caps.Result{}, err
	}
	if c.backend == nil {
		return caps.Result{}, fmt.Errorf("context filesystem is not configured")
	}
	path, err := normalizePath(args.Path, false)
	if err != nil {
		return caps.Result{}, err
	}
	result, err := c.backend.List(ctx, path, &filesystem.ListOptions{Recursive: args.Recursive, Limit: args.Limit})
	if err != nil {
		return caps.Result{}, err
	}
	return renderNodes(result), nil
}

func (c *Capability) cat(ctx context.Context, raw string) (caps.Result, error) {
	args, err := caps.DecodeArgs[catArgs](raw)
	if err != nil {
		return caps.Result{}, err
	}
	if c.backend == nil {
		return caps.Result{}, fmt.Errorf("context filesystem is not configured")
	}
	path, err := normalizePath(args.Path, false)
	if err != nil {
		return caps.Result{}, err
	}
	content, err := c.backend.Cat(ctx, path)
	if err != nil {
		return caps.Result{}, err
	}
	return caps.Result{Content: string(content)}, nil
}

func (c *Capability) search(ctx context.Context, raw string) (caps.Result, error) {
	args, err := caps.DecodeArgs[searchArgs](raw)
	if err != nil {
		return caps.Result{}, err
	}
	if c.backend == nil {
		return caps.Result{}, fmt.Errorf("context filesystem is not configured")
	}
	path, err := normalizePath(args.Path, false)
	if err != nil {
		return caps.Result{}, err
	}
	topK := args.TopK
	if topK <= 0 {
		topK = 10
	}
	result, err := c.backend.Search(ctx, path, &filesystem.SearchOptions{
		Query:     args.Query,
		TopK:      topK,
		Recursive: args.Recursive,
	})
	if err != nil {
		return caps.Result{}, err
	}
	return renderNodes(result), nil
}

func (c *Capability) write(ctx context.Context, raw string) (caps.Result, error) {
	args, err := caps.DecodeArgs[writeArgs](raw)
	if err != nil {
		return caps.Result{}, err
	}
	if c.writable == nil {
		return caps.Result{}, fmt.Errorf("context filesystem is read-only in this deployment")
	}
	path, err := normalizePath(args.Path, true)
	if err != nil {
		return caps.Result{}, err
	}
	revision, err := c.writable.Write(ctx, path, []byte(args.Content), args.BaseRevision)
	if err != nil {
		return caps.Result{}, err
	}
	return caps.Result{
		Content: fmt.Sprintf("wrote %s (revision=%s)", path, revision),
		Meta:    map[string]any{"revision": revision, "path": path},
	}, nil
}

func (c *Capability) edit(ctx context.Context, raw string) (caps.Result, error) {
	args, err := caps.DecodeArgs[editArgs](raw)
	if err != nil {
		return caps.Result{}, err
	}
	if c.writable == nil {
		return caps.Result{}, fmt.Errorf("context filesystem is read-only in this deployment")
	}
	if args.OldString == "" {
		return caps.Result{}, fmt.Errorf("old_string is required")
	}
	path, err := normalizePath(args.Path, true)
	if err != nil {
		return caps.Result{}, err
	}
	content, revision, err := c.writable.Read(ctx, path)
	if err != nil {
		return caps.Result{}, err
	}
	if args.BaseRevision != "" && args.BaseRevision != revision {
		return caps.Result{}, fmt.Errorf("%w: have %s, want %s", ErrConflict, revision, args.BaseRevision)
	}
	text := string(content)
	count := strings.Count(text, args.OldString)
	if count == 0 {
		return caps.Result{}, fmt.Errorf("old_string not found in %s", path)
	}
	if count > 1 {
		return caps.Result{}, fmt.Errorf("old_string appears %d times in %s; include more context", count, path)
	}
	updated := strings.Replace(text, args.OldString, args.NewString, 1)
	newRevision, err := c.writable.Write(ctx, path, []byte(updated), revision)
	if err != nil {
		return caps.Result{}, err
	}
	return caps.Result{
		Content: fmt.Sprintf("edited %s (revision=%s)", path, newRevision),
		Meta:    map[string]any{"revision": newRevision, "path": path},
	}, nil
}

// normalizePath rejects absolute paths, traversal and empty segments. When
// writing, it also enforces the memories/ or skills/ prefix.
func normalizePath(path string, writing bool) (string, error) {
	trimmed := strings.TrimSpace(path)
	if trimmed == "" {
		return "", fmt.Errorf("path is required")
	}
	if strings.HasPrefix(trimmed, "/") {
		return "", fmt.Errorf("absolute paths are not allowed: %s", path)
	}
	segments := strings.Split(trimmed, "/")
	cleaned := make([]string, 0, len(segments))
	for _, segment := range segments {
		switch segment {
		case "", ".":
			continue
		case "..":
			return "", fmt.Errorf("path traversal is not allowed: %s", path)
		default:
			cleaned = append(cleaned, segment)
		}
	}
	if len(cleaned) == 0 {
		return "", fmt.Errorf("path is required")
	}
	joined := strings.Join(cleaned, "/")
	if writing && !isWritablePath(joined) {
		return "", fmt.Errorf("%w: %s", ErrWriteDenied, joined)
	}
	return joined, nil
}

func isWritablePath(path string) bool {
	for _, segment := range strings.Split(path, "/") {
		if segment == "memories" || segment == "skills" {
			return true
		}
	}
	return false
}

func renderNodes(result *filesystem.Result) caps.Result {
	if result == nil || len(result.Nodes) == 0 {
		return caps.Result{Content: "no entries"}
	}
	var builder strings.Builder
	for _, node := range result.Nodes {
		if node == nil {
			continue
		}
		builder.WriteString(fmt.Sprintf("%s\t%s", node.Path, node.Type))
		if node.Size > 0 {
			builder.WriteString(fmt.Sprintf("\t%d bytes", node.Size))
		}
		builder.WriteString("\n")
	}
	return caps.Result{Content: strings.TrimSpace(builder.String())}
}

// MemoryWritable is an in-memory Writable used by tests and single-process runs.
type MemoryWritable struct {
	files map[string][]byte
}

// NewMemoryWritable builds an empty in-memory writable.
func NewMemoryWritable() *MemoryWritable {
	return &MemoryWritable{files: make(map[string][]byte)}
}

// Read implements Writable.
func (m *MemoryWritable) Read(_ context.Context, path string) ([]byte, string, error) {
	content, ok := m.files[path]
	if !ok {
		return nil, "", fmt.Errorf("fs: %s not found", path)
	}
	return content, revisionOf(content), nil
}

// Write implements Writable.
func (m *MemoryWritable) Write(_ context.Context, path string, content []byte, baseRevision string) (string, error) {
	if existing, ok := m.files[path]; ok {
		if baseRevision == "" {
			return "", fmt.Errorf("%w: %s already exists", ErrConflict, path)
		}
		if baseRevision != revisionOf(existing) {
			return "", fmt.Errorf("%w: have %s, want %s", ErrConflict, revisionOf(existing), baseRevision)
		}
	} else if baseRevision != "" {
		return "", fmt.Errorf("%w: %s does not exist", ErrConflict, path)
	}
	stored := make([]byte, len(content))
	copy(stored, content)
	m.files[path] = stored
	return revisionOf(stored), nil
}

func revisionOf(content []byte) string {
	sum := sha256.Sum256(content)
	return hex.EncodeToString(sum[:8])
}
