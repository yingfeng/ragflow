// Package rag exposes RAGFlow's retrieval surface as agent tools.
//
// The tools are the agentic pipeline's own tool implementations, instantiated
// here and invoked directly: names, argument bounds, XML rendering, snippet
// windows and compiled-knowledge gating are therefore identical by
// construction rather than by convention. Nothing is re-implemented and dataset
// files are never exposed as a filesystem.
package rag

import (
	"context"
	"encoding/json"
	"fmt"
	"html"
	"regexp"
	"sort"
	"strconv"
	"strings"

	einotool "github.com/cloudwego/eino/components/tool"

	"ragflow/internal/agentic_rag"
	"ragflow/internal/harness/agent/model"
	"ragflow/internal/harness/toolkit"
)

// toolFactory builds one tool bound to a tenant and dataset scope. The tools
// are stateless, so a per-call rebuild is cheap.
type toolFactory func(tenantID string, datasetIDs []string) einotool.InvokableTool

// retrievalTools are always offered.
var retrievalTools = map[string]toolFactory{
	"search_chunks": func(tenantID string, datasetIDs []string) einotool.InvokableTool {
		return agentic_rag.NewSearchChunksTool(tenantID, datasetIDs)
	},
	"search_bm25_chunks": func(tenantID string, datasetIDs []string) einotool.InvokableTool {
		return agentic_rag.NewSearchBm25ChunksTool(tenantID, datasetIDs)
	},
	"search_semantic_chunks": func(tenantID string, datasetIDs []string) einotool.InvokableTool {
		return agentic_rag.NewSearchSemanticChunksTool(tenantID, datasetIDs)
	},
	"search_metadata": func(tenantID string, datasetIDs []string) einotool.InvokableTool {
		return agentic_rag.NewMetadataSearchTool(tenantID, datasetIDs)
	},
	"grep_chunks": func(tenantID string, datasetIDs []string) einotool.InvokableTool {
		return agentic_rag.NewGrepChunksTool(tenantID, datasetIDs)
	},
	"list_chunks": func(tenantID string, datasetIDs []string) einotool.InvokableTool {
		return agentic_rag.NewListChunksTool(tenantID, datasetIDs)
	},
}

// compiledTools require compiled artifacts on the bound datasets, which is the
// same enablement rule the agentic pipeline applies.
var compiledTools = map[string]toolFactory{
	"navigate_tree": func(tenantID string, datasetIDs []string) einotool.InvokableTool {
		return agentic_rag.NewNavigateTreeTool(tenantID, datasetIDs)
	},
	"navigate_structure": func(tenantID string, datasetIDs []string) einotool.InvokableTool {
		return agentic_rag.NewNavigateStructureTool(tenantID, datasetIDs)
	},
	"graph_explore": func(tenantID string, datasetIDs []string) einotool.InvokableTool {
		return agentic_rag.NewGraphExploreTool(tenantID, datasetIDs)
	},
	"graph_doc_outline": func(tenantID string, datasetIDs []string) einotool.InvokableTool {
		return agentic_rag.NewGraphDocOutlineTool(tenantID, datasetIDs)
	},
}

// Config binds the capability to one run's data reach.
type Config struct {
	TenantID   string
	DatasetIDs []string
	// WebSearch enables the web_search tool when non-nil.
	WebSearch agentic_rag.WebSearchFunc
}

// Capability exposes the retrieval tools.
type Capability struct {
	tenantID   string
	datasetIDs []string
	webSearch  agentic_rag.WebSearchFunc
	factories  map[string]toolFactory
	specs      []model.ToolSpec
}

// New describes the tools available for the bound datasets.
func New(ctx context.Context, cfg Config) (*Capability, error) {
	factories := make(map[string]toolFactory, len(retrievalTools)+len(compiledTools)+1)
	for name, factory := range retrievalTools {
		factories[name] = factory
	}
	if cfg.WebSearch != nil {
		search := cfg.WebSearch
		factories["web_search"] = func(string, []string) einotool.InvokableTool {
			return agentic_rag.NewWebSearchTool(search)
		}
	}
	if agentic_rag.DatasetsHaveCompiledKnowledge(ctx, cfg.TenantID, cfg.DatasetIDs) {
		for name, factory := range compiledTools {
			factories[name] = factory
		}
	}

	specs, err := describe(ctx, factories, cfg.TenantID, cfg.DatasetIDs)
	if err != nil {
		return nil, fmt.Errorf("caps/rag: %w", err)
	}
	return &Capability{
		tenantID:   cfg.TenantID,
		datasetIDs: cfg.DatasetIDs,
		webSearch:  cfg.WebSearch,
		factories:  factories,
		specs:      specs,
	}, nil
}

// Name implements toolkit.Capability.
func (*Capability) Name() string { return "rag" }

// Tools implements toolkit.Capability.
func (c *Capability) Tools() []model.ToolSpec {
	out := make([]model.ToolSpec, len(c.specs))
	copy(out, c.specs)
	return out
}

// RequiredTools implements toolkit.RequiredToolList. A run whose only job is
// corpus evidence must gather some before it may plan or write: without the
// gate, its first round can go to think/todo_write and produce nothing the
// answer can stand on. The agentic pipeline drives the same gate through its
// retrieval model (agentic_rag/retrieval_model.go), with the same shape: any
// one locate tool lifts it.
func (c *Capability) RequiredTools() []string {
	required := make([]string, 0, len(retrievalTools))
	for name := range retrievalTools {
		if _, ok := c.factories[name]; !ok {
			continue
		}
		required = append(required, name)
	}
	sort.Strings(required)
	return required
}

// Invoke implements toolkit.Capability. The run scope narrows the dataset scope
// the tools were built with; the tools themselves still reject dataset_ids that
// fall outside it.
func (c *Capability) Invoke(ctx context.Context, call toolkit.Call) (toolkit.Result, error) {
	factory, ok := c.factories[call.Tool]
	if !ok {
		return toolkit.Result{}, fmt.Errorf("rag: unknown tool %q", call.Tool)
	}

	datasetIDs := c.datasetIDs
	if scope := toolkit.ScopeFrom(ctx); len(scope.Datasets) > 0 {
		datasetIDs = scope.Datasets
	}

	content, err := factory(c.tenantID, datasetIDs).InvokableRun(ctx, normalizedArgs(call.Arguments))
	if err != nil {
		return toolkit.Result{}, fmt.Errorf("%s: %w", call.Tool, err)
	}
	// Inline is what makes a retrieval result readable at all. Every tool behind
	// this capability already returns a bounded window (grep caps at 30 matches
	// with a snippet per match, list_chunks at one window of chunks), so spilling
	// it again would replace that window with a handle to itself: the model would
	// see a preview of the preview and have to spend a read_spill round to get
	// back to the text it already asked for. The agentic pipeline, which these
	// tools come from, hands the model the window verbatim.
	return toolkit.Result{
		Content:  content,
		Evidence: ParseEvidence(content),
		Inline:   true,
	}, nil
}

// describe renders the tool specs straight from each tool's own declaration.
func describe(ctx context.Context, factories map[string]toolFactory, tenantID string, datasetIDs []string) ([]model.ToolSpec, error) {
	names := make([]string, 0, len(factories))
	for name := range factories {
		names = append(names, name)
	}
	sort.Strings(names)

	specs := make([]model.ToolSpec, 0, len(names))
	for _, name := range names {
		info, err := factories[name](tenantID, datasetIDs).Info(ctx)
		if err != nil {
			return nil, fmt.Errorf("describe %s: %w", name, err)
		}
		spec := model.ToolSpec{Name: name, Description: info.Desc}
		if info.ParamsOneOf != nil {
			// ParamsOneOf keeps its fields unexported, so marshalling it directly
			// yields "{}" and the model would receive a tool with no arguments
			// declared at all. ToJSONSchema is the only correct conversion.
			declared, err := info.ParamsOneOf.ToJSONSchema()
			if err != nil {
				return nil, fmt.Errorf("schema %s: %w", name, err)
			}
			if declared != nil {
				raw, err := json.Marshal(declared)
				if err != nil {
					return nil, fmt.Errorf("schema %s: %w", name, err)
				}
				params := map[string]any{}
				if err := json.Unmarshal(raw, &params); err != nil {
					return nil, fmt.Errorf("decode schema %s: %w", name, err)
				}
				spec.Parameters = params
			}
		}
		specs = append(specs, spec)
	}
	return specs, nil
}

func normalizedArgs(arguments string) string {
	if strings.TrimSpace(arguments) == "" {
		return "{}"
	}
	return arguments
}

// chunkPattern matches one rendered chunk: the attributes plus the element body.
// The body is what carries the text, and the text is what makes an evidence
// entry usable later — the body is empty for a self-closing element.
var chunkPattern = regexp.MustCompile(`(?s)<chunk\s+([^>]*?)(?:/>|>(.*?)</chunk>)`)

// attrPattern extracts one attribute from a tag body.
var attrPattern = regexp.MustCompile(`([a-z_]+)="([^"]*)"`)

// matchSnippetPattern matches the matched-window element grep renders.
var matchSnippetPattern = regexp.MustCompile(`(?s)<match_snippet>(.*?)</match_snippet>`)

// tagPattern strips the remaining markup from a chunk body.
var tagPattern = regexp.MustCompile(`(?s)<[^>]+>`)

// evidenceExcerptRunes bounds one evidence line. It is short on purpose: the
// pool is re-sent every round, so each entry has to identify its passage without
// paying for the passage again.
const evidenceExcerptRunes = 120

// ParseEvidence extracts one citation-bearing evidence per rendered chunk.
//
// The chunk id is the ref because it is stable across runs and is what citation
// markers resolve back to; the owning document id is the source.
//
// The snippet carries the passage's opening text (or grep's matched window), not
// just the document name. The pool built from these entries is re-sent as the
// run's memory after the transcript has been compacted, so a bare document name
// would leave the model knowing only that it once read "三国演义" 40 times.
func ParseEvidence(content string) []toolkit.Evidence {
	if !strings.Contains(content, "<chunk") {
		return nil
	}
	matches := chunkPattern.FindAllStringSubmatch(content, -1)
	out := make([]toolkit.Evidence, 0, len(matches))
	seen := make(map[string]struct{}, len(matches))
	for _, match := range matches {
		attrs := make(map[string]string, 8)
		for _, attr := range attrPattern.FindAllStringSubmatch(match[1], -1) {
			attrs[attr[1]] = attr[2]
		}
		chunkID := attrs["chunk_id"]
		if chunkID == "" {
			continue
		}
		if _, dup := seen[chunkID]; dup {
			continue
		}
		seen[chunkID] = struct{}{}

		evidence := toolkit.Evidence{Ref: chunkID, Source: attrs["doc_id"]}
		if evidence.Source == "" {
			evidence.Source = chunkID
		}
		evidence.Snippet = evidenceExcerpt(attrs["doc_name"], match[2])
		if score, err := strconv.ParseFloat(attrs["score"], 64); err == nil {
			evidence.Score = score
		}
		out = append(out, evidence)
	}
	return out
}

// evidenceExcerpt condenses a chunk body into the one line kept in the pool.
func evidenceExcerpt(docName, body string) string {
	text := body
	if matched := matchSnippetPattern.FindStringSubmatch(body); matched != nil {
		text = matched[1]
	} else {
		text = tagPattern.ReplaceAllString(body, " ")
	}
	text = strings.Join(strings.Fields(html.UnescapeString(text)), " ")
	if runes := []rune(text); len(runes) > evidenceExcerptRunes {
		text = string(runes[:evidenceExcerptRunes]) + "…"
	}
	switch {
	case text == "":
		return docName
	case docName == "":
		return text
	default:
		return docName + " · " + text
	}
}
