// Package web exposes internet search and fetch. The capability is inert until a
// Searcher is supplied so a deployment without egress simply has no web tools
// answering.
package web

import (
	"context"
	"fmt"
	"strings"

	"ragflow/internal/harness/agent/model"
	"ragflow/internal/harness/caps"
)

// Hit is one search result.
type Hit struct {
	Title   string
	URL     string
	Snippet string
}

// Searcher is implemented by the platform's web integration.
type Searcher interface {
	Search(ctx context.Context, query string, limit int) ([]Hit, error)
	Fetch(ctx context.Context, url string) (string, error)
}

// Capability exposes web tools.
type Capability struct {
	searcher Searcher
}

// New builds the capability.
func New(searcher Searcher) *Capability {
	return &Capability{searcher: searcher}
}

// Name implements caps.Capability.
func (*Capability) Name() string { return "web" }

// Tools implements caps.Capability.
func (*Capability) Tools() []model.ToolSpec {
	return []model.ToolSpec{
		{
			Name:        "web_search",
			Description: "Search the public web.",
			Parameters: caps.ObjectSchema(map[string]any{
				"query": map[string]any{"type": "string"},
				"limit": map[string]any{"type": "integer", "description": "Default 5, max 20."},
			}, "query"),
		},
		{
			Name:        "web_fetch",
			Description: "Fetch a URL and return its text.",
			Parameters:  caps.ObjectSchema(map[string]any{"url": map[string]any{"type": "string"}}, "url"),
		},
	}
}

type searchArgs struct {
	Query string `json:"query"`
	Limit int    `json:"limit"`
}

type fetchArgs struct {
	URL string `json:"url"`
}

// Invoke implements caps.Capability.
func (c *Capability) Invoke(ctx context.Context, call caps.Call) (caps.Result, error) {
	if c.searcher == nil {
		return caps.Result{}, fmt.Errorf("web access is not configured for this deployment")
	}
	switch call.Tool {
	case "web_search":
		args, err := caps.DecodeArgs[searchArgs](call.Arguments)
		if err != nil {
			return caps.Result{}, err
		}
		limit := args.Limit
		if limit <= 0 {
			limit = 5
		}
		if limit > 20 {
			limit = 20
		}
		hits, err := c.searcher.Search(ctx, args.Query, limit)
		if err != nil {
			return caps.Result{}, err
		}
		return renderHits(hits), nil
	case "web_fetch":
		args, err := caps.DecodeArgs[fetchArgs](call.Arguments)
		if err != nil {
			return caps.Result{}, err
		}
		if strings.TrimSpace(args.URL) == "" {
			return caps.Result{}, fmt.Errorf("url is required")
		}
		text, err := c.searcher.Fetch(ctx, args.URL)
		if err != nil {
			return caps.Result{}, err
		}
		return caps.Result{
			Content:  text,
			Evidence: []caps.Evidence{{Source: "web", Ref: args.URL}},
		}, nil
	default:
		return caps.Result{}, fmt.Errorf("web: unknown tool %q", call.Tool)
	}
}

func renderHits(hits []Hit) caps.Result {
	if len(hits) == 0 {
		return caps.Result{Content: "no results"}
	}
	var builder strings.Builder
	evidence := make([]caps.Evidence, 0, len(hits))
	for index, hit := range hits {
		builder.WriteString(fmt.Sprintf("[%d] %s\n%s\n%s\n\n", index+1, hit.Title, hit.URL, hit.Snippet))
		evidence = append(evidence, caps.Evidence{Source: "web", Ref: hit.URL, Snippet: hit.Title})
	}
	return caps.Result{Content: strings.TrimSpace(builder.String()), Evidence: evidence}
}
