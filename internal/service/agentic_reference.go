//
//  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//

package service

import (
	"context"
	"fmt"
	"regexp"
	"strings"

	"go.uber.org/zap"

	"ragflow/internal/agentic_rag"
	"ragflow/internal/common"
	"ragflow/internal/engine"
	enginetypes "ragflow/internal/engine/types"
)

// chunkIDHandleRE matches a citation marker whose body is a chunk id instead of
// a handle number ("[ID:2f4f73f8e245db19]").
//
// Every passage a tool prints carries `chunk_id="..."` next to the `ref="n"`
// handle, and a model that copies the wrong attribute writes the chunk id inside
// the marker the prompt asks for. Nothing downstream could read that: the handle
// scan needs digits, and the provenance scan needs the `chunk_id:` field. The
// marker therefore reached the UI as literal text and the reference payload
// shipped empty, so the citation was unopenable.
var chunkIDHandleRE = regexp.MustCompile(`\[ID:\s*([A-Za-z0-9_-]{6,})\]`)

// chunkIDHandles returns the chunk ids the answer wrote inside citation markers,
// in first-appearance order, deduplicated. A pure-digit body is a handle number,
// not a chunk id, and is left to CitedIDsFromMarkers.
func chunkIDHandles(answer string) []string {
	if answer == "" || !strings.Contains(answer, "[ID:") {
		return nil
	}
	seen := make(map[string]struct{})
	var out []string
	for _, m := range chunkIDHandleRE.FindAllStringSubmatch(answer, -1) {
		id := m[1]
		if isAllDigits(id) {
			continue
		}
		if _, dup := seen[id]; dup {
			continue
		}
		seen[id] = struct{}{}
		out = append(out, id)
	}
	return out
}

// rewriteChunkIDHandles replaces every chunk-id marker with the marker numbering
// the reference payload ships, and drops a marker naming a chunk that did not
// resolve — the same rule ResolveCitations applies to unresolvable handles, since
// a marker must never point at an index that holds some other passage.
func rewriteChunkIDHandles(answer string, publishable []string) string {
	if answer == "" || len(publishable) == 0 || !strings.Contains(answer, "[ID:") {
		return answer
	}
	pos := make(map[string]int, len(publishable))
	for i, id := range publishable {
		if _, dup := pos[id]; !dup {
			pos[id] = i
		}
	}
	return chunkIDHandleRE.ReplaceAllStringFunc(answer, func(m string) string {
		sub := chunkIDHandleRE.FindStringSubmatch(m)
		if len(sub) < 2 || isAllDigits(sub[1]) {
			return m
		}
		n, ok := pos[sub[1]]
		if !ok {
			return ""
		}
		return fmt.Sprintf("[ID:%d]", n)
	})
}

func isAllDigits(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}

// mergeCitedIDs appends the ids of extra that cited does not hold yet, keeping
// first-appearance order: the payload's numbering is that order.
func mergeCitedIDs(cited, extra []string) []string {
	if len(extra) == 0 {
		return cited
	}
	seen := make(map[string]struct{}, len(cited)+len(extra))
	for _, id := range cited {
		seen[id] = struct{}{}
	}
	for _, id := range extra {
		if _, dup := seen[id]; dup {
			continue
		}
		seen[id] = struct{}{}
		cited = append(cited, id)
	}
	return cited
}

// buildAgenticReference turns the answer's citations into the reference payload
// the naive pipeline ships, and rewrites the answer's markers to that payload's
// numbering. The answer cites the run's evidence handles ([ID:n], the numbering
// the model was shown); an answer that named passages by raw chunk_id instead
// (a document-level question has no handle to write) falls back to that path.
// Web passages resolve from the run's registry; only corpus ids reach the index.
// With no resolvable citations both come back unchanged: an empty reference
// keeps the SSE shape the UI expects, and the untouched text is the honest state
// of an answer whose sources could not be loaded.
func (s *ChatPipelineService) buildAgenticReference(ctx context.Context, tenantID string, datasetIDs []string, final string, registry *agentic_rag.EvidenceRegistry) (map[string]interface{}, string) {
	// Normalize the answer's markers BEFORE anything reads them. A reasoning
	// model parks the whole answer inside <think>, so the normalization covers
	// the entire text rather than a post-`</think>` tail: a handle sitting in
	// the thinking stretch is cited exactly like one in the visible prose.
	//
	// Two shapes need repair, and both are invisible to every scanner
	// downstream — this package's `\[ID:(\d+)\]` and the client's marker regex
	// each stop at the first non-digit:
	//   - a near-canonical marker ("(ID:47)", "[ID: 47]", "【ID:47】"), and
	//   - several handles compressed into one marker ("[ID:47, ID:115]").
	normalized := agentic_rag.ExpandCommaCitations(
		RepairBadCitationFormats(final), len(registry.IDs()))
	// A model may cite a web result with a Markdown link instead of its
	// handle. Only links to sources actually shown during this run qualify.
	linkedURLs := make(map[string]bool)
	for n, id := range registry.IDs() {
		web, ok := registry.WebResult(id)
		if !ok || linkedURLs[web.URL] {
			continue
		}
		linkedURLs[web.URL] = true
		marker := fmt.Sprintf("[ID:%d]", n)
		if !strings.Contains(normalized, marker) {
			linkEnd := "](" + web.URL + ")"
			normalized = strings.ReplaceAll(normalized, linkEnd, linkEnd+" "+marker)
		}
	}

	// Preferred: the answer's own [ID:n] handles, resolved against the run's
	// evidence registry — the same numbering the model was shown, so a marker
	// can only name a passage that actually reached the context.
	cited := agentic_rag.CitedIDsFromMarkers(normalized, registry)
	byHandles := len(cited) > 0
	// Two more ways an answer names its sources: a raw chunk id inside the marker
	// ([ID:2f4f73f8e245db19], the chunk_id printed next to every passage) and the
	// `chunk_id: <id>` provenance field a document-level answer uses instead of a
	// handle. Both are merged in first-appearance order, which is the order the
	// reference payload ships in.
	handleChunkIDs := chunkIDHandles(normalized)
	cited = mergeCitedIDs(cited, handleChunkIDs)
	cited = mergeCitedIDs(cited, agentic_rag.ExtractCitedChunkIDs(normalized))
	if len(cited) == 0 {
		return map[string]interface{}{}, normalized
	}
	// Keep only ids that actually resolved, preserving first-appearance
	// order: the marker number IS the chunk's position in the payload array.
	byID := make(map[string]map[string]interface{}, len(cited))
	var corpusIDs []string
	for _, id := range cited {
		if web, ok := registry.WebResult(id); ok {
			title := web.Title
			if title == "" {
				title = web.URL
			}
			byID[id] = map[string]interface{}{
				"id": id, "doc_id": web.URL, "docnm_kwd": title,
				"content_with_weight": web.Content, "url": web.URL,
				"kb_id": []interface{}{}, "img_id": "", "positions": []interface{}{},
			}
		} else {
			corpusIDs = append(corpusIDs, id)
		}
	}
	rows := fetchChunksByIDs(ctx, tenantID, datasetIDs, corpusIDs)
	for _, r := range rows {
		if id, ok := r["id"].(string); ok {
			byID[id] = r
		}
	}
	resolved := make([]string, 0, len(cited))
	for _, id := range cited {
		if _, ok := byID[id]; ok {
			resolved = append(resolved, id)
		}
	}
	if len(resolved) == 0 {
		return map[string]interface{}{}, normalized
	}
	ordered := make([]map[string]interface{}, 0, len(resolved))
	for _, id := range resolved {
		ordered = append(ordered, byID[id])
	}
	marked := normalized
	if byHandles {
		// Compact the run-level handles ("[ID:265]") into the answer's own
		// first-citation order ("[ID:0]"), the numbering the payload ships.
		marked, _ = agentic_rag.ResolveCitations(normalized, registry, resolved)
	}
	// A chunk id inside a marker is rewritten to the same numbering, so the
	// marker the model wrote becomes the marker the client can resolve.
	marked = rewriteChunkIDHandles(marked, resolved)
	marked = agentic_rag.InsertCitationMarkers(marked, resolved)
	common.InfoCtx(ctx, "agentic citations built",
		zap.Int("cited", len(cited)),
		zap.Int("resolved", len(resolved)),
		zap.Bool("by_handle", byHandles),
		zap.Int("chunk_id_handles", len(handleChunkIDs)),
		zap.Int("final_bytes", len(marked)))
	return map[string]interface{}{
		"chunks":   chunksFormat(ordered),
		"doc_aggs": agenticDocAggs(ordered),
	}, marked
}

// fetchChunksByIDs loads the cited chunks from the tenant's index, keeping the
// engine row shape (chunk_id / content_with_weight / docnm_kwd / doc_id /
// kb_id / img_id / position_int) so chunksFormat can normalize it. The index is
// the chat's OWNING tenant's - the same scoping the agent's retrieval tools
// use.
func fetchChunksByIDs(ctx context.Context, tenantID string, datasetIDs, ids []string) []map[string]interface{} {
	de := engine.Get()
	if de == nil || len(ids) == 0 {
		return nil
	}
	req := &enginetypes.SearchRequest{
		IndexNames:   []string{fmt.Sprintf("ragflow_%s", tenantID)},
		KbIDs:        datasetIDs,
		Filter:       map[string]interface{}{"id": ids},
		SelectFields: []string{"content_with_weight", "docnm_kwd", "doc_id", "kb_id", "img_id", "position_int"},
		Limit:        len(ids),
	}
	res, err := de.Search(ctx, req)
	if err != nil {
		common.WarnCtx(ctx, "agentic citation chunk fetch failed", zap.Error(err))
		return nil
	}
	return res.Chunks
}

// agenticDocAggs aggregates the fetched chunks per document - the doc_aggs
// shape the UI's citation popover resolves documents from (doc_id + doc_name,
// plus a chunk count for parity with the naive payload).
func agenticDocAggs(chunks []map[string]interface{}) []interface{} {
	order := make([]string, 0, len(chunks))
	agg := make(map[string]map[string]interface{}, len(chunks))
	for _, ck := range chunks {
		docID, _ := ck["doc_id"].(string)
		if docID == "" {
			continue
		}
		if _, ok := agg[docID]; !ok {
			order = append(order, docID)
			agg[docID] = map[string]interface{}{
				"doc_id":   docID,
				"doc_name": ck["docnm_kwd"],
				"count":    0,
			}
			if url, _ := ck["url"].(string); url != "" {
				agg[docID]["url"] = url
			}
		}
		if n, ok := agg[docID]["count"].(int); ok {
			agg[docID]["count"] = n + 1
		}
	}
	out := make([]interface{}, 0, len(order))
	for _, docID := range order {
		out = append(out, agg[docID])
	}
	return out
}
