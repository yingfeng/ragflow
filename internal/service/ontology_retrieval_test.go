package service

import (
	"context"
	"errors"
	"testing"

	"ragflow/internal/service/nlp"
)

type fakeOntologyScope struct {
	scope ontologyABoxScope
	err   error
}

func (f fakeOntologyScope) Resolve(_ context.Context, _ string, _ []string) (ontologyABoxScope, error) {
	return f.scope, f.err
}

type fakeOntologySearcher struct {
	got    *nlp.RetrievalRequest
	result *nlp.RetrievalResult
	err    error
}

func (f *fakeOntologySearcher) Retrieval(_ context.Context, req *nlp.RetrievalRequest) (*nlp.RetrievalResult, error) {
	f.got = req
	return f.result, f.err
}

func newOntologyABoxTestService(
	scope ontologyABoxScope,
	filter map[string]interface{},
	filterErr error,
) (*OntologyABoxRetrievalService, *fakeOntologySearcher) {
	searcher := &fakeOntologySearcher{
		result: &nlp.RetrievalResult{
			Chunks: []map[string]interface{}{{"content_with_weight": "row"}},
			Total:  1,
		},
	}
	svc := &OntologyABoxRetrievalService{
		scopeResolver: fakeOntologyScope{scope: scope},
		searcher:      searcher,
		structureFilter: func(_ context.Context, _, _ string, _ OntologyStructureConstraints) (map[string]interface{}, error) {
			if filterErr != nil {
				return nil, filterErr
			}
			return filter, nil
		},
	}
	return svc, searcher
}

func validOntologyABoxRequest() *OntologyABoxRetrievalRequest {
	return &OntologyABoxRetrievalRequest{
		UserID:             "user-1",
		DatasetIDs:         []string{"kb-1"},
		Question:           "who leads the guild",
		OntologyTemplateID: "tpl-1",
		EntityTypes:        []string{"agent"},
	}
}

// Without a constraint this endpoint would be "search the compiled rows", which
// the main retrieval already owns. Refusing keeps one owner per question, and
// nothing should reach the engine.
func TestOntologyABoxRetrievalRequiresAConstraint(t *testing.T) {
	svc, searcher := newOntologyABoxTestService(ontologyABoxScope{TenantIDs: []string{"tenant-1"}}, nil, nil)
	req := validOntologyABoxRequest()
	req.EntityTypes = nil

	if _, err := svc.Retrieve(context.Background(), req); err == nil {
		t.Fatal("expected a constraint-less request to be refused")
	}
	if searcher.got != nil {
		t.Fatal("the engine must not be searched without a constraint")
	}
}

// The filter reaches the engine unchanged and comes back on the result: the
// expansion is the answer to "why these rows", so the caller has to see it.
func TestOntologyABoxRetrievalAppliesAndEchoesTheFilter(t *testing.T) {
	filter := map[string]interface{}{
		"entity_type_kwd":     []string{"agent", "person", "organization"},
		"knowledge_graph_kwd": []string{"entity"},
	}
	svc, searcher := newOntologyABoxTestService(
		ontologyABoxScope{TenantIDs: []string{"tenant-1"}},
		filter, nil,
	)

	result, err := svc.Retrieve(context.Background(), validOntologyABoxRequest())
	if err != nil {
		t.Fatalf("retrieve: %v", err)
	}
	if searcher.got == nil {
		t.Fatal("the engine was never searched")
	}
	if got := searcher.got.Filter; got["entity_type_kwd"] == nil {
		t.Fatalf("filter = %#v, want the expanded classes", got)
	}
	if result.Filter["knowledge_graph_kwd"] == nil {
		t.Fatalf("result filter = %#v, want the applied filter echoed", result.Filter)
	}
	if len(searcher.got.KbIDs) != 1 || searcher.got.KbIDs[0] != "kb-1" {
		t.Fatalf("kb ids = %v, want the requested scope", searcher.got.KbIDs)
	}
	if searcher.got.PageSize != 10 {
		t.Fatalf("page size = %d, want the default top_k", searcher.got.PageSize)
	}
	if len(result.Chunks) != 1 || result.Total != 1 {
		t.Fatalf("result = %#v, want the engine rows passed through", result)
	}
}

// A template that cannot be read fails the request instead of quietly searching
// without the narrowing the caller asked for.
func TestOntologyABoxRetrievalPropagatesAnUnreadableTemplate(t *testing.T) {
	svc, searcher := newOntologyABoxTestService(
		ontologyABoxScope{TenantIDs: []string{"tenant-1"}},
		nil, errors.New("template could not be read"),
	)

	if _, err := svc.Retrieve(context.Background(), validOntologyABoxRequest()); err == nil {
		t.Fatal("expected the template error to surface")
	}
	if searcher.got != nil {
		t.Fatal("nothing should be searched when the structure cannot be read")
	}
}

// The remaining refusals: each is a request that cannot be answered honestly.
func TestOntologyABoxRetrievalRefusesIncompleteRequests(t *testing.T) {
	cases := map[string]func(*OntologyABoxRetrievalRequest){
		"no caller":   func(r *OntologyABoxRetrievalRequest) { r.UserID = "" },
		"no datasets": func(r *OntologyABoxRetrievalRequest) { r.DatasetIDs = nil },
		"no question": func(r *OntologyABoxRetrievalRequest) { r.Question = "  " },
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			svc, searcher := newOntologyABoxTestService(ontologyABoxScope{TenantIDs: []string{"tenant-1"}}, nil, nil)
			req := validOntologyABoxRequest()
			mutate(req)

			if _, err := svc.Retrieve(context.Background(), req); err == nil {
				t.Fatalf("%s: expected a refusal", name)
			}
			if searcher.got != nil {
				t.Fatalf("%s: the engine must not be searched", name)
			}
		})
	}
}
