package api

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"ragflow/internal/entity"
	"ragflow/internal/harness/agent/loop"
	"ragflow/internal/harness/agent/run"
	"ragflow/internal/harness/events"
)

// Events records the query it received and returns the canned result.
func (f *fakeRuns) Events(_ context.Context, _ string, query EventQuery) ([]*events.Event, error) {
	f.eventQuery = query
	return f.eventItems, f.eventErr
}

type fakeRuns struct {
	invoked            chan run.Request
	events             chan run.Event
	cancelled          string
	steered            string
	starts             int
	resumedRunID       string
	resumeErr          error
	approvalResolved   string
	approvalPending    bool
	approvalRemembered bool
	eventQuery         EventQuery
	eventItems         []*events.Event
	eventErr           error
}

func newFakeRuns() *fakeRuns {
	return &fakeRuns{invoked: make(chan run.Request, 4), events: make(chan run.Event, 4)}
}

func (f *fakeRuns) Invoke(_ context.Context, req run.Request) (loop.Outcome, error) {
	f.invoked <- req
	return loop.Outcome{RunID: req.RunID, Text: "ok"}, nil
}

func (f *fakeRuns) Subscribe(string) chan run.Event { return f.events }

func (f *fakeRuns) Unsubscribe(string, chan run.Event) {}

func (f *fakeRuns) Cancel(runID string) bool {
	f.cancelled = runID
	return runID == "known"
}

func (f *fakeRuns) Steer(runID, message string) bool {
	f.steered = runID + ":" + message
	return runID == "known"
}

func (f *fakeRuns) Resume(_ context.Context, req run.Request) (loop.Outcome, error) {
	f.resumedRunID = req.RunID
	return loop.Outcome{RunID: req.RunID}, f.resumeErr
}

func (f *fakeRuns) ResolveApproval(runID, requestID string, allow bool, reason string, remember bool) bool {
	f.approvalRemembered = remember
	f.approvalResolved = runID + ":" + requestID + ":" + reason
	return f.approvalPending
}

func (f *fakeRuns) Steps(context.Context, string) ([]entity.HarnessAgentStep, error) {
	return []entity.HarnessAgentStep{{ID: "s1", RunID: "known", Seq: 1, Phase: "think"}}, nil
}

func newTestServer(t *testing.T, runs Runs) *gin.Engine {
	t.Helper()
	gin.SetMode(gin.TestMode)
	router := gin.New()
	New(runs).Register(router.Group("/v1"))
	return router
}

func TestCreateRunStartsInBackground(t *testing.T) {
	runs := newFakeRuns()
	router := newTestServer(t, runs)

	body := `{"tenant_id":"t1","agent_name":"rag","user_message":"who wrote it?","dataset_ids":["d1"]}`
	request := httptest.NewRequest(http.MethodPost, "/v1/agents/a1/runs", strings.NewReader(body))
	request.Header.Set("Content-Type", "application/json")
	recorder := httptest.NewRecorder()
	router.ServeHTTP(recorder, request)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", recorder.Code, recorder.Body.String())
	}

	var response struct {
		Code int `json:"code"`
		Data struct {
			RunID string `json:"run_id"`
		} `json:"data"`
	}
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if response.Data.RunID == "" {
		t.Fatal("run_id is empty")
	}

	select {
	case req := <-runs.invoked:
		if req.RunID != response.Data.RunID || req.TenantID != "t1" || req.Scope.Datasets[0] != "d1" {
			t.Fatalf("invoked request = %+v", req)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("run was never started")
	}
}

func TestCreateRunRejectsEmptyMessage(t *testing.T) {
	router := newTestServer(t, newFakeRuns())

	request := httptest.NewRequest(http.MethodPost, "/v1/agents/a1/runs", strings.NewReader(`{"tenant_id":"t1"}`))
	request.Header.Set("Content-Type", "application/json")
	recorder := httptest.NewRecorder()
	router.ServeHTTP(recorder, request)

	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", recorder.Code)
	}
}

func TestStreamRunRelaysEvents(t *testing.T) {
	runs := newFakeRuns()
	router := newTestServer(t, runs)

	server := httptest.NewServer(router)
	defer server.Close()

	response, err := http.Get(server.URL + "/v1/runs/r1/stream")
	if err != nil {
		t.Fatalf("get stream: %v", err)
	}
	defer response.Body.Close()

	if contentType := response.Header.Get("Content-Type"); !strings.Contains(contentType, "text/event-stream") {
		t.Fatalf("content type = %q", contentType)
	}

	runs.events <- run.Event{Type: run.EventStep, RunID: "r1"}

	buffer := make([]byte, 512)
	read, err := response.Body.Read(buffer)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	payload := string(buffer[:read])
	if !strings.Contains(payload, "event: step") || !strings.Contains(payload, `"run_id":"r1"`) {
		t.Fatalf("payload = %q", payload)
	}
}

func TestCancelRun(t *testing.T) {
	runs := newFakeRuns()
	router := newTestServer(t, runs)

	recorder := httptest.NewRecorder()
	router.ServeHTTP(recorder, httptest.NewRequest(http.MethodPost, "/v1/runs/known/cancel", nil))
	if recorder.Code != http.StatusOK || runs.cancelled != "known" {
		t.Fatalf("status = %d, cancelled = %q", recorder.Code, runs.cancelled)
	}

	recorder = httptest.NewRecorder()
	router.ServeHTTP(recorder, httptest.NewRequest(http.MethodPost, "/v1/runs/unknown/cancel", nil))
	if recorder.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want 404", recorder.Code)
	}
}

func TestSteerRun(t *testing.T) {
	runs := newFakeRuns()
	router := newTestServer(t, runs)

	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, "/v1/runs/known/steer", strings.NewReader(`{"message":"also check 2019"}`))
	request.Header.Set("Content-Type", "application/json")
	router.ServeHTTP(recorder, request)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", recorder.Code, recorder.Body.String())
	}
	if runs.steered != "known:also check 2019" {
		t.Fatalf("steered = %q", runs.steered)
	}

	recorder = httptest.NewRecorder()
	request = httptest.NewRequest(http.MethodPost, "/v1/runs/unknown/steer", strings.NewReader(`{"message":"x"}`))
	request.Header.Set("Content-Type", "application/json")
	router.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want 404", recorder.Code)
	}
}

func TestListEventsPassesFilters(t *testing.T) {
	runs := newFakeRuns()
	runs.eventItems = []*events.Event{{ID: "evt-1", Type: events.EventToolCallResult, Layer: events.LayerAudit}}
	router := newTestServer(t, runs)

	recorder := httptest.NewRecorder()
	router.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet,
		"/v1/runs/known/events?layer=C2&layer=C0&type=tool.call.result&limit=10", nil))

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", recorder.Code, recorder.Body.String())
	}
	if len(runs.eventQuery.Layers) != 2 || runs.eventQuery.Layers[0] != "C2" {
		t.Fatalf("layers = %v", runs.eventQuery.Layers)
	}
	if len(runs.eventQuery.Types) != 1 || runs.eventQuery.Types[0] != "tool.call.result" {
		t.Fatalf("types = %v", runs.eventQuery.Types)
	}
	if runs.eventQuery.Limit != 10 {
		t.Fatalf("limit = %d", runs.eventQuery.Limit)
	}

	recorder = httptest.NewRecorder()
	router.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/v1/runs/known/events?limit=0", nil))
	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400 for limit=0", recorder.Code)
	}
}

func TestResolveApprovalRemember(t *testing.T) {
	runs := newFakeRuns()
	runs.approvalPending = true
	router := newTestServer(t, runs)

	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, "/v1/runs/known/approvals/req-1",
		strings.NewReader(`{"allow":true,"reason":"ok","remember":true}`))
	request.Header.Set("Content-Type", "application/json")
	router.ServeHTTP(recorder, request)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", recorder.Code, recorder.Body.String())
	}
	if !runs.approvalRemembered {
		t.Fatal("remember flag was not forwarded")
	}
}

func TestListSteps(t *testing.T) {
	router := newTestServer(t, newFakeRuns())

	recorder := httptest.NewRecorder()
	router.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/v1/runs/known/steps", nil))
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d", recorder.Code)
	}
	if !strings.Contains(recorder.Body.String(), `"phase":"think"`) {
		t.Fatalf("body = %s", recorder.Body.String())
	}
}
