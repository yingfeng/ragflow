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
)

type fakeRuns struct {
	invoked   chan run.Request
	events    chan run.Event
	cancelled string
	starts    int
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
