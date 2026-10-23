package handler

import (
	"context"
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"gorm.io/gorm"

	"ragflow/internal/entity"
	"ragflow/internal/harness/agent/definition"
	"ragflow/internal/service"
)

// fakeHarnessAgents stands in for the agent service: the handler's job is the
// HTTP contract, not the storage rules.
type fakeHarnessAgents struct {
	detail  *service.AgentDetail
	err     error
	updated definition.Definition
	calls   int
}

func (f *fakeHarnessAgents) Detail(ctx context.Context, tenantID, agentID string) (*service.AgentDetail, error) {
	if f.err != nil {
		return nil, f.err
	}
	return f.detail, nil
}

func (f *fakeHarnessAgents) Update(ctx context.Context, tenantID, agentID, updatedBy string, def definition.Definition) (*service.AgentDetail, error) {
	f.calls++
	if f.err != nil {
		return nil, f.err
	}
	f.updated = def
	return f.detail, nil
}

// harnessAgentContext builds a gin context the way the harness middleware leaves
// it: an authenticated user, addressed by the path parameter.
func harnessAgentContext(t *testing.T, body string) (*gin.Context, *httptest.ResponseRecorder) {
	t.Helper()
	gin.SetMode(gin.TestMode)
	recorder := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(recorder)
	request := httptest.NewRequest("GET", "/api/v1/harness/agents/agent-1", strings.NewReader(body))
	request.Header.Set("Content-Type", "application/json")
	c.Request = request
	c.Params = gin.Params{{Key: "agent_id", Value: "agent-1"}}
	c.Set("user", &entity.User{ID: "tenant-1"})
	return c, recorder
}

// decodeEnvelope reads the {code,data,message} envelope the client parses.
func decodeEnvelope(t *testing.T, recorder *httptest.ResponseRecorder) (int, map[string]any) {
	t.Helper()
	var envelope map[string]any
	if err := json.Unmarshal(recorder.Body.Bytes(), &envelope); err != nil {
		t.Fatalf("decode response %q: %v", recorder.Body.String(), err)
	}
	code, _ := envelope["code"].(float64)
	return int(code), envelope
}

// The editor needs the live definition and its version, so a save can tell the
// user which version they replaced.
func TestGetHarnessAgentReturnsTheLiveVersion(t *testing.T) {
	agents := &fakeHarnessAgents{detail: &service.AgentDetail{
		AgentID:   "agent-1",
		VersionID: "version-2",
		Version:   2,
		Name:      "thread-agent",
		Definition: definition.Definition{
			Prompt: "p", Policy: definition.PolicyConfig{MaxRounds: 8},
		},
	}}
	h := NewHarnessAgentHandler(agents)

	c, recorder := harnessAgentContext(t, "")
	h.getAgent(c)

	code, envelope := decodeEnvelope(t, recorder)
	if code != 0 {
		t.Fatalf("code = %d, want success: %s", code, recorder.Body.String())
	}
	data, _ := envelope["data"].(map[string]any)
	if data["version"].(float64) != 2 || data["version_id"] != "version-2" {
		t.Fatalf("data = %v", data)
	}
}

// A body that edits the policy is forwarded as a definition: the handler must not
// invent fields or drop the ones it does not know.
func TestUpdateHarnessAgentForwardsTheDefinition(t *testing.T) {
	agents := &fakeHarnessAgents{detail: &service.AgentDetail{AgentID: "agent-1", Version: 2}}
	h := NewHarnessAgentHandler(agents)

	body := `{"name":"thread-agent","prompt":"new prompt","policy":{"max_rounds":0,"max_tool_calls":40}}`
	c, recorder := harnessAgentContext(t, body)
	h.updateAgent(c)

	code, _ := decodeEnvelope(t, recorder)
	if code != 0 {
		t.Fatalf("code = %d, want success: %s", code, recorder.Body.String())
	}
	if agents.calls != 1 {
		t.Fatalf("service calls = %d, want 1", agents.calls)
	}
	if agents.updated.Prompt != "new prompt" {
		t.Fatalf("prompt = %q", agents.updated.Prompt)
	}
	if agents.updated.Policy.MaxRounds != 0 || agents.updated.Policy.MaxToolCalls != 40 {
		t.Fatalf("policy = %+v", agents.updated.Policy)
	}
}

// A missing agent is answerable with a 404 code, not a server error: retrying with
// the same id cannot succeed.
func TestUpdateHarnessAgentAnswersNotFound(t *testing.T) {
	h := NewHarnessAgentHandler(&fakeHarnessAgents{err: gorm.ErrRecordNotFound})

	c, recorder := harnessAgentContext(t, `{"prompt":"p"}`)
	h.updateAgent(c)

	code, _ := decodeEnvelope(t, recorder)
	if code != 404 {
		t.Fatalf("code = %d, want 404: %s", code, recorder.Body.String())
	}
}

// An unchanged save and a taken name are both the client's to fix.
func TestUpdateHarnessAgentAnswersDataErrors(t *testing.T) {
	for _, tc := range []struct {
		name string
		err  error
	}{
		{"unchanged", service.ErrNoChange},
		{"name taken", service.ErrAgentNameTaken},
		{"empty prompt", definition.ErrEmptyPrompt},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := NewHarnessAgentHandler(&fakeHarnessAgents{err: tc.err})
			c, recorder := harnessAgentContext(t, `{"prompt":""}`)
			h.updateAgent(c)
			code, _ := decodeEnvelope(t, recorder)
			if code == 0 || code == 500 || code == 404 {
				t.Fatalf("code = %d, want a data error: %s", code, recorder.Body.String())
			}
		})
	}
}

// An unauthenticated caller never reaches the service.
func TestUpdateHarnessAgentRequiresAUser(t *testing.T) {
	agents := &fakeHarnessAgents{}
	h := NewHarnessAgentHandler(agents)

	c, recorder := harnessAgentContext(t, `{"prompt":"p"}`)
	c.Set("user", nil)
	h.updateAgent(c)

	if agents.calls != 0 {
		t.Fatalf("service calls = %d, want none", agents.calls)
	}
	if recorder.Body.Len() == 0 {
		t.Fatal("no response written")
	}
}
