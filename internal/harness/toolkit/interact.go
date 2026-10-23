package toolkit

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"ragflow/internal/harness/agent/model"
	"ragflow/internal/harness/events"
)

// InteractTimeout bounds one wait for a user's answer. It is deliberately far
// above the approval timeout: an approval is a decision that was already
// framed, while an open question may need the user to go find something.
const InteractTimeout = 15 * time.Minute

// Interactor is the channel a run uses to ask its user something. It is an
// interface, not a broker, so the chat path can report the wait to its stream
// while the API path serves it from the pending list.
type Interactor interface {
	// Ask blocks until the user answers, the timeout expires or the context is
	// cancelled. answered=false means nobody replied: the caller must proceed
	// without the answer and say so.
	Ask(ctx context.Context, requestID, question, context string) (answer string, answered bool)
}

// InteractRef is one waiting question as the API presents it. The key settles
// it: a client that reloaded and lost the stream can still answer.
type InteractRef struct {
	Key      string `json:"request_id"`
	RunID    string `json:"run_id"`
	Question string `json:"question"`
	Context  string `json:"context,omitempty"`
	TenantID string `json:"tenant_id"`
}

// pendingInteraction is one registered wait.
type pendingInteraction struct {
	ref     InteractRef
	answers chan string
}

// InteractBroker settles model→user questions across runs, the same way
// ApprovalBroker settles dangerous calls: one process-wide instance, requests
// addressed by "<run id>:<call id>", answered by key.
type InteractBroker struct {
	mu      sync.Mutex
	pending map[string]*pendingInteraction
}

// NewInteractBroker builds the broker.
func NewInteractBroker() *InteractBroker {
	return &InteractBroker{pending: make(map[string]*pendingInteraction)}
}

// Ask implements Interactor. The wait ends the moment the answer arrives, the
// run is cancelled, or the timeout expires; an expired wait is unregistered so
// a late answer cannot resolve a request nobody is waiting for.
func (b *InteractBroker) Ask(ctx context.Context, requestID, question, contextText string) (string, bool) {
	answers := make(chan string, 1)
	ref := InteractRef{
		Key:      requestID,
		Question: question,
		Context:  contextText,
		TenantID: ScopeFrom(ctx).TenantID,
	}
	if index := strings.LastIndex(requestID, ":"); index > 0 {
		ref.RunID = requestID[:index]
	}

	b.mu.Lock()
	b.pending[requestID] = &pendingInteraction{ref: ref, answers: answers}
	b.mu.Unlock()

	defer func() {
		b.mu.Lock()
		delete(b.pending, requestID)
		b.mu.Unlock()
	}()

	select {
	case answer := <-answers:
		return answer, true
	case <-ctx.Done():
		return "", false
	case <-time.After(InteractTimeout):
		return "", false
	}
}

// Pending returns one registered wait.
func (b *InteractBroker) Pending(runID, requestID string) (InteractRef, bool) {
	if requestID == "" {
		return InteractRef{}, false
	}
	if runID != "" && !strings.HasPrefix(requestID, runID+":") {
		return InteractRef{}, false
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if item, ok := b.pending[requestID]; ok {
		return item.ref, true
	}
	return InteractRef{}, false
}

// List returns every waiting question, newest last.
func (b *InteractBroker) List() []InteractRef {
	b.mu.Lock()
	defer b.mu.Unlock()
	refs := make([]InteractRef, 0, len(b.pending))
	for _, item := range b.pending {
		refs = append(refs, item.ref)
	}
	return refs
}

// Resolve delivers one answer. It reports whether the request was still
// waiting; a resolution after the run moved on (timeout, cancel) is dropped
// rather than buffered for a future request with the same id.
func (b *InteractBroker) Resolve(requestID, answer string) bool {
	if requestID == "" {
		return false
	}
	b.mu.Lock()
	item, ok := b.pending[requestID]
	b.mu.Unlock()
	if !ok {
		return false
	}
	select {
	case item.answers <- answer:
		return true
	default:
		// The waiter left between the map read and the send; the answer has no
		// recipient, so it must not count as delivered.
		return false
	}
}

// ReportInteractor wraps an Interactor so the caller observes every wait and
// every answer while the same broker settles it. The chat path uses this to put
// the question on its stream; the report must return immediately.
type ReportInteractor struct {
	Next Interactor
	// OnAsk reports the question going out. OnAnswer reports what came back,
	// carrying the question again so a renderer can show the pair without
	// keeping its own map. answered=false means the wait expired.
	OnAsk    func(requestID, question, contextText string)
	OnAnswer func(requestID, question, answer string, answered bool)
}

// Ask implements Interactor.
func (r ReportInteractor) Ask(ctx context.Context, requestID, question, contextText string) (string, bool) {
	if r.OnAsk != nil {
		r.OnAsk(requestID, question, contextText)
	}
	answer, answered := r.Next.Ask(ctx, requestID, question, contextText)
	if r.OnAnswer != nil {
		r.OnAnswer(requestID, question, answer, answered)
	}
	return answer, answered
}

// interactAskArgs is the ask_user body.
type interactAskArgs struct {
	Question string `json:"question"`
	Context  string `json:"context,omitempty"`
}

// NewInteractCapability builds the ask_user capability over an Interactor.
func NewInteractCapability(interactor Interactor) Capability {
	return &interactCapability{interactor: interactor}
}

// interactCapability exposes ask_user.
type interactCapability struct {
	interactor Interactor
}

// AskUser asks the run's user a question and waits for the answer.
const askUserDescription = "Ask the user a question and wait for their answer. " +
	"Use it when the task genuinely cannot proceed: a required credential, an " +
	"ambiguous instruction with materially different readings, or a choice only " +
	"the user can make. Do not use it for anything the corpus, the tools or a " +
	"reasonable default could settle — a question is the most expensive call in " +
	"the run, because it stops everything until a human replies. State options " +
	"when they exist; an open question with no context is hard to answer."

// Tools implements Capability.
func (c *interactCapability) Tools() []model.ToolSpec {
	return []model.ToolSpec{
		{
			Name:        "ask_user",
			Description: askUserDescription,
			Parameters: ObjectSchema(map[string]any{
				"question": map[string]any{"type": "string", "description": "One question, self-contained: the user reads it without the rest of the run."},
				"context":  map[string]any{"type": "string", "description": "Optional one-line background, for a question that needs more than itself."},
			}, "question"),
		},
	}
}

// ToolRisk implements RiskClassifier: asking changes nothing, and it is the
// cheapest way to resolve an ambiguity a dangerous tool call would have
// guessed at.
func (*interactCapability) ToolRisk(string) Risk { return RiskReadOnly }

// Name implements Capability.
func (*interactCapability) Name() string { return "interact" }

// Invoke implements Capability.
func (c *interactCapability) Invoke(ctx context.Context, call Call) (Result, error) {
	if c.interactor == nil {
		return Result{}, fmt.Errorf("no user channel is configured for this run")
	}
	args, err := DecodeArgs[interactAskArgs](call.Arguments)
	if err != nil {
		return Result{}, err
	}
	if strings.TrimSpace(args.Question) == "" {
		return Result{}, fmt.Errorf("question is required")
	}
	requestID := RunIDFrom(ctx) + ":" + call.ID
	if recorder := events.RecorderFromContext(ctx); recorder != nil {
		recorder.RecordInteractionRequest(ctx, requestID, args.Question, args.Context)
	}
	answer, answered := c.interactor.Ask(ctx, requestID, args.Question, args.Context)
	if recorder := events.RecorderFromContext(ctx); recorder != nil {
		recorder.RecordInteractionResolved(ctx, requestID, answer, answered)
	}
	if !answered {
		// No answer is a fact the model must act on, not an error to retry: the
		// same ask would wait another quarter hour on the same missing user.
		return Result{Content: "the user did not answer within the wait; proceed with what you have and state the assumption"}, nil
	}
	return Result{Content: answer}, nil
}
