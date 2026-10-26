package loop

import (
	"context"
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"time"

	"ragflow/internal/harness/agent/model"
)

// Provider failures are not all the same shape. The reference runtimes separate
// three classes and the harness has to do the same, because a single "retry
// everything" loop either hammers a broken provider or gives up on a transient
// 503:
//
//   - transient (429 / 5xx / timeout / reset): retry with backoff;
//   - prompt too long: shrink the prompt and retry, up to three times
//     (Astra aborts after three consecutive PTL errors);
//   - everything else: fail the run immediately.
const (
	// modelAttempts bounds one completion against transient failures.
	modelAttempts = 3
	// ptlAttempts bounds how many times one run may shrink its prompt.
	ptlAttempts = 3
	// modelRetryBase is the first backoff step; it doubles per attempt.
	modelRetryBase = 300 * time.Millisecond
	// modelRetryCeiling caps the backoff.
	modelRetryCeiling = 3 * time.Second
	// modelRetryAfterCeiling caps a provider-stated wait. It is separate from
	// the backoff ceiling because a rate limiter asking for 30 seconds is
	// telling the truth; the local curve is only a guess.
	modelRetryAfterCeiling = 60 * time.Second
	// maxRepeatedToolCalls is how often one identical call may run before act
	// refuses it. One repeat is a retry; three is a loop.
	maxRepeatedToolCalls = 3
	// maxAttemptsCeiling bounds a configured attempt count. A deployment may
	// raise the number of retries; nothing may raise it into a long stall.
	maxAttemptsCeiling = 20
)

// RetryPolicy is the deployment's retry budget. A zero field keeps the default;
// the distinction matters because "0 retries" is a legitimate operator choice
// and "unset" is not the same configuration.
type RetryPolicy struct {
	// Request bounds a non-streaming completion.
	Request int
	// Stream bounds a streaming completion. It can differ: a stream that has
	// not emitted yet is cheap to replay, and some deployments want fewer
	// retries once text may already be on screen.
	Stream int
}

// requestAttempts resolves the configured value against the defaults.
func (p RetryPolicy) requestAttempts() int { return boundedAttempts(p.Request) }

// streamAttempts resolves the configured value against the defaults.
func (p RetryPolicy) streamAttempts() int { return boundedAttempts(p.Stream) }

func boundedAttempts(configured int) int {
	if configured <= 0 {
		return modelAttempts
	}
	if configured > maxAttemptsCeiling {
		return maxAttemptsCeiling
	}
	return configured
}

var retryableMarkers = []string{
	"rate limit",
	"rate_limit",
	"429",
	"overloaded",
	"timeout",
	"timed out",
	"deadline exceeded",
	"connection reset",
	"connection refused",
	"temporarily unavailable",
	"502",
	"503",
	"504",
}

var promptTooLongMarkers = []string{
	"context length",
	"context_length",
	"maximum context",
	"prompt is too long",
	"reduce the length",
	"too many tokens",
	"token limit",
	"input is too long",
}

func errorContains(err error, markers []string) bool {
	if err == nil {
		return false
	}
	message := strings.ToLower(err.Error())
	for _, marker := range markers {
		if strings.Contains(message, marker) {
			return true
		}
	}
	return false
}

// isRetryableModelError reports whether the provider failure is transient.
func isRetryableModelError(err error) bool {
	return errorContains(err, retryableMarkers)
}

// contentFilterMarkers name a provider rejection for content, not for load. It
// is retried once with guidance because the model can phrase the same answer
// differently; retrying the identical request would be rejected identically.
var contentFilterMarkers = []string{
	"content filter",
	"content_filter",
	"content policy",
	"content_policy",
	"content management policy",
	"flagged",
	"responsible ai",
}

// authMarkers name a credential failure. It is never retried: the same key
// fails the same way, and retrying spends the run's budget on a fix that only
// a human can make.
var authMarkers = []string{
	"invalid api key",
	"invalid_api_key",
	"incorrect api key",
	"unauthorized",
	"authentication",
	"401",
}

// isContentFilterError reports a provider-side content rejection.
func isContentFilterError(err error) bool {
	return errorContains(err, contentFilterMarkers)
}

// isAuthError reports a credential failure.
func isAuthError(err error) bool {
	return errorContains(err, authMarkers)
}

// contentFilterGuidance is appended once when a provider rejects an answer for
// content. A bare retry reproduces the rejection; naming the constraint is what
// lets the model answer the question another way.
const contentFilterGuidance = "The provider rejected the previous answer for content policy. Answer the same question again without the material that was rejected, keeping the factual content the user asked for."

// withContentFilterGuidance appends the guidance once per request. The struct is
// copied wholesale so a field added to Request later is carried over too.
func withContentFilterGuidance(request model.Request) model.Request {
	updated := request
	messages := make([]model.Message, len(request.Messages), len(request.Messages)+1)
	copy(messages, request.Messages)
	updated.Messages = append(messages, model.Message{Role: model.RoleUser, Content: contentFilterGuidance})
	return updated
}

// isPromptTooLongError reports whether the prompt exceeded the model window.
func isPromptTooLongError(err error) bool {
	if errorContains(err, promptTooLongMarkers) {
		return true
	}
	// MiniMax reports an over-window request as a bare provider code instead of
	// prose ("minimax API error: ... (2064)"). Matching the number alone would
	// catch unrelated digits, so the provider name is required with it. Without
	// this the run failed outright where compaction could have recovered it.
	if err == nil {
		return false
	}
	message := strings.ToLower(err.Error())
	return strings.Contains(message, "minimax") && strings.Contains(message, "(2064)")
}

// modelBackoff doubles from modelRetryBase and stops growing at the ceiling.
func modelBackoff(attempt int) time.Duration {
	delay := modelRetryBase << (attempt - 1)
	if delay <= 0 || delay > modelRetryCeiling {
		return modelRetryCeiling
	}
	return delay
}

// retryAfterPattern matches the wait a provider driver appends to its error text
// when the response carried a Retry-After header.
var retryAfterPattern = regexp.MustCompile(`retry-after:\s*(\d+)`)

// modelRetryDelay prefers the provider's own wait over the local backoff curve.
//
// Retrying before a provider's stated wait is guaranteed to fail again, and it
// spends an attempt to learn that. The ceiling still applies: one provider
// answering Retry-After: 3600 must not stall the run for an hour.
func modelRetryDelay(err error, attempt int) time.Duration {
	if err != nil {
		if match := retryAfterPattern.FindStringSubmatch(strings.ToLower(err.Error())); len(match) == 2 {
			if seconds, convErr := strconv.Atoi(match[1]); convErr == nil && seconds > 0 {
				delay := time.Duration(seconds) * time.Second
				if delay < modelRetryAfterCeiling {
					return delay
				}
				return modelRetryAfterCeiling
			}
		}
	}
	return modelBackoff(attempt)
}

// callModel retries transient provider failures. Prompt-too-long is not retried
// here: it needs the prompt rebuilt, which is the caller's decision.
//
// The breaker is checked before the first attempt and updated once per call, not
// once per attempt: three attempts against a dead provider are one failure for
// the purpose of failing other runs fast.
func callModel(ctx context.Context, mdl model.Model, request model.Request, breaker *model.Breaker) (model.Response, error) {
	return callModelWith(ctx, mdl, request, breaker, 0)
}

// callModelWith is callModel with an explicit attempt budget. Zero means the
// default, so a caller that has no policy to apply does not have to name one.
func callModelWith(ctx context.Context, mdl model.Model, request model.Request, breaker *model.Breaker, attempts int) (model.Response, error) {
	attempts = boundedAttempts(attempts)
	if remaining, ok := breaker.Allow(time.Now()); !ok {
		return model.Response{}, &model.ErrProviderCoolingDown{Key: "provider", Remaining: remaining}
	}

	var lastErr error
	for attempt := 1; attempt <= attempts; attempt++ {
		response, err := mdl.Complete(ctx, request)
		if err == nil {
			breaker.Success()
			return response, nil
		}
		lastErr = err
		// A content rejection is the one failure where replaying the request is
		// pointless and replaying it with a constraint is not: the model has to
		// know what was rejected to phrase the answer differently.
		if isContentFilterError(err) && attempt < attempts {
			request = withContentFilterGuidance(request)
			continue
		}
		if !isRetryableModelError(err) || attempt == attempts {
			if isRetryableModelError(err) {
				breaker.Failure(time.Now())
			}
			return model.Response{}, explainModelError(err)
		}
		timer := time.NewTimer(modelRetryDelay(err, attempt))
		select {
		case <-ctx.Done():
			timer.Stop()
			return model.Response{}, ctx.Err()
		case <-timer.C:
		}
	}
	return model.Response{}, explainModelError(lastErr)
}

// explainModelError turns a credential failure into something an operator can
// act on. The provider's own text says what broke; what it does not say is that
// retrying cannot help, which is the fact the caller needs before spending the
// run's remaining budget on it.
func explainModelError(err error) error {
	if err == nil || !isAuthError(err) {
		return err
	}
	return fmt.Errorf("%w (the model provider rejected the credentials; retrying will not help — check the tenant's provider key for this model)", err)
}

// callModelStreamingWith retries a streaming call.
//
// A retry replays the request, which means the user would see the same output
// twice. So a retry is only allowed before the first delta reached either sink;
// once text was shown the error is final and the caller decides what to tell the
// user.
//
// sink receives visible text, thought receives reasoning. They are separate
// because the two have different homes in the UI: text becomes either the answer
// or the step panel's working note, while reasoning is always working material.
// Dropping thought — which this function used to do by accepting only the first
// sender argument — is why a run could search for twenty rounds with nothing
// shown about why.
func callModelStreamingWith(ctx context.Context, streamer model.Streamer, request model.Request, breaker *model.Breaker, attempts int, sink, thought func(string)) (model.Response, error) {
	attempts = boundedAttempts(attempts)
	if remaining, ok := breaker.Allow(time.Now()); !ok {
		return model.Response{}, &model.ErrProviderCoolingDown{Key: "provider", Remaining: remaining}
	}

	emitted := false
	sender := func(answer, reasoning *string) error {
		if reasoning != nil && *reasoning != "" && thought != nil {
			// Reasoning already reached the UI: a retry would print it twice, so
			// it counts as emitted exactly like answer text does.
			emitted = true
			thought(*reasoning)
		}
		if answer == nil || *answer == "" || sink == nil {
			return nil
		}
		emitted = true
		sink(*answer)
		return nil
	}

	var lastErr error
	for attempt := 1; attempt <= attempts; attempt++ {
		response, err := streamer.Stream(ctx, request, sender)
		if err == nil {
			breaker.Success()
			return response, nil
		}
		lastErr = err
		if isContentFilterError(err) && !emitted && attempt < attempts {
			request = withContentFilterGuidance(request)
			continue
		}
		retryable := isRetryableModelError(err) && !emitted
		if !retryable || attempt == attempts {
			if isRetryableModelError(err) {
				breaker.Failure(time.Now())
			}
			return model.Response{}, explainModelError(err)
		}
		timer := time.NewTimer(modelRetryDelay(err, attempt))
		select {
		case <-ctx.Done():
			timer.Stop()
			return model.Response{}, ctx.Err()
		case <-timer.C:
		}
	}
	return model.Response{}, explainModelError(lastErr)
}

// boostTier maps the number of prompt-too-long recoveries onto the compaction
// tier the next prepare must use.
func boostTier(boost int) int {
	if boost <= 0 {
		return 0
	}
	if boost > ptlAttempts {
		boost = ptlAttempts
	}
	return boost
}
