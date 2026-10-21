package loop

import (
	"context"
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
	// maxRepeatedToolCalls is how often one identical call may run before act
	// refuses it. One repeat is a retry; three is a loop.
	maxRepeatedToolCalls = 3
)

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

// isPromptTooLongError reports whether the prompt exceeded the model window.
func isPromptTooLongError(err error) bool {
	return errorContains(err, promptTooLongMarkers)
}

// modelBackoff doubles from modelRetryBase and stops growing at the ceiling.
func modelBackoff(attempt int) time.Duration {
	delay := modelRetryBase << (attempt - 1)
	if delay <= 0 || delay > modelRetryCeiling {
		return modelRetryCeiling
	}
	return delay
}

// callModel retries transient provider failures. Prompt-too-long is not retried
// here: it needs the prompt rebuilt, which is the caller's decision.
//
// The breaker is checked before the first attempt and updated once per call, not
// once per attempt: three attempts against a dead provider are one failure for
// the purpose of failing other runs fast.
func callModel(ctx context.Context, mdl model.Model, request model.Request, breaker *model.Breaker) (model.Response, error) {
	if remaining, ok := breaker.Allow(time.Now()); !ok {
		return model.Response{}, &model.ErrProviderCoolingDown{Key: "provider", Remaining: remaining}
	}

	var lastErr error
	for attempt := 1; attempt <= modelAttempts; attempt++ {
		response, err := mdl.Complete(ctx, request)
		if err == nil {
			breaker.Success()
			return response, nil
		}
		lastErr = err
		if !isRetryableModelError(err) || attempt == modelAttempts {
			if isRetryableModelError(err) {
				breaker.Failure(time.Now())
			}
			return model.Response{}, err
		}
		timer := time.NewTimer(modelBackoff(attempt))
		select {
		case <-ctx.Done():
			timer.Stop()
			return model.Response{}, ctx.Err()
		case <-timer.C:
		}
	}
	return model.Response{}, lastErr
}

// callModelStreaming retries a streaming call.
//
// A retry replays the request, which means the user would see the answer text
// twice. So a retry is only allowed before the first delta reached the sink; once
// text was shown the error is final and the caller decides what to tell the user.
func callModelStreaming(ctx context.Context, streamer model.Streamer, request model.Request, breaker *model.Breaker, sink func(string)) (model.Response, error) {
	if remaining, ok := breaker.Allow(time.Now()); !ok {
		return model.Response{}, &model.ErrProviderCoolingDown{Key: "provider", Remaining: remaining}
	}

	emitted := false
	sender := func(answer, _ *string) error {
		if answer == nil || *answer == "" || sink == nil {
			return nil
		}
		emitted = true
		sink(*answer)
		return nil
	}

	var lastErr error
	for attempt := 1; attempt <= modelAttempts; attempt++ {
		response, err := streamer.Stream(ctx, request, sender)
		if err == nil {
			breaker.Success()
			return response, nil
		}
		lastErr = err
		retryable := isRetryableModelError(err) && !emitted
		if !retryable || attempt == modelAttempts {
			if isRetryableModelError(err) {
				breaker.Failure(time.Now())
			}
			return model.Response{}, err
		}
		timer := time.NewTimer(modelBackoff(attempt))
		select {
		case <-ctx.Done():
			timer.Stop()
			return model.Response{}, ctx.Err()
		case <-timer.C:
		}
	}
	return model.Response{}, lastErr
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
