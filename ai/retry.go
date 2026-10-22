package ai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"net/http"
	"strconv"
	"strings"
	"sync/atomic"
	"time"
)

// Providers retry requests here rather than leaving it to their SDKs, so every
// provider follows one policy and enforcing MaxRetriesDelayMs does not depend
// on predicting an SDK's internals. A provider turns its SDK's retries off and
// sends each request through retryPolicy.do.

const (
	// DefaultMaxRetries is how many times a failed request is retried when
	// StreamOptions.MaxRetries is nil. It matches the OpenAI SDKs' default.
	DefaultMaxRetries = 2
	// DefaultTimeout is how long a request waits for its response to start when
	// StreamOptions.TimeoutMs is nil. It matches the OpenAI Node SDK's default.
	DefaultTimeout = 10 * time.Minute
	// DefaultMaxRetriesDelay is the longest server-requested retry delay waited
	// out when StreamOptions.MaxRetriesDelayMs is nil.
	DefaultMaxRetriesDelay = 60 * time.Second
)

const (
	// Backoff between retries when the server does not ask for a delay.
	initialRetryDelay = 500 * time.Millisecond
	maxRetryBackoff   = 8 * time.Second
	// The retry count header that Stainless-generated SDKs, such as OpenAI's,
	// send. With their retries off they always send 0, so do keeps it accurate.
	retryCountHeader = "X-Stainless-Retry-Count"
)

// retryPolicy is how a provider retries one request.
type retryPolicy struct {
	maxRetries int
	// maxDelay is the longest server-requested delay to wait out; zero means
	// no limit.
	maxDelay time.Duration
	// timeout bounds each attempt's wait for its response to start; zero means
	// no limit.
	timeout time.Duration
}

// maxDurationMs is the largest millisecond count a time.Duration can hold,
// about 292 years.
const maxDurationMs = math.MaxInt64 / int64(time.Millisecond)

// newRetryPolicy reads the retry options, rejecting negative values and
// durations too large to represent.
func newRetryPolicy(options StreamOptions) (retryPolicy, error) {
	policy := retryPolicy{maxRetries: DefaultMaxRetries}
	if options.MaxRetries != nil {
		if *options.MaxRetries < 0 {
			return retryPolicy{}, fmt.Errorf("MaxRetries must not be negative, got %d", *options.MaxRetries)
		}
		policy.maxRetries = int(*options.MaxRetries)
	}
	var err error
	if policy.maxDelay, err = durationOption("MaxRetriesDelayMs", options.MaxRetriesDelayMs, DefaultMaxRetriesDelay); err != nil {
		return retryPolicy{}, err
	}
	if policy.timeout, err = durationOption("TimeoutMs", options.TimeoutMs, DefaultTimeout); err != nil {
		return retryPolicy{}, err
	}
	return policy, nil
}

// durationOption returns the milliseconds in the option name as a duration, or
// def when ms is nil.
func durationOption(name string, ms *int64, def time.Duration) (time.Duration, error) {
	switch {
	case ms == nil:
		return def, nil
	case *ms < 0:
		return 0, fmt.Errorf("%s must not be negative, got %d", name, *ms)
	case *ms > maxDurationMs:
		return 0, fmt.Errorf("%s must be at most %d, got %d", name, maxDurationMs, *ms)
	}
	return time.Duration(*ms) * time.Millisecond, nil
}

// do sends req with send, retrying connection errors, timeouts, and retryable
// responses up to maxRetries times. It waits between attempts for the delay
// the server asks for, or else an exponential backoff. When the server asks
// for longer than maxDelay, do fails at once instead of waiting, with the
// requested delay in the error. Once retries run out, it returns the last
// attempt's result.
func (p retryPolicy) do(req *http.Request, send func(*http.Request) (*http.Response, error)) (*http.Response, error) {
	ctx := req.Context()
	for attempt := 0; ; attempt++ {
		attemptReq := req
		if attempt > 0 {
			var err error
			if attemptReq, err = resend(req); err != nil {
				return nil, err
			}
			if req.Header.Get(retryCountHeader) != "" {
				attemptReq.Header.Set(retryCountHeader, strconv.Itoa(attempt))
			}
		}

		resp, err := p.attempt(attemptReq, send)
		if ctxErr := ctx.Err(); ctxErr != nil {
			closeBody(resp)
			return nil, ctxErr
		}
		if attempt >= p.maxRetries || !shouldRetry(req, resp, err) {
			return resp, err
		}

		delay, requested := retryDelay(resp, attempt)
		if requested && p.maxDelay > 0 && delay > p.maxDelay {
			return nil, retryDelayError(resp, delay, p.maxDelay)
		}
		closeBody(resp)
		timer := time.NewTimer(delay)
		select {
		case <-ctx.Done():
			timer.Stop()
			return nil, ctx.Err()
		case <-timer.C:
		}
	}
}

// resend returns a copy of req with a fresh body for another attempt.
func resend(req *http.Request) (*http.Request, error) {
	next := req.Clone(req.Context())
	if req.GetBody != nil {
		body, err := req.GetBody()
		if err != nil {
			return nil, err
		}
		next.Body = body
	}
	return next, nil
}

// shouldRetry reports whether a request that ended with resp or err may be
// sent again.
func shouldRetry(req *http.Request, resp *http.Response, err error) bool {
	if req.Body != nil && req.Body != http.NoBody && req.GetBody == nil {
		return false
	}
	if err != nil {
		return true
	}
	switch resp.Header.Get("x-should-retry") {
	case "true":
		return true
	case "false":
		return false
	}
	return resp.StatusCode == http.StatusRequestTimeout ||
		resp.StatusCode == http.StatusConflict ||
		resp.StatusCode == http.StatusTooManyRequests ||
		resp.StatusCode >= http.StatusInternalServerError
}

// retryDelay returns how long to wait before retrying after resp, and whether
// the server asked for that delay. resp is nil after a connection error.
func retryDelay(resp *http.Response, attempt int) (time.Duration, bool) {
	if resp != nil {
		if delay, ok := requestedDelay(resp.Header); ok {
			return max(0, delay), true
		}
	}
	// Double one step at a time and stop at the cap, so a high attempt number
	// cannot overflow the delay.
	delay := initialRetryDelay
	for i := 0; i < attempt && delay < maxRetryBackoff; i++ {
		delay *= 2
	}
	delay = min(delay, maxRetryBackoff)
	// Jitter of up to a quarter keeps clients from retrying in step.
	return delay - rand.N(delay/4), false
}

// requestedDelay reads the delay a server asks for in Retry-After-Ms, as
// OpenAI sends it, or else in Retry-After, as seconds or an HTTP date.
func requestedDelay(header http.Header) (time.Duration, bool) {
	if value := header.Get("Retry-After-Ms"); value != "" {
		if ms, err := strconv.ParseFloat(value, 64); err == nil {
			return time.Duration(ms * float64(time.Millisecond)), true
		}
	}
	if value := header.Get("Retry-After"); value != "" {
		if seconds, err := strconv.ParseFloat(value, 64); err == nil {
			return time.Duration(seconds * float64(time.Second)), true
		}
		if at, err := http.ParseTime(value); err == nil {
			return time.Until(at), true
		}
	}
	return 0, false
}

// retryDelayError reports a requested delay over the limit, in pi-mono's
// format, so that its "retry delay" retry matching carries over. It consumes
// resp's body for the API's error message.
func retryDelayError(resp *http.Response, delay, limit time.Duration) error {
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<10))
	_ = resp.Body.Close()
	return fmt.Errorf(
		"Server requested %ds retry delay (max: %ds). %s",
		ceilSeconds(delay), ceilSeconds(limit), apiErrorMessage(body),
	)
}

// apiErrorMessage returns error.message from an API error body, or the body
// itself when it has none.
func apiErrorMessage(body []byte) string {
	var parsed struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if json.Unmarshal(body, &parsed) == nil && parsed.Error.Message != "" {
		return parsed.Error.Message
	}
	return strings.TrimSpace(string(body))
}

func ceilSeconds(d time.Duration) int64 {
	return int64(math.Ceil(d.Seconds()))
}

// closeBody drains a little of resp's body, so the connection can be reused,
// and closes it.
func closeBody(resp *http.Response) {
	if resp == nil || resp.Body == nil {
		return
	}
	_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 4<<10))
	_ = resp.Body.Close()
}

var errResponseTimeout = errors.New("response timeout")

// attempt sends req once, failing it when its response does not start within
// the timeout. Timing stops once the response starts, so a long stream is
// never cut off.
func (p retryPolicy) attempt(req *http.Request, send func(*http.Request) (*http.Response, error)) (*http.Response, error) {
	if p.timeout <= 0 {
		return send(req)
	}
	ctx, cancel := context.WithCancelCause(req.Context())
	// The response and the timer race to settle the attempt. Only a timer that
	// settles it first cancels, so a response that wins is never canceled after
	// it is returned.
	var settled atomic.Bool
	timer := time.AfterFunc(p.timeout, func() {
		if settled.CompareAndSwap(false, true) {
			cancel(errResponseTimeout)
		}
	})
	resp, err := send(req.WithContext(ctx))
	timer.Stop()

	if !settled.CompareAndSwap(false, true) {
		closeBody(resp)
		cancel(nil)
		return nil, fmt.Errorf("request timed out: no response within %s", p.timeout)
	}
	if err != nil || resp == nil {
		cancel(nil)
		return resp, err
	}
	resp.Body = cancelOnClose{ReadCloser: resp.Body, cancel: cancel}
	return resp, nil
}

// cancelOnClose releases a request's context once its response body is closed.
type cancelOnClose struct {
	io.ReadCloser
	cancel context.CancelCauseFunc
}

func (b cancelOnClose) Close() error {
	err := b.ReadCloser.Close()
	b.cancel(nil)
	return err
}
