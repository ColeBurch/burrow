package ai

import (
	"context"
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"
	"testing"
	"testing/synctest"
	"time"
)

func TestRequestedDelay(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		// Bubble time stands still, so a date two minutes out is exactly 120s.
		at := time.Now().Add(2 * time.Minute).UTC()
		tests := []struct {
			name   string
			header http.Header
			want   time.Duration
			wantOK bool
		}{
			{name: "Retry-After-Ms", header: http.Header{"Retry-After-Ms": {"1500.5"}}, want: 1500500 * time.Microsecond, wantOK: true},
			{name: "Retry-After-Ms wins", header: http.Header{"Retry-After-Ms": {"10"}, "Retry-After": {"30"}}, want: 10 * time.Millisecond, wantOK: true},
			{name: "bad Retry-After-Ms falls back", header: http.Header{"Retry-After-Ms": {"soon"}, "Retry-After": {"30"}}, want: 30 * time.Second, wantOK: true},
			{name: "seconds", header: http.Header{"Retry-After": {"30"}}, want: 30 * time.Second, wantOK: true},
			// RFC 9110 lets a Retry-After date take any of the three HTTP date formats.
			{name: "IMF-fixdate", header: http.Header{"Retry-After": {at.Format(http.TimeFormat)}}, want: 2 * time.Minute, wantOK: true},
			{name: "RFC 850 date", header: http.Header{"Retry-After": {at.Format("Monday, 02-Jan-06 15:04:05 GMT")}}, want: 2 * time.Minute, wantOK: true},
			{name: "asctime date", header: http.Header{"Retry-After": {at.Format(time.ANSIC)}}, want: 2 * time.Minute, wantOK: true},
			{name: "past date", header: http.Header{"Retry-After": {at.Add(-time.Hour).Format(http.TimeFormat)}}, want: -58 * time.Minute, wantOK: true},
			{name: "unparsable", header: http.Header{"Retry-After": {"later"}}},
			{name: "none", header: http.Header{}},
		}
		for _, tt := range tests {
			got, ok := requestedDelay(tt.header)
			if got != tt.want || ok != tt.wantOK {
				t.Errorf("%s: requestedDelay = %v, %v; want %v, %v", tt.name, got, ok, tt.want, tt.wantOK)
			}
		}
	})
}

func TestRetryDelayBackoff(t *testing.T) {
	for attempt, base := range []time.Duration{
		500 * time.Millisecond, time.Second, 2 * time.Second, 4 * time.Second, 8 * time.Second, 8 * time.Second,
	} {
		for range 50 {
			delay, requested := retryDelay(nil, attempt)
			if requested || delay > base || delay <= base*3/4 {
				t.Fatalf("attempt %d: retryDelay = %v, %v; want unrequested and in (%v, %v]", attempt, delay, requested, base*3/4, base)
			}
		}
	}
	// A requested delay in the past means retry now.
	resp := &http.Response{Header: http.Header{"Retry-After": {"-5"}}}
	if delay, requested := retryDelay(resp, 0); delay != 0 || !requested {
		t.Errorf("retryDelay for a negative Retry-After = %v, %v; want 0, true", delay, requested)
	}
}

// Backoff stays at the cap however many retries are allowed. Shifting the
// initial delay by the attempt number overflowed from attempt 35, and the
// jitter then panicked on a delay that was not positive.
func TestRetryDelayBackoffLateAttempts(t *testing.T) {
	for _, attempt := range []int{5, 34, 35, 62, 63, 64, 1000} {
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("attempt %d: retryDelay panicked: %v", attempt, r)
				}
			}()
			delay, requested := retryDelay(nil, attempt)
			if requested || delay > maxRetryBackoff || delay <= maxRetryBackoff*3/4 {
				t.Errorf("attempt %d: retryDelay = %v, %v; want unrequested and in (%v, %v]", attempt, delay, requested, maxRetryBackoff*3/4, maxRetryBackoff)
			}
		}()
	}
}

// A request whose body cannot be read again is sent once.
func TestRetryPolicyDoesNotResendUnrewindableBody(t *testing.T) {
	req, err := http.NewRequest(http.MethodPost, "https://api.test/", io.NopCloser(strings.NewReader("{}")))
	if err != nil {
		t.Fatal(err)
	}
	if req.GetBody != nil {
		t.Fatal("test request has GetBody; want a body that cannot be read again")
	}
	sends := 0
	resp, err := retryPolicy{maxRetries: 2}.do(req, func(*http.Request) (*http.Response, error) {
		sends++
		return &http.Response{StatusCode: http.StatusInternalServerError, Header: http.Header{}, Body: http.NoBody}, nil
	})
	if err != nil || resp.StatusCode != http.StatusInternalServerError {
		t.Fatalf("do = %v, %v; want the 500 response", resp, err)
	}
	if sends != 1 {
		t.Errorf("sent %d times, want 1", sends)
	}
}

// When a response arrives just as the timeout fires, the attempt must settle on
// one outcome: a timeout, or a response whose context the timer never cancels.
// Handing back the response and then canceling it would kill a good stream.
func TestRetryPolicyAttemptTimeoutRace(t *testing.T) {
	// Each run is a tie; the scheduler picks the winner, so run enough ties to
	// hit both orders.
	for range 200 {
		synctest.Test(t, func(t *testing.T) {
			var attemptCtx context.Context
			req, err := http.NewRequest(http.MethodGet, "https://api.test/", nil)
			if err != nil {
				t.Fatal(err)
			}
			resp, err := retryPolicy{timeout: time.Second}.attempt(req, func(r *http.Request) (*http.Response, error) {
				attemptCtx = r.Context()
				time.Sleep(time.Second)
				return &http.Response{StatusCode: http.StatusOK, Header: http.Header{}, Body: http.NoBody}, nil
			})
			// Let a timer that already fired finish.
			synctest.Wait()
			if err != nil {
				return
			}
			if attemptCtx.Err() != nil {
				t.Fatalf("attempt returned a response, then its context was canceled: %v", context.Cause(attemptCtx))
			}
			resp.Body.Close()
		})
	}
}

func TestNewRetryPolicy(t *testing.T) {
	ms := func(v int64) *int64 { return &v }
	// The largest millisecond count a time.Duration can hold, about 292 years.
	const maxMs = math.MaxInt64 / int64(time.Millisecond)
	tests := []struct {
		name    string
		options StreamOptions
		want    retryPolicy
		wantErr string
	}{
		{
			name:    "nil uses the defaults",
			options: StreamOptions{},
			want:    retryPolicy{maxRetries: DefaultMaxRetries, maxDelay: DefaultMaxRetriesDelay, timeout: DefaultTimeout},
		},
		{
			name:    "zero means no limit",
			options: StreamOptions{MaxRetries: ms(0), TimeoutMs: ms(0), MaxRetriesDelayMs: ms(0)},
			want:    retryPolicy{},
		},
		{
			name:    "values are milliseconds",
			options: StreamOptions{MaxRetries: ms(5), TimeoutMs: ms(1500), MaxRetriesDelayMs: ms(30000)},
			want:    retryPolicy{maxRetries: 5, maxDelay: 30 * time.Second, timeout: 1500 * time.Millisecond},
		},
		{
			name:    "largest duration",
			options: StreamOptions{TimeoutMs: ms(maxMs), MaxRetriesDelayMs: ms(maxMs)},
			want: retryPolicy{
				maxRetries: DefaultMaxRetries,
				maxDelay:   time.Duration(maxMs) * time.Millisecond,
				timeout:    time.Duration(maxMs) * time.Millisecond,
			},
		},
		{name: "negative MaxRetries", options: StreamOptions{MaxRetries: ms(-1)}, wantErr: "MaxRetries must not be negative, got -1"},
		{name: "negative TimeoutMs", options: StreamOptions{TimeoutMs: ms(-1)}, wantErr: "TimeoutMs must not be negative, got -1"},
		{name: "negative MaxRetriesDelayMs", options: StreamOptions{MaxRetriesDelayMs: ms(-1)}, wantErr: "MaxRetriesDelayMs must not be negative, got -1"},
		// Larger values would wrap: math.MaxInt64 to -1ms, which disabled the
		// limit, and 18446744073710 to 448µs, a near-instant timeout.
		{
			name:    "TimeoutMs too large",
			options: StreamOptions{TimeoutMs: ms(maxMs + 1)},
			wantErr: fmt.Sprintf("TimeoutMs must be at most %d, got %d", maxMs, maxMs+1),
		},
		{
			name:    "TimeoutMs that wraps to a short timeout",
			options: StreamOptions{TimeoutMs: ms(18_446_744_073_710)},
			wantErr: fmt.Sprintf("TimeoutMs must be at most %d, got 18446744073710", maxMs),
		},
		{
			name:    "MaxRetriesDelayMs too large",
			options: StreamOptions{MaxRetriesDelayMs: ms(math.MaxInt64)},
			wantErr: fmt.Sprintf("MaxRetriesDelayMs must be at most %d, got %d", maxMs, int64(math.MaxInt64)),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := newRetryPolicy(tt.options)
			if tt.wantErr != "" {
				if err == nil || err.Error() != tt.wantErr {
					t.Fatalf("newRetryPolicy error = %v, want %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if got != tt.want {
				t.Errorf("newRetryPolicy = %+v, want %+v", got, tt.want)
			}
		})
	}
}
