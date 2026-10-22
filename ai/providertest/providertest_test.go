package providertest

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"testing"

	"github.com/ColeBurch/burrow/ai"
)

func TestServerRecordsRequestsAndWritesSSE(t *testing.T) {
	server := NewServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		WriteSSE(w,
			Event{Name: "first", Data: `{"n":1}`},
			Event{Data: "line one\nline two"},
		)
	}))

	req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, "https://provider.test/v1/stream", strings.NewReader(`{"model":"m"}`))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Authorization", "Bearer k")
	resp, err := server.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}

	if got := resp.Header.Get("Content-Type"); got != "text/event-stream" {
		t.Errorf("Content-Type = %q, want text/event-stream", got)
	}
	wantBody := "event: first\ndata: {\"n\":1}\n\ndata: line one\ndata: line two\n\n"
	if string(body) != wantBody {
		t.Errorf("body = %q, want %q", body, wantBody)
	}

	requests := server.Requests()
	if len(requests) != 1 {
		t.Fatalf("got %d requests, want 1", len(requests))
	}
	got := requests[0]
	if got.Method != http.MethodPost || got.Path != "/v1/stream" || got.Header.Get("Authorization") != "Bearer k" {
		t.Errorf("request = %s %s Authorization=%q", got.Method, got.Path, got.Header.Get("Authorization"))
	}
	if model := got.JSON(t)["model"]; model != "m" {
		t.Errorf("body model = %#v, want m", model)
	}
}

func TestRouteSendsRegisteredProviderToServer(t *testing.T) {
	server := NewServer(t, nil)
	const api = ai.API("providertest-route-test")
	sourceID := string(api)
	type call struct {
		api    ai.API
		client *http.Client
		signal context.Context
	}
	var calls []call
	ai.RegisterApiProvider(ai.ApiProvider[ai.API, ai.StreamOptions]{
		Api: api,
		Stream: func(model ai.Model[ai.API], _ ai.ModelContext, options *ai.StreamOptions) (*ai.AssistantMessageEventStream, error) {
			calls = append(calls, call{model.API, options.HTTPClient, options.Signal})
			return ai.NewAssistantMessageEventStream(1), nil
		},
		StreamSimple: func(model ai.Model[ai.API], _ ai.ModelContext, options *ai.SimpleStreamOptions) (*ai.AssistantMessageEventStream, error) {
			calls = append(calls, call{model.API, options.HTTPClient, options.Signal})
			return ai.NewAssistantMessageEventStream(1), nil
		},
	}, &sourceID)
	t.Cleanup(func() { ai.UnregisterApiProvider(sourceID) })

	var routed ai.API
	t.Run("route", func(t *testing.T) {
		model := server.Route(t, ai.Model[ai.API]{ID: "m", API: api})
		routed = model.API
		if routed == api {
			t.Fatalf("routed API = %q, want a test-only API", routed)
		}
		ctx := t.Context()
		if _, err := ai.Stream(model, ai.ModelContext{}, &ai.StreamOptions{Signal: ctx}); err != nil {
			t.Fatal(err)
		}
		if _, err := ai.StreamSimple(model, ai.ModelContext{}, nil); err != nil {
			t.Fatal(err)
		}
		if len(calls) != 2 {
			t.Fatalf("got %d provider calls, want 2", len(calls))
		}
		for i, c := range calls {
			if c.api != api || c.client != server.Client() {
				t.Errorf("call %d: API = %q, client set = %v; want %q and the server's client", i, c.api, c.client == server.Client(), api)
			}
		}
		if calls[0].signal != ctx {
			t.Errorf("Stream options lost the caller's Signal")
		}
	})

	if _, ok := ai.GetApiProvider(routed); ok {
		t.Errorf("routed provider %q is still registered after its test ended", routed)
	}
}

// recordingTB records Errorf calls instead of failing the test.
type recordingTB struct {
	testing.TB
	mu     sync.Mutex
	errors []string
}

func (r *recordingTB) Errorf(format string, args ...any) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.errors = append(r.errors, fmt.Sprintf(format, args...))
}

func TestSequenceServesHandlersInOrderAndFailsPastTheEnd(t *testing.T) {
	tb := &recordingTB{TB: t}
	reply := func(body string) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { io.WriteString(w, body) })
	}
	server := NewServer(t, Sequence(tb, reply("one"), reply("two")))

	var bodies []string
	var statuses []int
	for range 3 {
		resp, err := server.Client().Get("https://provider.test/")
		if err != nil {
			t.Fatal(err)
		}
		body, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			t.Fatal(err)
		}
		bodies = append(bodies, string(body))
		statuses = append(statuses, resp.StatusCode)
	}

	if bodies[0] != "one" || bodies[1] != "two" {
		t.Errorf("bodies = %q, want one then two", bodies[:2])
	}
	if statuses[2] != http.StatusInternalServerError {
		t.Errorf("third status = %d, want 500", statuses[2])
	}
	tb.mu.Lock()
	defer tb.mu.Unlock()
	if len(tb.errors) != 1 {
		t.Errorf("got %d test errors, want 1 for the unexpected request: %q", len(tb.errors), tb.errors)
	}
}

func TestMatchServesFirstMatchingCaseAndFailsOnNone(t *testing.T) {
	tb := &recordingTB{TB: t}
	reply := func(body string) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// The handler still sees the body Match read.
			got, _ := io.ReadAll(r.Body)
			io.WriteString(w, body+":"+string(got))
		})
	}
	server := NewServer(t, Match(tb,
		Case{When: BodyContains("summary"), Handler: reply("first")},
		Case{When: BodyContains("summary prefix"), Handler: reply("unreachable")},
		Case{When: BodyContains("prompt"), Handler: reply("second")},
	))

	var bodies []string
	var statuses []int
	for _, sent := range []string{"prompt", "summary prefix", "other"} {
		resp, err := server.Client().Post("https://provider.test/", "text/plain", strings.NewReader(sent))
		if err != nil {
			t.Fatal(err)
		}
		body, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			t.Fatal(err)
		}
		bodies = append(bodies, string(body))
		statuses = append(statuses, resp.StatusCode)
	}

	if bodies[0] != "second:prompt" || bodies[1] != "first:summary prefix" {
		t.Errorf("bodies = %q, want second:prompt then first:summary prefix", bodies[:2])
	}
	if statuses[2] != http.StatusInternalServerError {
		t.Errorf("unmatched status = %d, want 500", statuses[2])
	}
	tb.mu.Lock()
	defer tb.mu.Unlock()
	if len(tb.errors) != 1 {
		t.Errorf("got %d test errors, want 1 for the unmatched request: %q", len(tb.errors), tb.errors)
	}
}
