// Package providertest fakes model provider APIs in tests. A Server runs on
// httptest's in-memory network and records every request. WriteSSE and
// WriteJSON write responses in any provider's wire format; the event and error
// bodies themselves are up to each test.
//
// Pass the server's client to code that accepts ai.StreamOptions:
//
//	server := providertest.NewServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
//		providertest.WriteSSE(w, providertest.Event{Name: "done", Data: `{"type":"done"}`})
//	}))
//	stream, err := ai.Stream(model, modelContext, &ai.StreamOptions{HTTPClient: server.Client()})
//
// Code that builds its own options, such as an agent, can use a model returned
// by Server.Route instead.
package providertest

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/ColeBurch/burrow/ai"
)

// Server is a fake provider API. The embedded httptest.Server's Client sends
// requests for any host to it.
type Server struct {
	*httptest.Server

	mu       sync.Mutex
	requests []Request
}

// Request is a request the server received.
type Request struct {
	Method string
	Path   string
	Header http.Header
	Body   []byte
}

// NewServer starts a Server that records each request and then passes it to
// handler. The server shuts down when the test ends.
func NewServer(t testing.TB, handler http.Handler) *Server {
	t.Helper()
	s := &Server{}
	s.Server = httptest.NewTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("providertest: read request body: %v", err)
		}
		r.Body = io.NopCloser(bytes.NewReader(body))
		s.mu.Lock()
		s.requests = append(s.requests, Request{
			Method: r.Method,
			Path:   r.URL.Path,
			Header: r.Header.Clone(),
			Body:   body,
		})
		s.mu.Unlock()
		handler.ServeHTTP(w, r)
	}))
	// The client may dial a connection it never uses, such as when a request
	// starts just before the previous response's connection returns to the
	// pool. Closing the server then aborts that connection's TLS handshake,
	// which the server would log as an error.
	s.Config.ErrorLog = log.New(skipHandshakeErrors{os.Stderr}, "", log.LstdFlags)
	return s
}

// skipHandshakeErrors drops TLS handshake errors from a server's log.
type skipHandshakeErrors struct{ w io.Writer }

func (s skipHandshakeErrors) Write(p []byte) (int, error) {
	if bytes.Contains(p, []byte("http: TLS handshake error")) {
		return len(p), nil
	}
	return s.w.Write(p)
}

// Requests returns the requests received so far, oldest first.
func (s *Server) Requests() []Request {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Clone(s.requests)
}

// JSON decodes the request body as a JSON object. It fails t if it is not one.
func (r Request) JSON(t testing.TB) map[string]any {
	t.Helper()
	var body map[string]any
	if err := json.Unmarshal(r.Body, &body); err != nil {
		t.Fatalf("providertest: decode %s %s body: %v", r.Method, r.Path, err)
	}
	return body
}

var routeCount atomic.Int64

// Route returns a copy of model whose requests go to s. It registers a provider
// under a test-only API that sets HTTPClient and calls the provider registered
// for model.API with the original model. The registration is removed when the
// test ends.
func (s *Server) Route(t testing.TB, model ai.Model[ai.API]) ai.Model[ai.API] {
	t.Helper()
	original := model.API
	inner, ok := ai.GetApiProvider(original)
	if !ok {
		t.Fatalf("providertest: no provider registered for API %q", original)
	}

	routed := ai.API(fmt.Sprintf("%s-providertest-%d", original, routeCount.Add(1)))
	sourceID := string(routed)
	ai.RegisterApiProvider(ai.ApiProvider[ai.API, ai.StreamOptions]{
		Api: routed,
		Stream: func(model ai.Model[ai.API], modelContext ai.ModelContext, options *ai.StreamOptions) (*ai.AssistantMessageEventStream, error) {
			var routedOptions ai.StreamOptions
			if options != nil {
				routedOptions = *options
			}
			routedOptions.HTTPClient = s.Client()
			model.API = original
			return inner.Stream(model, modelContext, &routedOptions)
		},
		StreamSimple: func(model ai.Model[ai.API], modelContext ai.ModelContext, options *ai.SimpleStreamOptions) (*ai.AssistantMessageEventStream, error) {
			var routedOptions ai.SimpleStreamOptions
			if options != nil {
				routedOptions = *options
			}
			routedOptions.HTTPClient = s.Client()
			model.API = original
			return inner.StreamSimple(model, modelContext, &routedOptions)
		},
	}, &sourceID)
	t.Cleanup(func() { ai.UnregisterApiProvider(sourceID) })

	model.API = routed
	return model
}

// Sequence returns a handler that serves the nth request with handlers[n], for
// tests that expect a fixed series of requests. It fails t and responds 500 to
// any request past the last handler.
func Sequence(t testing.TB, handlers ...http.Handler) http.Handler {
	var next atomic.Int64
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := int(next.Add(1))
		if n > len(handlers) {
			t.Errorf("providertest: unexpected request %d to %s %s; the sequence has %d", n, r.Method, r.URL.Path, len(handlers))
			http.Error(w, "providertest: unexpected request", http.StatusInternalServerError)
			return
		}
		handlers[n-1].ServeHTTP(w, r)
	})
}

// Case is a handler for the requests that When accepts.
type Case struct {
	When    func(Request) bool
	Handler http.Handler
}

// Match returns a handler that serves each request with the first case that
// accepts it, for tests whose requests have no fixed order, such as ones sent
// in parallel. It fails t and responds 500 to a request no case accepts.
func Match(t testing.TB, cases ...Case) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("providertest: read request body: %v", err)
		}
		r.Body = io.NopCloser(bytes.NewReader(body))
		request := Request{Method: r.Method, Path: r.URL.Path, Header: r.Header.Clone(), Body: body}
		for _, c := range cases {
			if c.When(request) {
				c.Handler.ServeHTTP(w, r)
				return
			}
		}
		t.Errorf("providertest: no case matches request to %s %s: %s", r.Method, r.URL.Path, body)
		http.Error(w, "providertest: unexpected request", http.StatusInternalServerError)
	})
}

// BodyContains reports whether a request body contains s, for use as a
// Case.When.
func BodyContains(s string) func(Request) bool {
	return func(r Request) bool { return bytes.Contains(r.Body, []byte(s)) }
}

// Event is one server-sent event. Name sets the event field and may be empty.
type Event struct {
	Name string
	Data string
}

// WriteSSE writes events as a text/event-stream response. It flushes after
// each event, so the client sees it before the handler blocks or returns.
func WriteSSE(w http.ResponseWriter, events ...Event) {
	w.Header().Set("Content-Type", "text/event-stream")
	flusher, _ := w.(http.Flusher)
	for _, event := range events {
		if event.Name != "" {
			fmt.Fprintf(w, "event: %s\n", event.Name)
		}
		for line := range strings.SplitSeq(event.Data, "\n") {
			fmt.Fprintf(w, "data: %s\n", line)
		}
		fmt.Fprint(w, "\n")
		if flusher != nil {
			flusher.Flush()
		}
	}
}

// WriteJSON writes body as a JSON response with the given status.
func WriteJSON(w http.ResponseWriter, status int, body string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	io.WriteString(w, body)
}
