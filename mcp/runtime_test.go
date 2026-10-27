package mcp

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ColeBurch/burrow/baseAgent/authstoretest"
	sdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

// httpMCP serves an MCP server over streamable HTTP. It can restart, so
// the server forgets its sessions, and fail requests with 503.
type httpMCP struct {
	*httptest.Server
	server  *sdk.Server
	handler atomic.Pointer[sdk.StreamableHTTPHandler]

	mu sync.Mutex
	// fail is how many more requests of each JSON-RPC method fail with 503.
	fail map[string]int
	// requests counts the requests of each method.
	requests map[string]int
	// hold, when set, holds initialize requests until their client gives
	// up, after signaling holding.
	hold    bool
	holding chan struct{}
}

func newHTTPMCP(t *testing.T, server *sdk.Server) *httpMCP {
	m := &httpMCP{server: server, fail: map[string]int{}, requests: map[string]int{}, holding: make(chan struct{}, 1)}
	m.restart()
	m.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			body, _ := io.ReadAll(r.Body)
			r.Body = io.NopCloser(bytes.NewReader(body))
			var message struct {
				Method string `json:"method"`
			}
			_ = json.Unmarshal(body, &message)
			m.mu.Lock()
			m.requests[message.Method]++
			fail := m.fail[message.Method] > 0
			if fail {
				m.fail[message.Method]--
			}
			hold := m.hold && message.Method == "initialize"
			m.mu.Unlock()
			if fail {
				http.Error(w, "overloaded", http.StatusServiceUnavailable)
				return
			}
			if hold {
				m.holding <- struct{}{}
				<-r.Context().Done()
				return
			}
		}
		m.handler.Load().ServeHTTP(w, r)
	}))
	t.Cleanup(m.Close)
	return m
}

// restart replaces the handler, which forgets the sessions.
func (m *httpMCP) restart() {
	m.handler.Store(sdk.NewStreamableHTTPHandler(func(*http.Request) *sdk.Server { return m.server }, nil))
}

func (m *httpMCP) failNext(method string, n int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.fail[method] = n
}

func (m *httpMCP) count(method string) int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.requests[method]
}

// echoServer offers an echo tool and resources.
func echoServer() *sdk.Server {
	server := newResourceServer(nil)
	server.AddTool(&sdk.Tool{Name: "echo", InputSchema: objectSchema},
		func(_ context.Context, req *sdk.CallToolRequest) (*sdk.CallToolResult, error) {
			var args struct{ Q string }
			_ = json.Unmarshal(req.Params.Arguments, &args)
			return textResult(args.Q), nil
		})
	return server
}

func echo(t *testing.T, ctx context.Context, connection *ServerConnection, q string) error {
	t.Helper()
	result, err := connection.CallTool(ctx, "echo", map[string]any{"q": q}, nil)
	if err != nil {
		return err
	}
	if got := textOf(ConvertResult("s", "echo", result, nil).Content); got != q {
		t.Fatalf("echo = %q, want %q", got, q)
	}
	return nil
}

func TestConnectionReportsServer(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	server := sdk.NewServer(&sdk.Implementation{Name: "docs", Version: "1.0.0"}, &sdk.ServerOptions{Instructions: "  Search the docs.\n"})
	server.AddTool(&sdk.Tool{Name: "search", InputSchema: objectSchema}, linkTool)
	var changes atomic.Int32
	connection, _ := newConnection(t, server, ServerConnectionOptions{OnChange: func(*ServerConnection) { changes.Add(1) }})
	if got := connection.Status(); got.State != StateConnecting {
		t.Fatalf("status before connecting = %+v", got)
	}
	if _, err := connection.GetClient(ctx); err != nil {
		t.Fatal(err)
	}
	if got := connection.Status(); got != (ServerStatus{State: StateConnected}) {
		t.Fatalf("status = %+v", got)
	}
	if tools := connection.Tools(); len(tools) != 1 || tools[0].Name != "search" {
		t.Fatalf("tools = %v", tools)
	}
	if connection.HasResources() || connection.Instructions() != "Search the docs." || connection.Timeout() != DefaultTimeout {
		t.Fatalf("resources %v, instructions %q, timeout %s", connection.HasResources(), connection.Instructions(), connection.Timeout())
	}
	if changes.Load() < 2 {
		t.Fatalf("%d changes, want connecting and connected", changes.Load())
	}

	// A server without tools is not asked for them.
	resourceOnly := sdk.NewServer(&sdk.Implementation{Name: "files", Version: "1.0.0"}, nil)
	resourceOnly.AddResource(&sdk.Resource{URI: "file:///a.txt", Name: "a.txt"},
		func(context.Context, *sdk.ReadResourceRequest) (*sdk.ReadResourceResult, error) { return nil, nil })
	m := newHTTPMCP(t, resourceOnly)
	resources, _ := newConnection(t, nil, ServerConnectionOptions{CreateTransport: HTTPTransport(m.URL, nil)})
	if _, err := resources.GetClient(ctx); err != nil {
		t.Fatal(err)
	}
	if !resources.HasResources() || resources.Tools() != nil || m.count("tools/list") != 0 {
		t.Fatalf("resources %v, tools %v, %d tools/list requests", resources.HasResources(), resources.Tools(), m.count("tools/list"))
	}

	if _, err := NewServerConnection(ServerConnectionOptions{CreateTransport: (&inMemory{}).transport}); err == nil {
		t.Fatal("connection without a name")
	}
	if _, err := NewServerConnection(ServerConnectionOptions{Name: "s"}); err == nil {
		t.Fatal("connection without a transport")
	}
}

// Concurrent callers share one connect.
func TestGetClientSharesConnect(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	connection, memory := newConnection(t, echoServer(), ServerConnectionOptions{})
	var wg sync.WaitGroup
	sessions := make([]*sdk.ClientSession, 8)
	for i := range sessions {
		wg.Go(func() {
			session, err := connection.GetClient(ctx)
			if err != nil {
				t.Error(err)
			}
			sessions[i] = session
		})
	}
	wg.Wait()
	for _, session := range sessions {
		if session != sessions[0] {
			t.Fatal("callers got different sessions")
		}
	}
	if memory.connects() != 1 {
		t.Fatalf("%d connects", memory.connects())
	}
}

// A dropped connection shows as disconnected, and the next call reconnects.
func TestReconnectsAfterDisconnect(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	connection, memory := newConnection(t, echoServer(), ServerConnectionOptions{})
	if err := echo(t, ctx, connection, "one"); err != nil {
		t.Fatal(err)
	}
	memory.mu.Lock()
	first := memory.sessions[0]
	memory.mu.Unlock()
	first.Close()
	eventually(t, "disconnected", func() bool { return connection.Status().State == StateDisconnected })
	if got := connection.Status().Error; got != "Connection closed" {
		t.Fatalf("error = %q", got)
	}
	if err := echo(t, ctx, connection, "two"); err != nil {
		t.Fatal(err)
	}
	if memory.connects() != 2 || connection.Status().State != StateConnected {
		t.Fatalf("%d connects, status %+v", memory.connects(), connection.Status())
	}

	// Reconnect replaces a working connection.
	if err := connection.Reconnect(ctx); err != nil {
		t.Fatal(err)
	}
	if err := echo(t, ctx, connection, "three"); err != nil {
		t.Fatal(err)
	}
	if memory.connects() != 3 {
		t.Fatalf("%d connects after Reconnect", memory.connects())
	}
}

// A request on a session the server forgot (a restart) is retried once on
// a new session, tool calls included, since the server did not run it.
func TestRetriesOnNewSessionWhenSessionMissing(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	m := newHTTPMCP(t, echoServer())
	connection, _ := newConnection(t, nil, ServerConnectionOptions{CreateTransport: HTTPTransport(m.URL, nil)})
	if err := echo(t, ctx, connection, "before"); err != nil {
		t.Fatal(err)
	}
	m.restart()
	if err := echo(t, ctx, connection, "after"); err != nil {
		t.Fatal(err)
	}
	if got := m.count("initialize"); got != 2 {
		t.Fatalf("%d initialize requests, want 2", got)
	}
	m.restart()
	if _, err := connection.ReadResource(ctx, "file:///a.txt", nil); err != nil {
		t.Fatal(err)
	}
	if connection.Status().State != StateConnected {
		t.Fatalf("status = %+v", connection.Status())
	}
}

// Reads are retried once after a transient HTTP error; tool calls are not,
// since the tool may have run.
func TestRetriesReadsAfterTransientError(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	m := newHTTPMCP(t, echoServer())
	connection, _ := newConnection(t, nil, ServerConnectionOptions{CreateTransport: HTTPTransport(m.URL, nil)})
	if _, err := connection.GetClient(ctx); err != nil {
		t.Fatal(err)
	}

	m.failNext("resources/read", 1)
	if _, err := connection.ReadResource(ctx, "file:///a.txt", nil); err != nil {
		t.Fatal(err)
	}
	if got := m.count("resources/read"); got != 2 {
		t.Fatalf("%d reads, want 2", got)
	}
	m.failNext("resources/read", 2)
	if _, err := connection.ReadResource(ctx, "file:///a.txt", nil); !isTransientHTTPError(err) {
		t.Fatalf("read failing twice: %v", err)
	}

	m.failNext("tools/call", 1)
	if err := echo(t, ctx, connection, "x"); !isTransientHTTPError(err) {
		t.Fatalf("tool call: %v", err)
	}
	if got := m.count("tools/call"); got != 1 {
		t.Fatalf("%d tool calls, want 1", got)
	}
	// The connection survives transient errors.
	if err := echo(t, ctx, connection, "y"); err != nil {
		t.Fatal(err)
	}
	if got := m.count("initialize"); got != 1 {
		t.Fatalf("%d initialize requests, want 1", got)
	}
}

// Connecting to an HTTP server is retried after transient errors, twice.
func TestRetriesConnectAfterTransientError(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	m := newHTTPMCP(t, echoServer())
	connection, _ := newConnection(t, nil, ServerConnectionOptions{CreateTransport: HTTPTransport(m.URL, nil)})
	m.failNext("initialize", 1)
	if err := echo(t, ctx, connection, "x"); err != nil {
		t.Fatal(err)
	}
	if got := m.count("initialize"); got != 2 {
		t.Fatalf("%d initialize requests, want 2", got)
	}

	failing, _ := newConnection(t, nil, ServerConnectionOptions{Name: "down", CreateTransport: HTTPTransport(m.URL, nil)})
	m.failNext("initialize", 10)
	_, err := failing.GetClient(ctx)
	if err == nil || !strings.Contains(err.Error(), `MCP server "down" failed to connect`) || !isTransientHTTPError(err) {
		t.Fatalf("err = %v", err)
	}
	if got := m.count("initialize"); got != 2+1+len(connectRetryDelays) {
		t.Fatalf("%d initialize requests, want %d", got, 2+1+len(connectRetryDelays))
	}
	if status := failing.Status(); status.State != StateFailed || status.Error == "" {
		t.Fatalf("status = %+v", status)
	}

	// Other transports are not retried.
	attempts := 0
	other, _ := newConnection(t, nil, ServerConnectionOptions{CreateTransport: func(*OAuthHandler) (sdk.Transport, error) {
		attempts++
		return &sdk.StreamableClientTransport{Endpoint: m.URL}, errors.New("no transport")
	}})
	if _, err := other.GetClient(ctx); err == nil || attempts != 1 {
		t.Fatalf("err %v after %d attempts", err, attempts)
	}
}

func TestRequestTimeout(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	server := sdk.NewServer(&sdk.Implementation{Name: "slow", Version: "1.0.0"}, nil)
	server.AddTool(&sdk.Tool{Name: "wait", InputSchema: objectSchema},
		func(ctx context.Context, _ *sdk.CallToolRequest) (*sdk.CallToolResult, error) {
			<-ctx.Done()
			return nil, ctx.Err()
		})
	connection, _ := newConnection(t, server, ServerConnectionOptions{})
	_, err := connection.CallTool(ctx, "wait", nil, &RequestOptions{Timeout: 50 * time.Millisecond})
	if !errors.Is(err, context.DeadlineExceeded) || !strings.Contains(err.Error(), "request timed out after 50ms") {
		t.Fatalf("err = %v", err)
	}
	// The caller's own cancellation is not reported as a timeout.
	cancelled, cancelCall := context.WithCancel(ctx)
	time.AfterFunc(50*time.Millisecond, cancelCall)
	_, err = connection.CallTool(cancelled, "wait", nil, &RequestOptions{Timeout: time.Minute})
	if err == nil || strings.Contains(err.Error(), "timed out") {
		t.Fatalf("cancelled call: %v", err)
	}
	if connection.Status().State != StateConnected {
		t.Fatalf("status = %+v", connection.Status())
	}
}

// Close stops a connect in progress and refuses later calls.
func TestCloseStopsConnect(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	m := newHTTPMCP(t, echoServer())
	m.hold = true
	connection, err := NewServerConnection(ServerConnectionOptions{Name: "s", CreateTransport: HTTPTransport(m.URL, nil)})
	if err != nil {
		t.Fatal(err)
	}
	connected := make(chan error, 1)
	go func() {
		_, err := connection.GetClient(ctx)
		connected <- err
	}()
	<-m.holding
	if err := connection.Close(ctx); err != nil {
		t.Fatal(err)
	}
	if err := <-connected; err == nil {
		t.Fatal("connect succeeded after Close")
	}
	if status := connection.Status(); status.State != StateClosed {
		t.Fatalf("status = %+v", status)
	}
	if _, err := connection.GetClient(ctx); err == nil || !strings.Contains(err.Error(), "shut down") {
		t.Fatalf("GetClient after Close: %v", err)
	}
	if got := m.count("initialize"); got != 1 {
		t.Fatalf("%d initialize requests: Close retried the connect", got)
	}
}

// A caller that stops waiting does not stop the connect others share.
func TestGetClientCancelStopsWaitingOnly(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	m := newHTTPMCP(t, echoServer())
	m.hold = true
	connection, _ := newConnection(t, nil, ServerConnectionOptions{CreateTransport: HTTPTransport(m.URL, nil), Timeout: 300 * time.Millisecond})
	waiting, stop := context.WithCancel(ctx)
	done := make(chan error, 1)
	go func() {
		_, err := connection.GetClient(waiting)
		done <- err
	}()
	<-m.holding
	stop()
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v", err)
	}
	if connection.Status().State != StateConnecting {
		t.Fatalf("status = %+v", connection.Status())
	}
	// The connect times out on its own.
	eventually(t, "the connect to fail", func() bool { return connection.Status().State == StateFailed })
}

func TestIsTransientError(t *testing.T) {
	for _, tc := range []struct {
		err  error
		want bool
	}{
		{&netError{}, true},
		{context.Canceled, false},
		{context.DeadlineExceeded, false},
		{ErrSignInRequired, false},
		{errors.New("Service Unavailable"), false},
	} {
		if got := isTransientError(tc.err); got != tc.want {
			t.Errorf("isTransientError(%v) = %v", tc.err, got)
		}
	}
}

type netError struct{}

func (*netError) Error() string   { return "connection refused" }
func (*netError) Timeout() bool   { return false }
func (*netError) Temporary() bool { return false }

// Configured headers reach the server, as pi's headers setting does, but do
// not replace the headers the transport sets.
func TestHTTPTransportHeaders(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	handler := sdk.NewStreamableHTTPHandler(func(*http.Request) *sdk.Server { return echoServer() }, nil)
	var accepts sync.Map
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Api-Key") != "secret" {
			http.Error(w, "no API key", http.StatusUnauthorized)
			return
		}
		accepts.Store(r.Header.Get("Accept"), true)
		handler.ServeHTTP(w, r)
	}))
	// After the connections close, which the cleanups of newConnection do.
	t.Cleanup(server.Close)
	connection, _ := newConnection(t, nil, ServerConnectionOptions{CreateTransport: HTTPTransport(server.URL, &HTTPTransportOptions{
		Headers: map[string]string{"x-api-key": "secret", "Accept": "text/plain"},
	})})
	result, err := connection.CallTool(ctx, "echo", map[string]any{"q": "hi"}, nil)
	if err != nil || textOf(ConvertResult("s", "echo", result, nil).Content) != "hi" {
		t.Fatalf("call = %v, %v", result, err)
	}
	if _, ok := accepts.Load("text/plain"); ok {
		t.Fatal("a configured header replaced the transport's Accept")
	}

	for name, value := range map[string]string{"Bad Name": "v", "": "v", "X-Ok": "secret\r\nInjected: yes"} {
		failing, _ := newConnection(t, nil, ServerConnectionOptions{CreateTransport: HTTPTransport(server.URL, &HTTPTransportOptions{
			Headers: map[string]string{name: value},
		})})
		_, err := failing.GetClient(ctx)
		if err == nil || !strings.Contains(err.Error(), "invalid HTTP header") || strings.Contains(err.Error(), "secret") {
			t.Errorf("header %q: %v", name, err)
		}
	}
}

// The OAuth access token wins over a configured Authorization header.
func TestHTTPTransportHeadersWithOAuth(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	f := newFakeOAuthServer(t, nil)
	h := newHandler(t, f, authstoretest.NewMemoryStore(), &fakeClock{now: time.Now()})
	signIn(t, ctx, h)
	connection, _ := newConnection(t, nil, ServerConnectionOptions{Name: "fake", OAuth: h, CreateTransport: HTTPTransport(f.URL+"/mcp", &HTTPTransportOptions{
		Headers: map[string]string{"Authorization": "Bearer configured"},
	})})
	if err := callWhoami(ctx, mustClient(t, ctx, connection)); err != nil {
		t.Fatal(err)
	}
}

// Headers go only to the server's origin, not to where it redirects.
func TestHeadersStayOnOrigin(t *testing.T) {
	var elsewhere atomic.Value
	other := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		elsewhere.Store(r.Header.Get("X-Api-Key"))
	}))
	defer other.Close()
	var here atomic.Value
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		here.Store(r.Header.Get("X-Api-Key"))
		http.Redirect(w, r, other.URL+"/landing", http.StatusTemporaryRedirect)
	}))
	defer server.Close()
	client, err := withHeaders(nil, server.URL+"/mcp", map[string]string{"X-Api-Key": "secret"})
	if err != nil {
		t.Fatal(err)
	}
	req, _ := http.NewRequestWithContext(t.Context(), http.MethodGet, server.URL+"/mcp", nil)
	resp, err := client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if here.Load() != "secret" || elsewhere.Load() != "" {
		t.Fatalf("API key at the server %q, after the redirect %q", here.Load(), elsewhere.Load())
	}
	if req.Header.Get("X-Api-Key") != "" {
		t.Fatal("the caller's request was modified")
	}
}
