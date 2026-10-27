// Adapted from pi's packages/coding-agent/src/extensions/mcp/runtime.ts
// (MIT, see LICENSE).

package mcp

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"iter"
	"maps"
	"net"
	"net/http"
	"net/url"
	"runtime/debug"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/modelcontextprotocol/go-sdk/jsonrpc"
	sdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

// DefaultTimeout bounds connecting and each request when
// ServerConnectionOptions.Timeout is zero.
const DefaultTimeout = 60 * time.Second

// connectRetryDelays are the delays between attempts to connect to an HTTP
// server that failed with a transient error.
var connectRetryDelays = []time.Duration{250 * time.Millisecond, time.Second}

// ServerState is the state of a [ServerConnection].
type ServerState string

const (
	StateConnecting ServerState = "connecting"
	StateConnected  ServerState = "connected"
	// StateDisconnected means the connection dropped, for example because
	// the stdio server exited. The next call reconnects.
	StateDisconnected ServerState = "disconnected"
	// StateNeedsAuth means the user must sign in; see [ErrSignInRequired].
	StateNeedsAuth ServerState = "needs-auth"
	StateFailed    ServerState = "failed"
	StateClosed    ServerState = "closed"
)

// ServerStatus is the state of a [ServerConnection] and why it is in it.
type ServerStatus struct {
	State ServerState
	// Error says why the server failed or disconnected, or that refreshing
	// its tools failed.
	Error string
}

// TransportFactory returns a new transport for each connection attempt.
// When oauth is not nil, the transport must authorize its requests with it,
// and send them with its HTTPClient.
type TransportFactory func(oauth *OAuthHandler) (sdk.Transport, error)

// HTTPTransportOptions configures [HTTPTransport].
type HTTPTransportOptions struct {
	// HTTPClient sends the requests. Nil means a default client. Apps that
	// let users add servers should pass a client that refuses private
	// addresses.
	HTTPClient *http.Client
	// Headers are sent with each request to the endpoint's origin, for
	// example an API key for a server without OAuth. Headers the transport
	// sets itself take precedence, among them the OAuth access token once the
	// user has signed in, as in pi.
	Headers map[string]string
}

// HTTPTransport connects to a streamable HTTP endpoint. Nil opts means the
// default client and no headers.
func HTTPTransport(endpoint string, opts *HTTPTransportOptions) TransportFactory {
	var httpClient *http.Client
	var headers map[string]string
	if opts != nil {
		httpClient, headers = opts.HTTPClient, maps.Clone(opts.Headers)
	}
	return func(oauth *OAuthHandler) (sdk.Transport, error) {
		client, err := withHeaders(httpClient, endpoint, headers)
		if err != nil {
			return nil, err
		}
		transport := &sdk.StreamableClientTransport{Endpoint: endpoint, HTTPClient: client}
		// A nil *OAuthHandler in the interface field would not be nil.
		if oauth != nil {
			transport.OAuthHandler = oauth
			transport.HTTPClient = oauth.HTTPClient(client)
		}
		return transport, nil
	}
}

// withHeaders returns base, or a default client when it is nil, sending
// headers to the origin of endpoint.
func withHeaders(base *http.Client, endpoint string, headers map[string]string) (*http.Client, error) {
	if len(headers) == 0 {
		return base, nil
	}
	for name, value := range headers {
		if !validHeader(name, value) {
			// The value may be a secret, so it is not shown.
			return nil, fmt.Errorf("mcp: invalid HTTP header %q", name)
		}
	}
	u, err := url.Parse(endpoint)
	if err != nil {
		return nil, fmt.Errorf("mcp: invalid server URL %q: %w", endpoint, err)
	}
	client := http.Client{}
	if base != nil {
		client = *base
	}
	next := client.Transport
	if next == nil {
		next = http.DefaultTransport
	}
	client.Transport = &headerTransport{next: next, origin: origin(u), headers: headers}
	return &client, nil
}

// headerTransport adds configured headers to requests to the MCP server's
// origin that do not set them already. A redirect elsewhere does not carry
// them, so an API key does not leak to another host.
type headerTransport struct {
	next    http.RoundTripper
	origin  string
	headers map[string]string
}

func (t *headerTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	if origin(req.URL) != t.origin {
		return t.next.RoundTrip(req)
	}
	// A RoundTripper must not modify the request.
	req = req.Clone(req.Context())
	for name, value := range t.headers {
		if req.Header.Get(name) == "" {
			req.Header.Set(name, value)
		}
	}
	return t.next.RoundTrip(req)
}

// validHeader reports whether name is a token (RFC 9110) and value has no
// line breaks or NULs, which net/http would refuse to send.
func validHeader(name, value string) bool {
	if name == "" {
		return false
	}
	for _, c := range []byte(name) {
		if !('a' <= c && c <= 'z' || 'A' <= c && c <= 'Z' || '0' <= c && c <= '9' || strings.IndexByte("!#$%&'*+-.^_`|~", c) >= 0) {
			return false
		}
	}
	return !strings.ContainsAny(value, "\r\n\x00")
}

// ServerConnectionOptions configures a [ServerConnection].
type ServerConnectionOptions struct {
	// Name names the server in tool names, labels and messages. Required.
	Name string
	// CreateTransport returns the transport of each connection attempt.
	// Required.
	CreateTransport TransportFactory
	// OAuth authorizes requests to a server that signs users in with OAuth.
	// It is passed to CreateTransport.
	OAuth *OAuthHandler
	// Timeout bounds connecting and each request; for tool calls, the time
	// between progress notifications. Zero means DefaultTimeout.
	Timeout time.Duration
	// Client identifies the client to the server. Nil means burrow.
	Client *sdk.Implementation
	// OnTools is called when the server's tools were listed: after each
	// connect and when the server reports that they changed.
	OnTools func(*ServerConnection)
	// OnChange is called when the status or the tools change.
	OnChange func(*ServerConnection)
}

// ServerConnection is one configured server. It connects on first use and
// reconnects lazily when a call finds the connection gone. Callbacks run
// without its lock held, so they may call its methods.
type ServerConnection struct {
	name            string
	timeout         time.Duration
	createTransport TransportFactory
	oauth           *OAuthHandler
	onTools         func(*ServerConnection)
	onChange        func(*ServerConnection)
	client          *sdk.Client
	progress        progressRouter
	// shutdown is cancelled by Close. It cancels a connect in progress,
	// including the wait between retries.
	shutdown context.Context
	stop     context.CancelFunc

	mu           sync.Mutex
	status       ServerStatus
	tools        []*sdk.Tool
	hasResources bool
	instructions string
	session      *sdk.ClientSession
	opening      *opening
}

// opening is a connect in progress, which concurrent callers share.
type opening struct {
	done    chan struct{}
	session *sdk.ClientSession
	err     error
}

// NewServerConnection returns a connection that has not connected yet. Call
// GetClient to connect.
func NewServerConnection(opts ServerConnectionOptions) (*ServerConnection, error) {
	if opts.Name == "" {
		return nil, errors.New("mcp: ServerConnectionOptions.Name is required")
	}
	if opts.CreateTransport == nil {
		return nil, errors.New("mcp: ServerConnectionOptions.CreateTransport is required")
	}
	c := &ServerConnection{
		name:            opts.Name,
		timeout:         cmp.Or(opts.Timeout, DefaultTimeout),
		createTransport: opts.CreateTransport,
		oauth:           opts.OAuth,
		onTools:         opts.OnTools,
		onChange:        opts.OnChange,
		status:          ServerStatus{State: StateConnecting},
	}
	c.shutdown, c.stop = context.WithCancel(context.Background())
	clientOptions := &sdk.ClientOptions{
		// Listing tools from the handler would block the connection's
		// reader, which has to deliver the response.
		ToolListChangedHandler:      func(_ context.Context, req *sdk.ToolListChangedRequest) { go c.refreshTools(req.Session) },
		ProgressNotificationHandler: c.progress.handle,
	}
	implementation := opts.Client
	if implementation == nil {
		implementation = &sdk.Implementation{Name: "burrow", Version: moduleVersion()}
	}
	c.client = sdk.NewClient(implementation, clientOptions)
	return c, nil
}

// moduleVersion is this module's version in the build, else "dev".
func moduleVersion() string {
	if info, ok := debug.ReadBuildInfo(); ok {
		for _, module := range append([]*debug.Module{&info.Main}, info.Deps...) {
			if module.Path == "github.com/ColeBurch/burrow/mcp" && module.Version != "" && module.Version != "(devel)" {
				return module.Version
			}
		}
	}
	return "dev"
}

// Name is the server's name.
func (c *ServerConnection) Name() string { return c.name }

// Timeout bounds connecting and each request.
func (c *ServerConnection) Timeout() time.Duration { return c.timeout }

// OAuth is the handler that authorizes requests, or nil.
func (c *ServerConnection) OAuth() *OAuthHandler { return c.oauth }

// Status is the connection's state and why it is in it.
func (c *ServerConnection) Status() ServerStatus {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.status
}

// Tools are the tools the server offered when last listed.
func (c *ServerConnection) Tools() []*sdk.Tool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return slices.Clone(c.tools)
}

// HasResources reports whether the server offers resources, which the
// resource tools list and read on demand. They are not listed at connect:
// listing every resource can take long (a large drive, say).
func (c *ServerConnection) HasResources() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.hasResources
}

// Instructions are the server's instructions from initialize, describing
// its tools as a group.
func (c *ServerConnection) Instructions() string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.instructions
}

func (c *ServerConnection) closed() bool { return c.shutdown.Err() != nil }

// GetClient returns the connected session, connecting first if needed.
// Concurrent callers share one connect. Cancelling ctx stops waiting, not
// the connect.
func (c *ServerConnection) GetClient(ctx context.Context) (*sdk.ClientSession, error) {
	c.mu.Lock()
	if c.closed() {
		c.mu.Unlock()
		return nil, fmt.Errorf("MCP server %q is shut down", c.name)
	}
	if session := c.session; session != nil {
		c.mu.Unlock()
		return session, nil
	}
	o := c.opening
	if o == nil {
		o = &opening{done: make(chan struct{})}
		c.opening = o
		go c.open(o)
	}
	c.mu.Unlock()
	select {
	case <-o.done:
		return o.session, o.err
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// CallTool calls the server's tool name. It is not retried after a
// transient error, since the tool may have run.
func (c *ServerConnection) CallTool(ctx context.Context, name string, args map[string]any, opts *RequestOptions) (*sdk.CallToolResult, error) {
	return withClient(ctx, c, false, opts, func(ctx context.Context, session *sdk.ClientSession, onProgress func(*sdk.ProgressNotificationParams)) (*sdk.CallToolResult, error) {
		params := &sdk.CallToolParams{Name: name, Arguments: args}
		if onProgress != nil {
			token, done := c.progress.register(onProgress)
			defer done()
			params.SetProgressToken(token)
		}
		return session.CallTool(ctx, params)
	})
}

// ReadResource reads the resource at uri.
func (c *ServerConnection) ReadResource(ctx context.Context, uri string, opts *RequestOptions) (*sdk.ReadResourceResult, error) {
	return withClient(ctx, c, true, opts, func(ctx context.Context, session *sdk.ClientSession, _ func(*sdk.ProgressNotificationParams)) (*sdk.ReadResourceResult, error) {
		return session.ReadResource(ctx, &sdk.ReadResourceParams{URI: uri})
	})
}

// ResourcesPage lists one page of resources, starting at cursor.
func (c *ServerConnection) ResourcesPage(ctx context.Context, cursor string, opts *RequestOptions) (*sdk.ListResourcesResult, error) {
	return withClient(ctx, c, true, opts, func(ctx context.Context, session *sdk.ClientSession, _ func(*sdk.ProgressNotificationParams)) (*sdk.ListResourcesResult, error) {
		return session.ListResources(ctx, &sdk.ListResourcesParams{Cursor: cursor})
	})
}

// ResourceTemplatesPage lists one page of resource templates, starting at
// cursor. Servers that do not implement resources/templates/list have none.
func (c *ServerConnection) ResourceTemplatesPage(ctx context.Context, cursor string, opts *RequestOptions) (*sdk.ListResourceTemplatesResult, error) {
	return withClient(ctx, c, true, opts, func(ctx context.Context, session *sdk.ClientSession, _ func(*sdk.ProgressNotificationParams)) (*sdk.ListResourceTemplatesResult, error) {
		result, err := session.ListResourceTemplates(ctx, &sdk.ListResourceTemplatesParams{Cursor: cursor})
		if isMethodNotFound(err) {
			return &sdk.ListResourceTemplatesResult{}, nil
		}
		return result, err
	})
}

// AllResources lists every page of resources.
func (c *ServerConnection) AllResources(ctx context.Context, opts *RequestOptions) ([]*sdk.Resource, error) {
	return withClient(ctx, c, true, opts, func(ctx context.Context, session *sdk.ClientSession, _ func(*sdk.ProgressNotificationParams)) ([]*sdk.Resource, error) {
		return collectAll(session.Resources(ctx, nil))
	})
}

// AllResourceTemplates lists every page of resource templates.
func (c *ServerConnection) AllResourceTemplates(ctx context.Context, opts *RequestOptions) ([]*sdk.ResourceTemplate, error) {
	return withClient(ctx, c, true, opts, func(ctx context.Context, session *sdk.ClientSession, _ func(*sdk.ProgressNotificationParams)) ([]*sdk.ResourceTemplate, error) {
		templates, err := collectAll(session.ResourceTemplates(ctx, nil))
		if isMethodNotFound(err) {
			return nil, nil
		}
		return templates, err
	})
}

// runFunc sends a request on session. onProgress, nil unless the request
// asked for progress, must receive the request's progress notifications.
type runFunc[T any] func(ctx context.Context, session *sdk.ClientSession, onProgress func(*sdk.ProgressNotificationParams)) (T, error)

// withClient runs a request, reconnecting when needed. readOnly requests are
// retried once after a transient HTTP error; tool calls are not, since they
// may have run.
func withClient[T any](ctx context.Context, c *ServerConnection, readOnly bool, opts *RequestOptions, run runFunc[T]) (T, error) {
	var zero T
	for attempt := 1; ; attempt++ {
		session, err := c.GetClient(ctx)
		if err != nil {
			return zero, err
		}
		result, err := request(ctx, session, opts, run)
		if err == nil {
			return result, nil
		}
		if readOnly && attempt == 1 && ctx.Err() == nil && isTransientHTTPError(err) {
			if sleep(ctx, connectRetryDelays[0]) != nil {
				return zero, err
			}
			continue
		}
		if attempt == 1 && errors.Is(err, sdk.ErrSessionMissing) {
			// The server no longer knows the session (restart, deploy), so it
			// did not run the request. Retry once on a new session. The old
			// session is detached but not closed: closing would fail its
			// other in-flight calls, which instead get the same 404 and retry
			// the same way.
			c.mu.Lock()
			if c.session == session {
				c.session = nil
			}
			c.mu.Unlock()
			continue
		}
		if !errors.Is(err, ErrSignInRequired) {
			return zero, err
		}
		c.dropClient(session)
		c.markNeedsAuth()
		return zero, c.signInRequired()
	}
}

// errRequestTimedOut cancels a request that ran out of time.
var errRequestTimedOut = errors.New("request timed out")

// request runs one request. opts.Timeout bounds the time without progress,
// as in pi: each progress notification restarts it, so a long-running tool
// that keeps reporting progress is not cut off.
func request[T any](ctx context.Context, session *sdk.ClientSession, opts *RequestOptions, run runFunc[T]) (T, error) {
	var onProgress func(*sdk.ProgressNotificationParams)
	if opts != nil {
		onProgress = opts.OnProgress
	}
	if opts == nil || opts.Timeout <= 0 {
		return run(ctx, session, onProgress)
	}
	requestCtx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)
	timer := time.AfterFunc(opts.Timeout, func() { cancel(errRequestTimedOut) })
	defer timer.Stop()
	if onProgress != nil {
		report := onProgress
		onProgress = func(p *sdk.ProgressNotificationParams) {
			timer.Reset(opts.Timeout)
			report(p)
		}
	}
	result, err := run(requestCtx, session, onProgress)
	if err != nil && ctx.Err() == nil && errors.Is(context.Cause(requestCtx), errRequestTimedOut) {
		err = fmt.Errorf("request timed out after %s: %w", opts.Timeout, context.DeadlineExceeded)
	}
	return result, err
}

// Reconnect connects again with fresh credentials, for example after
// signing in.
func (c *ServerConnection) Reconnect(ctx context.Context) error {
	if err := c.waitForOpening(ctx); err != nil {
		return err
	}
	c.mu.Lock()
	session := c.session
	c.mu.Unlock()
	if session != nil {
		c.dropClient(session)
	}
	_, err := c.GetClient(ctx)
	return err
}

// SignOut disconnects after the stored credentials were removed.
func (c *ServerConnection) SignOut(ctx context.Context) error {
	if err := c.waitForOpening(ctx); err != nil {
		return err
	}
	c.mu.Lock()
	session := c.session
	c.mu.Unlock()
	if session != nil {
		c.dropClient(session)
	}
	if !c.closed() {
		c.markNeedsAuth()
	}
	return nil
}

// Close closes the connection and returns once no transport of this server
// is open, including one that was still connecting, and no refresh of its
// tokens is saving. A refresh the server already answered may have rotated
// the refresh token; exiting before the new tokens are saved would lose the
// grant. It returns early with ctx's error.
func (c *ServerConnection) Close(ctx context.Context) error {
	c.stop()
	c.mu.Lock()
	c.status = ServerStatus{State: StateClosed}
	session, o := c.session, c.opening
	c.session = nil
	c.mu.Unlock()
	c.changed()
	if session != nil {
		_ = session.Close()
	}
	if o != nil {
		select {
		case <-o.done:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	if c.oauth != nil {
		return c.oauth.Wait(ctx)
	}
	return nil
}

// waitForOpening waits for a connect in progress, ignoring its result.
func (c *ServerConnection) waitForOpening(ctx context.Context) error {
	c.mu.Lock()
	o := c.opening
	c.mu.Unlock()
	if o == nil {
		return nil
	}
	select {
	case <-o.done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (c *ServerConnection) signInRequired() error {
	return fmt.Errorf("MCP server %s: %w", c.name, ErrSignInRequired)
}

func (c *ServerConnection) markNeedsAuth() {
	c.mu.Lock()
	c.status = ServerStatus{State: StateNeedsAuth}
	c.mu.Unlock()
	c.changed()
}

func (c *ServerConnection) changed() {
	if c.onChange != nil {
		c.onChange(c)
	}
}

func (c *ServerConnection) dropClient(session *sdk.ClientSession) {
	c.mu.Lock()
	if c.session == session {
		c.session = nil
	}
	c.mu.Unlock()
	_ = session.Close()
}

func (c *ServerConnection) open(o *opening) {
	c.mu.Lock()
	c.status = ServerStatus{State: StateConnecting}
	c.mu.Unlock()
	c.changed()
	for attempt := 0; ; attempt++ {
		session, retryable, err := c.connectOnce()
		if err == nil {
			o.session = session
			break
		}
		if c.closed() || !retryable || attempt >= len(connectRetryDelays) || !isTransientError(err) ||
			sleep(c.shutdown, connectRetryDelays[attempt]) != nil {
			o.err = c.connectFailed(err)
			break
		}
	}
	c.mu.Lock()
	if c.opening == o {
		c.opening = nil
	}
	c.mu.Unlock()
	close(o.done)
}

// connectOnce connects, lists the tools, and makes the session current.
// retryable reports an HTTP transport, whose transient failures are worth
// another attempt.
func (c *ServerConnection) connectOnce() (session *sdk.ClientSession, retryable bool, err error) {
	transport, err := c.createTransport(c.oauth)
	if err != nil {
		return nil, false, err
	}
	switch transport.(type) {
	case *sdk.StreamableClientTransport, *sdk.SSEClientTransport:
		retryable = true
	}
	ctx, cancel := context.WithTimeout(c.shutdown, c.timeout)
	defer cancel()
	session, err = c.client.Connect(ctx, transport, nil)
	if err != nil {
		return nil, retryable, withStderr(transport, err)
	}
	initialized := session.InitializeResult()
	capabilities := initialized.Capabilities
	if capabilities == nil {
		capabilities = &sdk.ServerCapabilities{}
	}
	// Servers without the tools capability (prompts or resources only) do
	// not answer tools/list.
	var tools []*sdk.Tool
	if capabilities.Tools != nil {
		if tools, err = collectAll(session.Tools(ctx, nil)); err != nil {
			_ = session.Close()
			return nil, retryable, withStderr(transport, err)
		}
	}
	c.mu.Lock()
	if c.closed() {
		c.mu.Unlock()
		_ = session.Close()
		return nil, false, errors.New("shut down while connecting")
	}
	c.session = session
	c.tools = tools
	c.hasResources = capabilities.Resources != nil
	c.instructions = strings.TrimSpace(initialized.Instructions)
	c.status = ServerStatus{State: StateConnected}
	c.mu.Unlock()
	go c.watch(session, transport)
	if c.onTools != nil {
		c.onTools(c)
	}
	c.changed()
	return session, retryable, nil
}

func (c *ServerConnection) connectFailed(err error) error {
	if errors.Is(err, ErrSignInRequired) && !c.closed() {
		c.markNeedsAuth()
		return c.signInRequired()
	}
	c.mu.Lock()
	state := StateFailed
	if c.closed() {
		state = StateClosed
	}
	c.status = ServerStatus{State: state, Error: err.Error()}
	c.mu.Unlock()
	c.changed()
	return fmt.Errorf("MCP server %q failed to connect: %w", c.name, err)
}

// watch notices when the transport drops. The next call reconnects; until
// then the status shows why, with the end of a stdio server's stderr.
func (c *ServerConnection) watch(session *sdk.ClientSession, transport sdk.Transport) {
	_ = session.Wait()
	c.mu.Lock()
	dropped := c.session == session && !c.closed()
	c.mu.Unlock()
	if !dropped {
		return
	}
	reason := "Connection closed"
	if tail := stderrOf(transport); tail != "" {
		reason += "\n" + tail
	}
	c.mu.Lock()
	if c.session != session || c.closed() {
		c.mu.Unlock()
		return
	}
	c.session = nil
	c.status = ServerStatus{State: StateDisconnected, Error: reason}
	c.mu.Unlock()
	c.changed()
}

func (c *ServerConnection) refreshTools(session *sdk.ClientSession) {
	ctx, cancel := context.WithTimeout(c.shutdown, c.timeout)
	defer cancel()
	tools, err := collectAll(session.Tools(ctx, nil))
	c.mu.Lock()
	if c.session != session || c.closed() {
		c.mu.Unlock()
		return
	}
	if err != nil {
		c.status.Error = "Failed to refresh tools: " + err.Error()
	} else {
		c.tools = tools
	}
	c.mu.Unlock()
	if err == nil && c.onTools != nil {
		c.onTools(c)
	}
	c.changed()
}

// rejected is the code of the SDK's errors for requests that failed without
// breaking the connection, such as transient HTTP statuses and network
// errors.
var rejected = &jsonrpc.Error{Code: -32005}

// transientStatusTexts are the statuses the SDK reports as transient. It
// reports them only as the text at the end of a rejected request's error.
var transientStatusTexts = []string{
	http.StatusText(http.StatusInternalServerError),
	http.StatusText(http.StatusBadGateway),
	http.StatusText(http.StatusServiceUnavailable),
	http.StatusText(http.StatusGatewayTimeout),
	http.StatusText(http.StatusTooManyRequests),
}

// isTransientHTTPError reports an overloaded or restarting server.
func isTransientHTTPError(err error) bool {
	if !errors.Is(err, rejected) {
		return false
	}
	message := err.Error()
	for _, status := range transientStatusTexts {
		if strings.HasSuffix(message, ": "+status) {
			return true
		}
	}
	return false
}

// isTransientError reports network failures and overloaded or restarting
// servers, which are worth another attempt.
func isTransientError(err error) bool {
	if errors.Is(err, ErrSignInRequired) || errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return false
	}
	var netErr net.Error
	return isTransientHTTPError(err) || errors.As(err, &netErr)
}

func sleep(ctx context.Context, d time.Duration) error {
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-timer.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func collectAll[T any](seq iter.Seq2[T, error]) ([]T, error) {
	var items []T
	for item, err := range seq {
		if err != nil {
			return nil, err
		}
		items = append(items, item)
	}
	return items, nil
}

// progressRouter delivers progress notifications to the requests that asked
// for them. The SDK takes one progress handler per client.
type progressRouter struct {
	mu    sync.Mutex
	next  uint64
	calls map[string]*progressCall
}

type progressCall struct {
	mu   sync.Mutex
	done bool
	fn   func(*sdk.ProgressNotificationParams)
}

// handle passes a progress notification to the request it belongs to.
func (p *progressRouter) handle(_ context.Context, req *sdk.ProgressNotificationClientRequest) {
	token, ok := req.Params.ProgressToken.(string)
	if !ok {
		return
	}
	p.mu.Lock()
	call := p.calls[token]
	p.mu.Unlock()
	if call == nil {
		return
	}
	call.mu.Lock()
	defer call.mu.Unlock()
	if !call.done {
		call.fn(req.Params)
	}
}

// register returns a new progress token for fn, and a function that stops
// delivery. Once done returns, fn is not called again.
func (p *progressRouter) register(fn func(*sdk.ProgressNotificationParams)) (token string, done func()) {
	call := &progressCall{fn: fn}
	p.mu.Lock()
	p.next++
	token = fmt.Sprintf("burrow-%d", p.next)
	if p.calls == nil {
		p.calls = map[string]*progressCall{}
	}
	p.calls[token] = call
	p.mu.Unlock()
	return token, func() {
		p.mu.Lock()
		delete(p.calls, token)
		p.mu.Unlock()
		call.mu.Lock()
		call.done = true
		call.mu.Unlock()
	}
}
