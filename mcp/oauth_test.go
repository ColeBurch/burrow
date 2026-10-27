package mcp

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ColeBurch/burrow/baseAgent"
	"github.com/ColeBurch/burrow/baseAgent/authstoretest"
	sdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

// fakeOAuthServer is an MCP server and its authorization server on one
// origin. Access and refresh tokens rotate on every refresh.
type fakeOAuthServer struct {
	*httptest.Server
	t *testing.T

	// Settings, fixed before Start.
	expiresIn     int
	issSupported  bool
	noMetadata    bool   // no resource or authorization server metadata
	requiredScope string // scope tool calls need, answered with insufficient_scope
	keepRefresh   bool   // refreshes keep the refresh token and do not return one
	cimd          bool   // accepts https client IDs as Client ID Metadata Documents
	// refreshGate, when set, holds each refresh until it is closed, after
	// signaling refreshing.
	refreshGate chan struct{}
	refreshing  chan struct{}

	mu      sync.Mutex
	clients map[string][]string // client ID to redirect URIs
	// applicationTypes are the application_type of each registration.
	applicationTypes []string
	codes            map[string]fakeCode
	access           map[string]string // access token to scope
	refresh          map[string]string // refresh token to scope
	next             int
	refreshN         atomic.Int32
	// rejectTokens makes the MCP endpoint reject every access token, while
	// refreshes still succeed.
	rejectTokens atomic.Bool
	// forbid makes the MCP endpoint answer a plain 403.
	forbid atomic.Bool
	// posts counts POSTs to the MCP endpoint.
	posts atomic.Int32
}

type fakeCode struct {
	clientID, redirectURI, challenge, scope, resource string
}

// newFakeOAuthServer starts the server once configure has set its settings.
func newFakeOAuthServer(t *testing.T, configure func(f *fakeOAuthServer)) *fakeOAuthServer {
	f := &fakeOAuthServer{
		t:         t,
		expiresIn: 3600,
		clients:   map[string][]string{},
		codes:     map[string]fakeCode{},
		access:    map[string]string{},
		refresh:   map[string]string{},
	}
	mcpServer := sdk.NewServer(&sdk.Implementation{Name: "fake", Version: "1.0.0"}, nil)
	mcpServer.AddTool(&sdk.Tool{Name: "whoami", InputSchema: objectSchema},
		func(context.Context, *sdk.CallToolRequest) (*sdk.CallToolResult, error) {
			return textResult("you"), nil
		})
	mcpHandler := sdk.NewStreamableHTTPHandler(func(*http.Request) *sdk.Server { return mcpServer }, nil)

	mux := http.NewServeMux()
	mux.HandleFunc("/mcp", func(w http.ResponseWriter, r *http.Request) {
		token, _ := strings.CutPrefix(r.Header.Get("Authorization"), "Bearer ")
		f.mu.Lock()
		scope, ok := f.access[token]
		f.mu.Unlock()
		if f.rejectTokens.Load() {
			ok = false
		}
		if !ok {
			challenge := `Bearer scope="tools"`
			if !f.noMetadata {
				challenge += fmt.Sprintf(`, resource_metadata="%s/prm"`, f.URL)
			}
			w.Header().Set("WWW-Authenticate", challenge)
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		if r.Method == http.MethodPost {
			f.posts.Add(1)
		}
		if f.forbid.Load() {
			http.Error(w, "forbidden", http.StatusForbidden)
			return
		}
		if f.requiredScope != "" && !slices.Contains(strings.Fields(scope), f.requiredScope) {
			w.Header().Set("WWW-Authenticate", fmt.Sprintf(`Bearer error="insufficient_scope", scope="%s"`, f.requiredScope))
			http.Error(w, "forbidden", http.StatusForbidden)
			return
		}
		mcpHandler.ServeHTTP(w, r)
	})
	// Resource metadata at the URL challenges name and at the well-known
	// path of the MCP endpoint.
	prm := func(w http.ResponseWriter, r *http.Request) {
		if f.noMetadata {
			http.NotFound(w, r)
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{
			"resource":              f.URL + "/mcp",
			"authorization_servers": []string{f.URL},
			"scopes_supported":      []string{"tools"},
		})
	}
	mux.HandleFunc("/prm", prm)
	mux.HandleFunc("/.well-known/oauth-protected-resource/mcp", prm)
	mux.HandleFunc("/.well-known/oauth-authorization-server", func(w http.ResponseWriter, r *http.Request) {
		if f.noMetadata {
			http.NotFound(w, r)
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{
			"issuer":                                         f.URL,
			"authorization_endpoint":                         f.URL + "/authorize",
			"token_endpoint":                                 f.URL + "/token",
			"registration_endpoint":                          f.URL + "/register",
			"response_types_supported":                       []string{"code"},
			"code_challenge_methods_supported":               []string{"S256"},
			"token_endpoint_auth_methods_supported":          []string{"none"},
			"scopes_supported":                               []string{"tools", "admin", "offline_access"},
			"authorization_response_iss_parameter_supported": f.issSupported,
			"client_id_metadata_document_supported":          f.cimd,
		})
	})
	mux.HandleFunc("/register", func(w http.ResponseWriter, r *http.Request) {
		var metadata struct {
			RedirectURIs    []string `json:"redirect_uris"`
			ApplicationType string   `json:"application_type"`
		}
		if err := json.NewDecoder(r.Body).Decode(&metadata); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]any{"error": "invalid_client_metadata"})
			return
		}
		f.mu.Lock()
		f.next++
		id := fmt.Sprintf("client-%d", f.next)
		f.clients[id] = metadata.RedirectURIs
		f.applicationTypes = append(f.applicationTypes, metadata.ApplicationType)
		f.mu.Unlock()
		writeJSON(w, http.StatusCreated, map[string]any{
			"client_id":                  id,
			"redirect_uris":              metadata.RedirectURIs,
			"token_endpoint_auth_method": "none",
		})
	})
	mux.HandleFunc("/authorize", func(w http.ResponseWriter, r *http.Request) {
		q := r.URL.Query()
		f.mu.Lock()
		defer f.mu.Unlock()
		// A real server would fetch the document and check its redirect URIs.
		metadataDocument := f.cimd && strings.HasPrefix(q.Get("client_id"), "https://")
		if !metadataDocument && !slices.Contains(f.clients[q.Get("client_id")], q.Get("redirect_uri")) {
			http.Error(w, "unknown client or redirect URI", http.StatusBadRequest)
			return
		}
		if q.Get("code_challenge_method") != "S256" || q.Get("response_type") != "code" {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		f.next++
		code := fmt.Sprintf("code-%d", f.next)
		f.codes[code] = fakeCode{q.Get("client_id"), q.Get("redirect_uri"), q.Get("code_challenge"), q.Get("scope"), q.Get("resource")}
		redirect, _ := url.Parse(q.Get("redirect_uri"))
		values := url.Values{"code": {code}, "state": {q.Get("state")}}
		if f.issSupported {
			values.Set("iss", f.URL)
		}
		redirect.RawQuery = values.Encode()
		http.Redirect(w, r, redirect.String(), http.StatusFound)
	})
	mux.HandleFunc("/token", func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseForm(); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]any{"error": "invalid_request"})
			return
		}
		f.mu.Lock()
		defer f.mu.Unlock()
		var scope string
		switch r.PostForm.Get("grant_type") {
		case "authorization_code":
			code, ok := f.codes[r.PostForm.Get("code")]
			delete(f.codes, r.PostForm.Get("code"))
			sum := sha256.Sum256([]byte(r.PostForm.Get("code_verifier")))
			if !ok || code.clientID != r.PostForm.Get("client_id") || code.redirectURI != r.PostForm.Get("redirect_uri") ||
				base64.RawURLEncoding.EncodeToString(sum[:]) != code.challenge || code.resource != r.PostForm.Get("resource") {
				writeJSON(w, http.StatusBadRequest, map[string]any{"error": "invalid_grant"})
				return
			}
			scope = code.scope
		case "refresh_token":
			var ok bool
			scope, ok = f.refresh[r.PostForm.Get("refresh_token")]
			if !ok {
				writeJSON(w, http.StatusBadRequest, map[string]any{"error": "invalid_grant", "error_description": "refresh token reused"})
				return
			}
			if !f.keepRefresh {
				delete(f.refresh, r.PostForm.Get("refresh_token"))
			}
			f.refreshN.Add(1)
			// Slow refreshes widen the window for duplicate ones.
			time.Sleep(20 * time.Millisecond)
			if f.refreshGate != nil {
				f.refreshing <- struct{}{}
				<-f.refreshGate
			}
		default:
			writeJSON(w, http.StatusBadRequest, map[string]any{"error": "unsupported_grant_type"})
			return
		}
		f.next++
		accessToken := fmt.Sprintf("at-%d", f.next)
		f.access[accessToken] = scope
		response := map[string]any{"access_token": accessToken, "token_type": "Bearer", "expires_in": f.expiresIn}
		if r.PostForm.Get("grant_type") == "authorization_code" || !f.keepRefresh {
			refreshToken := fmt.Sprintf("rt-%d", f.next)
			f.refresh[refreshToken] = scope
			response["refresh_token"] = refreshToken
		}
		writeJSON(w, http.StatusOK, response)
	})
	// Settings and URL are set before Start, so handlers see them.
	f.Server = httptest.NewUnstartedServer(mux)
	if configure != nil {
		configure(f)
	}
	f.Start()
	t.Cleanup(f.Close)
	return f
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

// revoke makes the server reject every access token, and with
// refresh, every refresh token.
func (f *fakeOAuthServer) revoke(refresh bool) {
	f.mu.Lock()
	defer f.mu.Unlock()
	clear(f.access)
	if refresh {
		clear(f.refresh)
	}
}

// fakeClock is a settable clock for handlers.
type fakeClock struct {
	mu  sync.Mutex
	now time.Time
}

func (c *fakeClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *fakeClock) Advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(d)
}

func newHandler(t *testing.T, f *fakeOAuthServer, store baseAgent.AuthStore, clock *fakeClock) *OAuthHandler {
	t.Helper()
	h, err := NewOAuthHandler(&OAuthOptions{
		Store:       store,
		OwnerID:     "user-1",
		Server:      "fake",
		ServerURL:   f.URL + "/mcp",
		RedirectURL: "https://app.example/mcp/callback/fake",
		Now:         clock.Now,
	})
	if err != nil {
		t.Fatal(err)
	}
	return h
}

func connectHTTP(ctx context.Context, f *fakeOAuthServer, h *OAuthHandler) (*sdk.ClientSession, error) {
	client := sdk.NewClient(&sdk.Implementation{Name: "test-client", Version: "1.0.0"}, nil)
	return client.Connect(ctx, &sdk.StreamableClientTransport{
		Endpoint: f.URL + "/mcp", OAuthHandler: h, HTTPClient: h.HTTPClient(nil), MaxRetries: -1,
	}, nil)
}

// authorize acts as the browser: it opens authURL and returns the query of
// the redirect to the app.
func authorize(t *testing.T, authURL string) url.Values {
	t.Helper()
	client := &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	resp, err := client.Get(authURL)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusFound {
		t.Fatalf("authorize returned %s", resp.Status)
	}
	location, err := url.Parse(resp.Header.Get("Location"))
	if err != nil {
		t.Fatal(err)
	}
	return location.Query()
}

func signIn(t *testing.T, ctx context.Context, h *OAuthHandler) {
	t.Helper()
	authURL, err := h.StartSignIn(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if err := h.CompleteSignIn(ctx, authorize(t, authURL)); err != nil {
		t.Fatal(err)
	}
}

func callWhoami(ctx context.Context, session *sdk.ClientSession) error {
	result, err := session.CallTool(ctx, &sdk.CallToolParams{Name: "whoami"})
	if err != nil {
		return err
	}
	if text := textOf(ConvertResult("fake", "whoami", result, nil).Content); text != "you" {
		return fmt.Errorf("whoami = %q", text)
	}
	return nil
}

func TestSignInAcrossProcesses(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	f := newFakeOAuthServer(t, func(f *fakeOAuthServer) { f.issSupported = true })
	store := authstoretest.NewMemoryStore()
	clock := &fakeClock{now: time.Now()}

	// Nothing stored: connecting asks for a sign-in instead of starting one.
	starter := newHandler(t, f, store, clock)
	if _, err := connectHTTP(ctx, f, starter); !errors.Is(err, ErrSignInRequired) {
		t.Fatalf("connect before sign-in: %v", err)
	}

	authURL, err := starter.StartSignIn(ctx)
	if err != nil {
		t.Fatal(err)
	}
	query, _ := url.Parse(authURL)
	params := query.Query()
	if got := params.Get("scope"); got != "tools offline_access" {
		t.Fatalf("scope = %q", got)
	}
	if params.Get("resource") != f.URL+"/mcp" || params.Get("prompt") != "consent" {
		t.Fatalf("authorization URL = %s", authURL)
	}
	response := authorize(t, authURL)

	// Another process completes the sign-in from the store alone.
	completer := newHandler(t, f, store, clock)
	if err := completer.CompleteSignIn(ctx, response); err != nil {
		t.Fatal(err)
	}
	if err := completer.CompleteSignIn(ctx, response); err == nil {
		t.Fatal("a response completed two sign-ins")
	}
	if ok, err := starter.SignedIn(ctx); err != nil || !ok {
		t.Fatalf("SignedIn = %v, %v", ok, err)
	}

	session, err := connectHTTP(ctx, f, starter)
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()
	if err := callWhoami(ctx, session); err != nil {
		t.Fatal(err)
	}

	if err := starter.SignOut(ctx); err != nil {
		t.Fatal(err)
	}
	if ok, _ := starter.SignedIn(ctx); ok {
		t.Fatal("signed in after SignOut")
	}
}

func TestExpiringTokenRefreshesOnceAcrossHandlers(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	f := newFakeOAuthServer(t, func(f *fakeOAuthServer) { f.expiresIn = 60 })
	store := authstoretest.NewMemoryStore()
	clock := &fakeClock{now: time.Now()}
	signIn(t, ctx, newHandler(t, f, store, clock))

	// Within 30 seconds of expiry, every request wants a refresh.
	clock.Advance(45 * time.Second)
	handlers := []*OAuthHandler{newHandler(t, f, store, clock), newHandler(t, f, store, clock)}
	tokens := make([]string, 8)
	var wg sync.WaitGroup
	for i := range tokens {
		wg.Go(func() {
			source, err := handlers[i%2].TokenSource(ctx)
			if err != nil || source == nil {
				t.Errorf("TokenSource = %v, %v", source, err)
				return
			}
			token, err := source.Token()
			if err != nil {
				t.Error(err)
				return
			}
			tokens[i] = token.AccessToken
		})
	}
	wg.Wait()
	if n := f.refreshN.Load(); n != 1 {
		t.Fatalf("refreshed %d times, want 1", n)
	}
	for _, token := range tokens {
		if token != tokens[0] {
			t.Fatalf("tokens differ: %v", tokens)
		}
	}

	session, err := connectHTTP(ctx, f, handlers[0])
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()
	if err := callWhoami(ctx, session); err != nil {
		t.Fatal(err)
	}
}

func TestRejectedTokenRefreshesAfter401(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	f := newFakeOAuthServer(t, nil)
	store := authstoretest.NewMemoryStore()
	clock := &fakeClock{now: time.Now()}
	h := newHandler(t, f, store, clock)
	signIn(t, ctx, h)

	session, err := connectHTTP(ctx, f, h)
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()

	f.revoke(false)
	if err := callWhoami(ctx, session); err != nil {
		t.Fatalf("call after revoked access token: %v", err)
	}
	if n := f.refreshN.Load(); n != 1 {
		t.Fatalf("refreshed %d times, want 1", n)
	}

	// Without a valid refresh token, only a new sign-in helps.
	f.revoke(true)
	if err := callWhoami(ctx, session); !errors.Is(err, ErrSignInRequired) {
		t.Fatalf("call after revoked grant: %v", err)
	}
	if ok, _ := h.SignedIn(ctx); ok {
		t.Fatal("rejected tokens are still stored")
	}
}

// A server that needs a sign-in marks its tools' results, and reconnects
// once the user signed in, for example in another request.
func TestSignInRequiredAndReconnectSignedIn(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	f := newFakeOAuthServer(t, nil)
	h := newHandler(t, f, authstoretest.NewMemoryStore(), &fakeClock{now: time.Now()})
	registry := newFakeRegistry()
	ext, err := NewExtension(ExtensionOptions{
		Servers: []ServerConnectionOptions{{Name: "fake", CreateTransport: HTTPTransport(f.URL+"/mcp", nil), OAuth: h}},
		Tools:   registry,
	})
	if err != nil {
		t.Fatal(err)
	}
	ext.Start()
	defer ext.Close(ctx)
	ext.WaitForStartup(ctx)
	connection := ext.Connection("fake")
	if status := connection.Status(); status.State != StateNeedsAuth {
		t.Fatalf("status before sign-in = %#v", status)
	}
	if err := ext.ReconnectSignedIn(ctx); err != nil || connection.Status().State != StateNeedsAuth {
		t.Fatalf("reconnected without new credentials: %v, %#v", err, connection.Status())
	}

	signIn(t, ctx, h)
	if err := ext.ReconnectSignedIn(ctx); err != nil {
		t.Fatal(err)
	}
	if status := connection.Status(); status.State != StateConnected {
		t.Fatalf("status after sign-in = %#v", status)
	}
	whoami := registry.tool(t, "mcp__fake__whoami")
	if result, err := whoami.Execute(ctx, "call-1", map[string]any{}, nil); err != nil || textOf(result.Content) != "you" {
		t.Fatalf("whoami = %#v, %v", result, err)
	}

	f.revoke(true)
	result, err := whoami.Execute(ctx, "call-2", map[string]any{}, nil)
	if err != nil {
		t.Fatalf("Execute returned error %v, want a sign-in result", err)
	}
	details, ok := result.Details.(ToolDetails)
	if !result.IsError || !ok || !details.SignInRequired || details.Server != "fake" || details.Tool != "whoami" {
		t.Fatalf("result = %#v", result)
	}
	if text := textOf(result.Content); !strings.Contains(text, `MCP server "fake" requires sign-in`) {
		t.Fatalf("text = %q", text)
	}
	if status := connection.Status(); status.State != StateNeedsAuth {
		t.Fatalf("status after revoke = %#v", status)
	}

	signIn(t, ctx, h)
	if err := ext.ReconnectSignedIn(ctx); err != nil {
		t.Fatal(err)
	}
	if result, err := whoami.Execute(ctx, "call-3", map[string]any{}, nil); err != nil || textOf(result.Content) != "you" {
		t.Fatalf("whoami after signing in again = %#v, %v", result, err)
	}

	if err := ext.Logout(ctx, "fake"); err != nil {
		t.Fatal(err)
	}
	if ok, _ := h.SignedIn(ctx); ok || connection.Status().State != StateNeedsAuth {
		t.Fatalf("after Logout: signed in %v, status %#v", ok, connection.Status())
	}
}

func TestWaitForRefreshBeforeExit(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	f := newFakeOAuthServer(t, func(f *fakeOAuthServer) {
		f.expiresIn = 60
		f.refreshGate = make(chan struct{})
		f.refreshing = make(chan struct{}, 1)
	})
	// A failed check must not leave the server holding the refresh, or
	// closing the server would hang.
	release := sync.OnceFunc(func() { close(f.refreshGate) })
	defer release()
	store := authstoretest.NewMemoryStore()
	clock := &fakeClock{now: time.Now()}
	h := newHandler(t, f, store, clock)
	wait := func(ctx context.Context) <-chan error {
		waited := make(chan error, 1)
		go func() { waited <- h.Wait(ctx) }()
		return waited
	}
	if err := h.Wait(ctx); err != nil {
		t.Fatalf("Wait with nothing running = %v", err)
	}
	signIn(t, ctx, h)
	before, err := h.load(ctx)
	if err != nil {
		t.Fatal(err)
	}

	// The request that started the refresh goes away while the server
	// holds the refresh; the refresh keeps running.
	clock.Advance(45 * time.Second)
	requestCtx, cancelRequest := context.WithCancel(ctx)
	go func() { _, _ = h.TokenSource(requestCtx) }()
	<-f.refreshing
	cancelRequest()

	expired, cancelExpired := context.WithCancel(ctx)
	cancelExpired()
	select {
	case err := <-wait(expired):
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("Wait with a done context = %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Wait with a done context did not return")
	}
	waited := wait(ctx)
	select {
	case err := <-waited:
		t.Fatalf("Wait returned %v during the refresh", err)
	case <-time.After(50 * time.Millisecond):
	}
	release()
	if err := <-waited; err != nil {
		t.Fatal(err)
	}

	// The rotated tokens were saved before Wait returned.
	state, err := newHandler(t, f, store, clock).load(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if state.Tokens == nil || state.Tokens.AccessToken == before.Tokens.AccessToken || f.refreshN.Load() != 1 {
		t.Fatalf("stored tokens = %#v after %d refreshes", state.Tokens, f.refreshN.Load())
	}
}

func TestRefreshKeepsRefreshTokenWhenNoneReturned(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	f := newFakeOAuthServer(t, func(f *fakeOAuthServer) { f.keepRefresh = true })
	store := authstoretest.NewMemoryStore()
	clock := &fakeClock{now: time.Now()}
	h := newHandler(t, f, store, clock)
	signIn(t, ctx, h)

	session, err := connectHTTP(ctx, f, h)
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()
	for i := range 2 {
		f.revoke(false)
		if err := callWhoami(ctx, session); err != nil {
			t.Fatalf("call %d after revoked access token: %v", i, err)
		}
	}
	if n := f.refreshN.Load(); n != 2 {
		t.Fatalf("refreshed %d times, want 2", n)
	}
}

func TestInsufficientScopeStepsUp(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	f := newFakeOAuthServer(t, func(f *fakeOAuthServer) { f.requiredScope = "admin" })
	store := authstoretest.NewMemoryStore()
	clock := &fakeClock{now: time.Now()}
	h := newHandler(t, f, store, clock)
	signIn(t, ctx, h)

	if _, err := connectHTTP(ctx, f, h); !errors.Is(err, ErrSignInRequired) {
		t.Fatalf("connect without scope: %v", err)
	}
	if n := f.refreshN.Load(); n != 0 {
		t.Fatalf("refreshed %d times for more scope", n)
	}

	// The next sign-in asks for the new scope on top of the granted one.
	authURL, err := h.StartSignIn(ctx)
	if err != nil {
		t.Fatal(err)
	}
	parsed, _ := url.Parse(authURL)
	if got := parsed.Query().Get("scope"); got != "tools offline_access admin" {
		t.Fatalf("step-up scope = %q", got)
	}
	if err := h.CompleteSignIn(ctx, authorize(t, authURL)); err != nil {
		t.Fatal(err)
	}
	session, err := connectHTTP(ctx, f, h)
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()
	if err := callWhoami(ctx, session); err != nil {
		t.Fatal(err)
	}
}

func TestSignInWithoutMetadata(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	f := newFakeOAuthServer(t, func(f *fakeOAuthServer) { f.noMetadata = true })
	store := authstoretest.NewMemoryStore()
	clock := &fakeClock{now: time.Now()}
	h := newHandler(t, f, store, clock)

	authURL, err := h.StartSignIn(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(authURL, f.URL+"/authorize?") || strings.Contains(authURL, "resource=") {
		t.Fatalf("authorization URL = %s", authURL)
	}
	response := authorize(t, authURL)

	// Without metadata there is no issuer to compare with.
	withIss := url.Values{"code": {response.Get("code")}, "state": {response.Get("state")}, "iss": {f.URL}}
	if err := h.CompleteSignIn(ctx, withIss); err == nil || !strings.Contains(err.Error(), "issuer") {
		t.Fatalf("iss without metadata: %v", err)
	}
	if err := h.CompleteSignIn(ctx, response); err != nil {
		t.Fatal(err)
	}
	session, err := connectHTTP(ctx, f, h)
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()
	if err := callWhoami(ctx, session); err != nil {
		t.Fatal(err)
	}
}

func TestCompleteSignInRejectsBadResponses(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	f := newFakeOAuthServer(t, func(f *fakeOAuthServer) { f.issSupported = true })
	store := authstoretest.NewMemoryStore()
	clock := &fakeClock{now: time.Now()}
	h := newHandler(t, f, store, clock)

	authURL, err := h.StartSignIn(ctx)
	if err != nil {
		t.Fatal(err)
	}
	response := authorize(t, authURL)
	with := func(key, value string) url.Values {
		changed := url.Values{}
		for k, v := range response {
			changed[k] = slices.Clone(v)
		}
		if value == "" {
			changed.Del(key)
		} else {
			changed.Set(key, value)
		}
		return changed
	}

	cases := map[string]url.Values{
		"wrong state":   with("state", "forged"),
		"missing iss":   with("iss", ""),
		"wrong iss":     with("iss", "https://evil.example"),
		"missing code":  with("code", ""),
		"error":         {"error": {"access_denied"}, "state": {response.Get("state")}},
		"no such state": {"code": {"x"}},
	}
	for name, query := range cases {
		if err := h.CompleteSignIn(ctx, query); err == nil {
			t.Errorf("%s: CompleteSignIn succeeded", name)
		}
	}

	clock.Advance(signInExpiry + time.Minute)
	if err := h.CompleteSignIn(ctx, response); err == nil || !strings.Contains(err.Error(), "expired") {
		t.Fatalf("expired sign-in: %v", err)
	}
	if ok, _ := h.SignedIn(ctx); ok {
		t.Fatal("signed in by a rejected response")
	}
}

func TestNewOAuthHandlerValidates(t *testing.T) {
	store := authstoretest.NewMemoryStore()
	for name, opts := range map[string]*OAuthOptions{
		"nil":        nil,
		"no store":   {Server: "s", ServerURL: "https://s.example/mcp"},
		"no server":  {Store: store, ServerURL: "https://s.example/mcp"},
		"bad url":    {Store: store, Server: "s", ServerURL: "ftp://s.example"},
		"no host":    {Store: store, Server: "s", ServerURL: "https:///mcp"},
		"long names": {Store: store, Server: strings.Repeat("s", 1100), ServerURL: "https://s.example/mcp"},
	} {
		if _, err := NewOAuthHandler(opts); err == nil {
			t.Errorf("%s: NewOAuthHandler succeeded", name)
		}
	}
	h, err := NewOAuthHandler(&OAuthOptions{Store: store, Server: "s", ServerURL: "https://s.example/mcp"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := h.StartSignIn(context.Background()); err == nil || !strings.Contains(err.Error(), "RedirectURL") {
		t.Fatalf("StartSignIn without RedirectURL: %v", err)
	}
}

func TestSelectClientAuthMethod(t *testing.T) {
	cases := []struct {
		client    oauthClient
		supported []string
		want      string
	}{
		{oauthClient{TokenEndpointAuthMethod: "none"}, nil, "none"},
		{oauthClient{TokenEndpointAuthMethod: "client_secret_post", ClientSecret: "s"}, []string{"client_secret_post"}, "client_secret_post"},
		{oauthClient{TokenEndpointAuthMethod: "client_secret_post", ClientSecret: "s"}, []string{"client_secret_basic"}, "client_secret_basic"},
		{oauthClient{ClientSecret: "s"}, nil, "client_secret_basic"},
		{oauthClient{}, nil, "none"},
		{oauthClient{ClientSecret: "s"}, []string{"none", "client_secret_post"}, "client_secret_post"},
		{oauthClient{}, []string{"client_secret_basic", "none"}, "none"},
		{oauthClient{ClientSecret: "s"}, []string{"private_key_jwt"}, "client_secret_post"},
	}
	for _, c := range cases {
		if got := selectClientAuthMethod(&c.client, c.supported); got != c.want {
			t.Errorf("selectClientAuthMethod(%+v, %v) = %s, want %s", c.client, c.supported, got, c.want)
		}
	}
}

func TestCheckEndpointAndScopes(t *testing.T) {
	for endpoint, ok := range map[string]bool{
		"https://auth.example/token": true,
		"http://127.0.0.1:8080/x":    true,
		"http://localhost/x":         true,
		"http://[::1]/x":             true,
		"http://auth.example/token":  false,
		"javascript:alert(1)":        false,
	} {
		if err := checkEndpoint(endpoint); (err == nil) != ok {
			t.Errorf("checkEndpoint(%q) = %v", endpoint, err)
		}
	}
	if got := mergeScopes("a b", "", "b  c", "a"); got != "a b c" {
		t.Fatalf("mergeScopes = %q", got)
	}
}

// Resource metadata naming the MCP server's origin or a parent path covers
// it, as in pi; only metadata for another origin or path is refused.
func TestDiscoverAcceptsCoveringResource(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	var as *httptest.Server
	as = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/.well-known/oauth-authorization-server" {
			http.NotFound(w, r)
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{
			"issuer":                           as.URL,
			"authorization_endpoint":           as.URL + "/authorize",
			"token_endpoint":                   as.URL + "/token",
			"response_types_supported":         []string{"code"},
			"code_challenge_methods_supported": []string{"S256"},
		})
	}))
	defer as.Close()

	const (
		pathWellKnown = "/.well-known/oauth-protected-resource/mcp"
		rootWellKnown = "/.well-known/oauth-protected-resource"
	)
	for _, tc := range []struct {
		name       string
		serverPath string
		// documents maps paths to the resource, after the origin, of the
		// metadata served there. An empty resource means status 500.
		documents map[string]string
		challenge string
		servers   []string // authorization servers; nil means the real one
		want      string   // "as", "origin" or an error
		resource  string   // the resource indicator, after the origin
	}{
		{name: "exact resource", serverPath: "/mcp", documents: map[string]string{pathWellKnown: "/mcp"}, want: "as", resource: "/mcp"},
		{name: "root server with trailing slash", documents: map[string]string{rootWellKnown: "/"}, want: "as", resource: "/"},
		{name: "endpoint with trailing slash", serverPath: "/mcp", documents: map[string]string{pathWellKnown: "/mcp/"}, want: "as", resource: "/mcp/"},
		{name: "root document naming the endpoint", serverPath: "/mcp", documents: map[string]string{rootWellKnown: "/mcp"}, want: "as", resource: "/mcp"},
		{name: "origin-wide via challenge", serverPath: "/mcp", documents: map[string]string{"/prm": ""}, challenge: "/prm", want: "as"},
		{name: "origin-wide at endpoint path", serverPath: "/mcp", documents: map[string]string{pathWellKnown: ""}, want: "as"},
		{name: "parent path", serverPath: "/v1/mcp", documents: map[string]string{"/.well-known/oauth-protected-resource/v1/mcp": "/v1"}, want: "as", resource: "/v1"},
		{name: "other path", serverPath: "/mcp", documents: map[string]string{pathWellKnown: "/other"}, want: "does not match"},
		{name: "path prefix within a segment", serverPath: "/mcpx", documents: map[string]string{rootWellKnown: "/mcp"}, want: "does not match"},
		{name: "other origin", serverPath: "/mcp", documents: map[string]string{pathWellKnown: "https://elsewhere.example/mcp"}, want: "does not match"},
		{name: "no metadata", serverPath: "/mcp", want: "origin"},
		{name: "no authorization servers", serverPath: "/mcp", documents: map[string]string{pathWellKnown: "/mcp"}, servers: []string{}, want: "origin", resource: "/mcp"},
		{name: "insecure authorization server", serverPath: "/mcp", documents: map[string]string{pathWellKnown: "/mcp"}, servers: []string{"http://as.example"}, want: "does not use HTTPS"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var mcp *httptest.Server
			mcp = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				resource, ok := tc.documents[r.URL.Path]
				if !ok {
					http.NotFound(w, r)
					return
				}
				if !strings.Contains(resource, "://") {
					resource = mcp.URL + resource
				}
				servers := tc.servers
				if servers == nil {
					servers = []string{as.URL}
				}
				writeJSON(w, http.StatusOK, map[string]any{"resource": resource, "authorization_servers": servers})
			}))
			defer mcp.Close()
			h, err := NewOAuthHandler(&OAuthOptions{
				Store: authstoretest.NewMemoryStore(), OwnerID: "user-1", Server: "s", ServerURL: mcp.URL + tc.serverPath,
			})
			if err != nil {
				t.Fatal(err)
			}
			challenge := ""
			if tc.challenge != "" {
				challenge = mcp.URL + tc.challenge
			}
			discovery, _, err := h.discover(ctx, challenge)
			switch tc.want {
			case "as", "origin":
				if err != nil {
					t.Fatal(err)
				}
				want := as.URL + "/authorize"
				if tc.want == "origin" {
					want = mcp.URL + "/authorize"
				}
				if got := discovery.AuthorizationServerMetadata.AuthorizationEndpoint; got != want {
					t.Fatalf("authorization endpoint %q, want %q", got, want)
				}
				wantResource := ""
				if tc.resource != "" || tc.want == "as" {
					wantResource = mcp.URL + tc.resource
				}
				if discovery.Resource != wantResource {
					t.Fatalf("resource %q, want %q", discovery.Resource, wantResource)
				}
			default:
				if err == nil || !strings.Contains(err.Error(), tc.want) {
					t.Fatalf("err = %v, want %q", err, tc.want)
				}
			}
		})
	}
}

// A path document that fails with a server error is not a miss, so the root
// document is not consulted, as in pi.
func TestDiscoverStopsAtServerError(t *testing.T) {
	var mcp *httptest.Server
	mcp = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/.well-known/oauth-protected-resource/mcp":
			http.Error(w, "down", http.StatusInternalServerError)
		case "/.well-known/oauth-protected-resource":
			writeJSON(w, http.StatusOK, map[string]any{"resource": mcp.URL + "/mcp", "authorization_servers": []string{"https://as.example"}})
		default:
			http.NotFound(w, r)
		}
	}))
	defer mcp.Close()
	h, err := NewOAuthHandler(&OAuthOptions{Store: authstoretest.NewMemoryStore(), OwnerID: "user-1", Server: "s", ServerURL: mcp.URL + "/mcp"})
	if err != nil {
		t.Fatal(err)
	}
	discovery, _, err := h.discover(t.Context(), "")
	if err != nil || discovery.AuthorizationServerURL != mcp.URL || discovery.Resource != "" {
		t.Fatalf("discovery = %+v, %v", discovery, err)
	}
}

// A server that rejects freshly refreshed tokens needs a sign-in. It costs
// one refresh, not one per request, and a token the server once accepted
// is refreshed normally when it later expires.
func TestRejectedRefreshedTokenNeedsSignIn(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	f := newFakeOAuthServer(t, nil)
	h := newHandler(t, f, authstoretest.NewMemoryStore(), &fakeClock{now: time.Now()})
	signIn(t, ctx, h)
	connection, _ := newConnection(t, nil, ServerConnectionOptions{Name: "fake", OAuth: h, CreateTransport: HTTPTransport(f.URL+"/mcp", nil)})
	if _, err := connection.GetClient(ctx); err != nil {
		t.Fatal(err)
	}
	expect := func(when string, refreshes int32, state ServerState) {
		t.Helper()
		if n := f.refreshN.Load(); n != refreshes || connection.Status().State != state {
			t.Fatalf("%s: %d refreshes, status %+v; want %d, %s", when, n, connection.Status(), refreshes, state)
		}
	}

	f.rejectTokens.Store(true)
	for i := range 2 {
		if _, err := connection.CallTool(ctx, "whoami", nil, nil); !errors.Is(err, ErrSignInRequired) {
			t.Fatalf("call %d: %v", i, err)
		}
		expect("rejected", 1, StateNeedsAuth)
	}

	f.rejectTokens.Store(false)
	if err := connection.Reconnect(ctx); err != nil {
		t.Fatal(err)
	}
	f.revoke(false)
	if err := callWhoami(ctx, mustClient(t, ctx, connection)); err != nil {
		t.Fatalf("call after the accepted token was revoked: %v", err)
	}
	expect("refreshed", 2, StateConnected)

	// Connecting costs one refresh too.
	f.rejectTokens.Store(true)
	if err := connection.Reconnect(ctx); !errors.Is(err, ErrSignInRequired) {
		t.Fatalf("reconnect: %v", err)
	}
	expect("reconnect rejected", 3, StateNeedsAuth)
}

func mustClient(t *testing.T, ctx context.Context, connection *ServerConnection) *sdk.ClientSession {
	t.Helper()
	session, err := connection.GetClient(ctx)
	if err != nil {
		t.Fatal(err)
	}
	return session
}

// Loopback and custom-scheme redirect URIs register native clients, as in
// pi (SEP-837): OpenID Connect servers assume web, which refuses them.
func TestRegistrationApplicationType(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	f := newFakeOAuthServer(t, nil)
	// A web app's https redirect URI.
	signIn(t, ctx, newHandler(t, f, authstoretest.NewMemoryStore(), &fakeClock{now: time.Now()}))
	// A TUI's loopback callback.
	h := newHandler(t, f, authstoretest.NewMemoryStore(), &fakeClock{now: time.Now()})
	if err := SignInMcpServer(ctx, h, SignInOptions{Prompt: &fakePrompt{t: t, browser: true}}); err != nil {
		t.Fatal(err)
	}
	f.mu.Lock()
	got := slices.Clone(f.applicationTypes)
	f.mu.Unlock()
	if !slices.Equal(got, []string{"web", "native"}) {
		t.Fatalf("application types = %v", got)
	}
	for redirect, want := range map[string]string{
		"http://localhost:8080/cb": "native",
		"http://[::1]/cb":          "native",
		"com.example.app:/cb":      "native",
		"https://app.example/cb":   "web",
		"http://app.example/cb":    "web",
	} {
		if got := applicationType(redirect); got != want {
			t.Errorf("applicationType(%q) = %q, want %q", redirect, got, want)
		}
	}
}

// A plain 403 is the answer: it is not sent again and does not break the
// connection, so other calls keep working, as in pi.
func TestPlainForbiddenKeepsConnection(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	f := newFakeOAuthServer(t, nil)
	h := newHandler(t, f, authstoretest.NewMemoryStore(), &fakeClock{now: time.Now()})
	signIn(t, ctx, h)
	connection, _ := newConnection(t, nil, ServerConnectionOptions{Name: "fake", OAuth: h, CreateTransport: HTTPTransport(f.URL+"/mcp", nil)})
	session := mustClient(t, ctx, connection)

	f.forbid.Store(true)
	before := f.posts.Load()
	var wg sync.WaitGroup
	for range 2 {
		wg.Go(func() {
			_, err := connection.CallTool(ctx, "whoami", nil, nil)
			if err == nil || !strings.Contains(err.Error(), "refused the request: 403 Forbidden") || errors.Is(err, ErrSignInRequired) {
				t.Errorf("forbidden call: %v", err)
			}
		})
	}
	wg.Wait()
	if n := f.posts.Load() - before; n != 2 {
		t.Fatalf("2 calls sent %d requests", n)
	}

	f.forbid.Store(false)
	if err := callWhoami(ctx, mustClient(t, ctx, connection)); err != nil {
		t.Fatal(err)
	}
	if mustClient(t, ctx, connection) != session || connection.Status().State != StateConnected || f.refreshN.Load() != 0 {
		t.Fatalf("status %+v, %d refreshes; the connection did not survive", connection.Status(), f.refreshN.Load())
	}
}

// AuthServerMetadataURL replaces discovery of the authorization server, as
// pi's authServerMetadataUrl does. The document is trusted: its issuer is
// another host than the one it is served from would suggest, and it lists
// no PKCE methods, which discovery refuses.
func TestAuthServerMetadataURL(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	f := newFakeOAuthServer(t, func(f *fakeOAuthServer) { f.noMetadata = true })
	var tokens atomic.Int32
	var as *httptest.Server
	as = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		document := map[string]any{
			"issuer":                   "https://issuer.example",
			"authorization_endpoint":   as.URL + "/authorize",
			"token_endpoint":           as.URL + "/token",
			"registration_endpoint":    as.URL + "/register",
			"response_types_supported": []string{"code"},
		}
		switch r.URL.Path {
		case "/meta":
			writeJSON(w, http.StatusOK, document)
		case "/partial":
			delete(document, "token_endpoint")
			writeJSON(w, http.StatusOK, document)
		case "/garbage":
			w.Write([]byte("<html>"))
		case "/token":
			tokens.Add(1)
			f.Config.Handler.ServeHTTP(w, r)
		case "/authorize", "/register":
			f.Config.Handler.ServeHTTP(w, r)
		default:
			http.NotFound(w, r)
		}
	}))
	defer as.Close()
	handler := func(serverURL, metadataPath string) *OAuthHandler {
		t.Helper()
		h, err := NewOAuthHandler(&OAuthOptions{
			Store: authstoretest.NewMemoryStore(), OwnerID: "user-1", Server: "fake", ServerURL: serverURL,
			RedirectURL: "https://app.example/mcp/callback/fake", AuthServerMetadataURL: as.URL + metadataPath,
		})
		if err != nil {
			t.Fatal(err)
		}
		return h
	}

	h := handler(f.URL+"/mcp", "/meta")
	authURL, err := h.StartSignIn(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(authURL, as.URL+"/authorize?") {
		t.Fatalf("authorization URL = %s", authURL)
	}
	// The response's issuer is checked against the configured document.
	response := authorize(t, authURL)
	withIss := url.Values{"code": {response.Get("code")}, "state": {response.Get("state")}, "iss": {as.URL}}
	if err := h.CompleteSignIn(ctx, withIss); err == nil || !strings.Contains(err.Error(), "issuer") {
		t.Fatalf("iss of another issuer: %v", err)
	}
	if err := h.CompleteSignIn(ctx, response); err != nil {
		t.Fatal(err)
	}
	session, err := connectHTTP(ctx, f, h)
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()
	if err := callWhoami(ctx, session); err != nil {
		t.Fatal(err)
	}
	if tokens.Load() != 1 {
		t.Fatalf("%d token requests to the configured authorization server", tokens.Load())
	}

	// It wins over the authorization server the resource metadata names, and
	// the resource is kept.
	withMetadata := newFakeOAuthServer(t, nil)
	discovery, scopes, err := handler(withMetadata.URL+"/mcp", "/meta").discover(ctx, "")
	if err != nil || discovery.AuthorizationServerURL != "https://issuer.example" ||
		discovery.Resource != withMetadata.URL+"/mcp" || !slices.Equal(scopes, []string{"tools"}) {
		t.Fatalf("discovery = %+v, %v, %v", discovery, scopes, err)
	}

	for path, want := range map[string]string{
		"/missing": "HTTP 404 loading authorization server metadata",
		"/garbage": "invalid authorization server metadata",
		"/partial": "lacks the issuer or an endpoint",
	} {
		if _, _, err := handler(f.URL+"/mcp", path).discover(ctx, ""); err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("%s: %v, want %q", path, err, want)
		}
	}
	if _, err := NewOAuthHandler(&OAuthOptions{
		Store: authstoretest.NewMemoryStore(), Server: "s", ServerURL: "https://s.example/mcp", AuthServerMetadataURL: "http://as.example/meta",
	}); err == nil {
		t.Fatal("accepted an authorization server metadata URL without HTTPS")
	}
}
