package mcp

import (
	"context"
	"errors"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/ColeBurch/burrow/baseAgent/authstoretest"
	"github.com/modelcontextprotocol/go-sdk/oauthex"
)

// fakePrompt acts as the user: the browser follows the authorization URL to
// the loopback callback, or the user pastes the redirect URL.
type fakePrompt struct {
	t *testing.T
	// browser follows the authorization URL.
	browser bool
	// paste returns what the user pastes given the authorization URL; nil
	// waits for ctx.
	paste func(authURL string) string

	mu   sync.Mutex
	urls []string
}

func (p *fakePrompt) ShowAuthorizationURL(authURL string) {
	p.mu.Lock()
	p.urls = append(p.urls, authURL)
	p.mu.Unlock()
	if !p.browser {
		return
	}
	go func() {
		resp, err := http.Get(authURL)
		if err != nil {
			p.t.Errorf("browser: %v", err)
			return
		}
		resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			p.t.Errorf("browser got %s", resp.Status)
		}
	}()
}

func (p *fakePrompt) PromptForRedirectURL(ctx context.Context) (string, error) {
	if p.paste == nil {
		<-ctx.Done()
		return "", ctx.Err()
	}
	return p.paste(p.lastURL()), nil
}

func (p *fakePrompt) lastURL() string {
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.urls) == 0 {
		return ""
	}
	return p.urls[len(p.urls)-1]
}

func (p *fakePrompt) shown() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.urls)
}

// redirectFrom is the redirect URL the authorization server sends the
// browser to.
func redirectFrom(t *testing.T, authURL string) string {
	t.Helper()
	client := &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	resp, err := client.Get(authURL)
	if err != nil {
		t.Error(err)
		return ""
	}
	resp.Body.Close()
	return resp.Header.Get("Location")
}

func authQuery(t *testing.T, authURL string) url.Values {
	t.Helper()
	u, err := url.Parse(authURL)
	if err != nil {
		t.Fatal(err)
	}
	return u.Query()
}

// Signing in from a TUI: the browser reaches the loopback callback, the
// server reconnects, a later sign-in refreshes without the browser, and
// logging out disconnects.
func TestExtensionSignInWithBrowser(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	f := newFakeOAuthServer(t, func(f *fakeOAuthServer) { f.issSupported = true })
	h := newHandler(t, f, authstoretest.NewMemoryStore(), &fakeClock{now: time.Now()})
	registry := newFakeRegistry()
	ext := newExtension(t, registry, ServerConnectionOptions{Name: "fake", CreateTransport: HTTPTransport(f.URL+"/mcp", nil), OAuth: h})
	ext.Start()
	ext.WaitForStartup(ctx)
	connection := ext.Connection("fake")
	if state := connection.Status().State; state != StateNeedsAuth {
		t.Fatalf("state before sign-in = %s", state)
	}

	prompt := &fakePrompt{t: t, browser: true}
	if err := ext.SignIn(ctx, "fake", SignInOptions{Prompt: prompt}); err != nil {
		t.Fatal(err)
	}
	redirect, err := url.Parse(authQuery(t, prompt.lastURL()).Get("redirect_uri"))
	if err != nil || redirect.Hostname() != "127.0.0.1" || redirect.Port() == "" || redirect.Path != "/callback" {
		t.Fatalf("redirect_uri = %v", redirect)
	}
	if connection.Status().State != StateConnected {
		t.Fatalf("status after sign-in = %+v", connection.Status())
	}
	if got, err := callRegistered(t, ctx, registry, "mcp__fake__whoami"); err != nil || got != "you" {
		t.Fatalf("whoami = %q, %v", got, err)
	}

	// Signing in again refreshes on the registered redirect URI's port.
	refreshes := f.refreshN.Load()
	again := &fakePrompt{t: t}
	if err := SignInMcpServer(ctx, h, SignInOptions{Prompt: again}); err != nil {
		t.Fatal(err)
	}
	if again.shown() != 0 || f.refreshN.Load() != refreshes+1 {
		t.Fatalf("shown %d URLs, %d refreshes", again.shown(), f.refreshN.Load()-refreshes)
	}

	if err := ext.Logout(ctx, "fake"); err != nil {
		t.Fatal(err)
	}
	if signedIn, err := h.SignedIn(ctx); err != nil || signedIn || connection.Status().State != StateNeedsAuth {
		t.Fatalf("after logout: signed in %v (%v), status %+v", signedIn, err, connection.Status())
	}
}

// Over SSH the browser cannot reach the callback; the user pastes the
// redirect URL instead.
func TestSignInWithPastedRedirectURL(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	f := newFakeOAuthServer(t, nil)
	h := newHandler(t, f, authstoretest.NewMemoryStore(), &fakeClock{now: time.Now()})
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := listener.Addr().(*net.TCPAddr).Port
	listener.Close()
	callbackURL := "http://localhost:" + strconv.Itoa(port) + "/oauth/done"

	prompt := &fakePrompt{t: t, paste: func(authURL string) string { return redirectFrom(t, authURL) }}
	if err := SignInMcpServer(ctx, h, SignInOptions{Prompt: prompt, CallbackURL: callbackURL}); err != nil {
		t.Fatal(err)
	}
	// A callback URL with a port is sent exactly as written.
	if got := authQuery(t, prompt.lastURL()).Get("redirect_uri"); got != callbackURL {
		t.Fatalf("redirect_uri = %q", got)
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

func TestSignInCancelled(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	f := newFakeOAuthServer(t, nil)
	h := newHandler(t, f, authstoretest.NewMemoryStore(), &fakeClock{now: time.Now()})

	if err := SignInMcpServer(ctx, h, SignInOptions{}); err == nil || !strings.Contains(err.Error(), "Prompt is required") {
		t.Fatalf("no prompt: %v", err)
	}
	empty := &fakePrompt{t: t, paste: func(string) string { return "  " }}
	if err := SignInMcpServer(ctx, h, SignInOptions{Prompt: empty}); !errors.Is(err, ErrSignInCancelled) {
		t.Fatalf("empty answer: %v", err)
	}
	waiting, stop := context.WithCancel(ctx)
	waitingPrompt := &fakePrompt{t: t}
	go func() {
		for waitingPrompt.shown() == 0 {
			time.Sleep(5 * time.Millisecond)
		}
		stop()
	}()
	if err := SignInMcpServer(waiting, h, SignInOptions{Prompt: waitingPrompt}); !errors.Is(err, ErrSignInCancelled) {
		t.Fatalf("cancelled: %v", err)
	}
	if err := SignInMcpServer(waiting, h, SignInOptions{Prompt: waitingPrompt}); !errors.Is(err, ErrSignInCancelled) {
		t.Fatalf("cancelled before: %v", err)
	}
	wrong := &fakePrompt{t: t, paste: func(authURL string) string {
		return strings.Replace(redirectFrom(t, authURL), "state=", "state=x", 1)
	}}
	if err := SignInMcpServer(ctx, h, SignInOptions{Prompt: wrong}); err == nil || !strings.Contains(err.Error(), "different sign-in") {
		t.Fatalf("wrong state: %v", err)
	}
	if signedIn, _ := h.SignedIn(ctx); signedIn {
		t.Fatal("signed in")
	}
	if _, err := newCallbackSettings(SignInOptions{CallbackURL: "https://127.0.0.1/cb"}); err == nil {
		t.Fatal("accepted an https callback URL")
	}
}

func TestResponseFromRedirectURL(t *testing.T) {
	redirect, _ := url.Parse("http://127.0.0.1:1234/callback")
	for _, tc := range []struct {
		input, code, iss, err string
	}{
		{input: " http://127.0.0.1:1234/callback?code=c&state=s&iss=https://as \n", code: "c", iss: "https://as"},
		{input: "code=c&state=s", err: "full redirect URL"},
		{input: "http://127.0.0.1:1234/other?code=c&state=s", err: "does not match"},
		{input: "http://127.0.0.1:9999/callback?code=c&state=s", err: "does not match"},
		{input: "http://127.0.0.1:1234/callback?error=access_denied&state=s", err: "access_denied"},
		{input: "http://127.0.0.1:1234/callback?error=x&error_description=Denied&state=s", err: "Denied"},
		{input: "http://127.0.0.1:1234/callback?code=c&state=t", err: "different sign-in"},
		{input: "http://127.0.0.1:1234/callback?state=s", err: "authorization code"},
	} {
		code, iss, err := responseFromRedirectURL(tc.input, "s", redirect)
		if tc.err != "" {
			if err == nil || !strings.Contains(err.Error(), tc.err) {
				t.Errorf("%q: err = %v, want %q", tc.input, err, tc.err)
			}
			continue
		}
		if err != nil || code != tc.code || iss != tc.iss {
			t.Errorf("%q: %q %q %v", tc.input, code, iss, err)
		}
	}
}

// With a Client ID Metadata Document the app does not register a client.
// Without the iss parameter the document and redirect URI are specific to
// the server.
func TestSignInWithClientMetadataDocument(t *testing.T) {
	for _, iss := range []bool{false, true} {
		t.Run("iss="+strconv.FormatBool(iss), func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
			defer cancel()
			f := newFakeOAuthServer(t, func(f *fakeOAuthServer) { f.cimd, f.issSupported = true, iss })
			h, err := NewOAuthHandler(&OAuthOptions{
				Store:                  authstoretest.NewMemoryStore(),
				OwnerID:                "user-1",
				Server:                 "fake",
				ServerURL:              f.URL + "/mcp",
				ClientMetadataDocument: HostedClientMetadataDocument("https://app.example/oauth/"),
			})
			if err != nil {
				t.Fatal(err)
			}
			prompt := &fakePrompt{t: t, browser: true}
			if err := SignInMcpServer(ctx, h, SignInOptions{Prompt: prompt}); err != nil {
				t.Fatal(err)
			}
			id, err := CallbackID(f.URL + "/mcp")
			if err != nil {
				t.Fatal(err)
			}
			query := authQuery(t, prompt.lastURL())
			redirect, _ := url.Parse(query.Get("redirect_uri"))
			wantClient, wantPath := "https://app.example/oauth/"+id+"/client.json", "/callback/"+id
			if iss {
				wantClient, wantPath = "https://app.example/oauth/client.json", "/callback"
			}
			if query.Get("client_id") != wantClient || redirect.Path != wantPath {
				t.Fatalf("client_id %q, redirect_uri %q", query.Get("client_id"), query.Get("redirect_uri"))
			}
			f.mu.Lock()
			registered := len(f.clients)
			f.mu.Unlock()
			if registered != 0 {
				t.Fatalf("registered %d clients", registered)
			}
			session, err := connectHTTP(ctx, f, h)
			if err != nil {
				t.Fatal(err)
			}
			defer session.Close()
			if err := callWhoami(ctx, session); err != nil {
				t.Fatal(err)
			}
		})
	}

	// Without support the sign-in fails rather than registering.
	f := newFakeOAuthServer(t, nil)
	h, err := NewOAuthHandler(&OAuthOptions{
		Store: authstoretest.NewMemoryStore(), OwnerID: "user-1", Server: "fake", ServerURL: f.URL + "/mcp",
		RedirectURL:            "https://app.example/callback",
		ClientMetadataDocument: HostedClientMetadataDocument("https://app.example/oauth"),
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := h.StartSignIn(t.Context()); err == nil || !strings.Contains(err.Error(), "does not support Client ID Metadata Documents") {
		t.Fatalf("err = %v", err)
	}
	invalid, err := NewOAuthHandler(&OAuthOptions{
		Store: authstoretest.NewMemoryStore(), OwnerID: "user-1", Server: "fake", ServerURL: f.URL + "/mcp",
		RedirectURL: "https://app.example/callback",
		ClientMetadataDocument: func(string, string, *oauthex.AuthServerMeta) (*ClientMetadataDocument, error) {
			return &ClientMetadataDocument{URL: "https://app.example/", RedirectURL: "https://app.example/callback"}, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := invalid.StartSignIn(t.Context()); err == nil || !strings.Contains(err.Error(), "invalid OAuth client metadata URL") {
		t.Fatalf("root document URL: %v", err)
	}
}

func TestCallbackID(t *testing.T) {
	id, err := CallbackID("https://mcp.example.com/mcp")
	if err != nil || len(id) != 12 || strings.ContainsAny(id, "+/=") {
		t.Fatalf("id = %q, %v", id, err)
	}
	for _, same := range []string{"https://mcp.example.com/mcp#frag", "https://MCP.example.com/mcp"} {
		if other, _ := CallbackID(same); other != id {
			t.Errorf("CallbackID(%q) = %q, want %q", same, other, id)
		}
	}
	if root, _ := CallbackID("https://mcp.example.com"); root == id {
		t.Fatal("different URLs share an ID")
	}
	if slash, _ := CallbackID("https://mcp.example.com/"); slash != func() string { s, _ := CallbackID("https://mcp.example.com"); return s }() {
		t.Fatal("an empty path differs from /")
	}
}

// A shared callback finds the sign-in its response belongs to, among the
// signed-in user's sign-ins only.
func TestFindPendingSignIn(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	f := newFakeOAuthServer(t, nil)
	store := authstoretest.NewMemoryStore()
	h := newHandler(t, f, store, &fakeClock{now: time.Now()})
	authURL, err := h.StartSignIn(ctx)
	if err != nil {
		t.Fatal(err)
	}
	response := authorize(t, authURL)
	state := response.Get("state")

	pending, err := FindPendingSignIn(ctx, store, "user-1", state)
	if err != nil || *pending != (PendingSignIn{Server: "fake", ServerURL: f.URL + "/mcp"}) {
		t.Fatalf("pending = %+v, %v", pending, err)
	}
	for _, tc := range []struct{ owner, state string }{{"user-2", state}, {"user-1", state + "x"}, {"user-1", ""}} {
		if _, err := FindPendingSignIn(ctx, store, tc.owner, tc.state); !errors.Is(err, ErrNoPendingSignIn) {
			t.Errorf("FindPendingSignIn(%q, %q) = %v", tc.owner, tc.state, err)
		}
	}
	if err := h.CompleteSignIn(ctx, response); err != nil {
		t.Fatal(err)
	}
	if _, err := FindPendingSignIn(ctx, store, "user-1", state); !errors.Is(err, ErrNoPendingSignIn) {
		t.Fatalf("after completing: %v", err)
	}
}

// A custom Client ID Metadata Document may name any redirect path on the
// loopback callback; one elsewhere fails at once rather than waiting.
func TestSignInWithCustomMetadataDocumentRedirect(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	f := newFakeOAuthServer(t, func(f *fakeOAuthServer) { f.cimd, f.issSupported = true, true })
	handler := func(redirect func(*url.URL)) *OAuthHandler {
		h, err := NewOAuthHandler(&OAuthOptions{
			Store: authstoretest.NewMemoryStore(), OwnerID: "user-1", Server: "fake", ServerURL: f.URL + "/mcp",
			ClientMetadataDocument: func(_, redirectURL string, _ *oauthex.AuthServerMeta) (*ClientMetadataDocument, error) {
				u, err := url.Parse(redirectURL)
				if err != nil {
					return nil, err
				}
				redirect(u)
				return &ClientMetadataDocument{URL: "https://app.example/client.json", RedirectURL: u.String()}, nil
			},
		})
		if err != nil {
			t.Fatal(err)
		}
		return h
	}

	h := handler(func(u *url.URL) { u.Path = "/oauth/done" })
	prompt := &fakePrompt{t: t, browser: true}
	if err := SignInMcpServer(ctx, h, SignInOptions{Prompt: prompt}); err != nil {
		t.Fatal(err)
	}
	if redirect, _ := url.Parse(authQuery(t, prompt.lastURL()).Get("redirect_uri")); redirect.Path != "/oauth/done" {
		t.Fatalf("redirect_uri = %v", redirect)
	}
	if signedIn, err := h.SignedIn(ctx); err != nil || !signedIn {
		t.Fatalf("signed in %v, %v", signedIn, err)
	}

	elsewhere := handler(func(u *url.URL) { u.Host = "127.0.0.1:1" })
	unused := &fakePrompt{t: t}
	if err := SignInMcpServer(ctx, elsewhere, SignInOptions{Prompt: unused}); err == nil || !strings.Contains(err.Error(), "not on the loopback callback") {
		t.Fatalf("redirect elsewhere: %v", err)
	}
	if unused.shown() != 0 {
		t.Fatal("showed an authorization URL that cannot complete")
	}
}
