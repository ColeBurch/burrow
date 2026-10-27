// Adapted from pi's packages/mcp/src/oauth/callback.ts (MIT, see LICENSE).

package mcp

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"slices"
	"strconv"
	"sync"
	"time"
)

// OAuthCallback is an authorization response.
type OAuthCallback struct {
	Code  string
	State string
	Iss   string
}

// OAuthCallbackPage is the outcome shown on the browser page after the
// redirect.
type OAuthCallbackPage struct {
	OK      bool
	Message string
	Details string
}

// OAuthCallbackServerOptions configures [ListenOAuthCallback].
type OAuthCallbackServerOptions struct {
	// Host is the address to listen on. Empty means 127.0.0.1.
	Host string
	// RedirectHost is the host name in the redirect URL, for example
	// localhost for a client registered with it while listening on
	// 127.0.0.1. Empty means Host.
	RedirectHost string
	// Port is the port to listen on. Zero means a free port.
	Port int
	// Path receives the callback. Empty means /callback.
	Path string
	// ExtraPaths receive the callback too, for example the server-specific
	// path of a redirect URI.
	ExtraPaths []string
	// Timeout bounds each wait. Zero means 5 minutes.
	Timeout time.Duration
	// RenderPage renders the browser page as HTML. Nil means a plain-text
	// message.
	RenderPage func(OAuthCallbackPage) string
}

// OAuthCallbackServer receives authorization responses on a loopback
// address, for apps such as a TUI where the browser runs on the user's
// machine.
type OAuthCallbackServer struct {
	redirectURL string
	server      *http.Server
	timeout     time.Duration
	renderPage  func(OAuthCallbackPage) string

	mu sync.Mutex
	// paths receive callbacks: Path, ExtraPaths and the paths of waits.
	paths   []string
	pending map[string]*pendingCallback
}

type pendingCallback struct {
	path   string
	result chan callbackResult
}

type callbackResult struct {
	callback *OAuthCallback
	err      error
}

// ListenOAuthCallback starts a callback server. Close it when done.
func ListenOAuthCallback(opts OAuthCallbackServerOptions) (*OAuthCallbackServer, error) {
	host := cmp.Or(opts.Host, "127.0.0.1")
	redirectHost := cmp.Or(opts.RedirectHost, host)
	path := cmp.Or(opts.Path, "/callback")
	listener, err := net.Listen("tcp", net.JoinHostPort(host, strconv.Itoa(opts.Port)))
	if err != nil {
		return nil, err
	}
	port := listener.Addr().(*net.TCPAddr).Port
	s := &OAuthCallbackServer{
		redirectURL: "http://" + net.JoinHostPort(redirectHost, strconv.Itoa(port)) + path,
		paths:       append([]string{path}, opts.ExtraPaths...),
		timeout:     cmp.Or(opts.Timeout, 5*time.Minute),
		renderPage:  opts.RenderPage,
		pending:     map[string]*pendingCallback{},
	}
	s.server = &http.Server{Handler: http.HandlerFunc(s.handle), ReadHeaderTimeout: 10 * time.Second}
	go func() { _ = s.server.Serve(listener) }()
	return s, nil
}

// RedirectURL is the URL of Path on the port the server listens on.
func (s *OAuthCallbackServer) RedirectURL() string { return s.redirectURL }

// WaitForCallback expects the authorization response with state, and
// returns a function that waits for it. Call it before sending the user to
// the authorization URL, since a response nobody expects is refused. With a
// path, the server receives responses on it too, and a response on another
// path fails, so a server-specific redirect URI can tell authorization
// servers apart (RFC 9700 section 4.4.2.2).
func (s *OAuthCallbackServer) WaitForCallback(state, path string) (func(ctx context.Context) (*OAuthCallback, error), error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.pending == nil {
		return nil, errors.New("OAuth callback server closed")
	}
	if _, ok := s.pending[state]; ok {
		return nil, errors.New("OAuth state is already pending")
	}
	if path != "" && !slices.Contains(s.paths, path) {
		s.paths = append(s.paths, path)
	}
	pending := &pendingCallback{path: path, result: make(chan callbackResult, 1)}
	s.pending[state] = pending
	return func(ctx context.Context) (*OAuthCallback, error) {
		timer := time.NewTimer(s.timeout)
		defer timer.Stop()
		select {
		case result := <-pending.result:
			return result.callback, result.err
		case <-timer.C:
			s.forget(state, pending)
			return nil, errors.New("OAuth callback timed out")
		case <-ctx.Done():
			s.forget(state, pending)
			return nil, ctx.Err()
		}
	}, nil
}

func (s *OAuthCallbackServer) forget(state string, pending *pendingCallback) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.pending[state] == pending {
		delete(s.pending, state)
	}
}

// Close fails the waits in progress and stops the server.
func (s *OAuthCallbackServer) Close() error {
	s.mu.Lock()
	for _, pending := range s.pending {
		pending.result <- callbackResult{err: errors.New("OAuth callback server closed")}
	}
	s.pending = nil
	s.mu.Unlock()
	return s.server.Close()
}

func (s *OAuthCallbackServer) reply(w http.ResponseWriter, status int, page OAuthCallbackPage) {
	w.Header().Set("Cache-Control", "no-store")
	if s.renderPage != nil {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.WriteHeader(status)
		_, _ = w.Write([]byte(s.renderPage(page)))
		return
	}
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.WriteHeader(status)
	text := "Authorization complete. You may close this window."
	if !page.OK {
		text = page.Message
		if page.Details != "" {
			text = fmt.Sprintf("%s\n\n%s", page.Message, page.Details)
		}
	}
	_, _ = w.Write([]byte(text))
}

func (s *OAuthCallbackServer) handle(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	known := slices.Contains(s.paths, r.URL.Path)
	s.mu.Unlock()
	if !known {
		s.reply(w, http.StatusNotFound, OAuthCallbackPage{Message: "Not found"})
		return
	}
	query := r.URL.Query()
	state := query.Get("state")
	s.mu.Lock()
	pending := s.pending[state]
	if pending != nil {
		delete(s.pending, state)
	}
	s.mu.Unlock()
	if state == "" || pending == nil {
		s.reply(w, http.StatusBadRequest, OAuthCallbackPage{Message: "Invalid or expired OAuth state"})
		return
	}
	if pending.path != "" && r.URL.Path != pending.path {
		pending.result <- callbackResult{err: errors.New("the authorization response arrived on another redirect URI")}
		s.reply(w, http.StatusBadRequest, OAuthCallbackPage{Message: "Unexpected redirect URI"})
		return
	}
	if code := query.Get("error"); code != "" {
		description := cmp.Or(query.Get("error_description"), code)
		pending.result <- callbackResult{err: errors.New(description)}
		s.reply(w, http.StatusOK, OAuthCallbackPage{Message: "Authorization failed. You may close this window.", Details: description})
		return
	}
	code := query.Get("code")
	if code == "" {
		pending.result <- callbackResult{err: errors.New("OAuth callback did not include an authorization code")}
		s.reply(w, http.StatusBadRequest, OAuthCallbackPage{Message: "Missing authorization code"})
		return
	}
	pending.result <- callbackResult{callback: &OAuthCallback{Code: code, State: state, Iss: query.Get("iss")}}
	s.reply(w, http.StatusOK, OAuthCallbackPage{OK: true})
}
