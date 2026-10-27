// Adapted from pi's packages/coding-agent/src/extensions/mcp/oauth.ts
// (MIT, see LICENSE).

package mcp

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/url"
	"slices"
	"strconv"
	"strings"
)

// ErrSignInCancelled is returned by [SignInMcpServer] when the user or ctx
// cancels the sign-in.
var ErrSignInCancelled = errors.New("sign-in cancelled")

// SignInPrompt asks the user to authorize in a browser.
type SignInPrompt interface {
	// ShowAuthorizationURL shows the authorization URL to the user and opens
	// it in a browser.
	ShowAuthorizationURL(url string)
	// PromptForRedirectURL asks for the redirect URL from the browser
	// address bar, for when the browser cannot reach the loopback callback
	// (for example over SSH). ctx is cancelled once the callback arrives or
	// the sign-in is cancelled. An empty answer cancels the sign-in. A
	// prompt that cannot ask waits for ctx and returns its error.
	PromptForRedirectURL(ctx context.Context) (string, error)
}

// SignInOptions configures [SignInMcpServer].
type SignInOptions struct {
	// Prompt shows the authorization URL and asks for the redirect URL.
	// Required.
	Prompt SignInPrompt
	// CallbackURL is the loopback redirect URI. Empty means
	// http://127.0.0.1/callback. One with a port is sent exactly as
	// written, since servers compare it as a string; localhost is served on
	// 127.0.0.1.
	CallbackURL string
	// CallbackPort is the port of a CallbackURL without one. Zero means the
	// port of the stored client's redirect URI, so it stays valid, else a
	// free port.
	CallbackPort int
	// RenderPage renders the browser page after the redirect as HTML. Nil
	// means a plain-text message.
	RenderPage func(OAuthCallbackPage) string
}

// callbackSettings are where the loopback callback server listens and the
// redirect URI it serves.
type callbackSettings struct {
	host         string
	redirectHost string
	port         int
	path         string
	// fixedRedirectURL is the exact redirect URI, when the port is fixed.
	fixedRedirectURL string
}

func newCallbackSettings(opts SignInOptions) (callbackSettings, error) {
	raw := opts.CallbackURL
	if raw == "" {
		raw = "http://127.0.0.1/callback"
	}
	u, err := url.Parse(raw)
	if err != nil || u.Scheme != "http" || u.Hostname() == "" {
		return callbackSettings{}, fmt.Errorf("mcp: invalid callback URL %q", raw)
	}
	settings := callbackSettings{
		host:         u.Hostname(),
		redirectHost: u.Hostname(),
		port:         opts.CallbackPort,
		path:         u.Path,
	}
	if settings.path == "" {
		settings.path = "/"
	}
	if u.Port() != "" {
		settings.port, _ = strconv.Atoi(u.Port())
		settings.fixedRedirectURL = opts.CallbackURL
	} else if settings.port != 0 {
		u.Host = net.JoinHostPort(u.Hostname(), strconv.Itoa(settings.port))
		settings.fixedRedirectURL = u.String()
	}
	// Browsers fall back to 127.0.0.1 when ::1 refuses localhost.
	if settings.host == "localhost" {
		settings.host = "127.0.0.1"
	}
	return settings, nil
}

// SignInMcpServer signs the user in to handler's server from a local app,
// such as a TUI: it uses the stored refresh token when possible, and
// otherwise runs the browser authorization code flow against a loopback
// callback. The tokens are stored like CompleteSignIn stores them.
// Cancelling ctx stops the sign-in with ErrSignInCancelled; requests to
// the authorization server are also time-limited.
func SignInMcpServer(ctx context.Context, handler *OAuthHandler, opts SignInOptions) error {
	if opts.Prompt == nil {
		return errors.New("mcp: SignInOptions.Prompt is required")
	}
	if ctx.Err() != nil {
		return ErrSignInCancelled
	}
	err := signInMcpServer(ctx, handler, opts)
	if err != nil && ctx.Err() != nil {
		// Cancelled requests fail with the context's error; report them as
		// the cancellation they are.
		return ErrSignInCancelled
	}
	return err
}

func signInMcpServer(ctx context.Context, handler *OAuthHandler, opts SignInOptions) error {
	stored, err := handler.load(ctx)
	if err != nil {
		return err
	}
	var storedClient *oauthClient
	if stored != nil {
		storedClient = stored.Client
	}
	stepUp := stored != nil && stored.Challenge != nil && stored.Challenge.Error == "insufficient_scope"
	settings, err := newCallbackSettings(opts)
	if err != nil {
		return err
	}
	// Reuse the port of the registered redirect URI so the registered
	// client stays valid.
	preferredPort := settings.port
	if preferredPort == 0 && storedClient != nil && !storedClient.MetadataDocument && len(storedClient.RedirectURIs) > 0 {
		if registered, err := url.Parse(storedClient.RedirectURIs[0]); err == nil {
			preferredPort, _ = strconv.Atoi(registered.Port())
		}
	}
	cimd := handler.opts.ClientID == "" && handler.opts.ClientMetadataDocument != nil
	callback, err := listenForCallback(settings, preferredPort, settings.port != 0, opts.RenderPage)
	if err != nil {
		return err
	}
	defer callback.Close()
	redirectURL := settings.fixedRedirectURL
	if redirectURL == "" {
		redirectURL = callback.RedirectURL()
	}

	// A registered client cannot use another redirect URI, and its tokens
	// belong to it, so with another redirect URI the sign-in registers a new
	// client and cannot refresh. A refresh keeps the granted scope, so a
	// server asking for more needs the browser flow too.
	keepClient := handler.opts.ClientID != "" ||
		(cimd && (storedClient == nil || storedClient.MetadataDocument)) ||
		(!cimd && storedClient != nil && !storedClient.MetadataDocument && slices.Contains(storedClient.RedirectURIs, redirectURL))
	if keepClient && !stepUp && stored != nil && stored.Tokens != nil && stored.Tokens.RefreshToken != "" {
		if _, err := handler.refresh(ctx, stored.Tokens.AccessToken, nil); err == nil {
			return nil
		}
	}

	authURL, err := handler.startSignIn(ctx, redirectURL)
	if err != nil {
		return err
	}
	parsed, err := url.Parse(authURL)
	if err != nil {
		return err
	}
	state := parsed.Query().Get("state")
	// The flow picks the redirect URI: a Client ID Metadata Document may
	// name one specific to the MCP server, on any path of the callback.
	authorizationRedirect, err := url.Parse(parsed.Query().Get("redirect_uri"))
	if err != nil {
		return err
	}
	if listening, _ := url.Parse(redirectURL); authorizationRedirect.Scheme != listening.Scheme || authorizationRedirect.Host != listening.Host {
		return fmt.Errorf("mcp: the redirect URI %s is not on the loopback callback %s", authorizationRedirect, redirectURL)
	}
	waitForBrowser, err := callback.WaitForCallback(state, authorizationRedirect.Path)
	if err != nil {
		return err
	}
	opts.Prompt.ShowAuthorizationURL(authURL)
	code, iss, err := waitForAuthorizationResponse(ctx, waitForBrowser, state, authorizationRedirect, opts.Prompt)
	if err != nil {
		return err
	}
	query := url.Values{"code": {code}, "state": {state}}
	if iss != "" {
		query.Set("iss", iss)
	}
	return handler.CompleteSignIn(ctx, query)
}

// listenForCallback listens on port, or on a free port when it is taken and
// not required.
func listenForCallback(settings callbackSettings, port int, required bool, renderPage func(OAuthCallbackPage) string) (*OAuthCallbackServer, error) {
	opts := OAuthCallbackServerOptions{
		Host:         settings.host,
		RedirectHost: settings.redirectHost,
		Port:         port,
		Path:         settings.path,
		RenderPage:   renderPage,
	}
	server, err := ListenOAuthCallback(opts)
	if err != nil && !required && port != 0 {
		opts.Port = 0
		return ListenOAuthCallback(opts)
	}
	return server, err
}

// waitForAuthorizationResponse waits for the browser callback or a pasted
// redirect URL, whichever comes first.
func waitForAuthorizationResponse(
	ctx context.Context,
	waitForBrowser func(context.Context) (*OAuthCallback, error),
	state string,
	redirect *url.URL,
	prompt SignInPrompt,
) (code, iss string, err error) {
	type response struct {
		code, iss string
		err       error
	}
	ctx, cancel := context.WithCancel(ctx)
	// The losing side returns once ctx is cancelled.
	defer cancel()
	responses := make(chan response, 2)
	go func() {
		callback, err := waitForBrowser(ctx)
		if err != nil {
			responses <- response{err: err}
			return
		}
		responses <- response{code: callback.Code, iss: callback.Iss}
	}()
	go func() {
		input, err := prompt.PromptForRedirectURL(ctx)
		switch {
		case err != nil:
			responses <- response{err: err}
		case strings.TrimSpace(input) == "":
			responses <- response{err: ErrSignInCancelled}
		default:
			code, iss, err := responseFromRedirectURL(input, state, redirect)
			responses <- response{code: code, iss: iss, err: err}
		}
	}()
	first := <-responses
	return first.code, first.iss, first.err
}

// responseFromRedirectURL is the authorization response in a redirect URL
// the user pasted.
func responseFromRedirectURL(input, state string, redirect *url.URL) (code, iss string, err error) {
	u, err := url.Parse(strings.TrimSpace(input))
	if err != nil || u.Scheme == "" || u.Host == "" {
		return "", "", errors.New("expected the full redirect URL from the browser address bar")
	}
	// A server-specific redirect URI tells authorization servers apart, so
	// it must match exactly.
	if u.Scheme != redirect.Scheme || u.Host != redirect.Host || u.Path != redirect.Path {
		return "", "", errors.New("the redirect URL does not match this sign-in's redirect URI")
	}
	query := u.Query()
	if errorCode := query.Get("error"); errorCode != "" {
		if description := query.Get("error_description"); description != "" {
			return "", "", errors.New(description)
		}
		return "", "", errors.New(errorCode)
	}
	if query.Get("state") != state {
		return "", "", errors.New("the redirect URL belongs to a different sign-in")
	}
	code = query.Get("code")
	if code == "" {
		return "", "", errors.New("the redirect URL does not contain an authorization code")
	}
	return code, query.Get("iss"), nil
}
