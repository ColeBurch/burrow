// Adapted from pi's packages/coding-agent/src/extensions/mcp/oauth.ts and
// packages/mcp/src/oauth/flow.ts (MIT, see LICENSE). Discovery and client
// registration use github.com/modelcontextprotocol/go-sdk/oauthex.

package mcp

import (
	"cmp"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/ColeBurch/burrow/baseAgent"
	"github.com/modelcontextprotocol/go-sdk/auth"
	"github.com/modelcontextprotocol/go-sdk/oauthex"
	"golang.org/x/oauth2"
)

// OAuthNamespace is the AuthStore namespace of MCP OAuth state.
const OAuthNamespace = "mcp"

const (
	// refreshSkew is how close to expiry an access token is refreshed
	// before it is sent.
	refreshSkew = 30 * time.Second
	// defaultOAuthRequestTimeout bounds each request to the authorization
	// server, so an unresponsive one cannot hold the store lock for long.
	defaultOAuthRequestTimeout = 15 * time.Second
	// signInExpiry is how long a started sign-in can be completed.
	signInExpiry = 30 * time.Minute
	// maxStoreNameLength is the longest name an AuthStore accepts.
	maxStoreNameLength = 1024
	oauthStateVersion  = 1
)

// ErrSignInRequired is returned, wrapped, when only a new sign-in gives
// access to a server: nothing is stored, the refresh token is gone or was
// rejected, or the server asks for more scope. Connect and CallTool errors
// wrap it, so check them with errors.Is and offer StartSignIn.
var ErrSignInRequired = errors.New("sign-in required")

// ErrNoPendingSignIn is returned when no started sign-in matches an
// authorization response's state.
var ErrNoPendingSignIn = errors.New("mcp: no sign-in is waiting for this response; sign in again")

// OAuthError is an error response of an authorization server's token
// endpoint (RFC 6749 section 5.2).
type OAuthError struct {
	Code        string
	Description string
	StatusCode  int
}

func (e *OAuthError) Error() string {
	if e.Description != "" {
		return fmt.Sprintf("oauth error %s: %s", e.Code, e.Description)
	}
	return "oauth error " + e.Code
}

// OAuthOptions configures an OAuthHandler.
type OAuthOptions struct {
	// Store keeps the state of each owner's sign-in: client registration,
	// tokens and the pending authorization request. Required.
	Store baseAgent.AuthStore
	// OwnerID is the user or tenant the credentials belong to.
	OwnerID string
	// Server names the server. With ServerURL it keys the stored state, so
	// servers sharing a URL keep separate accounts. Required.
	Server string
	// ServerURL is the MCP endpoint. Required.
	ServerURL string
	// HTTPClient sends discovery, registration and token requests. Apps that
	// let users add servers should pass a client that refuses private
	// addresses, and use it for the transport too. Nil means a default client.
	HTTPClient *http.Client
	// RequestTimeout bounds each request to the authorization server. Zero
	// means 15 seconds.
	RequestTimeout time.Duration

	// RedirectURL receives the authorization response; pass its query to
	// CompleteSignIn. It may be specific to the server, so the callback
	// knows which handler to complete. Required to sign in.
	RedirectURL string
	// ClientID and ClientSecret identify a client registered with the
	// authorization server beforehand. Without ClientID, sign-in registers
	// one dynamically (RFC 7591) and stores it.
	ClientID     string
	ClientSecret string
	// ClientName is the client_name of dynamic registration. Empty means
	// "burrow".
	ClientName string
	// ClientMetadataDocument, when set and ClientID is empty, identifies the
	// client with a Client ID Metadata Document instead of registering it
	// dynamically. See [HostedClientMetadataDocument].
	ClientMetadataDocument ClientMetadataDocumentFunc
	// Scope lists scopes, separated by spaces, to request in addition to the
	// ones the server asks for.
	Scope string
	// AuthServerMetadataURL is an authorization server metadata document
	// (RFC 8414 or OpenID Connect discovery) to use instead of discovery
	// through the server, for servers that advertise a wrong authorization
	// server or none. The document is trusted as configured, so its issuer is
	// not checked. It must use HTTPS, except on loopback hosts.
	AuthServerMetadataURL string

	// Now returns the current time. Nil means time.Now.
	Now func() time.Time
}

// OAuthHandler authorizes requests to one MCP server for one owner. Set it
// as the OAuthHandler of an sdk.StreamableClientTransport, with
// [OAuthHandler.HTTPClient] as its HTTPClient, as [HTTPTransport] does.
//
// Connections never start a browser flow on their own. They send the stored
// access token, refresh it when it is about to expire or after a 401, and
// otherwise fail with ErrSignInRequired. The app then signs the user in with
// StartSignIn and CompleteSignIn, which may run in different processes.
//
// Many servers rotate refresh tokens, so two refreshes with the same refresh
// token lose the grant. Refreshes hold the store's lock from reading the
// tokens to saving new ones, in every process, and tokens that changed
// meanwhile are used without refreshing.
type OAuthHandler struct {
	opts      OAuthOptions
	serverURL string
	key       string
	client    *http.Client

	// Store updates in progress, which Wait waits for.
	mu      sync.Mutex
	running int
	idle    chan struct{}
	// unproven is the access token Authorize got after a 401, until the
	// server accepts it. See HTTPClient.
	unproven string
}

var _ auth.OAuthHandler = (*OAuthHandler)(nil)

// NewOAuthHandler returns a handler for opts.Server and opts.OwnerID.
func NewOAuthHandler(opts *OAuthOptions) (*OAuthHandler, error) {
	if opts == nil || opts.Store == nil {
		return nil, errors.New("mcp: OAuthOptions.Store is required")
	}
	if opts.Server == "" {
		return nil, errors.New("mcp: OAuthOptions.Server is required")
	}
	serverURL, err := url.Parse(opts.ServerURL)
	if err != nil || (serverURL.Scheme != "https" && serverURL.Scheme != "http") || serverURL.Host == "" {
		return nil, fmt.Errorf("mcp: invalid server URL %q", opts.ServerURL)
	}
	serverURL.Fragment = ""
	if opts.AuthServerMetadataURL != "" {
		if err := checkEndpoint(opts.AuthServerMetadataURL); err != nil {
			return nil, err
		}
	}
	h := &OAuthHandler{opts: *opts, serverURL: serverURL.String()}
	h.key = opts.Server + "|" + h.serverURL
	if len(h.key) > maxStoreNameLength {
		return nil, errors.New("mcp: server name and URL are too long")
	}
	client := http.Client{}
	if opts.HTTPClient != nil {
		client = *opts.HTTPClient
	}
	if client.Timeout == 0 {
		client.Timeout = cmp.Or(opts.RequestTimeout, defaultOAuthRequestTimeout)
	}
	h.client = &client
	if h.opts.ClientName == "" {
		h.opts.ClientName = "burrow"
	}
	if h.opts.Now == nil {
		h.opts.Now = time.Now
	}
	return h, nil
}

// oauthState is what the store keeps per owner and server. It follows pi's
// McpOAuthState.
type oauthState struct {
	Version   int    `json:"version"`
	ServerURL string `json:"serverUrl"`
	// Client is the dynamically registered client. A configured ClientID is
	// not stored.
	Client *oauthClient `json:"client,omitempty"`
	Tokens *oauthTokens `json:"tokens,omitempty"`
	// TokensExpireAt is when the access token expires, in Unix milliseconds,
	// or 0 when unknown.
	TokensExpireAt int64           `json:"tokensExpireAt,omitempty"`
	Discovery      *oauthDiscovery `json:"discovery,omitempty"`
	// Challenge is the server's last WWW-Authenticate challenge, so the next
	// sign-in uses its resource metadata URL and scope.
	Challenge *oauthChallenge `json:"challenge,omitempty"`
	Pending   *pendingSignIn  `json:"pending,omitempty"`
}

type oauthClient struct {
	ClientID                string   `json:"client_id"`
	ClientSecret            string   `json:"client_secret,omitempty"`
	RedirectURIs            []string `json:"redirect_uris,omitempty"`
	TokenEndpointAuthMethod string   `json:"token_endpoint_auth_method,omitempty"`
	// MetadataDocument marks a client identified by a Client ID Metadata
	// Document, whose URL is ClientID. pi does not store it; it is stored
	// here so refreshes, which may run in another process, have it.
	MetadataDocument bool `json:"metadataDocument,omitempty"`
}

type oauthTokens struct {
	AccessToken  string `json:"access_token"`
	TokenType    string `json:"token_type,omitempty"`
	RefreshToken string `json:"refresh_token,omitempty"`
	Scope        string `json:"scope,omitempty"`
}

type oauthDiscovery struct {
	AuthorizationServerURL      string                  `json:"authorizationServerUrl"`
	AuthorizationServerMetadata *oauthex.AuthServerMeta `json:"authorizationServerMetadata"`
	// Resource is the resource indicator (RFC 8707) sent with authorization
	// and token requests, or empty when the server has no resource metadata.
	Resource string `json:"resource,omitempty"`
}

type oauthChallenge struct {
	ResourceMetadataURL string `json:"resourceMetadataUrl,omitempty"`
	Scope               string `json:"scope,omitempty"`
	Error               string `json:"error,omitempty"`
}

type pendingSignIn struct {
	State        string `json:"state"`
	CodeVerifier string `json:"codeVerifier"`
	RedirectURL  string `json:"redirectUrl"`
	Scope        string `json:"scope,omitempty"`
	CreatedAt    int64  `json:"createdAt"`
}

func decodeState(data []byte) (*oauthState, error) {
	if len(data) == 0 {
		return nil, nil
	}
	var state oauthState
	if err := json.Unmarshal(data, &state); err != nil {
		return nil, fmt.Errorf("decode MCP OAuth state: %w", err)
	}
	if state.Version != oauthStateVersion {
		return nil, fmt.Errorf("decode MCP OAuth state: unsupported version %d", state.Version)
	}
	return &state, nil
}

func encodeState(state *oauthState) ([]byte, error) {
	state.Version = oauthStateVersion
	return json.Marshal(state)
}

func (h *OAuthHandler) load(ctx context.Context) (*oauthState, error) {
	data, err := h.opts.Store.Get(ctx, h.opts.OwnerID, OAuthNamespace, h.key)
	if err != nil {
		return nil, err
	}
	return decodeState(data)
}

// modify runs fn on the stored state under the store's lock and stores
// what fn returns, unless fn returns nil. Once fn starts, cancelling ctx no
// longer stops it, so fn gets a context of its own: a token request the
// server answered may have rotated the refresh token, so its answer must be
// saved. Each request is bounded by the HTTP client's timeout instead.
func (h *OAuthHandler) modify(ctx context.Context, fn func(ctx context.Context, state *oauthState) (*oauthState, error)) (*oauthState, error) {
	h.mu.Lock()
	if h.running == 0 {
		h.idle = make(chan struct{})
	}
	h.running++
	h.mu.Unlock()
	defer func() {
		h.mu.Lock()
		h.running--
		if h.running == 0 {
			close(h.idle)
		}
		h.mu.Unlock()
	}()
	lockCtx, cancelLock := context.WithCancel(context.WithoutCancel(ctx))
	defer cancelLock()
	stopCancel := context.AfterFunc(ctx, cancelLock)
	data, err := h.opts.Store.Modify(lockCtx, h.opts.OwnerID, OAuthNamespace, h.key, func(current []byte) ([]byte, error) {
		if !stopCancel() {
			// ctx was cancelled while waiting for the lock.
			return nil, ctx.Err()
		}
		state, err := decodeState(current)
		if err != nil {
			return nil, err
		}
		next, err := fn(lockCtx, state)
		if err != nil || next == nil {
			return nil, err
		}
		next.ServerURL = h.serverURL
		return encodeState(next)
	})
	if err != nil {
		return nil, err
	}
	return decodeState(data)
}

// Wait returns once no store update of this handler is running, or with
// ctx's error when ctx is done first. Call it before the process exits: a
// refresh or code exchange the server already answered may have rotated
// the refresh token, and exiting before the new tokens are saved loses the
// grant. Updates keep running when the request that started them is
// cancelled, so they can still be in progress after every request returned.
func (h *OAuthHandler) Wait(ctx context.Context) error {
	h.mu.Lock()
	if h.running == 0 {
		h.mu.Unlock()
		return nil
	}
	idle := h.idle
	h.mu.Unlock()
	select {
	case <-idle:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (h *OAuthHandler) signInRequired(reason string) error {
	return fmt.Errorf("MCP server %s: %w: %s", h.opts.Server, ErrSignInRequired, reason)
}

func (h *OAuthHandler) expiring(state *oauthState) bool {
	return state.TokensExpireAt != 0 && time.UnixMilli(state.TokensExpireAt).Add(-refreshSkew).Before(h.opts.Now())
}

// TokenSource returns the stored access token, refreshed first when it is
// about to expire, or nil when there is none. It implements
// auth.OAuthHandler.
func (h *OAuthHandler) TokenSource(ctx context.Context) (oauth2.TokenSource, error) {
	state, err := h.load(ctx)
	if err != nil {
		return nil, err
	}
	if state != nil && state.Tokens != nil && state.Tokens.RefreshToken != "" && h.expiring(state) {
		// Failures fall through: the request goes out with the old token and
		// a 401 decides what happens.
		if refreshed, err := h.refresh(ctx, state.Tokens.AccessToken, nil); err == nil || errors.Is(err, ErrSignInRequired) {
			state = refreshed
		}
	}
	if state == nil || state.Tokens == nil {
		return nil, nil
	}
	return oauth2.StaticTokenSource(&oauth2.Token{AccessToken: state.Tokens.AccessToken, TokenType: "Bearer"}), nil
}

// Authorize handles a 401 or 403 from the server. It refreshes the rejected
// token, or returns ErrSignInRequired when only a new sign-in helps. It
// implements auth.OAuthHandler.
func (h *OAuthHandler) Authorize(ctx context.Context, req *http.Request, resp *http.Response) error {
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 1<<20))

	challenge := parseChallenge(resp.Header.Values("WWW-Authenticate"))
	if resp.StatusCode == http.StatusForbidden && challenge.Error != "insufficient_scope" {
		// Only insufficient_scope calls for authorization. Any other 403 is
		// the answer, as in pi. Returning nil would make the transport send
		// the request again, and break the connection on the second 403.
		return fmt.Errorf("MCP server %s refused the request: %s", h.opts.Server, resp.Status)
	}
	stale, _ := strings.CutPrefix(req.Header.Get("Authorization"), "Bearer ")
	state, err := h.refresh(ctx, stale, &challenge)
	if err != nil {
		return err
	}
	// The transport retries with these tokens; a 401 then means the server
	// rejects fresh tokens too, which only a sign-in fixes.
	if state != nil && state.Tokens != nil {
		h.mu.Lock()
		h.unproven = state.Tokens.AccessToken
		h.mu.Unlock()
	}
	return nil
}

// HTTPClient returns base, nil meaning a default client, with a transport
// that reports ErrSignInRequired when the server rejects the tokens
// Authorize just refreshed. Use it as the HTTPClient of the
// sdk.StreamableClientTransport the handler authorizes; [HTTPTransport]
// does. The SDK retries a request once after Authorize and reports a
// second 401 as a plain error that breaks the connection, so without it a
// server that rejects every token makes each connect refresh again and fail
// instead of asking the user to sign in.
func (h *OAuthHandler) HTTPClient(base *http.Client) *http.Client {
	client := http.Client{}
	if base != nil {
		client = *base
	}
	next := client.Transport
	if next == nil {
		next = http.DefaultTransport
	}
	client.Transport = &unprovenTokenTransport{handler: h, next: next}
	return &client
}

// unprovenTokenTransport turns a 401 for the tokens Authorize just
// refreshed into an error wrapping ErrSignInRequired. The SDK reports
// errors of the HTTP client as rejected requests, which keep the
// connection.
type unprovenTokenTransport struct {
	handler *OAuthHandler
	next    http.RoundTripper
}

func (t *unprovenTokenTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	resp, err := t.next.RoundTrip(req)
	if err != nil {
		return resp, err
	}
	token, ok := strings.CutPrefix(req.Header.Get("Authorization"), "Bearer ")
	if !ok || token == "" {
		return resp, nil
	}
	h := t.handler
	h.mu.Lock()
	unproven := h.unproven != "" && h.unproven == token
	if unproven && resp.StatusCode != http.StatusUnauthorized {
		// Accepted, so a later 401 means it expired: refresh it.
		h.unproven = ""
	}
	h.mu.Unlock()
	if !unproven || resp.StatusCode != http.StatusUnauthorized {
		return resp, nil
	}
	_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 1<<20))
	resp.Body.Close()
	return nil, h.signInRequired("the server rejected the refreshed access token")
}

// parseChallenge returns the parameters of the Bearer challenge.
func parseChallenge(headers []string) oauthChallenge {
	challenges, _ := oauthex.ParseWWWAuthenticate(headers)
	for _, c := range challenges {
		if c.Scheme == "bearer" {
			return oauthChallenge{
				ResourceMetadataURL: c.Params["resource_metadata"],
				Scope:               c.Params["scope"],
				Error:               c.Params["error"],
			}
		}
	}
	return oauthChallenge{}
}

// refresh replaces stale, the access token that expired or was rejected,
// and records challenge, if any. It returns the state it leaves stored.
func (h *OAuthHandler) refresh(ctx context.Context, stale string, challenge *oauthChallenge) (*oauthState, error) {
	var result error
	state, err := h.modify(ctx, func(ctx context.Context, state *oauthState) (*oauthState, error) {
		changed := false
		if challenge != nil && *challenge != (oauthChallenge{}) {
			if state == nil {
				state = &oauthState{}
			}
			if state.Challenge == nil || *state.Challenge != *challenge {
				state.Challenge = challenge
				changed = true
			}
		}
		keep := func() (*oauthState, error) {
			if changed {
				return state, nil
			}
			return nil, nil
		}
		switch {
		case challenge != nil && challenge.Error == "insufficient_scope":
			// A refresh keeps the granted scope, so more scope needs a new
			// sign-in.
			result = h.signInRequired("the server asks for more scope")
			return keep()
		case state == nil || state.Tokens == nil:
			result = h.signInRequired("not signed in")
			return keep()
		case state.Tokens.AccessToken != stale:
			// Another request or process refreshed the tokens, or the user
			// signed in.
			return keep()
		case state.Tokens.RefreshToken == "":
			result = h.signInRequired("the access token expired and there is no refresh token")
			return keep()
		}

		client := h.clientFor(state)
		if client == nil || state.Discovery == nil {
			result = h.signInRequired("the client registration is missing")
			return keep()
		}
		form := url.Values{"grant_type": {"refresh_token"}, "refresh_token": {state.Tokens.RefreshToken}}
		tokens, expireAt, err := h.tokenRequest(ctx, state.Discovery, client, form)
		var oauthErr *OAuthError
		if errors.As(err, &oauthErr) {
			switch oauthErr.Code {
			case "invalid_grant":
				state.Tokens, state.TokensExpireAt = nil, 0
				result = h.signInRequired("the refresh token was rejected: " + oauthErr.Error())
				return state, nil
			case "invalid_client", "unauthorized_client":
				state.Client, state.Tokens, state.TokensExpireAt = nil, nil, 0
				result = h.signInRequired("the client was rejected: " + oauthErr.Error())
				return state, nil
			}
		}
		if err != nil {
			return nil, fmt.Errorf("refresh MCP server %s token: %w", h.opts.Server, err)
		}
		// A refresh without a new refresh token keeps the old one, and one
		// without scope keeps the scope of the grant (RFC 6749 section 6).
		if tokens.RefreshToken == "" {
			tokens.RefreshToken = state.Tokens.RefreshToken
		}
		if tokens.Scope == "" {
			tokens.Scope = state.Tokens.Scope
		}
		state.Tokens, state.TokensExpireAt = tokens, expireAt
		return state, nil
	})
	if err != nil {
		return nil, err
	}
	return state, result
}

// SignedIn reports whether tokens are stored for the server. They may still
// be rejected.
func (h *OAuthHandler) SignedIn(ctx context.Context) (bool, error) {
	state, err := h.load(ctx)
	if err != nil {
		return false, err
	}
	return state != nil && state.Tokens != nil, nil
}

// SignOut deletes everything stored for the server, including the client
// registration.
func (h *OAuthHandler) SignOut(ctx context.Context) error {
	return h.opts.Store.Delete(ctx, h.opts.OwnerID, OAuthNamespace, h.key)
}

// StartSignIn begins signing in. It discovers the authorization server,
// registers a client if needed, stores the pending request, and returns the
// URL to send the user to. The authorization server redirects the user to
// RedirectURL; pass that request's query to CompleteSignIn, in any process.
// Starting again replaces the pending request.
func (h *OAuthHandler) StartSignIn(ctx context.Context) (string, error) {
	if h.opts.RedirectURL == "" {
		return "", errors.New("mcp: OAuthOptions.RedirectURL is required to sign in")
	}
	return h.startSignIn(ctx, h.opts.RedirectURL)
}

// startSignIn is StartSignIn with redirectURL, such as a loopback callback's.
func (h *OAuthHandler) startSignIn(ctx context.Context, redirectURL string) (string, error) {
	stored, err := h.load(ctx)
	if err != nil {
		return "", err
	}
	var challenge oauthChallenge
	if stored != nil && stored.Challenge != nil {
		challenge = *stored.Challenge
	}

	discovery, scopesSupported, err := h.discover(ctx, challenge.ResourceMetadataURL)
	if err != nil {
		return "", err
	}
	asm := discovery.AuthorizationServerMetadata
	if len(asm.ResponseTypesSupported) > 0 && !slices.Contains(asm.ResponseTypesSupported, "code") {
		return "", errors.New("mcp: the authorization server does not support authorization codes")
	}
	if len(asm.CodeChallengeMethodsSupported) > 0 && !slices.Contains(asm.CodeChallengeMethodsSupported, "S256") {
		return "", errors.New("mcp: the authorization server does not support PKCE S256")
	}
	if err := checkEndpoint(asm.AuthorizationEndpoint); err != nil {
		return "", err
	}

	// A server asking for more scope gets it on top of the configured scope
	// and, since the challenge may list only the missing scopes, on top of
	// the scope granted so far (SEP-2350).
	challenged := challenge.Scope
	if challenge.Error == "insufficient_scope" && stored != nil && stored.Tokens != nil && challenged != "" {
		challenged = mergeScopes(stored.Tokens.Scope, challenged)
	}
	scope := mergeScopes(h.opts.Scope, challenged)
	if scope == "" {
		scope = strings.Join(scopesSupported, " ")
	}
	// Ask for refresh tokens where the server supports offline_access
	// (SEP-2207), so connections outlive the access token.
	if slices.Contains(asm.ScopesSupported, "offline_access") {
		scope = mergeScopes(scope, "offline_access")
	}

	// registered replaces the stored client. Tokens belong to the client
	// that got them, so they are dropped when its ID changes.
	var client, registered *oauthClient
	var storedClient *oauthClient
	if stored != nil {
		storedClient = stored.Client
	}
	switch {
	case h.opts.ClientID != "":
		client = h.clientFor(nil)
	case h.opts.ClientMetadataDocument != nil:
		// A dynamically registered client stored before is replaced.
		if client, err = h.metadataDocumentClient(redirectURL, asm); err != nil {
			return "", err
		}
		if storedClient == nil || storedClient.ClientID != client.ClientID || !slices.Equal(storedClient.RedirectURIs, client.RedirectURIs) {
			registered = client
		}
	case storedClient != nil && !storedClient.MetadataDocument && slices.Contains(storedClient.RedirectURIs, redirectURL):
		client = storedClient
	default:
		// A registered client cannot use another redirect URI, so register
		// again.
		if registered, err = h.register(ctx, asm, scope, redirectURL); err != nil {
			return "", err
		}
		client = registered
	}
	if client.MetadataDocument {
		// The document's redirect URI may differ from the one given, for
		// example by a server-specific path.
		redirectURL = client.RedirectURIs[0]
	}

	pending := &pendingSignIn{
		State:        rand.Text(),
		CodeVerifier: oauth2.GenerateVerifier(),
		RedirectURL:  redirectURL,
		Scope:        scope,
		CreatedAt:    h.opts.Now().UnixMilli(),
	}
	authURL, err := url.Parse(asm.AuthorizationEndpoint)
	if err != nil {
		return "", fmt.Errorf("mcp: invalid authorization endpoint: %w", err)
	}
	query := authURL.Query()
	query.Set("response_type", "code")
	query.Set("client_id", client.ClientID)
	query.Set("code_challenge", oauth2.S256ChallengeFromVerifier(pending.CodeVerifier))
	query.Set("code_challenge_method", "S256")
	query.Set("redirect_uri", pending.RedirectURL)
	query.Set("state", pending.State)
	if scope != "" {
		query.Set("scope", scope)
	}
	if slices.Contains(strings.Fields(scope), "offline_access") {
		query.Set("prompt", "consent")
	}
	if discovery.Resource != "" {
		query.Set("resource", discovery.Resource)
	}
	authURL.RawQuery = query.Encode()

	_, err = h.modify(ctx, func(_ context.Context, state *oauthState) (*oauthState, error) {
		if state == nil {
			state = &oauthState{}
		}
		if registered != nil {
			if state.Client == nil || state.Client.ClientID != registered.ClientID {
				state.Tokens, state.TokensExpireAt = nil, 0
			}
			state.Client = registered
		}
		state.Discovery = discovery
		state.Pending = pending
		return state, nil
	})
	if err != nil {
		return "", err
	}
	return authURL.String(), nil
}

// CompleteSignIn finishes the sign-in StartSignIn began, given the query of
// the authorization server's redirect to RedirectURL. It checks the state
// and issuer, exchanges the code for tokens, and stores them.
func (h *OAuthHandler) CompleteSignIn(ctx context.Context, query url.Values) error {
	if code := query.Get("error"); code != "" {
		if description := query.Get("error_description"); description != "" {
			return fmt.Errorf("mcp: sign-in to %s failed: %s: %s", h.opts.Server, code, description)
		}
		return fmt.Errorf("mcp: sign-in to %s failed: %s", h.opts.Server, code)
	}
	code := query.Get("code")
	if code == "" {
		return errors.New("mcp: the authorization response has no code")
	}
	_, err := h.modify(ctx, func(ctx context.Context, state *oauthState) (*oauthState, error) {
		if state == nil || state.Pending == nil || state.Discovery == nil ||
			subtle.ConstantTimeCompare([]byte(state.Pending.State), []byte(query.Get("state"))) != 1 {
			return nil, ErrNoPendingSignIn
		}
		pending := state.Pending
		if h.opts.Now().Sub(time.UnixMilli(pending.CreatedAt)) > signInExpiry {
			return nil, errors.New("mcp: the sign-in expired; sign in again")
		}
		// RFC 9207: never send a code from another authorization server to
		// this one. Without metadata there is no issuer to compare with, so
		// any iss is rejected.
		asm := state.Discovery.AuthorizationServerMetadata
		if iss := query.Get("iss"); iss != "" || asm.AuthorizationResponseIssParameterSupported {
			if iss != asm.Issuer {
				return nil, fmt.Errorf("mcp: the authorization response comes from issuer %q, not %q", iss, asm.Issuer)
			}
		}
		client := h.clientFor(state)
		if client == nil {
			return nil, errors.New("mcp: the client registration is missing; sign in again")
		}
		form := url.Values{
			"grant_type":    {"authorization_code"},
			"code":          {code},
			"redirect_uri":  {pending.RedirectURL},
			"code_verifier": {pending.CodeVerifier},
		}
		tokens, expireAt, err := h.tokenRequest(ctx, state.Discovery, client, form)
		if err != nil {
			return nil, fmt.Errorf("mcp: sign-in to %s failed: %w", h.opts.Server, err)
		}
		// A response without scope grants the requested scope (RFC 6749
		// section 5.1). It is recorded so a step-up can keep it.
		if tokens.Scope == "" {
			tokens.Scope = pending.Scope
		}
		state.Tokens, state.TokensExpireAt = tokens, expireAt
		state.Pending, state.Challenge = nil, nil
		return state, nil
	})
	return err
}

// clientFor returns the configured client, else the stored one.
func (h *OAuthHandler) clientFor(state *oauthState) *oauthClient {
	if h.opts.ClientID != "" {
		return &oauthClient{ClientID: h.opts.ClientID, ClientSecret: h.opts.ClientSecret}
	}
	if state == nil {
		return nil
	}
	return state.Client
}

// discover finds the authorization server of the MCP server, following the
// discovery order of the MCP authorization spec, and returns it with the
// scopes the resource supports.
func (h *OAuthHandler) discover(ctx context.Context, resourceMetadataURL string) (*oauthDiscovery, []string, error) {
	server, err := url.Parse(h.serverURL)
	if err != nil {
		return nil, nil, err
	}
	origin := server.Scheme + "://" + server.Host

	prm, err := h.protectedResourceMetadata(ctx, server, resourceMetadataURL)
	if err != nil {
		return nil, nil, fmt.Errorf("mcp: protected resource metadata of %s: %w", h.opts.Server, err)
	}
	discovery := &oauthDiscovery{AuthorizationServerURL: origin}
	var scopesSupported []string
	if prm != nil {
		if err := selectResource(server, prm.Resource); err != nil {
			return nil, nil, err
		}
		if len(prm.AuthorizationServers) > 0 {
			discovery.AuthorizationServerURL = prm.AuthorizationServers[0]
		}
		discovery.Resource = prm.Resource
		scopesSupported = prm.ScopesSupported
	}
	// Without resource metadata the MCP server's origin is the authorization
	// server (MCP 2025-03-26).

	if h.opts.AuthServerMetadataURL != "" {
		asm, err := h.fetchAuthServerMetadata(ctx, h.opts.AuthServerMetadataURL)
		if err != nil {
			return nil, nil, fmt.Errorf("mcp: authorization server metadata of %s: %w", h.opts.Server, err)
		}
		discovery.AuthorizationServerURL = asm.Issuer
		discovery.AuthorizationServerMetadata = asm
		return discovery, scopesSupported, nil
	}
	asm, err := auth.GetAuthServerMetadata(ctx, discovery.AuthorizationServerURL, h.client)
	if err != nil {
		return nil, nil, fmt.Errorf("mcp: authorization server metadata of %s: %w", h.opts.Server, err)
	}
	if asm == nil {
		// Servers without metadata use the default endpoints (MCP
		// 2025-03-26). Their issuer is unknown.
		base := strings.TrimSuffix(discovery.AuthorizationServerURL, "/")
		asm = &oauthex.AuthServerMeta{
			AuthorizationEndpoint: base + "/authorize",
			TokenEndpoint:         base + "/token",
			RegistrationEndpoint:  base + "/register",
		}
	}
	discovery.AuthorizationServerMetadata = asm
	return discovery, scopesSupported, nil
}

// protectedResourceMetadata fetches the server's protected resource
// metadata (RFC 9728) as pi does: from the URL the challenge names, else
// from the well-known path of the MCP endpoint and, when nothing is there,
// the well-known root. It returns nil when the server publishes none, or
// none that parses, so its origin is the authorization server.
//
// It does not use oauthex.GetProtectedResourceMetadata, which requires the
// metadata to name exactly the URL it was looked up for. Servers commonly
// name their origin, or the endpoint with a trailing slash; selectResource
// accepts those, as pi does.
func (h *OAuthHandler) protectedResourceMetadata(ctx context.Context, server *url.URL, challengeURL string) (*oauthex.ProtectedResourceMetadata, error) {
	origin := server.Scheme + "://" + server.Host
	path := strings.TrimSuffix(server.EscapedPath(), "/")
	metadataURL := challengeURL
	if metadataURL == "" {
		metadataURL = origin + "/.well-known/oauth-protected-resource" + path
	}
	prm, status, err := h.fetchResourceMetadata(ctx, metadataURL)
	if err == nil && challengeURL == "" && path != "" && isDiscoveryMiss(status) {
		prm, _, err = h.fetchResourceMetadata(ctx, origin+"/.well-known/oauth-protected-resource")
	}
	return prm, err
}

// isDiscoveryMiss reports a status meaning the metadata is not at this
// URL, so discovery tries the next one.
func isDiscoveryMiss(status int) bool {
	return (status >= 400 && status < 500) || status == http.StatusBadGateway
}

// fetchResourceMetadata fetches one metadata document. A response that is
// not a valid document returns nil and its status.
func (h *OAuthHandler) fetchResourceMetadata(ctx context.Context, metadataURL string) (*oauthex.ProtectedResourceMetadata, int, error) {
	if err := checkEndpoint(metadataURL); err != nil {
		return nil, 0, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, metadataURL, nil)
	if err != nil {
		return nil, 0, err
	}
	req.Header.Set("Accept", "application/json")
	resp, err := h.client.Do(req)
	if err != nil {
		return nil, 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, resp.StatusCode, nil
	}
	var prm oauthex.ProtectedResourceMetadata
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&prm); err != nil || prm.Resource == "" {
		return nil, resp.StatusCode, nil
	}
	// GetAuthServerMetadata checks the authorization server's URL.
	return &prm, resp.StatusCode, nil
}

// fetchAuthServerMetadata loads the configured authorization server
// metadata document as pi does. Unlike discovery, it trusts the document: it
// neither compares the issuer with the URL nor requires a list of PKCE
// methods. The endpoints are checked where they are used.
func (h *OAuthHandler) fetchAuthServerMetadata(ctx context.Context, metadataURL string) (*oauthex.AuthServerMeta, error) {
	if err := checkEndpoint(metadataURL); err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, metadataURL, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "application/json")
	resp, err := h.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP %d loading authorization server metadata from %s", resp.StatusCode, metadataURL)
	}
	var asm oauthex.AuthServerMeta
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&asm); err != nil {
		return nil, fmt.Errorf("invalid authorization server metadata from %s: %w", metadataURL, err)
	}
	if asm.Issuer == "" || asm.AuthorizationEndpoint == "" || asm.TokenEndpoint == "" {
		return nil, fmt.Errorf("authorization server metadata from %s lacks the issuer or an endpoint", metadataURL)
	}
	return &asm, nil
}

// selectResource checks that resource, from protected resource metadata,
// covers the MCP server as pi's selectResource does: the same origin, and
// a path that is a prefix of the server's path segment by segment.
func selectResource(server *url.URL, resource string) error {
	configured, err := url.Parse(resource)
	if err != nil || origin(configured) != origin(server) {
		return fmt.Errorf("mcp: protected resource %s does not match MCP server %s", resource, server)
	}
	withSlash := func(path string) string {
		if strings.HasSuffix(path, "/") {
			return path
		}
		return path + "/"
	}
	if !strings.HasPrefix(withSlash(server.EscapedPath()), withSlash(configured.EscapedPath())) {
		return fmt.Errorf("mcp: protected resource %s does not match MCP server %s", resource, server)
	}
	return nil
}

// origin is u's scheme, host and port, without a default port, as browsers
// compare origins.
func origin(u *url.URL) string {
	scheme, host, port := strings.ToLower(u.Scheme), strings.ToLower(u.Hostname()), u.Port()
	if (scheme == "https" && port == "443") || (scheme == "http" && port == "80") {
		port = ""
	}
	if port != "" {
		host = net.JoinHostPort(host, port)
	} else if strings.Contains(host, ":") {
		host = "[" + host + "]"
	}
	return scheme + "://" + host
}

// applicationType is the OpenID Connect application_type of a client with
// redirectURL (MCP SEP-837), as pi picks it. OpenID Connect servers assume
// web without one, and refuse http loopback redirect URIs for web clients.
// Loopback hosts and custom schemes are native (RFC 8252).
func applicationType(redirectURL string) string {
	u, err := url.Parse(redirectURL)
	if err != nil {
		return "web"
	}
	host := u.Hostname()
	if (u.Scheme != "http" && u.Scheme != "https") || host == "localhost" || host == "127.0.0.1" || host == "::1" {
		return "native"
	}
	return "web"
}

// register registers a client dynamically (RFC 7591).
func (h *OAuthHandler) register(ctx context.Context, asm *oauthex.AuthServerMeta, scope, redirectURL string) (*oauthClient, error) {
	if asm.RegistrationEndpoint == "" {
		return nil, fmt.Errorf("mcp: the authorization server of %s does not support dynamic client registration; set OAuthOptions.ClientID", h.opts.Server)
	}
	if err := checkEndpoint(asm.RegistrationEndpoint); err != nil {
		return nil, err
	}
	resp, err := oauthex.RegisterClient(ctx, asm.RegistrationEndpoint, &oauthex.ClientRegistrationMetadata{
		RedirectURIs:            []string{redirectURL},
		TokenEndpointAuthMethod: "none",
		GrantTypes:              []string{"authorization_code", "refresh_token"},
		ResponseTypes:           []string{"code"},
		ClientName:              h.opts.ClientName,
		Scope:                   scope,
		ApplicationType:         applicationType(redirectURL),
	}, h.client)
	if err != nil {
		return nil, fmt.Errorf("mcp: register a client with %s: %w", h.opts.Server, err)
	}
	client := &oauthClient{
		ClientID:                resp.ClientID,
		ClientSecret:            resp.ClientSecret,
		RedirectURIs:            resp.RedirectURIs,
		TokenEndpointAuthMethod: resp.TokenEndpointAuthMethod,
	}
	if len(client.RedirectURIs) == 0 {
		client.RedirectURIs = []string{redirectURL}
	}
	return client, nil
}

// tokenRequest posts form to the token endpoint and returns the tokens and
// when the access token expires, in Unix milliseconds (0 when unknown).
func (h *OAuthHandler) tokenRequest(ctx context.Context, discovery *oauthDiscovery, client *oauthClient, form url.Values) (*oauthTokens, int64, error) {
	asm := discovery.AuthorizationServerMetadata
	if err := checkEndpoint(asm.TokenEndpoint); err != nil {
		return nil, 0, err
	}
	if discovery.Resource != "" {
		form.Set("resource", discovery.Resource)
	}
	header := http.Header{
		"Accept":       {"application/json"},
		"Content-Type": {"application/x-www-form-urlencoded"},
	}
	switch selectClientAuthMethod(client, asm.TokenEndpointAuthMethodsSupported) {
	case "client_secret_basic":
		header.Set("Authorization", "Basic "+basicAuth(client.ClientID, client.ClientSecret))
	case "client_secret_post":
		form.Set("client_id", client.ClientID)
		form.Set("client_secret", client.ClientSecret)
	default:
		form.Set("client_id", client.ClientID)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, asm.TokenEndpoint, strings.NewReader(form.Encode()))
	if err != nil {
		return nil, 0, err
	}
	req.Header = header
	resp, err := h.client.Do(req)
	if err != nil {
		return nil, 0, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return nil, 0, err
	}
	var parsed struct {
		Error            string      `json:"error"`
		ErrorDescription string      `json:"error_description"`
		AccessToken      string      `json:"access_token"`
		TokenType        string      `json:"token_type"`
		ExpiresIn        json.Number `json:"expires_in"`
		RefreshToken     string      `json:"refresh_token"`
		Scope            string      `json:"scope"`
	}
	jsonErr := json.Unmarshal(body, &parsed)
	// Servers may report OAuth errors with any status, so check the body
	// before the status.
	if jsonErr == nil && parsed.Error != "" {
		return nil, 0, &OAuthError{Code: parsed.Error, Description: parsed.ErrorDescription, StatusCode: resp.StatusCode}
	}
	if resp.StatusCode != http.StatusOK {
		return nil, 0, fmt.Errorf("token endpoint returned %s", resp.Status)
	}
	if jsonErr != nil {
		return nil, 0, fmt.Errorf("decode token response: %w", jsonErr)
	}
	if parsed.AccessToken == "" {
		return nil, 0, errors.New("token response has no access_token")
	}
	var expireAt int64
	if seconds, err := parsed.ExpiresIn.Int64(); err == nil && seconds > 0 {
		expireAt = h.opts.Now().Add(time.Duration(seconds) * time.Second).UnixMilli()
	}
	return &oauthTokens{
		AccessToken:  parsed.AccessToken,
		TokenType:    parsed.TokenType,
		RefreshToken: parsed.RefreshToken,
		Scope:        parsed.Scope,
	}, expireAt, nil
}

// selectClientAuthMethod picks how the client authenticates at the token
// endpoint: the method it registered with if the server supports it, else
// the best the server supports.
func selectClientAuthMethod(client *oauthClient, supported []string) string {
	hinted := client.TokenEndpointAuthMethod
	if slices.Contains([]string{"client_secret_basic", "client_secret_post", "none"}, hinted) &&
		(len(supported) == 0 || slices.Contains(supported, hinted)) {
		return hinted
	}
	hasSecret := client.ClientSecret != ""
	switch {
	case len(supported) == 0 && hasSecret:
		return "client_secret_basic"
	case len(supported) == 0:
		return "none"
	case hasSecret && slices.Contains(supported, "client_secret_basic"):
		return "client_secret_basic"
	case hasSecret && slices.Contains(supported, "client_secret_post"):
		return "client_secret_post"
	case slices.Contains(supported, "none"):
		return "none"
	case hasSecret:
		return "client_secret_post"
	default:
		return "none"
	}
}

// basicAuth encodes credentials for client_secret_basic, which form-encodes
// them first (RFC 6749 section 2.3.1).
func basicAuth(id, secret string) string {
	req := http.Request{Header: http.Header{}}
	req.SetBasicAuth(url.QueryEscape(id), url.QueryEscape(secret))
	encoded, _ := strings.CutPrefix(req.Header.Get("Authorization"), "Basic ")
	return encoded
}

// checkEndpoint refuses endpoints that are neither HTTPS nor loopback.
// Authorization server metadata endpoints are checked by oauthex already;
// the default ones and resource metadata URLs are not.
func checkEndpoint(endpoint string) error {
	u, err := url.Parse(endpoint)
	if err != nil {
		return fmt.Errorf("mcp: invalid OAuth endpoint %q: %w", endpoint, err)
	}
	if u.Scheme == "https" {
		return nil
	}
	host := u.Hostname()
	if ip := net.ParseIP(host); u.Scheme == "http" && (host == "localhost" || (ip != nil && ip.IsLoopback())) {
		return nil
	}
	return fmt.Errorf("mcp: OAuth endpoint %q does not use HTTPS", endpoint)
}

// mergeScopes returns the scopes of all lists, each once, in order.
func mergeScopes(lists ...string) string {
	var merged []string
	for _, list := range lists {
		for _, scope := range strings.Fields(list) {
			if !slices.Contains(merged, scope) {
				merged = append(merged, scope)
			}
		}
	}
	return strings.Join(merged, " ")
}

// ClientMetadataDocument is a Client ID Metadata Document: an https URL the
// client uses as its client_id, and a redirect URI the document lists.
type ClientMetadataDocument struct {
	URL         string
	RedirectURL string
}

// ClientMetadataDocumentFunc chooses the Client ID Metadata Document to sign
// in to serverURL with, given the redirect URI of the sign-in and the
// authorization server's metadata. It is called at every sign-in, and the
// result is stored with the tokens.
type ClientMetadataDocumentFunc func(serverURL, redirectURL string, metadata *oauthex.AuthServerMeta) (*ClientMetadataDocument, error)

// HostedClientMetadataDocument chooses documents served under baseURL, as
// pi chooses its own and Codex its. When the authorization server sends
// the iss parameter in authorization responses (RFC 9207), the document is
// baseURL/client.json and the redirect URI the one given. Otherwise the
// document and redirect URI are specific to the MCP server, so a response
// cannot be mixed up with one from another authorization server (RFC 9700
// section 4.4.2.2): baseURL/<id>/client.json with the redirect path
// followed by /<id>, where id is [CallbackID] of the server URL. The app
// serves those documents and receives the authorization response on those
// paths too.
func HostedClientMetadataDocument(baseURL string) ClientMetadataDocumentFunc {
	base := strings.TrimSuffix(baseURL, "/")
	return func(serverURL, redirectURL string, metadata *oauthex.AuthServerMeta) (*ClientMetadataDocument, error) {
		if !metadata.ClientIDMetadataDocumentSupported || !slices.Contains(metadata.TokenEndpointAuthMethodsSupported, "none") {
			return nil, errors.New("mcp: the authorization server does not support Client ID Metadata Documents for public clients")
		}
		if metadata.AuthorizationResponseIssParameterSupported {
			return &ClientMetadataDocument{URL: base + "/client.json", RedirectURL: redirectURL}, nil
		}
		id, err := CallbackID(serverURL)
		if err != nil {
			return nil, err
		}
		redirect, err := url.Parse(redirectURL)
		if err != nil {
			return nil, fmt.Errorf("mcp: invalid redirect URL: %w", err)
		}
		redirect.Path = strings.TrimSuffix(redirect.Path, "/") + "/" + id
		redirect.RawPath = ""
		return &ClientMetadataDocument{URL: base + "/" + id + "/client.json", RedirectURL: redirect.String()}, nil
	}
}

// CallbackID is 12 characters identifying an MCP server URL in callback
// paths, computed like Codex does: the first 9 bytes of the SHA-256 of the
// URL without its fragment, in unpadded base64url.
func CallbackID(serverURL string) (string, error) {
	u, err := url.Parse(serverURL)
	if err != nil {
		return "", fmt.Errorf("mcp: invalid server URL: %w", err)
	}
	// As a browser serializes it.
	u.Fragment, u.RawFragment = "", ""
	u.Host = strings.ToLower(u.Host)
	if u.Path == "" {
		u.Path = "/"
	}
	sum := sha256.Sum256([]byte(u.String()))
	return base64.RawURLEncoding.EncodeToString(sum[:9]), nil
}

// metadataDocumentClient is the client of the Client ID Metadata Document
// OAuthOptions.ClientMetadataDocument chooses.
func (h *OAuthHandler) metadataDocumentClient(redirectURL string, asm *oauthex.AuthServerMeta) (*oauthClient, error) {
	document, err := h.opts.ClientMetadataDocument(h.serverURL, redirectURL, asm)
	if err != nil {
		return nil, err
	}
	u, err := url.Parse(document.URL)
	if err != nil || u.Scheme != "https" || u.Host == "" || u.Path == "" || u.Path == "/" {
		return nil, fmt.Errorf("mcp: invalid OAuth client metadata URL %q", document.URL)
	}
	if document.RedirectURL == "" {
		return nil, errors.New("mcp: the client metadata document has no redirect URL")
	}
	return &oauthClient{
		ClientID:                document.URL,
		RedirectURIs:            []string{document.RedirectURL},
		TokenEndpointAuthMethod: "none",
		MetadataDocument:        true,
	}, nil
}

// PendingSignIn identifies the server of a started sign-in.
type PendingSignIn struct {
	Server    string
	ServerURL string
}

// FindPendingSignIn finds ownerID's started sign-in that the authorization
// response with state belongs to, for apps with one redirect URL for every
// server. Construct the result's OAuthHandler and pass the response to its
// CompleteSignIn. It returns ErrNoPendingSignIn when none matches.
//
// ownerID must come from the app's own authenticated session, never from
// the response. A lookup across owners would let an attacker send a victim
// the authorization URL of the attacker's sign-in: the victim's grant would
// then complete it, storing the victim's tokens under the attacker's
// account (login CSRF, RFC 9700 section 4.7).
func FindPendingSignIn(ctx context.Context, store baseAgent.AuthStore, ownerID, state string) (*PendingSignIn, error) {
	if state == "" {
		return nil, ErrNoPendingSignIn
	}
	names, err := store.List(ctx, ownerID, OAuthNamespace)
	if err != nil {
		return nil, err
	}
	for _, name := range names {
		data, err := store.Get(ctx, ownerID, OAuthNamespace, name)
		if err != nil {
			return nil, err
		}
		stored, err := decodeState(data)
		if err != nil || stored == nil || stored.Pending == nil ||
			subtle.ConstantTimeCompare([]byte(stored.Pending.State), []byte(state)) != 1 {
			continue
		}
		if server, ok := strings.CutSuffix(name, "|"+stored.ServerURL); ok {
			return &PendingSignIn{Server: server, ServerURL: stored.ServerURL}, nil
		}
	}
	return nil, ErrNoPendingSignIn
}
