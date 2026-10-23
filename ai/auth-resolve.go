// Ported from pi's packages/ai/src/auth/resolve.ts and the auth checks in
// packages/ai/src/models.ts (MIT, see LICENSE).

package ai

import (
	"context"
	"errors"
	"fmt"
	"time"
)

// oauthRefreshTimeout bounds one OAuth refresh request. The refresh ignores
// the caller's cancellation, so this is all that stops a hung provider.
const oauthRefreshTimeout = 15 * time.Second

// defaultOAuthMinimumValidity is how close to expiry a stored OAuth token is
// refreshed before use, so it cannot expire during a request.
const defaultOAuthMinimumValidity = 5 * time.Minute

// AuthErrorCode says which part of auth resolution failed.
type AuthErrorCode string

const (
	// AuthErrorStore means the credential store or API key resolution failed.
	// Retrying may succeed.
	AuthErrorStore AuthErrorCode = "auth"
	// AuthErrorOAuth means an OAuth refresh failed. The stored credential is
	// kept, but the user likely has to log in again.
	AuthErrorOAuth AuthErrorCode = "oauth"
)

// AuthError is returned when auth resolution fails, as pi's ModelsError is.
type AuthError struct {
	Code     AuthErrorCode
	Provider Provider
	Message  string
	Err      error
}

func (e *AuthError) Error() string {
	if e.Err == nil {
		return e.Message
	}
	return fmt.Sprintf("%s: %v", e.Message, e.Err)
}

func (e *AuthError) Unwrap() error { return e.Err }

// AuthResolutionOverrides changes how ResolveProviderAuth resolves one request.
type AuthResolutionOverrides struct {
	// APIKey replaces stored and ambient auth when set, even to "".
	APIKey *string
	// Env is laid over the ambient environment and a stored credential's Env.
	Env map[string]string
	// MinOAuthValidity, when longer than the default five minutes, is how long
	// an OAuth token must stay valid; a refresh that leaves less is an error.
	MinOAuthValidity time.Duration
}

// CheckProviderAuth reports whether provider has auth configured, and from
// where, without refreshing tokens. It returns nil, nil when the provider is
// not configured. Use it for status UI and model lists; requests use
// ResolveProviderAuth.
//
// It follows ResolveProviderAuth's precedence: a stored OAuth credential
// counts as configured even when expired, and a stored credential whose type
// has no handler in auth means unconfigured. API key auth is checked with
// auth.APIKey.Check when set, otherwise by resolving it. A nil env means no
// ambient auth.
func CheckProviderAuth(
	ctx context.Context,
	provider Provider,
	auth ProviderAuth,
	store CredentialStore,
	env AuthContext,
) (*AuthCheck, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if env == nil {
		env = NoAmbientAuth()
	}

	stored, err := store.Read(ctx, provider)
	if err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return nil, ctxErr
		}
		return nil, &AuthError{Code: AuthErrorStore, Provider: provider, Message: fmt.Sprintf("credential store read failed for %s", provider), Err: err}
	}

	if stored != nil {
		switch {
		case stored.Type == CredentialOAuth && auth.OAuth != nil:
			return &AuthCheck{Type: CredentialOAuth, Source: "OAuth"}, nil
		case stored.Type != CredentialAPIKey || auth.APIKey == nil:
			return nil, nil
		}
	}
	if auth.APIKey == nil {
		return nil, nil
	}

	if auth.APIKey.Check != nil {
		check, err := auth.APIKey.Check(ctx, env, stored)
		if err != nil {
			if ctxErr := ctx.Err(); ctxErr != nil {
				return nil, ctxErr
			}
			return nil, &AuthError{Code: AuthErrorStore, Provider: provider, Message: fmt.Sprintf("API key auth check failed for provider %s", provider), Err: err}
		}
		return check, nil
	}
	result, err := resolveAPIKey(ctx, env, *auth.APIKey, provider, stored)
	if err != nil || result == nil {
		return nil, err
	}
	return &AuthCheck{Type: CredentialAPIKey, Source: result.Source}, nil
}

// ResolveProviderAuth resolves the request auth for provider. It returns nil,
// nil when the provider is not configured.
//
// A stored credential owns the provider: env is consulted only when nothing
// is stored. There is no fallback to env after a failed refresh, or when the
// stored credential's type has no handler in auth, so a broken login never
// silently switches to an ambient key. The order is:
//
//  1. overrides.APIKey, when auth has API key support
//  2. a stored OAuth credential, refreshed when it expires within five minutes
//     (or overrides.MinOAuthValidity, if longer)
//  3. a stored API key credential
//  4. ambient auth from env, such as environment variables
//
// overrides.Env is laid over env and over a stored API key credential's Env.
// A nil env means no ambient auth.
func ResolveProviderAuth(
	ctx context.Context,
	provider Provider,
	auth ProviderAuth,
	store CredentialStore,
	env AuthContext,
	overrides *AuthResolutionOverrides,
) (*AuthResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	var o AuthResolutionOverrides
	if overrides != nil {
		o = *overrides
	}
	if env == nil {
		env = NoAmbientAuth()
	}
	requestEnv := env
	if len(o.Env) > 0 {
		requestEnv = overlayAuthContext{base: env, env: o.Env}
	}

	if o.APIKey != nil && auth.APIKey != nil {
		credential := &Credential{Type: CredentialAPIKey, Key: *o.APIKey, Env: o.Env}
		return resolveAPIKey(ctx, requestEnv, *auth.APIKey, provider, credential)
	}

	stored, err := store.Read(ctx, provider)
	if err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return nil, ctxErr
		}
		return nil, &AuthError{Code: AuthErrorStore, Provider: provider, Message: fmt.Sprintf("credential store read failed for %s", provider), Err: err}
	}

	if stored != nil {
		switch {
		case stored.Type == CredentialOAuth && auth.OAuth != nil:
			minValidity := max(defaultOAuthMinimumValidity, o.MinOAuthValidity)
			return resolveStoredOAuth(ctx, store, provider, *auth.OAuth, stored, minValidity, o.MinOAuthValidity > 0)
		case stored.Type == CredentialAPIKey && auth.APIKey != nil:
			credential := stored
			if len(o.Env) > 0 {
				merged := make(map[string]string, len(stored.Env)+len(o.Env))
				for k, v := range stored.Env {
					merged[k] = v
				}
				for k, v := range o.Env {
					merged[k] = v
				}
				copied := *stored
				copied.Env = merged
				credential = &copied
			}
			return resolveAPIKey(ctx, requestEnv, *auth.APIKey, provider, credential)
		}
		return nil, nil
	}

	if auth.APIKey == nil {
		return nil, nil
	}
	return resolveAPIKey(ctx, requestEnv, *auth.APIKey, provider, nil)
}

// resolveStoredOAuth turns a stored OAuth credential into request auth,
// refreshing it first if it expires within minValidity. The expiry check here
// is optimistic; RefreshStoredOAuthCredential repeats it under the lock.
// With requireValidity, a refresh that still leaves less than minValidity is
// an error; the default window only decides when to refresh.
func resolveStoredOAuth(
	ctx context.Context,
	store CredentialStore,
	provider Provider,
	oauth OAuthAuth,
	stored *Credential,
	minValidity time.Duration,
	requireValidity bool,
) (*AuthResult, error) {
	if stored.OAuth == nil {
		return nil, &AuthError{Code: AuthErrorStore, Provider: provider, Message: fmt.Sprintf("stored OAuth credential for %s has no token", provider)}
	}
	expiresSoon := func(c *Credential) bool {
		return !time.Now().Add(minValidity).Before(c.OAuth.Expires)
	}

	credential := stored
	if expiresSoon(credential) {
		refreshed, err := RefreshStoredOAuthCredential(ctx, store, provider, oauth, expiresSoon)
		if err != nil {
			return nil, err
		}
		if refreshed == nil {
			return nil, nil // logged out meanwhile
		}
		credential = refreshed
		if requireValidity && expiresSoon(credential) {
			return nil, &AuthError{Code: AuthErrorOAuth, Provider: provider, Message: fmt.Sprintf("OAuth refresh returned a token that expires too soon for %s", provider)}
		}
	}

	if oauth.ToAuth == nil {
		return nil, &AuthError{Code: AuthErrorOAuth, Provider: provider, Message: fmt.Sprintf("OAuth for %s cannot build request auth", provider)}
	}
	modelAuth, err := oauth.ToAuth(credential)
	if err != nil {
		return nil, &AuthError{Code: AuthErrorOAuth, Provider: provider, Message: fmt.Sprintf("OAuth auth derivation failed for %s", provider), Err: err}
	}
	return &AuthResult{Auth: modelAuth, Source: "OAuth"}, nil
}

func resolveAPIKey(ctx context.Context, env AuthContext, apiKey APIKeyAuth, provider Provider, credential *Credential) (*AuthResult, error) {
	if apiKey.Resolve == nil {
		return nil, &AuthError{Code: AuthErrorStore, Provider: provider, Message: fmt.Sprintf("API key auth for %s cannot resolve", provider)}
	}
	result, err := apiKey.Resolve(ctx, env, credential)
	if err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return nil, ctxErr
		}
		return nil, &AuthError{Code: AuthErrorStore, Provider: provider, Message: fmt.Sprintf("API key auth failed for provider %s", provider), Err: err}
	}
	return result, nil
}

// overlayAuthContext reads env first and base for anything env does not set.
type overlayAuthContext struct {
	base AuthContext
	env  map[string]string
}

func (c overlayAuthContext) Env(name string) (string, bool) {
	if value := c.env[name]; value != "" {
		return value, true
	}
	return c.base.Env(name)
}

func (c overlayAuthContext) FileExists(path string) bool { return c.base.FileExists(path) }

// RefreshStoredOAuthCredential refreshes provider's stored OAuth credential
// under the store's lock and stores the result before the lock is released.
// needsRefresh is checked again under the lock, so when several requests or
// processes find the token expired, only the first refreshes and the rest use
// its result.
//
// ctx cancels only the wait for the lock. Once the refresh starts, the
// provider may already have rotated the refresh token, so the refresh and its
// write ignore ctx and are bounded by oauthRefreshTimeout; otherwise a
// cancelled caller could throw away the only valid refresh token. A caller
// cancelled during the refresh gets ctx.Err() at once while the refresh
// finishes in the background.
//
// It returns the stored OAuth credential afterwards, or nil if the provider no
// longer has one, such as after a logout.
func RefreshStoredOAuthCredential(
	ctx context.Context,
	store CredentialStore,
	provider Provider,
	oauth OAuthAuth,
	needsRefresh func(*Credential) bool,
) (*Credential, error) {
	if oauth.Refresh == nil {
		return nil, &AuthError{Code: AuthErrorOAuth, Provider: provider, Message: fmt.Sprintf("OAuth for %s cannot refresh", provider)}
	}

	// lockCtx is what the store sees. ctx cancels it only until fn starts:
	// stopping the AfterFunc detaches it, so the store's transaction outlives
	// the caller.
	lockCtx, cancelLock := context.WithCancel(context.WithoutCancel(ctx))
	stopCancel := context.AfterFunc(ctx, cancelLock)

	fn := func(current *Credential) (*Credential, error) {
		if !stopCancel() {
			// ctx was cancelled while waiting for the lock.
			return nil, ctx.Err()
		}
		if current == nil || current.Type != CredentialOAuth || current.OAuth == nil {
			return nil, nil // logged out meanwhile
		}
		if !needsRefresh(current) {
			return nil, nil // another request or process refreshed it
		}

		refreshCtx, cancel := context.WithTimeout(lockCtx, oauthRefreshTimeout)
		defer cancel()
		refreshed, err := oauth.Refresh(refreshCtx, current)
		if err == nil && (refreshed == nil || refreshed.Type != CredentialOAuth || refreshed.OAuth == nil) {
			err = errors.New("refresh returned no OAuth credential")
		}
		if err != nil {
			return nil, &AuthError{Code: AuthErrorOAuth, Provider: provider, Message: fmt.Sprintf("OAuth refresh failed for %s", provider), Err: err}
		}
		return refreshed, nil
	}

	type result struct {
		post *Credential
		err  error
	}
	done := make(chan result, 1)
	go func() {
		defer cancelLock()
		defer stopCancel()
		post, err := store.Modify(lockCtx, provider, fn)
		done <- result{post, err}
	}()

	var r result
	select {
	case r = <-done:
	case <-ctx.Done():
		return nil, ctx.Err()
	}

	if r.err != nil {
		var authErr *AuthError
		if errors.As(r.err, &authErr) {
			return nil, authErr
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		return nil, &AuthError{Code: AuthErrorStore, Provider: provider, Message: fmt.Sprintf("credential store modify failed for %s", provider), Err: r.err}
	}
	if r.post == nil || r.post.Type != CredentialOAuth {
		return nil, nil
	}
	return r.post, nil
}
