// Ported from pi's packages/ai/src/auth/types.ts,
// packages/ai/src/auth/context.ts and the header merging and applyAuth in
// packages/ai/src/models.ts (MIT, see LICENSE).

package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"maps"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// CredentialType is the kind of a stored credential.
type CredentialType string

const (
	CredentialAPIKey CredentialType = "api_key"
	CredentialOAuth  CredentialType = "oauth"
)

// OAuthToken is the token set of an OAuth credential.
type OAuthToken struct {
	Access  string
	Refresh string
	Expires time.Time
	// Extra keeps provider fields, such as the Codex account ID, through a
	// round trip.
	Extra map[string]json.RawMessage
}

// Credential is one stored credential for a provider: an API key or an
// OAuth token set, as Type says.
type Credential struct {
	Type CredentialType
	// Key is the API key of an api_key credential.
	Key string
	// Env holds provider-scoped config of an api_key credential, such as an
	// account ID.
	Env map[string]string
	// OAuth is the token set of an oauth credential.
	OAuth *OAuthToken
}

// Clone returns a deep copy of c, so a store can hand out credentials without
// sharing its own. Clone of nil is nil.
func (c *Credential) Clone() *Credential {
	if c == nil {
		return nil
	}
	cloned := *c
	cloned.Env = maps.Clone(c.Env)
	if c.OAuth != nil {
		token := *c.OAuth
		if c.OAuth.Extra != nil {
			token.Extra = make(map[string]json.RawMessage, len(c.OAuth.Extra))
			for key, value := range c.OAuth.Extra {
				token.Extra[key] = bytes.Clone(value)
			}
		}
		cloned.OAuth = &token
	}
	return &cloned
}

// ModelAuth is the auth for a single model request. Anything that cannot be
// expressed as an API key, headers, or a base URL is provider config, not auth.
type ModelAuth struct {
	APIKey  *string
	Headers map[string]string
	BaseURL *string
}

// MergeHeaders merges header maps into a new map. Later maps win, and names
// compare case-insensitively, so "authorization" replaces "Authorization".
// The winning spelling is kept. It returns nil when there are no headers.
func MergeHeaders(layers ...map[string]string) map[string]string {
	var merged map[string]string
	for _, layer := range layers {
		for name, value := range layer {
			if merged == nil {
				merged = map[string]string{}
			}
			for existing := range merged {
				if existing != name && strings.EqualFold(existing, name) {
					delete(merged, existing)
				}
			}
			merged[name] = value
		}
	}
	return merged
}

// ApplyModelAuth applies auth to a request: it sets options.ApiKey, lays
// auth.Headers under options.Headers, and returns model with auth.BaseURL. The
// caller's headers win, so an app can still override a header per request.
// It changes neither the caller's header map nor model's.
func ApplyModelAuth[T API](model Model[T], options *StreamOptions, auth ModelAuth) Model[T] {
	if auth.APIKey != nil {
		key := *auth.APIKey
		options.ApiKey = &key
	}
	if len(auth.Headers) > 0 {
		options.Headers = MergeHeaders(auth.Headers, options.Headers)
	}
	if auth.BaseURL != nil {
		model.BaseURL = *auth.BaseURL
	}
	return model
}

// AuthResult is resolved request auth for a provider.
type AuthResult struct {
	Auth ModelAuth
	// Env holds provider-scoped config resolved from the credential and
	// ambient sources.
	Env map[string]string
	// Source labels where the auth came from for status UI, such as
	// "OPENAI_API_KEY" or "OAuth".
	Source string
}

// AuthCheck reports that a provider has auth configured, found without
// refreshing tokens or doing other request-time work.
type AuthCheck struct {
	Type   CredentialType
	Source string
}

// CredentialInfo is credential metadata for listing, without secrets.
type CredentialInfo struct {
	Provider Provider
	Type     CredentialType
}

// CredentialStore stores one credential per provider, for one user. Apps
// provide persistent stores; NewInMemoryCredentialStore is the default.
//
// Modify is the only write path, so every change is a serialized
// read-modify-write. ResolveProviderAuth refreshes OAuth tokens inside Modify,
// so concurrent requests cannot refresh a rotated token twice.
//
//   - Read returns nil, nil when nothing is stored. The credential may be
//     expired; request auth comes from ResolveProviderAuth.
//   - List returns metadata only. It must not resolve or expose secrets.
//   - Modify runs fn with the current credential (nil if none) while holding
//     a lock on the provider, across processes too where the backing store
//     supports it. fn may make a network call; hold the lock across it. fn
//     returning nil leaves the credential unchanged. Modify returns the
//     credential after the write. An error from fn stores nothing and is
//     returned, wrapped or not, so errors.As still finds it. If ctx is done
//     before the lock is acquired, Modify returns ctx.Err() without running fn.
//   - Delete removes the credential, serialized against Modify. Deleting
//     nothing succeeds.
//
// Methods return errors only when storage fails. Stores must not keep the
// credentials passed to them, nor return ones they keep; see Credential.Clone.
type CredentialStore interface {
	Read(ctx context.Context, provider Provider) (*Credential, error)
	List(ctx context.Context) ([]CredentialInfo, error)
	Modify(ctx context.Context, provider Provider, fn func(current *Credential) (*Credential, error)) (*Credential, error)
	Delete(ctx context.Context, provider Provider) error
}

// AuthContext is where ambient auth comes from: environment variables and
// credential files. It is injectable so that multi-tenant apps can turn
// ambient auth off with NoAmbientAuth.
type AuthContext interface {
	// Env returns the named variable, and false when it is unset or blank.
	Env(name string) (string, bool)
	// FileExists reports whether path exists. A leading ~ is the user's home
	// directory.
	FileExists(path string) bool
}

// DefaultAuthContext reads the process environment and file system.
func DefaultAuthContext() AuthContext { return defaultAuthContext{} }

// NoAmbientAuth is an AuthContext with no variables and no files, so only
// stored credentials and overrides configure a provider.
func NoAmbientAuth() AuthContext { return noAmbientAuth{} }

type defaultAuthContext struct{}

func (defaultAuthContext) Env(name string) (string, bool) {
	value := os.Getenv(name)
	if strings.TrimSpace(value) == "" {
		return "", false
	}
	return value, true
}

func (defaultAuthContext) FileExists(path string) bool {
	if path == "~" || strings.HasPrefix(path, "~/") {
		home, err := os.UserHomeDir()
		if err != nil {
			return false
		}
		path = filepath.Join(home, path[1:])
	}
	_, err := os.Stat(path)
	return err == nil
}

type noAmbientAuth struct{}

func (noAmbientAuth) Env(string) (string, bool) { return "", false }

func (noAmbientAuth) FileExists(string) bool { return false }
