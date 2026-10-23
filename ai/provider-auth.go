// Ported from pi's packages/ai/src/auth/types.ts, packages/ai/src/auth/helpers.ts
// and packages/ai/src/providers/openai.ts (MIT, see LICENSE).

package ai

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strings"
)

// AuthPromptType is the kind of input an AuthPrompt asks for.
type AuthPromptType string

const (
	AuthPromptText   AuthPromptType = "text"
	AuthPromptSecret AuthPromptType = "secret"
	AuthPromptSelect AuthPromptType = "select"
)

// AuthPromptOption is one choice of a select prompt.
type AuthPromptOption struct {
	ID          string
	Label       string
	Description string
}

// AuthPrompt asks the user for input during login.
type AuthPrompt struct {
	Type        AuthPromptType
	Message     string
	Placeholder string
	// Options are the choices of a select prompt.
	Options []AuthPromptOption
}

// AuthEventType is the kind of an AuthEvent.
type AuthEventType string

const (
	AuthEventInfo     AuthEventType = "info"
	AuthEventProgress AuthEventType = "progress"
)

// AuthEvent tells the user about login progress. It needs no answer.
type AuthEvent struct {
	Type    AuthEventType
	Message string
}

// AuthInteraction is how a login flow talks to the user, for both API key and
// OAuth logins. The app provides it: a TUI prompts in the terminal, a web app
// renders a form.
type AuthInteraction struct {
	// Prompt returns the entered text, or the chosen option's ID for a select
	// prompt. It returns an error if the user cancels or ctx is done.
	Prompt func(ctx context.Context, prompt AuthPrompt) (string, error)
	// Notify shows event to the user. It may be nil.
	Notify func(event AuthEvent)
}

// APIKeyAuth is API key auth: a stored key or provider config, plus ambient
// sources such as environment variables.
type APIKeyAuth struct {
	// Name is shown to the user, such as "OpenAI API key".
	Name string
	// Login asks the user for a key and returns the credential to store. Nil
	// means the provider only uses ambient auth.
	Login func(ctx context.Context, in AuthInteraction) (*Credential, error)
	// Check optionally reports whether auth is configured without request-time
	// work. Set it when Resolve runs commands or makes network calls; without
	// it, checks call Resolve.
	Check func(ctx context.Context, env AuthContext, cred *Credential) (*AuthCheck, error)
	// Resolve builds request auth from cred, which may be nil, and env. It
	// returns nil, nil when the provider is not configured.
	Resolve func(ctx context.Context, env AuthContext, cred *Credential) (*AuthResult, error)
}

// OAuthAuth is OAuth auth. Refresh and ToAuth are separate so that
// ResolveProviderAuth can refresh under the store's lock: Refresh produces a
// credential, and ToAuth derives request auth from whichever credential ends
// up stored.
type OAuthAuth struct {
	// Name is shown to the user, such as "OpenAI (ChatGPT subscription)".
	Name string
	// IsSubscription reports whether access is backed by a provider
	// subscription rather than API billing.
	IsSubscription bool
	// Login runs the provider's login flow and returns the credential to store.
	Login func(ctx context.Context, in AuthInteraction) (*Credential, error)
	// Refresh exchanges cred's refresh token over the network. It runs under
	// the store's lock with its own timeout. It must return the complete new
	// credential, keeping the old refresh token if the provider sent none.
	Refresh func(ctx context.Context, cred *Credential) (*Credential, error)
	// ToAuth derives request auth from a valid credential, including a base
	// URL or headers specific to the credential. It must have no side effects.
	ToAuth func(cred *Credential) (ModelAuth, error)
}

// ProviderAuth is how a provider authenticates. At least one of APIKey and
// OAuth is set; even providers that only use ambient auth or no key at all
// have APIKey auth, whose Resolve reports whether they are configured.
type ProviderAuth struct {
	APIKey *APIKeyAuth
	OAuth  *OAuthAuth
}

// EnvAPIKeyAuth is standard API key auth: a stored key wins, otherwise the
// first of envVars that is set. Login asks for the key. Providers that
// resolve differently, such as from provider config, credential files or IAM,
// write their own APIKeyAuth.
func EnvAPIKeyAuth(name string, envVars ...string) *APIKeyAuth {
	envVars = slices.Clone(envVars)
	return &APIKeyAuth{
		Name: name,
		Login: func(ctx context.Context, in AuthInteraction) (*Credential, error) {
			if in.Prompt == nil {
				return nil, errors.New("login needs a prompt")
			}
			key, err := in.Prompt(ctx, AuthPrompt{Type: AuthPromptSecret, Message: "Enter " + name})
			if err != nil {
				return nil, err
			}
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			key = strings.TrimSpace(key)
			if key == "" {
				return nil, fmt.Errorf("no %s entered", name)
			}
			return &Credential{Type: CredentialAPIKey, Key: key}, nil
		},
		Resolve: func(ctx context.Context, env AuthContext, cred *Credential) (*AuthResult, error) {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			if cred != nil && cred.Key != "" {
				key := cred.Key
				return &AuthResult{Auth: ModelAuth{APIKey: &key}, Env: maps.Clone(cred.Env), Source: "stored credential"}, nil
			}
			for _, envVar := range envVars {
				if value, ok := env.Env(envVar); ok {
					return &AuthResult{Auth: ModelAuth{APIKey: &value}, Source: envVar}, nil
				}
			}
			return nil, nil
		},
	}
}

// DefaultProviderAuth returns the auth of the built-in providers. Each call
// returns a new map, so callers may change it.
func DefaultProviderAuth() map[Provider]ProviderAuth {
	return map[Provider]ProviderAuth{
		ProviderOpenAI: {APIKey: EnvAPIKeyAuth("OpenAI API key", "OPENAI_API_KEY")},
	}
}
