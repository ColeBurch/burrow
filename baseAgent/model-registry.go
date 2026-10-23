// Adapted from pi's packages/coding-agent/src/core/model-registry.ts and the
// auth methods of Models in packages/ai/src/models.ts (MIT, see LICENSE).

package baseAgent

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"sort"

	"github.com/ColeBurch/burrow/ai"
)

// ErrNoAuth means a provider has no auth configured. Check for it with
// errors.Is to ask the user to log in.
var ErrNoAuth = errors.New("no auth configured")

// ModelRegistryOptions configures a ModelRegistry. The zero value uses the
// built-in models and providers, an in-memory credential store, and no
// ambient auth.
type ModelRegistryOptions struct {
	// Credentials stores logins. Nil means a new in-memory store, so logins
	// last only as long as the registry. Multi-user apps pass
	// ScopeAuthStore(store, ownerID).
	Credentials ai.CredentialStore
	// AmbientAuth is where auth comes from for a provider with no stored
	// credential. Nil means none, so environment variables such as
	// OPENAI_API_KEY are ignored; pass ai.DefaultAuthContext() to read them.
	// Leave it nil when the process environment belongs to a server rather
	// than to the user.
	AmbientAuth ai.AuthContext
	// ProviderAuth says how each provider authenticates. Nil means
	// ai.DefaultProviderAuth(). A provider missing from it is never
	// configured.
	ProviderAuth map[ai.Provider]ai.ProviderAuth
	// Models is the model list. Nil means ai.Models.
	Models map[ai.Provider]map[string]ai.Model[ai.API]
}

// ModelRegistry provides model lookup, auth checks, and request auth.
type ModelRegistry struct {
	models       map[ai.Provider]map[string]ai.Model[ai.API]
	providerAuth map[ai.Provider]ai.ProviderAuth
	credentials  ai.CredentialStore
	ambientAuth  ai.AuthContext
}

// NewModelRegistry creates a registry. options may be nil.
func NewModelRegistry(options *ModelRegistryOptions) *ModelRegistry {
	var o ModelRegistryOptions
	if options != nil {
		o = *options
	}
	if o.Credentials == nil {
		o.Credentials = ai.NewInMemoryCredentialStore()
	}
	if o.AmbientAuth == nil {
		o.AmbientAuth = ai.NoAmbientAuth()
	}
	if o.ProviderAuth == nil {
		o.ProviderAuth = ai.DefaultProviderAuth()
	}
	if o.Models == nil {
		o.Models = ai.Models
	}

	return &ModelRegistry{
		models:       cloneModels(o.Models),
		providerAuth: maps.Clone(o.ProviderAuth),
		credentials:  o.Credentials,
		ambientAuth:  o.AmbientAuth,
	}
}

// All returns every model in deterministic provider/id order.
func (r *ModelRegistry) All() []ai.Model[ai.API] {
	providers := make([]ai.Provider, 0, len(r.models))
	for provider := range r.models {
		providers = append(providers, provider)
	}
	sort.Slice(providers, func(i, j int) bool { return providers[i] < providers[j] })

	out := make([]ai.Model[ai.API], 0)
	for _, provider := range providers {
		ids := make([]string, 0, len(r.models[provider]))
		for id := range r.models[provider] {
			ids = append(ids, id)
		}
		sort.Strings(ids)

		for _, id := range ids {
			out = append(out, cloneModel(r.models[provider][id]))
		}
	}

	return out
}

// Available returns models whose provider has auth configured. It checks
// each provider once and never refreshes tokens, so a listed model can still
// fail at request time, such as when an OAuth refresh fails.
func (r *ModelRegistry) Available(ctx context.Context) ([]ai.Model[ai.API], error) {
	models := r.All()
	configured := map[ai.Provider]bool{}
	out := make([]ai.Model[ai.API], 0, len(models))
	for _, model := range models {
		ok, checked := configured[model.Provider]
		if !checked {
			check, err := r.CheckAuth(ctx, model.Provider)
			if err != nil {
				return nil, err
			}
			ok = check != nil
			configured[model.Provider] = ok
		}
		if ok {
			out = append(out, model)
		}
	}
	return out, nil
}

// Find returns a model by provider and model ID.
func (r *ModelRegistry) Find(provider ai.Provider, modelID string) (ai.Model[ai.API], bool) {
	providerModels, ok := r.models[provider]
	if !ok {
		return ai.Model[ai.API]{}, false
	}

	model, ok := providerModels[modelID]
	if !ok {
		return ai.Model[ai.API]{}, false
	}
	return cloneModel(model), true
}

// CheckAuth reports whether provider has auth configured, and from where,
// without refreshing tokens. It returns nil, nil when it is not configured.
// See ai.CheckProviderAuth.
func (r *ModelRegistry) CheckAuth(ctx context.Context, provider ai.Provider) (*ai.AuthCheck, error) {
	auth, ok := r.providerAuth[provider]
	if !ok {
		return nil, ctx.Err()
	}
	return ai.CheckProviderAuth(ctx, provider, auth, r.credentials, r.ambientAuth)
}

// Auth resolves the auth for a request to model, refreshing an OAuth token if
// it is about to expire. The model's own headers are laid over the auth's.
// Apply the result with ai.ApplyModelAuth.
//
// It returns an error wrapping ErrNoAuth when the provider is not configured,
// and an *ai.AuthError when the credential store or a refresh fails. See
// ai.ResolveProviderAuth.
func (r *ModelRegistry) Auth(ctx context.Context, model ai.Model[ai.API]) (ai.ModelAuth, error) {
	var result *ai.AuthResult
	if auth, ok := r.providerAuth[model.Provider]; ok {
		var err error
		result, err = ai.ResolveProviderAuth(ctx, model.Provider, auth, r.credentials, r.ambientAuth, nil)
		if err != nil {
			return ai.ModelAuth{}, err
		}
	}
	if result == nil {
		if err := ctx.Err(); err != nil {
			return ai.ModelAuth{}, err
		}
		return ai.ModelAuth{}, fmt.Errorf("provider %q: %w", model.Provider, ErrNoAuth)
	}

	// result.Env is provider config, which no built-in provider reads yet.
	auth := result.Auth
	if model.Headers != nil {
		auth.Headers = ai.MergeHeaders(auth.Headers, *model.Headers)
	}
	return auth, nil
}

// Login runs provider's login flow for the credential type typ and stores the
// credential, replacing any stored one. The flow runs before the store is
// locked, since it waits on the user.
func (r *ModelRegistry) Login(ctx context.Context, provider ai.Provider, typ ai.CredentialType, in ai.AuthInteraction) error {
	auth := r.providerAuth[provider]
	var login func(context.Context, ai.AuthInteraction) (*ai.Credential, error)
	switch {
	case typ == ai.CredentialAPIKey && auth.APIKey != nil:
		login = auth.APIKey.Login
	case typ == ai.CredentialOAuth && auth.OAuth != nil:
		login = auth.OAuth.Login
	}
	if login == nil {
		return fmt.Errorf("provider %q has no %s login", provider, typ)
	}

	credential, err := login(ctx, in)
	if err != nil {
		return err
	}
	if credential == nil || credential.Type != typ {
		return fmt.Errorf("%s login for provider %q returned no %s credential", typ, provider, typ)
	}

	if _, err := r.credentials.Modify(ctx, provider, func(*ai.Credential) (*ai.Credential, error) {
		return credential, nil
	}); err != nil {
		return storeError(ctx, provider, "modify", err)
	}
	return nil
}

// Logout deletes provider's stored credential. Ambient auth, such as an
// environment variable, may still configure the provider afterwards.
func (r *ModelRegistry) Logout(ctx context.Context, provider ai.Provider) error {
	if err := r.credentials.Delete(ctx, provider); err != nil {
		return storeError(ctx, provider, "delete", err)
	}
	return nil
}

// storeError wraps a credential store failure as ai.ResolveProviderAuth does.
func storeError(ctx context.Context, provider ai.Provider, op string, err error) error {
	if ctxErr := ctx.Err(); ctxErr != nil {
		return ctxErr
	}
	return &ai.AuthError{Code: ai.AuthErrorStore, Provider: provider, Message: fmt.Sprintf("credential store %s failed for %s", op, provider), Err: err}
}

// ProviderDisplayName returns a human-friendly provider name.
func (r *ModelRegistry) ProviderDisplayName(provider ai.Provider) string {
	switch provider {
	case ai.ProviderOpenAI:
		return "OpenAI"
	default:
		return string(provider)
	}
}

func cloneModels(src map[ai.Provider]map[string]ai.Model[ai.API]) map[ai.Provider]map[string]ai.Model[ai.API] {
	dst := make(map[ai.Provider]map[string]ai.Model[ai.API], len(src))
	for provider, models := range src {
		dst[provider] = make(map[string]ai.Model[ai.API], len(models))
		for id, model := range models {
			dst[provider][id] = cloneModel(model)
		}
	}
	return dst
}

func cloneModel(model ai.Model[ai.API]) ai.Model[ai.API] {
	model.Input = append([]string(nil), model.Input...)

	if model.ThinkingLevelMap != nil {
		thinkingLevelMap := make(ai.ThinkingLevelMap, len(model.ThinkingLevelMap))
		for level, value := range model.ThinkingLevelMap {
			if value == nil {
				thinkingLevelMap[level] = nil
				continue
			}
			copied := *value
			thinkingLevelMap[level] = &copied
		}
		model.ThinkingLevelMap = thinkingLevelMap
	}

	if model.Headers != nil {
		headers := make(map[string]string, len(*model.Headers))
		for key, value := range *model.Headers {
			headers[key] = value
		}
		model.Headers = &headers
	}

	if model.Compat != nil {
		compat := *model.Compat
		model.Compat = &compat
	}

	return model
}
