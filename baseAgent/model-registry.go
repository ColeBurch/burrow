package baseAgent

import (
	"fmt"
	"os"
	"sort"

	"github.com/ColeBurch/burrow/ai"
)

// AuthStore resolves provider API keys for model requests.
type AuthStore interface {
	APIKey(provider ai.Provider) (string, bool)
}

// EnvAuthStore resolves API keys from environment variables.
type EnvAuthStore struct{}

func (EnvAuthStore) APIKey(provider ai.Provider) (string, bool) {
	var envName string
	switch provider {
	case ai.ProviderOpenAI:
		envName = "OPENAI_API_KEY"
	default:
		return "", false
	}

	value := os.Getenv(envName)
	return value, value != ""
}

// RequestAuth is the request-level auth and headers resolved for a model.
type RequestAuth struct {
	APIKey  *string
	Headers map[string]string
}

// ModelRegistry provides model lookup and request auth resolution.
type ModelRegistry struct {
	models map[ai.Provider]map[string]ai.Model[ai.API]
	auth   AuthStore
}

// NewModelRegistry creates a registry using the built-in ai.Models list.
func NewModelRegistry(auth AuthStore) *ModelRegistry {
	if auth == nil {
		auth = EnvAuthStore{}
	}

	return &ModelRegistry{
		models: cloneModels(ai.Models),
		auth:   auth,
	}
}

// NewModelRegistryWithModels creates a registry using an explicit model set.
func NewModelRegistryWithModels(auth AuthStore, models map[ai.Provider]map[string]ai.Model[ai.API]) *ModelRegistry {
	if auth == nil {
		auth = EnvAuthStore{}
	}

	return &ModelRegistry{
		models: cloneModels(models),
		auth:   auth,
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

// Available returns models whose provider has configured auth.
func (r *ModelRegistry) Available() []ai.Model[ai.API] {
	models := r.All()
	out := make([]ai.Model[ai.API], 0, len(models))
	for _, model := range models {
		if r.HasConfiguredAuth(model) {
			out = append(out, model)
		}
	}
	return out
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

// HasConfiguredAuth reports whether a model's provider has an API key.
func (r *ModelRegistry) HasConfiguredAuth(model ai.Model[ai.API]) bool {
	_, ok := r.auth.APIKey(model.Provider)
	return ok
}

// APIKeyAndHeaders resolves the API key and headers to use for a model request.
func (r *ModelRegistry) APIKeyAndHeaders(model ai.Model[ai.API]) (RequestAuth, error) {
	apiKey, ok := r.auth.APIKey(model.Provider)
	if !ok {
		return RequestAuth{}, fmt.Errorf("no API key found for provider %q", model.Provider)
	}

	headers := map[string]string{}
	if model.Headers != nil {
		for key, value := range *model.Headers {
			headers[key] = value
		}
	}

	return RequestAuth{APIKey: &apiKey, Headers: headers}, nil
}

// APIKeyForProvider resolves a provider API key.
func (r *ModelRegistry) APIKeyForProvider(provider ai.Provider) (string, bool) {
	return r.auth.APIKey(provider)
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
