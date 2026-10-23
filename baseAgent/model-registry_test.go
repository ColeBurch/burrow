package baseAgent_test

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ColeBurch/burrow/ai"
	"github.com/ColeBurch/burrow/baseAgent"
)

const registryTestProvider ai.Provider = "test"

// registryTestModels returns two models of registryTestProvider and one of
// OpenAI.
func registryTestModels() map[ai.Provider]map[string]ai.Model[ai.API] {
	headers := map[string]string{"X-Model": "model", "x-shared": "model"}
	return map[ai.Provider]map[string]ai.Model[ai.API]{
		registryTestProvider: {
			"a": {ID: "a", Provider: registryTestProvider, BaseURL: "https://model.example", Headers: &headers},
			"b": {ID: "b", Provider: registryTestProvider},
		},
		ai.ProviderOpenAI: {"gpt": {ID: "gpt", Provider: ai.ProviderOpenAI}},
	}
}

// testOAuth is OAuth auth whose Refresh counts its calls and returns a token
// valid for an hour.
func testOAuth(refreshes *atomic.Int32) ai.ProviderAuth {
	return ai.ProviderAuth{OAuth: &ai.OAuthAuth{
		Name: "Test OAuth",
		Login: func(context.Context, ai.AuthInteraction) (*ai.Credential, error) {
			return &ai.Credential{Type: ai.CredentialOAuth, OAuth: &ai.OAuthToken{Access: "logged-in", Expires: time.Now().Add(time.Hour)}}, nil
		},
		Refresh: func(_ context.Context, cred *ai.Credential) (*ai.Credential, error) {
			refreshes.Add(1)
			return &ai.Credential{Type: ai.CredentialOAuth, OAuth: &ai.OAuthToken{Access: "refreshed", Refresh: cred.OAuth.Refresh, Expires: time.Now().Add(time.Hour)}}, nil
		},
		ToAuth: func(cred *ai.Credential) (ai.ModelAuth, error) {
			baseURL := "https://oauth.example"
			return ai.ModelAuth{
				APIKey:  &cred.OAuth.Access,
				Headers: map[string]string{"X-Shared": "auth", "X-Auth": "auth"},
				BaseURL: &baseURL,
			}, nil
		},
	}}
}

func storeCredential(t *testing.T, store ai.CredentialStore, provider ai.Provider, credential *ai.Credential) {
	t.Helper()
	if _, err := store.Modify(context.Background(), provider, func(*ai.Credential) (*ai.Credential, error) {
		return credential, nil
	}); err != nil {
		t.Fatal(err)
	}
}

func modelIDs(models []ai.Model[ai.API]) []string {
	ids := []string{}
	for _, model := range models {
		ids = append(ids, string(model.Provider)+"/"+model.ID)
	}
	return ids
}

func authKey(auth ai.ModelAuth) string {
	if auth.APIKey == nil {
		return "<nil>"
	}
	return *auth.APIKey
}

// enterKey is an AuthInteraction that enters key at every prompt.
func enterKey(key string) ai.AuthInteraction {
	return ai.AuthInteraction{Prompt: func(context.Context, ai.AuthPrompt) (string, error) { return key, nil }}
}

func TestModelRegistryIgnoresEnvByDefault(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "sk-env")
	ctx := context.Background()
	registry := baseAgent.NewModelRegistry(nil)
	model, ok := registry.Find(ai.ProviderOpenAI, "gpt-5")
	if !ok {
		t.Fatal("no built-in gpt-5 model")
	}

	if check, err := registry.CheckAuth(ctx, ai.ProviderOpenAI); check != nil || err != nil {
		t.Errorf("CheckAuth = %+v, %v; want unconfigured", check, err)
	}
	if available, err := registry.Available(ctx); len(available) != 0 || err != nil {
		t.Errorf("Available = %v, %v; want none", modelIDs(available), err)
	}
	if _, err := registry.Auth(ctx, model); !errors.Is(err, baseAgent.ErrNoAuth) {
		t.Errorf("Auth error = %v, want ErrNoAuth", err)
	}
}

func TestModelRegistryAmbientAuth(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "sk-env")
	ctx := context.Background()
	registry := baseAgent.NewModelRegistry(&baseAgent.ModelRegistryOptions{
		AmbientAuth: ai.DefaultAuthContext(),
		Models:      registryTestModels(),
	})

	check, err := registry.CheckAuth(ctx, ai.ProviderOpenAI)
	if err != nil || check == nil || check.Source != "OPENAI_API_KEY" {
		t.Errorf("CheckAuth = %+v, %v; want OPENAI_API_KEY", check, err)
	}
	available, err := registry.Available(ctx)
	if want := []string{"openai/gpt"}; err != nil || !reflect.DeepEqual(modelIDs(available), want) {
		t.Errorf("Available = %v, %v; want %v", modelIDs(available), err, want)
	}
	gpt, _ := registry.Find(ai.ProviderOpenAI, "gpt")
	if auth, err := registry.Auth(ctx, gpt); err != nil || authKey(auth) != "sk-env" {
		t.Errorf("Auth = %s, %v; want sk-env", authKey(auth), err)
	}
}

func TestModelRegistryLoginOwnsProvider(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "sk-env")
	ctx := context.Background()
	credentials := ai.NewInMemoryCredentialStore()
	registry := baseAgent.NewModelRegistry(&baseAgent.ModelRegistryOptions{
		Credentials: credentials,
		AmbientAuth: ai.DefaultAuthContext(),
		Models:      registryTestModels(),
	})
	gpt, _ := registry.Find(ai.ProviderOpenAI, "gpt")

	if err := registry.Login(ctx, ai.ProviderOpenAI, ai.CredentialAPIKey, enterKey("  sk-stored \n")); err != nil {
		t.Fatal(err)
	}
	if stored, _ := credentials.Read(ctx, ai.ProviderOpenAI); stored == nil || stored.Key != "sk-stored" {
		t.Errorf("stored credential = %+v, want the entered key", stored)
	}
	if auth, err := registry.Auth(ctx, gpt); err != nil || authKey(auth) != "sk-stored" {
		t.Errorf("Auth after login = %s, %v; want sk-stored over the environment", authKey(auth), err)
	}

	if err := registry.Logout(ctx, ai.ProviderOpenAI); err != nil {
		t.Fatal(err)
	}
	if auth, err := registry.Auth(ctx, gpt); err != nil || authKey(auth) != "sk-env" {
		t.Errorf("Auth after logout = %s, %v; want sk-env", authKey(auth), err)
	}
}

func TestModelRegistryAvailableNeverRefreshes(t *testing.T) {
	ctx := context.Background()
	var refreshes atomic.Int32
	credentials := &countingCredentialStore{CredentialStore: ai.NewInMemoryCredentialStore()}
	storeCredential(t, credentials, registryTestProvider, &ai.Credential{
		Type:  ai.CredentialOAuth,
		OAuth: &ai.OAuthToken{Access: "expired", Refresh: "refresh", Expires: time.Now().Add(-time.Hour)},
	})
	registry := baseAgent.NewModelRegistry(&baseAgent.ModelRegistryOptions{
		Credentials:  credentials,
		ProviderAuth: map[ai.Provider]ai.ProviderAuth{registryTestProvider: testOAuth(&refreshes)},
		Models:       registryTestModels(),
	})

	available, err := registry.Available(ctx)
	if want := []string{"test/a", "test/b"}; err != nil || !reflect.DeepEqual(modelIDs(available), want) {
		t.Errorf("Available = %v, %v; want %v", modelIDs(available), err, want)
	}
	if n := refreshes.Load(); n != 0 {
		t.Errorf("Available refreshed %d times, want 0", n)
	}
	if n := credentials.reads.Load(); n != 1 {
		t.Errorf("Available read the store %d times, want once for the one provider with auth", n)
	}
}

func TestModelRegistryAuthAppliesOAuth(t *testing.T) {
	ctx := context.Background()
	var refreshes atomic.Int32
	credentials := ai.NewInMemoryCredentialStore()
	storeCredential(t, credentials, registryTestProvider, &ai.Credential{
		Type:  ai.CredentialOAuth,
		OAuth: &ai.OAuthToken{Access: "expired", Refresh: "refresh", Expires: time.Now().Add(-time.Hour)},
	})
	registry := baseAgent.NewModelRegistry(&baseAgent.ModelRegistryOptions{
		Credentials:  credentials,
		ProviderAuth: map[ai.Provider]ai.ProviderAuth{registryTestProvider: testOAuth(&refreshes)},
		Models:       registryTestModels(),
	})
	model, _ := registry.Find(registryTestProvider, "a")

	auth, err := registry.Auth(ctx, model)
	if err != nil {
		t.Fatal(err)
	}
	if authKey(auth) != "refreshed" || refreshes.Load() != 1 {
		t.Errorf("Auth key = %s after %d refreshes, want the refreshed token after 1", authKey(auth), refreshes.Load())
	}
	if auth.BaseURL == nil || *auth.BaseURL != "https://oauth.example" {
		t.Errorf("BaseURL = %v, want the OAuth credential's", auth.BaseURL)
	}
	// Model headers win, whatever the case of their names.
	if want := map[string]string{"X-Model": "model", "x-shared": "model", "X-Auth": "auth"}; !reflect.DeepEqual(auth.Headers, want) {
		t.Errorf("Headers = %v, want %v", auth.Headers, want)
	}
	if again, _ := registry.Auth(ctx, model); authKey(again) != "refreshed" || refreshes.Load() != 1 {
		t.Errorf("second Auth = %s after %d refreshes, want the stored token with no new refresh", authKey(again), refreshes.Load())
	}
}

func TestModelRegistryAuthErrors(t *testing.T) {
	ctx := context.Background()
	failing := &countingCredentialStore{CredentialStore: ai.NewInMemoryCredentialStore(), err: errors.New("database down")}
	registry := baseAgent.NewModelRegistry(&baseAgent.ModelRegistryOptions{Credentials: failing, Models: registryTestModels()})
	gpt, _ := registry.Find(ai.ProviderOpenAI, "gpt")
	unknown := ai.Model[ai.API]{ID: "m", Provider: "unknown"}

	var authErr *ai.AuthError
	if _, err := registry.Auth(ctx, gpt); !errors.As(err, &authErr) || authErr.Code != ai.AuthErrorStore || errors.Is(err, baseAgent.ErrNoAuth) {
		t.Errorf("Auth with a failing store = %v, want an *ai.AuthError with code %q", err, ai.AuthErrorStore)
	}
	if _, err := registry.Available(ctx); !errors.As(err, &authErr) {
		t.Errorf("Available with a failing store = %v, want an *ai.AuthError", err)
	}
	if _, err := registry.Auth(ctx, unknown); !errors.Is(err, baseAgent.ErrNoAuth) || !strings.Contains(err.Error(), `"unknown"`) {
		t.Errorf("Auth for an unknown provider = %v, want ErrNoAuth naming it", err)
	}
	if check, err := registry.CheckAuth(ctx, "unknown"); check != nil || err != nil {
		t.Errorf("CheckAuth for an unknown provider = %+v, %v; want nil, nil", check, err)
	}

	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := registry.Auth(cancelled, unknown); !errors.Is(err, context.Canceled) {
		t.Errorf("Auth with a cancelled context = %v, want context.Canceled", err)
	}
}

func TestModelRegistryLoginErrors(t *testing.T) {
	ctx := context.Background()
	credentials := &countingCredentialStore{CredentialStore: ai.NewInMemoryCredentialStore()}
	wrongType := testOAuth(new(atomic.Int32))
	wrongType.OAuth.Login = func(context.Context, ai.AuthInteraction) (*ai.Credential, error) {
		return &ai.Credential{Type: ai.CredentialAPIKey, Key: "sk"}, nil
	}
	registry := baseAgent.NewModelRegistry(&baseAgent.ModelRegistryOptions{
		Credentials: credentials,
		ProviderAuth: map[ai.Provider]ai.ProviderAuth{
			ai.ProviderOpenAI:    ai.DefaultProviderAuth()[ai.ProviderOpenAI],
			registryTestProvider: wrongType,
		},
	})
	cancelled := errors.New("cancelled by user")

	for _, test := range []struct {
		name     string
		provider ai.Provider
		typ      ai.CredentialType
		in       ai.AuthInteraction
		want     string
	}{
		{name: "unknown provider", provider: "unknown", typ: ai.CredentialAPIKey, want: `provider "unknown" has no api_key login`},
		{name: "unsupported type", provider: ai.ProviderOpenAI, typ: ai.CredentialOAuth, want: `provider "openai" has no oauth login`},
		{name: "flow fails", provider: ai.ProviderOpenAI, typ: ai.CredentialAPIKey, in: ai.AuthInteraction{
			Prompt: func(context.Context, ai.AuthPrompt) (string, error) { return "", cancelled },
		}, want: cancelled.Error()},
		{name: "wrong credential type", provider: registryTestProvider, typ: ai.CredentialOAuth, want: "returned no oauth credential"},
	} {
		t.Run(test.name, func(t *testing.T) {
			if err := registry.Login(ctx, test.provider, test.typ, test.in); err == nil || !strings.Contains(err.Error(), test.want) {
				t.Errorf("Login error = %v, want %q", err, test.want)
			}
		})
	}
	if n := credentials.writes.Load(); n != 0 {
		t.Errorf("failed logins wrote the store %d times", n)
	}

	credentials.err = errors.New("database down")
	var authErr *ai.AuthError
	if err := registry.Login(ctx, ai.ProviderOpenAI, ai.CredentialAPIKey, enterKey("sk")); !errors.As(err, &authErr) || authErr.Code != ai.AuthErrorStore {
		t.Errorf("Login with a failing store = %v, want an *ai.AuthError with code %q", err, ai.AuthErrorStore)
	}
	if err := registry.Logout(ctx, ai.ProviderOpenAI); !errors.As(err, &authErr) || authErr.Code != ai.AuthErrorStore {
		t.Errorf("Logout with a failing store = %v, want an *ai.AuthError with code %q", err, ai.AuthErrorStore)
	}
}

func TestCreateAgentSessionAuthOptions(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "sk-env")
	ctx := context.Background()

	_, err := baseAgent.CreateAgentSession(ctx, baseAgent.CreateAgentSessionOptions{})
	if err == nil || !strings.Contains(err.Error(), "no models available") {
		t.Errorf("CreateAgentSession without AmbientAuth = %v, want no models available", err)
	}

	result, err := baseAgent.CreateAgentSession(ctx, baseAgent.CreateAgentSessionOptions{AmbientAuth: ai.DefaultAuthContext()})
	if err != nil {
		t.Fatalf("CreateAgentSession with AmbientAuth: %v", err)
	}
	model := result.Session.Agent.State().Model
	if model.Provider != ai.ProviderOpenAI {
		t.Errorf("model = %s/%s, want an OpenAI model", model.Provider, model.ID)
	}
	if auth, err := result.Session.Agent.ResolveAuth(ctx, model); err != nil || authKey(auth) != "sk-env" {
		t.Errorf("ResolveAuth = %s, %v; want sk-env", authKey(auth), err)
	}

	credentials := ai.NewInMemoryCredentialStore()
	storeCredential(t, credentials, ai.ProviderOpenAI, &ai.Credential{Type: ai.CredentialAPIKey, Key: "sk-stored"})
	result, err = baseAgent.CreateAgentSession(ctx, baseAgent.CreateAgentSessionOptions{Credentials: credentials})
	if err != nil {
		t.Fatalf("CreateAgentSession with Credentials: %v", err)
	}
	if auth, err := result.Session.Agent.ResolveAuth(ctx, result.Session.Agent.State().Model); err != nil || authKey(auth) != "sk-stored" {
		t.Errorf("ResolveAuth = %s, %v; want sk-stored", authKey(auth), err)
	}

	_, err = baseAgent.CreateAgentSession(ctx, baseAgent.CreateAgentSessionOptions{
		ModelRegistry: baseAgent.NewModelRegistry(nil),
		AmbientAuth:   ai.DefaultAuthContext(),
	})
	if err == nil || !strings.Contains(err.Error(), "not both") {
		t.Errorf("CreateAgentSession with a registry and AmbientAuth = %v, want an error", err)
	}
}

// countingCredentialStore counts reads and writes, and fails every call with
// err when it is set.
type countingCredentialStore struct {
	ai.CredentialStore
	err           error
	reads, writes atomic.Int32
}

func (s *countingCredentialStore) Read(ctx context.Context, provider ai.Provider) (*ai.Credential, error) {
	s.reads.Add(1)
	if s.err != nil {
		return nil, s.err
	}
	return s.CredentialStore.Read(ctx, provider)
}

func (s *countingCredentialStore) Modify(ctx context.Context, provider ai.Provider, fn func(*ai.Credential) (*ai.Credential, error)) (*ai.Credential, error) {
	s.writes.Add(1)
	if s.err != nil {
		return nil, s.err
	}
	return s.CredentialStore.Modify(ctx, provider, fn)
}

func (s *countingCredentialStore) Delete(ctx context.Context, provider ai.Provider) error {
	if s.err != nil {
		return s.err
	}
	return s.CredentialStore.Delete(ctx, provider)
}
