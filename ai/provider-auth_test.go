package ai

import (
	"context"
	"errors"
	"testing"
)

func TestEnvAPIKeyAuthResolve(t *testing.T) {
	t.Setenv("BURROW_TEST_FIRST", "")
	t.Setenv("BURROW_TEST_SECOND", "second")
	t.Setenv("BURROW_TEST_THIRD", "third")
	auth := EnvAPIKeyAuth("Test key", "BURROW_TEST_FIRST", "BURROW_TEST_SECOND", "BURROW_TEST_THIRD")
	env := DefaultAuthContext()
	ctx := context.Background()

	tests := []struct {
		name       string
		cred       *Credential
		env        AuthContext
		wantKey    string
		wantSource string
	}{
		{"stored key wins over env", &Credential{Type: CredentialAPIKey, Key: "stored"}, env, "stored", "stored credential"},
		{"first set env var in order", nil, env, "second", "BURROW_TEST_SECOND"},
		{"stored credential without key uses env", &Credential{Type: CredentialAPIKey}, env, "second", "BURROW_TEST_SECOND"},
		{"nothing configured", nil, NoAmbientAuth(), "", ""},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result, err := auth.Resolve(ctx, tc.env, tc.cred)
			if err != nil {
				t.Fatal(err)
			}
			if tc.wantKey == "" {
				if result != nil {
					t.Fatalf("result = %+v, want nil", result)
				}
				return
			}
			if result == nil || result.Auth.APIKey == nil || *result.Auth.APIKey != tc.wantKey || result.Source != tc.wantSource {
				t.Fatalf("result = %+v, want key %q from %q", result, tc.wantKey, tc.wantSource)
			}
		})
	}
}

func TestEnvAPIKeyAuthResolveDoesNotShareCredential(t *testing.T) {
	auth := EnvAPIKeyAuth("Test key")
	cred := &Credential{Type: CredentialAPIKey, Key: "stored", Env: map[string]string{"ACCOUNT": "a"}}
	result, err := auth.Resolve(context.Background(), NoAmbientAuth(), cred)
	if err != nil {
		t.Fatal(err)
	}
	*result.Auth.APIKey = "changed"
	result.Env["ACCOUNT"] = "changed"
	if cred.Key != "stored" || cred.Env["ACCOUNT"] != "a" {
		t.Fatalf("changing the result changed the credential: %+v", cred)
	}
}

func TestEnvAPIKeyAuthKeepsEnvVars(t *testing.T) {
	t.Setenv("BURROW_TEST_ORIGINAL", "original")
	envVars := []string{"BURROW_TEST_ORIGINAL"}
	auth := EnvAPIKeyAuth("Test key", envVars...)
	envVars[0] = "BURROW_TEST_CHANGED"
	result, err := auth.Resolve(context.Background(), DefaultAuthContext(), nil)
	if err != nil || result == nil || result.Source != "BURROW_TEST_ORIGINAL" {
		t.Fatalf("result = %+v, %v; changing the caller's slice changed the auth", result, err)
	}
}

func TestEnvAPIKeyAuthResolveCancelled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := EnvAPIKeyAuth("Test key").Resolve(ctx, NoAmbientAuth(), nil); !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want canceled", err)
	}
}

func TestEnvAPIKeyAuthLogin(t *testing.T) {
	auth := EnvAPIKeyAuth("Test key")
	var asked AuthPrompt
	in := AuthInteraction{Prompt: func(ctx context.Context, prompt AuthPrompt) (string, error) {
		asked = prompt
		return "  sk-test\n", nil
	}}

	cred, err := auth.Login(context.Background(), in)
	if err != nil {
		t.Fatal(err)
	}
	if asked.Type != AuthPromptSecret || asked.Message != "Enter Test key" {
		t.Errorf("prompt = %+v, want a secret prompt for the key", asked)
	}
	if cred.Type != CredentialAPIKey || cred.Key != "sk-test" {
		t.Errorf("credential = %+v, want trimmed api_key", cred)
	}
}

func TestEnvAPIKeyAuthLoginErrors(t *testing.T) {
	auth := EnvAPIKeyAuth("Test key")
	cancelled := errors.New("user cancelled")
	answer := func(value string, err error) AuthInteraction {
		return AuthInteraction{Prompt: func(context.Context, AuthPrompt) (string, error) { return value, err }}
	}

	if _, err := auth.Login(context.Background(), AuthInteraction{}); err == nil {
		t.Error("login without a prompt succeeded")
	}
	if _, err := auth.Login(context.Background(), answer("", cancelled)); !errors.Is(err, cancelled) {
		t.Errorf("err = %v, want the prompt's error", err)
	}
	if _, err := auth.Login(context.Background(), answer("   ", nil)); err == nil {
		t.Error("blank key accepted")
	}

	ctx, cancel := context.WithCancel(context.Background())
	in := AuthInteraction{Prompt: func(context.Context, AuthPrompt) (string, error) {
		cancel()
		return "sk-test", nil
	}}
	if _, err := auth.Login(ctx, in); !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want canceled after the prompt", err)
	}
}

func TestDefaultProviderAuth(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "sk-env")
	auths := DefaultProviderAuth()
	openai, ok := auths[ProviderOpenAI]
	if !ok || openai.APIKey == nil {
		t.Fatalf("openai auth = %+v, want API key auth", openai)
	}
	result, err := openai.APIKey.Resolve(context.Background(), DefaultAuthContext(), nil)
	if err != nil || result == nil || *result.Auth.APIKey != "sk-env" || result.Source != "OPENAI_API_KEY" {
		t.Fatalf("result = %+v, %v", result, err)
	}

	delete(auths, ProviderOpenAI)
	if _, ok := DefaultProviderAuth()[ProviderOpenAI]; !ok {
		t.Fatal("changing one result changed the next")
	}
}
