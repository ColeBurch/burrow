package baseAgent_test

import (
	"context"
	"errors"
	"testing"

	"github.com/ColeBurch/burrow/ai"
	"github.com/ColeBurch/burrow/baseAgent"
	"github.com/ColeBurch/burrow/baseAgent/authstoretest"
)

func TestScopeAuthStoreCorruptCredential(t *testing.T) {
	ctx := context.Background()
	store := authstoretest.NewMemoryStore()
	if _, err := store.Modify(ctx, "owner", baseAgent.CredentialNamespace, "openai", func([]byte) ([]byte, error) {
		return []byte("sk-written-without-the-codec"), nil
	}); err != nil {
		t.Fatal(err)
	}
	creds := baseAgent.ScopeAuthStore(store, "owner")
	login := func(*ai.Credential) (*ai.Credential, error) {
		return &ai.Credential{Type: ai.CredentialAPIKey, Key: "sk-new"}, nil
	}

	if _, err := creds.Read(ctx, "openai"); err == nil {
		t.Error("Read decoded a corrupt credential")
	}
	if _, err := creds.List(ctx); err == nil {
		t.Error("List decoded a corrupt credential")
	}
	if _, err := creds.Modify(ctx, "openai", login); err == nil {
		t.Error("Modify replaced a corrupt credential it could not decode")
	}

	// Logging out and in again replaces it.
	if err := creds.Delete(ctx, "openai"); err != nil {
		t.Fatal(err)
	}
	if got, err := creds.Modify(ctx, "openai", login); err != nil || got.Key != "sk-new" {
		t.Fatalf("login after logout = %+v, %v", got, err)
	}
}

func TestScopeAuthStoreModifyErrors(t *testing.T) {
	ctx := context.Background()
	creds := baseAgent.ScopeAuthStore(authstoretest.NewMemoryStore(), "owner")
	original := &ai.Credential{Type: ai.CredentialAPIKey, Key: "sk-original"}
	if _, err := creds.Modify(ctx, "openai", func(*ai.Credential) (*ai.Credential, error) { return original, nil }); err != nil {
		t.Fatal(err)
	}

	// RefreshStoredOAuthCredential tells a failed refresh from a store failure
	// with errors.As.
	refreshErr := &ai.AuthError{Code: ai.AuthErrorOAuth, Provider: "openai", Message: "refresh failed"}
	_, err := creds.Modify(ctx, "openai", func(*ai.Credential) (*ai.Credential, error) { return nil, refreshErr })
	var authErr *ai.AuthError
	if !errors.As(err, &authErr) || authErr != refreshErr {
		t.Errorf("Modify error = %v, want fn's *ai.AuthError", err)
	}

	// A credential the codec rejects stores nothing.
	_, err = creds.Modify(ctx, "openai", func(*ai.Credential) (*ai.Credential, error) {
		return &ai.Credential{Type: ai.CredentialOAuth}, nil
	})
	if err == nil {
		t.Error("Modify stored an OAuth credential without a token")
	}
	if got, err := creds.Read(ctx, "openai"); err != nil || got.Key != "sk-original" {
		t.Errorf("Read = %+v, %v; want the original credential", got, err)
	}
}

func TestScopeAuthStoreModifyUnchanged(t *testing.T) {
	ctx := context.Background()
	creds := baseAgent.ScopeAuthStore(authstoretest.NewMemoryStore(), "owner")
	unchanged := func(*ai.Credential) (*ai.Credential, error) { return nil, nil }

	if got, err := creds.Modify(ctx, "openai", unchanged); got != nil || err != nil {
		t.Fatalf("Modify of nothing = %+v, %v; want nil, nil", got, err)
	}
	stored := &ai.Credential{Type: ai.CredentialAPIKey, Key: "sk"}
	creds.Modify(ctx, "openai", func(*ai.Credential) (*ai.Credential, error) { return stored, nil })
	if got, err := creds.Modify(ctx, "openai", unchanged); err != nil || got.Key != "sk" {
		t.Fatalf("Modify returning nil = %+v, %v; want the current credential", got, err)
	}
}
