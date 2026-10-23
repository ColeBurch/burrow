package ai

import (
	"context"
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

const testProvider Provider = "test"

var envWithKey = mapAuthContext{"TEST_KEY": "from-env"}

type mapAuthContext map[string]string

func (m mapAuthContext) Env(name string) (string, bool) {
	value, ok := m[name]
	return value, ok && value != ""
}

func (mapAuthContext) FileExists(string) bool { return false }

// failingStore fails reads or writes, as a store does when its database is down.
type failingStore struct {
	CredentialStore
	readErr, modifyErr error
}

func (s failingStore) Read(ctx context.Context, provider Provider) (*Credential, error) {
	if s.readErr != nil {
		return nil, s.readErr
	}
	return s.CredentialStore.Read(ctx, provider)
}

func (s failingStore) Modify(ctx context.Context, provider Provider, fn func(*Credential) (*Credential, error)) (*Credential, error) {
	if s.modifyErr != nil {
		return nil, s.modifyErr
	}
	return s.CredentialStore.Modify(ctx, provider, fn)
}

func storeWith(t *testing.T, cred *Credential) *InMemoryCredentialStore {
	t.Helper()
	s := NewInMemoryCredentialStore()
	if cred != nil {
		if _, err := s.Modify(context.Background(), testProvider, func(*Credential) (*Credential, error) { return cred, nil }); err != nil {
			t.Fatal(err)
		}
	}
	return s
}

func stored(t *testing.T, s CredentialStore) *Credential {
	t.Helper()
	c, err := s.Read(context.Background(), testProvider)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

// holdLock takes the store's lock on testProvider until the returned func is
// called.
func holdLock(t *testing.T, s CredentialStore) (release func()) {
	t.Helper()
	held, done := make(chan struct{}), make(chan struct{})
	go s.Modify(context.Background(), testProvider, func(*Credential) (*Credential, error) {
		close(held)
		<-done
		return nil, nil
	})
	<-held
	return func() { close(done) }
}

func oauthCred(access string, expiresIn time.Duration) *Credential {
	return &Credential{Type: CredentialOAuth, OAuth: &OAuthToken{Access: access, Refresh: "r1", Expires: time.Now().Add(expiresIn)}}
}

func expired() *Credential { return oauthCred("old", -time.Minute) }

func needs(c *Credential) bool { return time.Now().After(c.OAuth.Expires) }

// rotatingOAuth rejects reuse of a refresh token, as rotating providers do.
func rotatingOAuth(calls *atomic.Int32, delay time.Duration) OAuthAuth {
	var mu sync.Mutex
	used := map[string]bool{}
	return OAuthAuth{Refresh: func(ctx context.Context, c *Credential) (*Credential, error) {
		calls.Add(1)
		mu.Lock()
		if used[c.OAuth.Refresh] {
			mu.Unlock()
			return nil, errors.New("invalid_grant: refresh token reused")
		}
		used[c.OAuth.Refresh] = true
		mu.Unlock()
		select {
		case <-time.After(delay):
		case <-ctx.Done():
			return nil, ctx.Err()
		}
		return &Credential{Type: CredentialOAuth, OAuth: &OAuthToken{Access: "new", Refresh: "r2", Expires: time.Now().Add(time.Hour)}}, nil
	}}
}

func oauthWith(calls *atomic.Int32, refreshTo time.Duration, refreshErr error) *OAuthAuth {
	return &OAuthAuth{
		Refresh: func(ctx context.Context, c *Credential) (*Credential, error) {
			calls.Add(1)
			if refreshErr != nil {
				return nil, refreshErr
			}
			return &Credential{Type: CredentialOAuth, OAuth: &OAuthToken{Access: "refreshed", Refresh: "r2", Expires: time.Now().Add(refreshTo)}}, nil
		},
		ToAuth: func(c *Credential) (ModelAuth, error) {
			access := c.OAuth.Access
			return ModelAuth{APIKey: &access}, nil
		},
	}
}

func testKeyAuth() *APIKeyAuth { return EnvAPIKeyAuth("Test key", "TEST_KEY") }

func apiKeyOf(t *testing.T, r *AuthResult) string {
	t.Helper()
	if r == nil || r.Auth.APIKey == nil {
		t.Fatalf("no auth: %+v", r)
	}
	return *r.Auth.APIKey
}

func ptr(s string) *string { return &s }

func TestRefreshStoredOAuthCredentialRefreshesOnce(t *testing.T) {
	s := storeWith(t, expired())
	var calls atomic.Int32
	oauth := rotatingOAuth(&calls, 20*time.Millisecond)
	var wg sync.WaitGroup
	errs := make(chan error, 10)
	for range 10 {
		wg.Go(func() {
			c, err := RefreshStoredOAuthCredential(context.Background(), s, testProvider, oauth, needs)
			if err != nil {
				errs <- err
				return
			}
			if c == nil || c.OAuth.Access != "new" {
				errs <- errors.New("did not get new token")
			}
		})
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatal(err)
	}
	if n := calls.Load(); n != 1 {
		t.Fatalf("Refresh called %d times, want 1", n)
	}
}

func TestRefreshStoredOAuthCredentialCancelBeforeLock(t *testing.T) {
	s := storeWith(t, expired())
	release := holdLock(t, s)
	var calls atomic.Int32
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	_, err := RefreshStoredOAuthCredential(ctx, s, testProvider, rotatingOAuth(&calls, 0), needs)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err = %v, want deadline exceeded", err)
	}
	release()
	time.Sleep(20 * time.Millisecond)
	if calls.Load() != 0 {
		t.Fatal("Refresh ran after the caller gave up waiting for the lock")
	}
	if stored(t, s).OAuth.Access != "old" {
		t.Fatal("credential changed")
	}
}

func TestRefreshStoredOAuthCredentialCancelDuringRefreshKeepsRotatedToken(t *testing.T) {
	s := storeWith(t, expired())
	var calls atomic.Int32
	ctx, cancel := context.WithCancel(context.Background())
	time.AfterFunc(20*time.Millisecond, cancel)
	start := time.Now()
	_, err := RefreshStoredOAuthCredential(ctx, s, testProvider, rotatingOAuth(&calls, 100*time.Millisecond), needs)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want canceled", err)
	}
	if d := time.Since(start); d > 70*time.Millisecond {
		t.Fatalf("returned after %v, want prompt return", d)
	}
	time.Sleep(200 * time.Millisecond)
	if c := stored(t, s); c.OAuth.Access != "new" || c.OAuth.Refresh != "r2" {
		t.Fatalf("rotated token not stored: %+v", c.OAuth)
	}
}

func TestRefreshStoredOAuthCredentialRefreshError(t *testing.T) {
	s := storeWith(t, expired())
	boom := errors.New("boom")
	oauth := OAuthAuth{Refresh: func(context.Context, *Credential) (*Credential, error) { return nil, boom }}
	_, err := RefreshStoredOAuthCredential(context.Background(), s, testProvider, oauth, needs)
	var authErr *AuthError
	if !errors.As(err, &authErr) || authErr.Code != AuthErrorOAuth || !errors.Is(err, boom) {
		t.Fatalf("err = %v, want oauth AuthError wrapping boom", err)
	}
	if stored(t, s).OAuth.Access != "old" {
		t.Fatal("credential changed after failed refresh")
	}
}

func TestRefreshStoredOAuthCredentialLoggedOut(t *testing.T) {
	var calls atomic.Int32
	c, err := RefreshStoredOAuthCredential(context.Background(), storeWith(t, nil), testProvider, rotatingOAuth(&calls, 0), needs)
	if c != nil || err != nil || calls.Load() != 0 {
		t.Fatalf("got %v, %v, %d calls; want nil, nil, 0", c, err, calls.Load())
	}
}

func TestRefreshStoredOAuthCredentialStoreError(t *testing.T) {
	s := failingStore{CredentialStore: storeWith(t, expired()), modifyErr: errors.New("db down")}
	var calls atomic.Int32
	_, err := RefreshStoredOAuthCredential(context.Background(), s, testProvider, rotatingOAuth(&calls, 0), needs)
	var authErr *AuthError
	if !errors.As(err, &authErr) || authErr.Code != AuthErrorStore {
		t.Fatalf("err = %v, want store AuthError", err)
	}
}

func TestResolveStoredAPIKeyWinsOverEnv(t *testing.T) {
	s := storeWith(t, &Credential{Type: CredentialAPIKey, Key: "stored"})
	r, err := ResolveProviderAuth(context.Background(), testProvider, ProviderAuth{APIKey: testKeyAuth()}, s, envWithKey, nil)
	if err != nil || apiKeyOf(t, r) != "stored" {
		t.Fatalf("got %v, %v", r, err)
	}
}

func TestResolveAmbientWhenNothingStored(t *testing.T) {
	r, err := ResolveProviderAuth(context.Background(), testProvider, ProviderAuth{APIKey: testKeyAuth()}, storeWith(t, nil), envWithKey, nil)
	if err != nil || apiKeyOf(t, r) != "from-env" || r.Source != "TEST_KEY" {
		t.Fatalf("got %+v, %v", r, err)
	}
}

func TestResolveNilEnvMeansNoAmbient(t *testing.T) {
	r, err := ResolveProviderAuth(context.Background(), testProvider, ProviderAuth{APIKey: testKeyAuth()}, storeWith(t, nil), nil, nil)
	if r != nil || err != nil {
		t.Fatalf("got %+v, %v; want nil, nil", r, err)
	}
}

func TestResolveNoEnvFallbackAfterFailedRefresh(t *testing.T) {
	s := storeWith(t, expired())
	var calls atomic.Int32
	auth := ProviderAuth{APIKey: testKeyAuth(), OAuth: oauthWith(&calls, time.Hour, errors.New("invalid_grant"))}
	r, err := ResolveProviderAuth(context.Background(), testProvider, auth, s, envWithKey, nil)
	var authErr *AuthError
	if r != nil || !errors.As(err, &authErr) || authErr.Code != AuthErrorOAuth {
		t.Fatalf("got %+v, %v; want oauth AuthError and no env key", r, err)
	}
}

func TestResolveStoredTypeWithoutHandlerIsUnconfigured(t *testing.T) {
	s := storeWith(t, oauthCred("tok", time.Hour))
	r, err := ResolveProviderAuth(context.Background(), testProvider, ProviderAuth{APIKey: testKeyAuth()}, s, envWithKey, nil)
	if r != nil || err != nil {
		t.Fatalf("got %+v, %v; want nil, nil (no env fallback)", r, err)
	}
}

func TestResolveOverrideKeyWins(t *testing.T) {
	s := storeWith(t, expired())
	var calls atomic.Int32
	auth := ProviderAuth{APIKey: testKeyAuth(), OAuth: oauthWith(&calls, time.Hour, nil)}
	r, err := ResolveProviderAuth(context.Background(), testProvider, auth, s, envWithKey, &AuthResolutionOverrides{APIKey: ptr("override")})
	if err != nil || apiKeyOf(t, r) != "override" || calls.Load() != 0 {
		t.Fatalf("got %v, %v, %d refreshes", r, err, calls.Load())
	}
}

func TestResolveEmptyOverrideKeyStillOverrides(t *testing.T) {
	s := storeWith(t, &Credential{Type: CredentialAPIKey, Key: "stored"})
	var seen *Credential
	auth := ProviderAuth{APIKey: &APIKeyAuth{Resolve: func(ctx context.Context, env AuthContext, cred *Credential) (*AuthResult, error) {
		seen = cred
		return nil, nil
	}}}
	_, err := ResolveProviderAuth(context.Background(), testProvider, auth, s, nil, &AuthResolutionOverrides{APIKey: ptr("")})
	if err != nil || seen == nil || seen.Key != "" {
		t.Fatalf("seen %+v, err %v; want the empty override, not the stored key", seen, err)
	}
}

func TestResolveOAuthRefreshWindow(t *testing.T) {
	tests := []struct {
		name      string
		expiresIn time.Duration
		want      string
		refreshes int32
	}{
		{"fresh", time.Hour, "tok", 0},
		{"within five minutes", 2 * time.Minute, "refreshed", 1},
		{"expired", -time.Minute, "refreshed", 1},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			s := storeWith(t, oauthCred("tok", tc.expiresIn))
			var calls atomic.Int32
			r, err := ResolveProviderAuth(context.Background(), testProvider, ProviderAuth{OAuth: oauthWith(&calls, time.Hour, nil)}, s, nil, nil)
			if err != nil || apiKeyOf(t, r) != tc.want || calls.Load() != tc.refreshes || r.Source != "OAuth" {
				t.Fatalf("got %+v, %v, %d refreshes", r, err, calls.Load())
			}
		})
	}
}

func TestResolveMinValidity(t *testing.T) {
	var calls atomic.Int32
	auth := ProviderAuth{OAuth: oauthWith(&calls, 10*time.Minute, nil)}
	s := storeWith(t, oauthCred("tok", 30*time.Minute))
	_, err := ResolveProviderAuth(context.Background(), testProvider, auth, s, nil, &AuthResolutionOverrides{MinOAuthValidity: time.Hour})
	if err == nil || !strings.Contains(err.Error(), "expires too soon") {
		t.Fatalf("err = %v, want expires too soon", err)
	}

	// Without the override, a 10-minute token is fine.
	s = storeWith(t, expired())
	r, err := ResolveProviderAuth(context.Background(), testProvider, auth, s, nil, nil)
	if err != nil || apiKeyOf(t, r) != "refreshed" {
		t.Fatalf("got %v, %v", r, err)
	}
}

func TestResolveEnvOverlay(t *testing.T) {
	auth := ProviderAuth{APIKey: testKeyAuth()}
	r, err := ResolveProviderAuth(context.Background(), testProvider, auth, storeWith(t, nil), envWithKey, &AuthResolutionOverrides{Env: map[string]string{"TEST_KEY": "overlay"}})
	if err != nil || apiKeyOf(t, r) != "overlay" {
		t.Fatalf("got %v, %v", r, err)
	}

	s := storeWith(t, &Credential{Type: CredentialAPIKey, Key: "stored", Env: map[string]string{"A": "stored", "B": "stored"}})
	r, err = ResolveProviderAuth(context.Background(), testProvider, auth, s, nil, &AuthResolutionOverrides{Env: map[string]string{"B": "override"}})
	if err != nil || r.Env["A"] != "stored" || r.Env["B"] != "override" {
		t.Fatalf("got %+v, %v", r, err)
	}
	if stored(t, s).Env["B"] != "stored" {
		t.Fatal("merging env modified the stored credential")
	}
}

func TestResolveStoreReadError(t *testing.T) {
	s := failingStore{CredentialStore: storeWith(t, nil), readErr: errors.New("db down")}
	_, err := ResolveProviderAuth(context.Background(), testProvider, ProviderAuth{APIKey: testKeyAuth()}, s, envWithKey, nil)
	var authErr *AuthError
	if !errors.As(err, &authErr) || authErr.Code != AuthErrorStore {
		t.Fatalf("err = %v, want store AuthError", err)
	}
}

func TestResolveCancelled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := ResolveProviderAuth(ctx, testProvider, ProviderAuth{APIKey: testKeyAuth()}, storeWith(t, nil), envWithKey, nil)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want canceled", err)
	}
}

func TestCheckProviderAuth(t *testing.T) {
	var calls atomic.Int32
	oauth := oauthWith(&calls, time.Hour, nil)
	tests := []struct {
		name   string
		stored *Credential
		auth   ProviderAuth
		env    AuthContext
		want   *AuthCheck
	}{
		{"expired OAuth counts without refreshing", expired(), ProviderAuth{APIKey: testKeyAuth(), OAuth: oauth}, envWithKey, &AuthCheck{Type: CredentialOAuth, Source: "OAuth"}},
		{"stored OAuth without handler", expired(), ProviderAuth{APIKey: testKeyAuth()}, envWithKey, nil},
		{"stored API key", &Credential{Type: CredentialAPIKey, Key: "k"}, ProviderAuth{APIKey: testKeyAuth()}, nil, &AuthCheck{Type: CredentialAPIKey, Source: "stored credential"}},
		{"stored API key without handler", &Credential{Type: CredentialAPIKey, Key: "k"}, ProviderAuth{OAuth: oauth}, envWithKey, nil},
		{"ambient", nil, ProviderAuth{APIKey: testKeyAuth()}, envWithKey, &AuthCheck{Type: CredentialAPIKey, Source: "TEST_KEY"}},
		{"nil env means no ambient", nil, ProviderAuth{APIKey: testKeyAuth()}, nil, nil},
		{"OAuth only and nothing stored", nil, ProviderAuth{OAuth: oauth}, envWithKey, nil},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := CheckProviderAuth(context.Background(), testProvider, tc.auth, storeWith(t, tc.stored), tc.env)
			if err != nil {
				t.Fatal(err)
			}
			if (got == nil) != (tc.want == nil) || (got != nil && *got != *tc.want) {
				t.Fatalf("got %+v, want %+v", got, tc.want)
			}
		})
	}
	if calls.Load() != 0 {
		t.Fatalf("check refreshed %d times", calls.Load())
	}
}

func TestCheckProviderAuthUsesCheck(t *testing.T) {
	var seen *Credential
	resolved := false
	auth := ProviderAuth{APIKey: &APIKeyAuth{
		Check: func(ctx context.Context, env AuthContext, cred *Credential) (*AuthCheck, error) {
			seen = cred
			return &AuthCheck{Type: CredentialAPIKey, Source: "profile"}, nil
		},
		Resolve: func(context.Context, AuthContext, *Credential) (*AuthResult, error) {
			resolved = true
			return nil, nil
		},
	}}
	s := storeWith(t, &Credential{Type: CredentialAPIKey, Key: "k"})
	got, err := CheckProviderAuth(context.Background(), testProvider, auth, s, nil)
	if err != nil || got == nil || got.Source != "profile" {
		t.Fatalf("got %+v, %v", got, err)
	}
	if resolved || seen == nil || seen.Key != "k" {
		t.Fatalf("resolved %v, Check saw %+v; want Check with the stored credential only", resolved, seen)
	}
}

func TestCheckProviderAuthErrors(t *testing.T) {
	boom := errors.New("boom")
	failingCheck := ProviderAuth{APIKey: &APIKeyAuth{Check: func(context.Context, AuthContext, *Credential) (*AuthCheck, error) { return nil, boom }}}
	tests := []struct {
		name  string
		store CredentialStore
		auth  ProviderAuth
	}{
		{"store read", failingStore{CredentialStore: storeWith(t, nil), readErr: boom}, ProviderAuth{APIKey: testKeyAuth()}},
		{"check", storeWith(t, nil), failingCheck},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := CheckProviderAuth(context.Background(), testProvider, tc.auth, tc.store, nil)
			var authErr *AuthError
			if !errors.As(err, &authErr) || authErr.Code != AuthErrorStore || !errors.Is(err, boom) {
				t.Fatalf("err = %v, want store AuthError wrapping boom", err)
			}
		})
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := CheckProviderAuth(ctx, testProvider, ProviderAuth{APIKey: testKeyAuth()}, storeWith(t, nil), nil); !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want canceled", err)
	}
}
