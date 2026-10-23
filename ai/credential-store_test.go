package ai

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"sync"
	"testing"
	"time"
)

func TestInMemoryCredentialStoreReadEmpty(t *testing.T) {
	c, err := NewInMemoryCredentialStore().Read(context.Background(), testProvider)
	if c != nil || err != nil {
		t.Fatalf("got %+v, %v; want nil, nil", c, err)
	}
}

func TestInMemoryCredentialStoreModify(t *testing.T) {
	ctx := context.Background()
	s := NewInMemoryCredentialStore()
	want := &Credential{Type: CredentialAPIKey, Key: "k"}

	var current *Credential
	post, err := s.Modify(ctx, testProvider, func(c *Credential) (*Credential, error) {
		current = c
		return want, nil
	})
	if err != nil || current != nil || !reflect.DeepEqual(post, want) {
		t.Fatalf("first Modify: current %+v, post %+v, %v", current, post, err)
	}

	post, err = s.Modify(ctx, testProvider, func(c *Credential) (*Credential, error) {
		current = c
		return nil, nil
	})
	if err != nil || !reflect.DeepEqual(current, want) || !reflect.DeepEqual(post, want) {
		t.Fatalf("nil from fn: current %+v, post %+v, %v; want unchanged", current, post, err)
	}
	if got := stored(t, s); !reflect.DeepEqual(got, want) {
		t.Fatalf("Read = %+v, want %+v", got, want)
	}
}

func TestInMemoryCredentialStoreModifyError(t *testing.T) {
	s := storeWith(t, &Credential{Type: CredentialAPIKey, Key: "old"})
	boom := &AuthError{Code: AuthErrorOAuth, Message: "boom"}
	_, err := s.Modify(context.Background(), testProvider, func(*Credential) (*Credential, error) {
		return &Credential{Type: CredentialAPIKey, Key: "new"}, boom
	})
	var authErr *AuthError
	if !errors.As(err, &authErr) || authErr != boom {
		t.Fatalf("err = %v, want fn's error", err)
	}
	if stored(t, s).Key != "old" {
		t.Fatal("failed Modify stored a credential")
	}
}

func TestInMemoryCredentialStoreCancelledBeforeLock(t *testing.T) {
	s := storeWith(t, &Credential{Type: CredentialAPIKey, Key: "old"})
	ran := false
	fn := func(*Credential) (*Credential, error) {
		ran = true
		return &Credential{Type: CredentialAPIKey, Key: "new"}, nil
	}

	// Done before the call, with the lock free.
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := s.Modify(ctx, testProvider, fn); !errors.Is(err, context.Canceled) || ran {
		t.Fatalf("err = %v, ran %v; want canceled without running fn", err, ran)
	}
	if err := s.Delete(ctx, testProvider); !errors.Is(err, context.Canceled) {
		t.Fatalf("Delete err = %v, want canceled", err)
	}

	// Done while waiting for the lock.
	release := holdLock(t, s)
	defer release()
	ctx, cancel = context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	if _, err := s.Modify(ctx, testProvider, fn); !errors.Is(err, context.DeadlineExceeded) || ran {
		t.Fatalf("err = %v, ran %v; want deadline exceeded without running fn", err, ran)
	}
	if stored(t, s).Key != "old" {
		t.Fatal("credential changed")
	}
}

func TestInMemoryCredentialStoreCancelledDuringFn(t *testing.T) {
	s := storeWith(t, &Credential{Type: CredentialAPIKey, Key: "old"})
	ctx, cancel := context.WithCancel(context.Background())
	_, err := s.Modify(ctx, testProvider, func(*Credential) (*Credential, error) {
		cancel()
		return &Credential{Type: CredentialAPIKey, Key: "new"}, nil
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want canceled", err)
	}
	if stored(t, s).Key != "old" {
		t.Fatal("Modify stored after ctx was done")
	}
}

func TestInMemoryCredentialStoreNoAliasing(t *testing.T) {
	ctx := context.Background()
	s := NewInMemoryCredentialStore()
	in := &Credential{Type: CredentialOAuth, OAuth: &OAuthToken{Access: "a", Extra: map[string]json.RawMessage{"x": json.RawMessage(`1`)}}}
	post, err := s.Modify(ctx, testProvider, func(*Credential) (*Credential, error) { return in, nil })
	if err != nil {
		t.Fatal(err)
	}
	in.OAuth.Access = "changed input"
	post.OAuth.Access = "changed result"
	read := stored(t, s)
	read.OAuth.Extra["x"][0] = '2'
	s.Modify(ctx, testProvider, func(c *Credential) (*Credential, error) {
		c.OAuth.Access = "changed current"
		return nil, nil
	})

	if c := stored(t, s); c.OAuth.Access != "a" || string(c.OAuth.Extra["x"]) != "1" {
		t.Fatalf("stored = %+v; the store shares memory with its callers", c.OAuth)
	}
}

func TestInMemoryCredentialStoreList(t *testing.T) {
	ctx := context.Background()
	s := NewInMemoryCredentialStore()
	for _, p := range []Provider{"zeta", "alpha", "mid"} {
		typ := CredentialAPIKey
		if p == "mid" {
			typ = CredentialOAuth
		}
		s.Modify(ctx, p, func(*Credential) (*Credential, error) { return &Credential{Type: typ, Key: "secret"}, nil })
	}
	got, err := s.List(ctx)
	want := []CredentialInfo{{"alpha", CredentialAPIKey}, {"mid", CredentialOAuth}, {"zeta", CredentialAPIKey}}
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("List = %+v, %v; want %+v", got, err, want)
	}
}

func TestInMemoryCredentialStoreDelete(t *testing.T) {
	ctx := context.Background()
	s := storeWith(t, &Credential{Type: CredentialAPIKey, Key: "k"})

	// Delete waits for a Modify in progress.
	release := holdLock(t, s)
	deleted := make(chan error, 1)
	go func() { deleted <- s.Delete(ctx, testProvider) }()
	select {
	case <-deleted:
		t.Fatal("Delete did not wait for Modify")
	case <-time.After(20 * time.Millisecond):
	}
	release()
	if err := <-deleted; err != nil {
		t.Fatal(err)
	}
	if c := stored(t, s); c != nil {
		t.Fatalf("Read after Delete = %+v", c)
	}
	if err := s.Delete(ctx, testProvider); err != nil {
		t.Fatalf("deleting nothing: %v", err)
	}
}

func TestInMemoryCredentialStoreSerializesModify(t *testing.T) {
	ctx := context.Background()
	s := NewInMemoryCredentialStore()
	var wg sync.WaitGroup
	for range 50 {
		wg.Go(func() {
			s.Modify(ctx, testProvider, func(c *Credential) (*Credential, error) {
				key := "x"
				if c != nil {
					key = c.Key + "x"
				}
				return &Credential{Type: CredentialAPIKey, Key: key}, nil
			})
		})
	}
	wg.Wait()
	if n := len(stored(t, s).Key); n != 50 {
		t.Fatalf("key length %d, want 50; concurrent Modify calls lost updates", n)
	}
}

func TestInMemoryCredentialStoreProvidersLockSeparately(t *testing.T) {
	s := storeWith(t, nil)
	release := holdLock(t, s)
	defer release()
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if _, err := s.Modify(ctx, "other", func(*Credential) (*Credential, error) { return nil, nil }); err != nil {
		t.Fatalf("Modify on another provider: %v", err)
	}
}
