// Adapted from pi's packages/coding-agent/src/core/auth-storage.ts (MIT, see LICENSE).

package baseAgent

import (
	"cmp"
	"context"
	"fmt"
	"slices"

	"github.com/ColeBurch/burrow/ai"
)

// AuthStore persists secrets, such as provider credentials and MCP OAuth
// state, for many owners. baseAgent ships no implementation, so any database
// can back it. authstoretest.Run checks an implementation against the rules
// below.
//
// Keys and values:
//   - A value is keyed by (ownerID, namespace, name). Owners never see each
//     other's values. "" is a valid owner ID, for single-user apps.
//   - namespace separates kinds of secret, such as "provider" and "mcp". The
//     code that owns a namespace encodes its values; ScopeAuthStore uses
//     MarshalCredential for "provider". Stores never look inside values.
//   - Names are opaque: compare them byte for byte, without parsing, trimming
//     or case folding. Names are at most 1024 bytes; index a hash of the name
//     if your database cannot index that.
//   - Get returns nil, nil when nothing is stored. Stored values are never
//     empty; nil and empty both mean nothing.
//   - List returns the names stored for ownerID in namespace, in any order,
//     without their values.
//   - Stores must not keep the slices passed to them, nor return slices they
//     keep; copy them with bytes.Clone.
//   - Encrypt values at rest, binding each ciphertext to its owner, namespace
//     and name (for example as AES-GCM additional data), so a value copied
//     into another row fails to decrypt.
//
// Locking:
//   - Modify runs fn once, holding a lock on (ownerID, namespace, name) that
//     every store instance over the same database respects, including ones in
//     other processes. A store may lock more than that key, never less.
//   - fn may make network calls; hold the lock across them.
//   - fn returning nil or empty stores nothing, and Modify returns the
//     current value.
//   - An error from fn stores nothing and is returned, wrapped or not, so
//     errors.As still finds it.
//   - If ctx is done before the lock is taken, return ctx.Err() without
//     running fn.
//   - Delete takes the same lock. Deleting nothing succeeds.
//   - fn never calls the store, so stores need not be reentrant.
type AuthStore interface {
	Get(ctx context.Context, ownerID, namespace, name string) ([]byte, error)
	List(ctx context.Context, ownerID, namespace string) ([]string, error) // names, any order
	Modify(ctx context.Context, ownerID, namespace, name string, fn func(current []byte) ([]byte, error)) ([]byte, error)
	Delete(ctx context.Context, ownerID, namespace, name string) error
}

// ScopeAuthStore returns ownerID's provider credentials in store as an
// ai.CredentialStore, as SessionManager binds one session. Credentials live in
// CredentialNamespace, named by provider ID and encoded with
// MarshalCredential.
//
// A stored value that does not decode fails Read, List and Modify, including
// the Modify of a new login. Delete still works, so logging out and in again
// replaces it.
func ScopeAuthStore(store AuthStore, ownerID string) ai.CredentialStore {
	return scopedAuthStore{store: store, ownerID: ownerID}
}

type scopedAuthStore struct {
	store   AuthStore
	ownerID string
}

var _ ai.CredentialStore = scopedAuthStore{}

func (s scopedAuthStore) Read(ctx context.Context, provider ai.Provider) (*ai.Credential, error) {
	data, err := s.store.Get(ctx, s.ownerID, CredentialNamespace, string(provider))
	if err != nil {
		return nil, err
	}
	return decodeStoredCredential(provider, data)
}

func (s scopedAuthStore) List(ctx context.Context) ([]ai.CredentialInfo, error) {
	names, err := s.store.List(ctx, s.ownerID, CredentialNamespace)
	if err != nil {
		return nil, err
	}
	// The type is inside the value, so read each credential. One deleted
	// since the list is skipped.
	infos := make([]ai.CredentialInfo, 0, len(names))
	for _, name := range names {
		credential, err := s.Read(ctx, ai.Provider(name))
		if err != nil {
			return nil, err
		}
		if credential != nil {
			infos = append(infos, ai.CredentialInfo{Provider: ai.Provider(name), Type: credential.Type})
		}
	}
	slices.SortFunc(infos, func(a, b ai.CredentialInfo) int { return cmp.Compare(a.Provider, b.Provider) })
	return infos, nil
}

func (s scopedAuthStore) Modify(ctx context.Context, provider ai.Provider, fn func(current *ai.Credential) (*ai.Credential, error)) (*ai.Credential, error) {
	data, err := s.store.Modify(ctx, s.ownerID, CredentialNamespace, string(provider), func(current []byte) ([]byte, error) {
		credential, err := decodeStoredCredential(provider, current)
		if err != nil {
			return nil, err
		}
		next, err := fn(credential)
		if err != nil || next == nil {
			// Returned as is, so errors.As finds an *ai.AuthError from fn.
			return nil, err
		}
		return MarshalCredential(next)
	})
	if err != nil {
		return nil, err
	}
	return decodeStoredCredential(provider, data)
}

func (s scopedAuthStore) Delete(ctx context.Context, provider ai.Provider) error {
	return s.store.Delete(ctx, s.ownerID, CredentialNamespace, string(provider))
}

func decodeStoredCredential(provider ai.Provider, data []byte) (*ai.Credential, error) {
	credential, err := UnmarshalCredential(data)
	if err != nil {
		return nil, fmt.Errorf("stored credential for %s: %w", provider, err)
	}
	return credential, nil
}
