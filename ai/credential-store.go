// Ported from pi's packages/ai/src/auth/credential-store.ts (MIT, see LICENSE).

package ai

import (
	"context"
	"slices"
	"strings"
	"sync"
)

// InMemoryCredentialStore is a CredentialStore that keeps credentials in
// memory, for tests and apps that do not persist auth. Modify and Delete are
// serialized per provider, within this process only.
type InMemoryCredentialStore struct {
	mu          sync.Mutex
	credentials map[Provider]*Credential
	// locks holds one lock per provider: a send takes it, a receive releases
	// it. Unlike a sync.Mutex, waiting for it can stop when ctx is done.
	locks map[Provider]chan struct{}
}

var _ CredentialStore = (*InMemoryCredentialStore)(nil)

// NewInMemoryCredentialStore returns an empty InMemoryCredentialStore.
func NewInMemoryCredentialStore() *InMemoryCredentialStore {
	return &InMemoryCredentialStore{
		credentials: map[Provider]*Credential{},
		locks:       map[Provider]chan struct{}{},
	}
}

func (s *InMemoryCredentialStore) Read(ctx context.Context, provider Provider) (*Credential, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.credentials[provider].Clone(), nil
}

func (s *InMemoryCredentialStore) List(ctx context.Context) ([]CredentialInfo, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	s.mu.Lock()
	infos := make([]CredentialInfo, 0, len(s.credentials))
	for provider, credential := range s.credentials {
		infos = append(infos, CredentialInfo{Provider: provider, Type: credential.Type})
	}
	s.mu.Unlock()
	slices.SortFunc(infos, func(a, b CredentialInfo) int { return strings.Compare(string(a.Provider), string(b.Provider)) })
	return infos, nil
}

func (s *InMemoryCredentialStore) Modify(ctx context.Context, provider Provider, fn func(current *Credential) (*Credential, error)) (*Credential, error) {
	unlock, err := s.lock(ctx, provider)
	if err != nil {
		return nil, err
	}
	defer unlock()

	s.mu.Lock()
	current := s.credentials[provider].Clone()
	s.mu.Unlock()

	next, err := fn(current.Clone())
	if err != nil {
		return nil, err
	}
	// Store nothing once ctx is done, as a database transaction rolls back.
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if next == nil {
		return current, nil
	}

	s.mu.Lock()
	s.credentials[provider] = next.Clone()
	s.mu.Unlock()
	return next.Clone(), nil
}

func (s *InMemoryCredentialStore) Delete(ctx context.Context, provider Provider) error {
	unlock, err := s.lock(ctx, provider)
	if err != nil {
		return err
	}
	defer unlock()

	s.mu.Lock()
	delete(s.credentials, provider)
	s.mu.Unlock()
	return nil
}

// lock takes provider's lock, or returns ctx.Err() if ctx is done first.
func (s *InMemoryCredentialStore) lock(ctx context.Context, provider Provider) (unlock func(), err error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	s.mu.Lock()
	l := s.locks[provider]
	if l == nil {
		l = make(chan struct{}, 1)
		s.locks[provider] = l
	}
	s.mu.Unlock()

	select {
	case l <- struct{}{}:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	// select picks at random when both are ready; never run fn after ctx is done.
	if err := ctx.Err(); err != nil {
		<-l
		return nil, err
	}
	return func() { <-l }, nil
}
