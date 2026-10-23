package authstoretest

import (
	"bytes"
	"context"
	"sync"

	"github.com/ColeBurch/burrow/baseAgent"
)

// MemoryStore is an in-memory baseAgent.AuthStore. It is a reference
// implementation of the contract and a store for tests and single-process
// apps. Values are not encrypted, and it keeps a small lock for every key it
// has seen.
type MemoryStore struct {
	mu     sync.Mutex
	values map[memoryKey][]byte
	// locks holds one lock per key: a send takes it, a receive releases it.
	// Unlike a sync.Mutex, waiting for it can stop when ctx is done.
	locks map[memoryKey]chan struct{}
}

type memoryKey struct{ ownerID, namespace, name string }

var _ baseAgent.AuthStore = (*MemoryStore)(nil)

// NewMemoryStore returns an empty MemoryStore.
func NewMemoryStore() *MemoryStore {
	return &MemoryStore{values: map[memoryKey][]byte{}, locks: map[memoryKey]chan struct{}{}}
}

func (s *MemoryStore) Get(ctx context.Context, ownerID, namespace, name string) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return bytes.Clone(s.values[memoryKey{ownerID, namespace, name}]), nil
}

func (s *MemoryStore) List(ctx context.Context, ownerID, namespace string) ([]string, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	var names []string
	for key := range s.values {
		if key.ownerID == ownerID && key.namespace == namespace {
			names = append(names, key.name)
		}
	}
	return names, nil
}

func (s *MemoryStore) Modify(ctx context.Context, ownerID, namespace, name string, fn func(current []byte) ([]byte, error)) ([]byte, error) {
	key := memoryKey{ownerID, namespace, name}
	unlock, err := s.lock(ctx, key)
	if err != nil {
		return nil, err
	}
	defer unlock()

	s.mu.Lock()
	current := s.values[key]
	s.mu.Unlock()

	next, err := fn(bytes.Clone(current))
	if err != nil {
		return nil, err
	}
	// Store nothing once ctx is done, as a database transaction rolls back.
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(next) == 0 {
		return bytes.Clone(current), nil
	}

	s.mu.Lock()
	s.values[key] = bytes.Clone(next)
	s.mu.Unlock()
	return bytes.Clone(next), nil
}

func (s *MemoryStore) Delete(ctx context.Context, ownerID, namespace, name string) error {
	key := memoryKey{ownerID, namespace, name}
	unlock, err := s.lock(ctx, key)
	if err != nil {
		return err
	}
	defer unlock()

	s.mu.Lock()
	delete(s.values, key)
	s.mu.Unlock()
	return nil
}

// lock takes key's lock, or returns ctx.Err() if ctx is done first.
func (s *MemoryStore) lock(ctx context.Context, key memoryKey) (unlock func(), err error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	s.mu.Lock()
	l := s.locks[key]
	if l == nil {
		l = make(chan struct{}, 1)
		s.locks[key] = l
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
