package ai

import (
	"errors"
	"sync"
)

type SessionResourceCleanup func(SessionID *string) error

var (
	mu              sync.Mutex
	nextCleanupID   uint64
	sessionCleanups = make(map[uint64]SessionResourceCleanup)
)

func RegisterSessionResourceCleanup(cleanup SessionResourceCleanup) func() {
	mu.Lock()
	id := nextCleanupID
	nextCleanupID++
	sessionCleanups[id] = cleanup
	mu.Unlock()

	return func() {
		mu.Lock()
		delete(sessionCleanups, id)
		mu.Unlock()
	}
}

func CleanupSessionResources(sessionID *string) error {
	mu.Lock()
	cleanups := make([]SessionResourceCleanup, 0, len(sessionCleanups))
	for _, cleanup := range sessionCleanups {
		cleanups = append(cleanups, cleanup)
	}
	mu.Unlock()

	var errs []error
	for _, cleanup := range cleanups {
		if err := cleanup(sessionID); err != nil {
			errs = append(errs, err)
		}
	}

	return errors.Join(errs...)
}
