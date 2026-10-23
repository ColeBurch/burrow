package authstoretest

import (
	"testing"

	"github.com/ColeBurch/burrow/baseAgent"
)

// Both stores of a subtest must share state, as two processes over one
// database do, so every call returns the same MemoryStore.
func TestMemoryStore(t *testing.T) {
	store := NewMemoryStore()
	Run(t, func(*testing.T) baseAgent.AuthStore { return store })
}
