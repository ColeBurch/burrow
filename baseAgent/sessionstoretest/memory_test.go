package sessionstoretest

import (
	"testing"

	"github.com/ColeBurch/burrow/baseAgent"
)

func TestMemoryStore(t *testing.T) {
	Run(t, func(*testing.T) baseAgent.SessionStore { return NewMemoryStore() })
}
