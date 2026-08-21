package baseAgent

import (
	"context"
	"errors"
	"testing"

	"github.com/ColeBurch/burrow/agent"
	"github.com/ColeBurch/burrow/ai"
)

type appendErrorSessionStore struct {
	appendErr error
}

func (*appendErrorSessionStore) CreateSession(context.Context, *SessionHeader) error {
	return nil
}

func (*appendErrorSessionStore) LoadSession(context.Context, string) (*SessionHeader, []SessionEntry, error) {
	panic("unexpected LoadSession call")
}

func (s *appendErrorSessionStore) AppendEntry(context.Context, *SessionHeader, SessionEntry) error {
	return s.appendErr
}

func (*appendErrorSessionStore) ListSessions(context.Context, map[string]any) ([]SessionInfo, error) {
	panic("unexpected ListSessions call")
}

func (*appendErrorSessionStore) FindMostRecentSession(context.Context, map[string]any) (*SessionInfo, error) {
	panic("unexpected FindMostRecentSession call")
}

func (*appendErrorSessionStore) CreateBranch(context.Context, string, SessionHeader) error {
	panic("unexpected CreateBranch call")
}

func TestAgentSessionPromptReturnsSessionStoreError(t *testing.T) {
	ctx := context.Background()
	appendErr := errors.New("session store append failed")
	store := &appendErrorSessionStore{appendErr: appendErr}

	sessionManager, err := NewSessionManager(ctx, nil, nil, true, store, nil)
	if err != nil {
		t.Fatalf("NewSessionManager() error = %v", err)
	}

	model := ai.Model[ai.API]{ID: "test-model", Name: "test-model"}
	a := agent.NewAgent(&agent.AgentOptions{
		InitialState: &agent.InitialAgentState{Model: &model},
	})
	session := NewAgentSession(a, sessionManager)

	err = session.Prompt(ctx, "hello", nil)
	if !errors.Is(err, appendErr) {
		t.Fatalf("Prompt() error = %v, want %v", err, appendErr)
	}
}
