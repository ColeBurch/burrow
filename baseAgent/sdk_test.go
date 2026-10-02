package baseAgent

import (
	"context"
	"errors"
	"testing"

	"github.com/ColeBurch/burrow/ai"
)

func callProviderWithoutPanic(call func() error) (err error, panicValue any) {
	defer func() {
		panicValue = recover()
	}()

	err = call()
	return
}

func prepareDefaultProviderTest(t *testing.T) ai.Model[ai.API] {
	t.Helper()

	ai.ClearApiProviders()
	t.Cleanup(ai.ClearApiProviders)
	ensureDefaultAPIProvidersRegistered()

	return ai.Model[ai.API]{
		ID:       "test-model",
		API:      ai.APIOpenAIResponses,
		Provider: ai.ProviderOpenAI,
	}
}

func streamOptionsWithPayloadError(payloadErr error) *ai.StreamOptions {
	apiKey := "test-api-key"
	return &ai.StreamOptions{
		ApiKey: &apiKey,
		OnPayload: func(any, ai.Model[ai.API]) (any, error) {
			return nil, payloadErr
		},
	}
}

func TestDefaultAPIProviderStreamReturnsErrorWithoutPanic(t *testing.T) {
	model := prepareDefaultProviderTest(t)
	payloadErr := errors.New("stop before request")

	err, panicValue := callProviderWithoutPanic(func() error {
		_, err := ai.Stream(model, ai.ModelContext{}, streamOptionsWithPayloadError(payloadErr))
		return err
	})
	if panicValue != nil {
		t.Fatalf("ai.Stream() panicked: %v", panicValue)
	}
	if !errors.Is(err, payloadErr) {
		t.Fatalf("ai.Stream() error = %v, want %v", err, payloadErr)
	}
}

func TestDefaultAPIProviderCompleteReturnsErrorWithoutPanic(t *testing.T) {
	model := prepareDefaultProviderTest(t)
	payloadErr := errors.New("stop before request")

	err, panicValue := callProviderWithoutPanic(func() error {
		_, err := ai.Complete(model, ai.ModelContext{}, streamOptionsWithPayloadError(payloadErr))
		return err
	})
	if panicValue != nil {
		t.Fatalf("ai.Complete() panicked: %v", panicValue)
	}
	if !errors.Is(err, payloadErr) {
		t.Fatalf("ai.Complete() error = %v, want %v", err, payloadErr)
	}
}

func reasoningTestModel() ai.Model[ai.API] {
	return ai.Model[ai.API]{
		ID:        "test-model",
		API:       ai.APIOpenAIResponses,
		Provider:  ai.ProviderOpenAI,
		Reasoning: true,
	}
}

func createTestAgentSession(t *testing.T, sm *SessionManager) *AgentSession {
	t.Helper()

	result, err := CreateAgentSession(context.Background(), CreateAgentSessionOptions{
		Model:          reasoningTestModel(),
		SessionManager: sm,
	})
	if err != nil {
		t.Fatalf("CreateAgentSession() error = %v", err)
	}
	return result.Session
}

func TestCreateAgentSessionUsesDefaultThinkingLevelWithoutThinkingEntry(t *testing.T) {
	sm, err := InMemorySessionManager(context.Background())
	if err != nil {
		t.Fatalf("InMemorySessionManager() error = %v", err)
	}
	if _, err := sm.AppendMessage(context.Background(), testUserMessage("hello")); err != nil {
		t.Fatalf("AppendMessage() error = %v", err)
	}

	level, err := createTestAgentSession(t, sm).GetThinkingLevel()
	if err != nil {
		t.Fatalf("GetThinkingLevel() error = %v", err)
	}
	if level != DefaultThinkingLevel {
		t.Fatalf("thinking level = %q, want %q", level, DefaultThinkingLevel)
	}
}

func TestCreateAgentSessionRestoresThinkingLevelFromEntry(t *testing.T) {
	sm, err := InMemorySessionManager(context.Background())
	if err != nil {
		t.Fatalf("InMemorySessionManager() error = %v", err)
	}
	if _, err := sm.AppendThinkingLevelChange(context.Background(), string(ai.ThinkingLevelHigh)); err != nil {
		t.Fatalf("AppendThinkingLevelChange() error = %v", err)
	}
	if _, err := sm.AppendMessage(context.Background(), testUserMessage("hello")); err != nil {
		t.Fatalf("AppendMessage() error = %v", err)
	}

	level, err := createTestAgentSession(t, sm).GetThinkingLevel()
	if err != nil {
		t.Fatalf("GetThinkingLevel() error = %v", err)
	}
	if level != ai.ThinkingLevelHigh {
		t.Fatalf("thinking level = %q, want %q", level, ai.ThinkingLevelHigh)
	}
}

func TestCreateAgentSessionWritesModelAndThinkingLevelForNewSession(t *testing.T) {
	sm, err := InMemorySessionManager(context.Background())
	if err != nil {
		t.Fatalf("InMemorySessionManager() error = %v", err)
	}
	createTestAgentSession(t, sm)

	var model *ModelChangeEntry
	var thinking *ThinkingLevelChangeEntry
	for _, entry := range sm.GetBranch() {
		switch e := entry.(type) {
		case *ModelChangeEntry:
			model = e
		case *ThinkingLevelChangeEntry:
			thinking = e
		}
	}
	if model == nil || model.ModelID != "test-model" {
		t.Fatalf("model entry = %+v, want test-model", model)
	}
	if thinking == nil || thinking.ThinkingLevel != string(DefaultThinkingLevel) {
		t.Fatalf("thinking entry = %+v, want %q", thinking, DefaultThinkingLevel)
	}
}
