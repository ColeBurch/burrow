package baseAgent

import (
	"context"
	"errors"

	"github.com/ColeBurch/burrow/agent"
	"github.com/ColeBurch/burrow/ai"
)

const DefaultThinkingLevel = ai.ThinkingLevelMedium

type CreateAgentSessionOptions struct {
	Model          ai.Model[ai.API]
	ThinkingLevel  ai.ModelThinkingLevel
	SessionManager *SessionManager
	ModelRegistry  *ModelRegistry
	AuthStore      AuthStore
}

type CreateAgentSessionResult struct {
	Session              *AgentSession
	ModelFallbackMessage *string
}

func FindInitialModel(registry *ModelRegistry) (ai.Model[ai.API], bool) {
	if registry == nil {
		registry = NewModelRegistry(nil)
	}

	available := registry.Available()
	if len(available) == 0 {
		return ai.Model[ai.API]{}, false
	}
	return available[0], true
}

// CreateAgentSession creates a minimal, generic agent session with no tools.
func CreateAgentSession(ctx context.Context, options CreateAgentSessionOptions) (*CreateAgentSessionResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}

	ensureDefaultAPIProvidersRegistered()

	registry := options.ModelRegistry
	if registry == nil {
		registry = NewModelRegistry(options.AuthStore)
	}

	sessionManager := options.SessionManager
	if sessionManager == nil {
		var err error
		sessionManager, err = InMemorySessionManager(ctx)
		if err != nil {
			return nil, err
		}
	}

	existingSession := sessionManager.BuildSessionContext()
	hasExistingSession := len(existingSession.Messages) > 0

	model := options.Model
	var modelFallbackMessage *string
	if model.ID == "" && hasExistingSession && existingSession.Model != nil {
		if restored, ok := registry.Find(ai.Provider(existingSession.Model.Provider), existingSession.Model.ModelID); ok && registry.HasConfiguredAuth(restored) {
			model = restored
		} else {
			msg := "could not restore model " + existingSession.Model.Provider + "/" + existingSession.Model.ModelID
			modelFallbackMessage = &msg
		}
	}

	if model.ID == "" {
		initial, ok := FindInitialModel(registry)
		if !ok {
			return nil, errors.New("no models available; set OPENAI_API_KEY or configure a model provider")
		}
		model = initial
		if modelFallbackMessage != nil {
			msg := *modelFallbackMessage + ". Using " + string(model.Provider) + "/" + model.ID
			modelFallbackMessage = &msg
		}
	}

	thinkingLevel := options.ThinkingLevel
	if thinkingLevel == "" {
		if hasExistingSession && existingSession.ThinkingLevel != "" {
			thinkingLevel = ai.ModelThinkingLevel(existingSession.ThinkingLevel)
		} else {
			thinkingLevel = DefaultThinkingLevel
		}
	}
	thinkingLevel = ai.ClampThinkingLevel(model, thinkingLevel)

	sessionID := sessionManager.GetSessionID()
	a := agent.NewAgent(&agent.AgentOptions{
		InitialState: &agent.InitialAgentState{
			SystemPrompt:  stringPtr(""),
			Model:         &model,
			ThinkingLevel: &thinkingLevel,
			Tools:         []agent.AgentTool[any, any]{},
			Messages:      existingSession.Messages,
		},
		ConvertToLLM: ConvertToLLM,
		GetAPIKey: func(provider string) *string {
			key, ok := registry.APIKeyForProvider(ai.Provider(provider))
			if !ok {
				return nil
			}
			return &key
		},
		SessionID: &sessionID,
	})

	if !hasExistingSession {
		if _, err := sessionManager.AppendModelChange(ctx, string(model.Provider), model.ID); err != nil {
			return nil, err
		}
		if _, err := sessionManager.AppendThinkingLevelChange(ctx, string(thinkingLevel)); err != nil {
			return nil, err
		}
	}

	return &CreateAgentSessionResult{
		Session:              NewAgentSession(a, sessionManager),
		ModelFallbackMessage: modelFallbackMessage,
	}, nil
}

func stringPtr(value string) *string { return &value }

func ensureDefaultAPIProvidersRegistered() {
	if _, ok := ai.GetApiProvider(ai.APIOpenAIResponses); ok {
		return
	}

	ai.RegisterApiProvider(ai.ApiProvider[ai.API, ai.StreamOptions]{
		Api:          ai.APIOpenAIResponses,
		StreamSimple: ai.StreamSimpleOpenAIResponses,
	}, nil)
}
