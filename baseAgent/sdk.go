// Adapted from pi's packages/coding-agent/src/core/sdk.ts (MIT, see LICENSE).

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
	// ModelRegistry finds models and resolves their auth. Nil means
	// NewModelRegistry with Credentials and AmbientAuth.
	ModelRegistry *ModelRegistry
	// Credentials and AmbientAuth configure the default registry; see
	// ModelRegistryOptions. Set them or ModelRegistry, not both. With neither,
	// nothing is configured until a login, so to use OPENAI_API_KEY set
	// AmbientAuth to ai.DefaultAuthContext().
	Credentials        ai.CredentialStore
	AmbientAuth        ai.AuthContext
	CompactionSettings *CompactionSettings
	// CompactionHook wraps manual and automatic compaction. See CompactionHook.
	CompactionHook CompactionHook

	SystemPrompt      *string
	Tools             []agent.AgentTool[any, any]
	ToolExecutionMode *agent.ToolExecutionMode

	/* Optional func to call before each tool call is executed. */
	BeforeToolCall func(
		ctx context.Context,
		call agent.BeforeToolCallContext,
	) (*agent.BeforeToolCallResult, error)

	/* Optional func to call after each tool call is executed. */
	AfterToolCall func(
		ctx context.Context,
		call agent.AfterToolCallContext[any],
	) (*agent.AfterToolCallResult, error)

	/* Optional func to call before each payload is sent to the provider. */
	OnPayload func(
		payload any,
		model ai.Model[ai.API],
	) (any, error)

	/* Optional func to call after each response is received from the provider. */
	OnResponse func(
		response ai.ProviderResponse,
		model ai.Model[ai.API],
	) error
}

type CreateAgentSessionResult struct {
	Session              *AgentSession
	ModelFallbackMessage *string
}

// FindInitialModel returns the first available model in registry, or false
// when no provider has auth configured.
func FindInitialModel(ctx context.Context, registry *ModelRegistry) (ai.Model[ai.API], bool, error) {
	if registry == nil {
		registry = NewModelRegistry(nil)
	}

	available, err := registry.Available(ctx)
	if err != nil || len(available) == 0 {
		return ai.Model[ai.API]{}, false, err
	}
	return available[0], true, nil
}

// CreateAgentSession creates a minimal, generic agent session with no tools.
func CreateAgentSession(ctx context.Context, options CreateAgentSessionOptions) (*CreateAgentSessionResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}

	ensureDefaultAPIProvidersRegistered()

	registry := options.ModelRegistry
	if registry == nil {
		registry = NewModelRegistry(&ModelRegistryOptions{
			Credentials: options.Credentials,
			AmbientAuth: options.AmbientAuth,
		})
	} else if options.Credentials != nil || options.AmbientAuth != nil {
		return nil, errors.New("set ModelRegistry or Credentials and AmbientAuth, not both")
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
	// BuildSessionContext reports "off" when the branch has no thinking-level
	// entry, so only trust its value when an entry actually exists.
	hasThinkingEntry := false
	for _, entry := range sessionManager.GetBranch() {
		if _, ok := entry.(*ThinkingLevelChangeEntry); ok {
			hasThinkingEntry = true
			break
		}
	}

	model := options.Model
	var modelFallbackMessage *string
	if model.ID == "" && hasExistingSession && existingSession.Model != nil {
		restored, ok := registry.Find(ai.Provider(existingSession.Model.Provider), existingSession.Model.ModelID)
		if ok {
			check, err := registry.CheckAuth(ctx, restored.Provider)
			if err != nil {
				return nil, err
			}
			ok = check != nil
		}
		if ok {
			model = restored
		} else {
			msg := "could not restore model " + existingSession.Model.Provider + "/" + existingSession.Model.ModelID
			modelFallbackMessage = &msg
		}
	}

	if model.ID == "" {
		initial, ok, err := FindInitialModel(ctx, registry)
		if err != nil {
			return nil, err
		}
		if !ok {
			return nil, errors.New("no models available; log in to a provider, or set AmbientAuth to ai.DefaultAuthContext() to use OPENAI_API_KEY")
		}
		model = initial
		if modelFallbackMessage != nil {
			msg := *modelFallbackMessage + ". Using " + string(model.Provider) + "/" + model.ID
			modelFallbackMessage = &msg
		}
	}

	thinkingLevel := options.ThinkingLevel
	if thinkingLevel == "" {
		if hasExistingSession && hasThinkingEntry {
			thinkingLevel = ai.ModelThinkingLevel(existingSession.ThinkingLevel)
		} else {
			thinkingLevel = DefaultThinkingLevel
		}
	}
	thinkingLevel = ai.ClampThinkingLevel(model, thinkingLevel)

	systemPrompt := ""
	if options.SystemPrompt != nil {
		systemPrompt = *options.SystemPrompt
	}

	tools := options.Tools
	if tools == nil {
		tools = []agent.AgentTool[any, any]{}
	}

	toolExecution := agent.ToolExecutionModeParallel
	if options.ToolExecutionMode != nil {
		toolExecution = *options.ToolExecutionMode
	}

	sessionID := sessionManager.GetSessionID()
	a := agent.NewAgent(&agent.AgentOptions{
		InitialState: &agent.InitialAgentState{
			SystemPrompt:  &systemPrompt,
			Model:         &model,
			ThinkingLevel: &thinkingLevel,
			Tools:         tools,
			Messages:      existingSession.Messages,
		},
		ConvertToLLM:   ConvertToLLM,
		ResolveAuth:    registry.Auth,
		SessionID:      &sessionID,
		ToolExecution:  toolExecution,
		BeforeToolCall: options.BeforeToolCall,
		AfterToolCall:  options.AfterToolCall,
		OnPayload:      options.OnPayload,
		OnResponse:     options.OnResponse,
	})

	if !hasExistingSession {
		if _, err := sessionManager.AppendModelChange(ctx, string(model.Provider), model.ID); err != nil {
			return nil, err
		}
		if _, err := sessionManager.AppendThinkingLevelChange(ctx, string(thinkingLevel)); err != nil {
			return nil, err
		}
	}

	session := NewAgentSession(a, sessionManager)
	if options.CompactionSettings != nil {
		session.SetCompactionSettings(*options.CompactionSettings)
	}
	session.SetCompactionHook(options.CompactionHook)

	return &CreateAgentSessionResult{
		Session:              session,
		ModelFallbackMessage: modelFallbackMessage,
	}, nil
}

func ensureDefaultAPIProvidersRegistered() {
	if _, ok := ai.GetApiProvider(ai.APIOpenAIResponses); ok {
		return
	}

	ai.RegisterApiProvider(ai.ApiProvider[ai.API, ai.StreamOptions]{
		Api:          ai.APIOpenAIResponses,
		Stream:       ai.StreamOpenAIResponses,
		StreamSimple: ai.StreamSimpleOpenAIResponses,
	}, nil)
}
