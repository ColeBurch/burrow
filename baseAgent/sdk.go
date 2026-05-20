package baseAgent

import "github.com/ColeBurch/burrow/ai"

type CreateAgentSessionOptions struct {
	CWD             *string
	AgentDir        *string
	AuthStorage     *AuthStorage
	ModelRegistry   *ModelRegistry
	Model           ai.Model[ai.API]
	ThinkingLevel   ai.ModelThinkingLevel
	ResourceLoader  *ResourceLoader
	SessionManager  *SessionManager
	SettingsManager *SettingsManager
}

type CreateAgentSessionResult struct {
	Session              AgentSession
	ModelFallbackMessage *string
}
