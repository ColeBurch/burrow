package baseAgent

import "github.com/ColeBurch/burrow/ai"

type CreateAgentSessionOptions struct {
	CWD            *string
	AgentDir       *string
	Model          ai.Model[ai.API]
	ThinkingLevel  ai.ModelThinkingLevel
	SessionManager *SessionManager
}

type CreateAgentSessionResult struct {
	Session              AgentSession
	ModelFallbackMessage *string
}
