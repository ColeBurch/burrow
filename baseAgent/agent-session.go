package baseAgent

import "github.com/ColeBurch/burrow/agent"

type AgentSession struct {
	Agent          agent.Agent
	SessionManager SessionManager
}
