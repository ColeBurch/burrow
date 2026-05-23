package baseAgent

import (
	"context"
	"errors"
	"time"

	"github.com/ColeBurch/burrow/agent"
	"github.com/ColeBurch/burrow/ai"
)

type StreamingBehavior string

const (
	StreamingBehaviorSteer    StreamingBehavior = "steer"
	StreamingBehaviorFollowUp StreamingBehavior = "followUp"
)

type PromptOptions struct {
	Images []ai.ImageContent
	/** When streaming, how to queue the message: "steer" (interrupt) or "followUp" (wait). Required if streaming. */
	StreamingBehavior StreamingBehavior
}

type AgentSession struct {
	Agent          *agent.Agent
	SessionManager *SessionManager
}

func NewAgentSession(a *agent.Agent, sm *SessionManager) *AgentSession {
	session := &AgentSession{Agent: a, SessionManager: sm}

	if a != nil && sm != nil {
		a.Subscribe(func(event agent.AgentEvent, ctx context.Context) error {
			switch e := event.(type) {
			case agent.MessageEndEvent:
				_, err := sm.AppendMessageWithID(ctx, e.MessageID, e.Message)
				return err
			}
			return nil
		})
	}

	return session
}

/** Whether agent is currently streaming a response */
func (s *AgentSession) IsStreaming() bool {
	return s.Agent != nil && s.Agent.State().IsStreaming
}

/** Return full agent state */
func (s *AgentSession) GetState() (agent.AgentState, error) {
	if s.Agent == nil {
		return agent.AgentState{}, errors.New("no agent set")
	}
	state := s.Agent.State()
	return state, nil
}

/** Current model (may be undefined if not yet selected) */
func (s *AgentSession) GetModel() (ai.Model[ai.API], error) {
	if s.Agent == nil {
		return ai.Model[ai.API]{}, errors.New("no agent set")
	}
	return s.Agent.State().Model, nil
}

/** Current thinking level */
func (s *AgentSession) GetThinkingLevel() (ai.ModelThinkingLevel, error) {
	if s.Agent == nil {
		return ai.ModelThinkingLevel(""), errors.New("no agent set")
	}
	return s.Agent.State().ThinkingLevel, nil
}

/** Current effective system prompt */
func (s *AgentSession) GetSystemPrompt() (string, error) {
	if s.Agent == nil {
		return "", errors.New("no agent set")
	}
	return s.Agent.State().SystemPrompt, nil
}

/*
 * Get the names of currently active tools.
 * Returns the names of tools currently set on the agent.
 */
func (s *AgentSession) GetActiveToolNames() ([]string, error) {
	if s.Agent == nil {
		return nil, errors.New("no agent set")
	}
	tools := s.Agent.State().Tools
	names := make([]string, 0, len(tools))
	for _, tool := range tools {
		names = append(names, tool.Name)
	}
	return names, nil
}

/** Current messages in the conversation */
func (s *AgentSession) GetMessages() ([]agent.AgentMessage, error) {
	if s.Agent == nil {
		return nil, errors.New("no agent set")
	}
	return s.Agent.State().Messages, nil
}

/** Current steering mode */
func (s *AgentSession) GetSteeringMode() (string, error) {
	if s.Agent == nil {
		return "", errors.New("no agent set")
	}
	return string(s.Agent.SteeringMode()), nil
}

/** Current follow-up mode */
func (s *AgentSession) GetFollowUpMode() (string, error) {
	if s.Agent == nil {
		return "", errors.New("no agent set")
	}
	return string(s.Agent.FollowUpMode()), nil
}

/** Current session ID */
func (s *AgentSession) GetSessionID() (string, error) {
	if s.Agent == nil {
		return "", errors.New("no agent set")
	}
	return s.SessionManager.sessionID, nil
}

/** Current session name */
func (s *AgentSession) GetSessionName() (*string, error) {
	if s.Agent == nil {
		return nil, errors.New("no agent set")
	}
	return s.SessionManager.GetSessionName(), nil
}

/**
 * Send a prompt to the agent.
 * - During streaming, queues via steer() or followUp() based on streamingBehavior option
 * - Validates model and API key before sending (when not streaming)
 * Error if streaming and no streamingBehavior specified
 * Error if no model selected or no API key available (when not streaming)
 */
func (s *AgentSession) Prompt(ctx context.Context, text string, options *PromptOptions) error {
	if s.Agent == nil {
		return errors.New("no agent set")
	}

	var message ai.UserMessage
	message.Role = ai.RoleUser

	message.Content = append(message.Content, ai.TextContent{
		Type: "text",
		Text: text,
	})

	if options != nil {
		for _, image := range options.Images {
			message.Content = append(message.Content, image)
		}
	}

	message.Timestamp = time.Now().UnixMilli()

	if s.IsStreaming() {
		if options == nil || options.StreamingBehavior == "" {
			return errors.New("streamingBehavior required when streaming")
		}

		if options.StreamingBehavior != StreamingBehaviorSteer && options.StreamingBehavior != StreamingBehaviorFollowUp {
			return errors.New("invalid streamingBehavior")
		}

		switch options.StreamingBehavior {
		case StreamingBehaviorSteer:
			s.Agent.Steer(message)
			return nil
		case StreamingBehaviorFollowUp:
			s.Agent.FollowUp(message)
			return nil
		}
	}

	if model, err := s.GetModel(); err != nil {
		return err
	} else {
		if model.Name == "unknown" {
			return errors.New("model is undefined")
		}
	}

	s.Agent.PromptMessage(ctx, message)
	return nil
}
