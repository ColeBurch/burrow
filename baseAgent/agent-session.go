package baseAgent

import (
	"context"
	"errors"
	"fmt"
	"sync"
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

type ManualCompactionOptions struct {
	CustomInstructions string
}

type AgentSession struct {
	Agent          *agent.Agent
	SessionManager *SessionManager

	mu                 sync.Mutex
	compactionSettings CompactionSettings
	compactionCancel   context.CancelFunc
	unsubscribeAgent   func()
}

func NewAgentSession(a *agent.Agent, sm *SessionManager) *AgentSession {
	session := &AgentSession{
		Agent:              a,
		SessionManager:     sm,
		compactionSettings: DEFAULT_COMPACTION_SETTINGS,
	}
	session.subscribeToAgentEvents()
	return session
}

func (s *AgentSession) subscribeToAgentEvents() {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.Agent == nil || s.SessionManager == nil || s.unsubscribeAgent != nil {
		return
	}
	s.unsubscribeAgent = s.Agent.Subscribe(func(event agent.AgentEvent, ctx context.Context) error {
		switch e := event.(type) {
		case agent.MessageEndEvent:
			_, err := s.SessionManager.AppendMessageWithID(ctx, e.MessageID, e.Message)
			return err
		}
		return nil
	})
}

func (s *AgentSession) SetCompactionSettings(settings CompactionSettings) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.compactionSettings = settings
}

func (s *AgentSession) GetCompactionSettings() CompactionSettings {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.compactionSettings
}

func (s *AgentSession) IsCompacting() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.compactionCancel != nil
}

func (s *AgentSession) AbortCompaction() {
	s.mu.Lock()
	cancel := s.compactionCancel
	s.mu.Unlock()

	if cancel != nil {
		cancel()
	}
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

// Compact manually compacts the active session branch and replaces the agent context.
func (s *AgentSession) Compact(ctx context.Context, options *ManualCompactionOptions) (*CompactionResult, error) {
	if s.Agent == nil {
		return nil, errors.New("no agent set")
	}
	if s.SessionManager == nil {
		return nil, errors.New("no session manager set")
	}

	if ctx == nil {
		ctx = context.Background()
	}

	s.mu.Lock()
	if s.compactionCancel != nil {
		s.mu.Unlock()
		return nil, errors.New("compaction already in progress")
	}
	compactionCtx, cancel := context.WithCancel(ctx)
	s.compactionCancel = cancel
	unsubscribe := s.unsubscribeAgent
	s.unsubscribeAgent = nil
	s.mu.Unlock()

	if unsubscribe != nil {
		unsubscribe()
	}
	defer func() {
		s.subscribeToAgentEvents()
		cancel()
		s.mu.Lock()
		s.compactionCancel = nil
		s.mu.Unlock()
	}()

	s.Agent.Abort()
	s.Agent.WaitForIdle()

	state := s.Agent.State()
	model := state.Model
	if model.ID == "" || model.Name == "unknown" {
		return nil, errors.New("model is undefined")
	}
	if s.Agent.GetAPIKey == nil {
		return nil, fmt.Errorf("no API key found for provider %q", model.Provider)
	}
	apiKey := s.Agent.GetAPIKey(string(model.Provider))
	if apiKey == nil || *apiKey == "" {
		return nil, fmt.Errorf("no API key found for provider %q", model.Provider)
	}

	pathEntries := s.SessionManager.GetBranch()
	preparation := PrepareCompaction(pathEntries, s.GetCompactionSettings())
	if preparation == nil {
		if len(pathEntries) > 0 {
			if _, ok := pathEntries[len(pathEntries)-1].(*CompactionEntry); ok {
				return nil, errors.New("already compacted")
			}
		}
		return nil, errors.New("nothing to compact")
	}

	var headers map[string]string
	if model.Headers != nil {
		headers = make(map[string]string, len(*model.Headers))
		for key, value := range *model.Headers {
			headers[key] = value
		}
	}

	customInstructions := ""
	if options != nil {
		customInstructions = options.CustomInstructions
	}
	result, err := Compact(
		compactionCtx,
		preparation,
		model,
		*apiKey,
		&CompactOptions{
			Headers:            headers,
			CustomInstructions: customInstructions,
			ThinkingLevel:      state.ThinkingLevel,
		},
	)
	if err != nil {
		if compactionCtx.Err() != nil {
			return nil, fmt.Errorf("compaction cancelled: %w", compactionCtx.Err())
		}
		return nil, err
	}
	if err := compactionCtx.Err(); err != nil {
		return nil, fmt.Errorf("compaction cancelled: %w", err)
	}

	if _, err := s.SessionManager.AppendCompaction(
		compactionCtx,
		result.Summary,
		result.FirstKeptEntryID,
		result.TokensBefore,
		nil,
		nil,
	); err != nil {
		return nil, err
	}

	sessionContext := s.SessionManager.BuildSessionContext()
	if err := s.Agent.ReplaceMessages(sessionContext.Messages); err != nil {
		// A prompt can race with the final replacement after its compaction-state check.
		// Abort that prompt before the persisted and in-memory contexts diverge.
		s.Agent.Abort()
		s.Agent.WaitForIdle()
		if retryErr := s.Agent.ReplaceMessages(sessionContext.Messages); retryErr != nil {
			return nil, retryErr
		}
	}

	return result, nil
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
	if s.IsCompacting() {
		return errors.New("compaction in progress")
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

	return s.Agent.PromptMessage(ctx, message)
}
