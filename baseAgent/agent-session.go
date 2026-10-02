package baseAgent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/ColeBurch/burrow/agent"
	"github.com/ColeBurch/burrow/ai"
)

type StreamingBehavior string

const (
	StreamingBehaviorSteer    StreamingBehavior = "steer"
	StreamingBehaviorFollowUp StreamingBehavior = "followUp"
)

// messagePersistTimeout bounds each message and compaction write, independent
// of run and compaction cancellation.
var messagePersistTimeout = 5 * time.Second

type PromptOptions struct {
	Images []ai.ImageContent
	/** When streaming, how to queue the message: "steer" (interrupt) or "followUp" (wait). Required if streaming. */
	StreamingBehavior StreamingBehavior
}

type ManualCompactionOptions struct {
	CustomInstructions string
}

// ============================================================================
// Session Events
// ============================================================================

// AgentSessionEvent includes ordinary agent events and session-specific events.
// Consumers can use EventType or a type switch to distinguish them.
type AgentSessionEvent interface {
	agent.AgentEvent
}

// AgentSessionEventListener receives synchronous session notifications.
// Listeners should return promptly and must not synchronously start or wait for
// session operations. They may subscribe, unsubscribe, inspect state, or cancel
// compaction. Queueing with Prompt's StreamingBehavior is also supported.
type AgentSessionEventListener func(event AgentSessionEvent)

type CompactionReason string

const (
	CompactionReasonManual    CompactionReason = "manual"
	CompactionReasonThreshold CompactionReason = "threshold"
	CompactionReasonOverflow  CompactionReason = "overflow"
)

type CompactionStartEvent struct {
	Type   string           `json:"type"`
	Reason CompactionReason `json:"reason"`
}

func (CompactionStartEvent) EventType() string { return "compaction_start" }

// CompactionEndEvent is emitted after persistence and context replacement on
// success, before any retry starts. Result is nil when no compaction completed.
// Exhausted overflow recovery emits an end event without another start event.
type CompactionEndEvent struct {
	Type    string            `json:"type"`
	Reason  CompactionReason  `json:"reason"`
	Result  *CompactionResult `json:"result,omitempty"`
	Aborted bool              `json:"aborted"`
	// WillRetry refers to overflow recovery, not queued-message continuation.
	WillRetry    bool   `json:"willRetry"`
	ErrorMessage string `json:"errorMessage,omitempty"`
	// RestoredMessages are steering and follow-up messages, steering first, that
	// were queued for the run a manual compaction aborted. They are removed from
	// the agent and will not be sent; resend them with Prompt to deliver them.
	RestoredMessages []agent.AgentMessage `json:"restoredMessages,omitempty"`
}

func (CompactionEndEvent) EventType() string { return "compaction_end" }

// ============================================================================
// Compaction Hook
// ============================================================================

// CompactionRequest describes one compaction of the active branch. Hooks may
// change a copy before passing it to next, but must not modify the entries.
type CompactionRequest struct {
	Reason             CompactionReason
	Preparation        *CompactionPreparation
	BranchEntries      []SessionEntry
	CustomInstructions string
	// Prompts starts as DefaultSummaryPrompts.
	Prompts SummaryPrompts
}

// CompactFunc produces the compaction result for a request.
type CompactFunc func(ctx context.Context, req CompactionRequest) (*CompactionResult, error)

// CompactionHook wraps manual and automatic compaction. It can call next for
// the default summary and adjust the result, return its own result, or return
// ErrCompactionCancelled to skip compaction.
//
// The hook runs with session operations blocked and the agent idle. It must
// not call AgentSession methods other than AbortCompaction and IsCompacting,
// and must not write to the SessionManager.
type CompactionHook func(ctx context.Context, req CompactionRequest, next CompactFunc) (*CompactionResult, error)

// ErrCompactionCancelled is returned by a CompactionHook to cancel compaction.
// It is reported as an aborted compaction; automatic compaction lets the
// prompt continue.
var ErrCompactionCancelled = errors.New("compaction cancelled")

// ErrNothingToCompact is returned when everything since the last compaction
// fits in KeepRecentTokens, so compacting would not shrink the context.
// Threshold compaction treats it as a no-op rather than a failure.
var ErrNothingToCompact = errors.New("nothing to compact")

type AgentSession struct {
	// Use Prompt for session orchestration. Direct Agent runs bypass automatic
	// compaction; their persisted messages are checked on the next session prompt.
	Agent          *agent.Agent
	SessionManager *SessionManager

	// operationMu serializes context replacement and run startup. Event listeners
	// must never acquire it: the owning operation may be waiting for the run.
	operationMu        sync.Mutex
	mu                 sync.Mutex
	managedRun         bool
	runCancel          context.CancelFunc
	pendingEnd         bool
	autoRunning        bool
	overflowAttempted  bool
	compactionSettings CompactionSettings
	compactionHook     CompactionHook
	compactionCancel   context.CancelFunc
	// manualCompactionDone is closed when a pending or running manual Compact
	// finishes, waking prompts that arrived during it.
	manualCompactionDone chan struct{}
	unsubscribeAgent     func()
	eventListeners       []*AgentSessionEventListener
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

// ============================================================================
// Event Subscription
// ============================================================================

// Subscribe receives both forwarded agent events and session-specific events.
// The returned function removes this registration and is safe to call repeatedly.
// Changes to subscriptions during an emission apply to subsequent emissions.
func (s *AgentSession) Subscribe(listener AgentSessionEventListener) func() {
	if listener == nil {
		return func() {}
	}
	// Each registration has its own identity, even for the same callback.
	registration := &listener
	s.mu.Lock()
	s.eventListeners = append(s.eventListeners, registration)
	s.mu.Unlock()

	return func() {
		s.mu.Lock()
		defer s.mu.Unlock()
		for i, registered := range s.eventListeners {
			if registered == registration {
				copy(s.eventListeners[i:], s.eventListeners[i+1:])
				s.eventListeners[len(s.eventListeners)-1] = nil
				s.eventListeners = s.eventListeners[:len(s.eventListeners)-1]
				return
			}
		}
	}
}

func (s *AgentSession) emit(event AgentSessionEvent) {
	s.mu.Lock()
	listeners := append([]*AgentSessionEventListener(nil), s.eventListeners...)
	s.mu.Unlock()
	for _, listener := range listeners {
		(*listener)(event)
	}
}

func (s *AgentSession) subscribeToAgentEvents() {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.Agent == nil || s.SessionManager == nil || s.unsubscribeAgent != nil {
		return
	}
	s.unsubscribeAgent = s.Agent.Subscribe(func(event agent.AgentEvent, ctx context.Context) error {
		s.emit(event)
		switch e := event.(type) {
		case agent.MessageEndEvent:
			// The run ctx is already cancelled when an aborted message ends, but the
			// message must still be recorded. Bound the write so a stuck store fails.
			persistCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), messagePersistTimeout)
			_, err := s.SessionManager.AppendMessageWithID(persistCtx, e.MessageID, e.Message)
			cancel()
			if err != nil {
				return err
			}
			s.mu.Lock()
			switch message := e.Message.(type) {
			case ai.UserMessage:
				s.overflowAttempted = false
			case ai.AssistantMessage:
				if message.StopReason != ai.StopReasonError && message.StopReason != ai.StopReasonAborted &&
					!ai.IsContextOverflow(message, s.Agent.State().Model.ContextWindow) {
					s.overflowAttempted = false
				}
			}
			s.mu.Unlock()
		case agent.EndEvent:
			s.mu.Lock()
			s.pendingEnd = true
			s.mu.Unlock()
			// Prompt drains this work after the synchronous agent call returns.
			// Waiting here would deadlock: subscribers finish before the run is idle.
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

// SetCompactionHook replaces the compaction hook. Nil restores the default.
func (s *AgentSession) SetCompactionHook(hook CompactionHook) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.compactionHook = hook
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

// ============================================================================
// Manual Compaction
// ============================================================================

// Compact manually compacts the active session branch and replaces the agent context.
//
// An active run is aborted first. Messages queued for it with Steer or FollowUp
// are not sent: they are removed and reported in the compaction_end event's
// RestoredMessages. Prompts that arrive during compaction wait for it to finish
// and then run on the compacted context.
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
	done := make(chan struct{})
	s.manualCompactionDone = done
	runCancel := s.runCancel
	s.mu.Unlock()

	// Persistence stays subscribed: messages that end during the abort are
	// recorded, so the live context and the session cannot diverge.
	if runCancel != nil {
		runCancel()
	}
	s.Agent.Abort()
	s.operationMu.Lock()
	// Clear the compaction before releasing operationMu, so a waiting prompt
	// that takes it next does not see this compaction as still pending.
	defer func() {
		cancel()
		s.mu.Lock()
		s.compactionCancel = nil
		s.manualCompactionDone = nil
		close(done)
		s.mu.Unlock()
		s.operationMu.Unlock()
	}()
	s.Agent.WaitForIdle()
	// Prompts wait during compaction rather than queue, so nothing is added
	// after this.
	restored := s.Agent.TakeQueuedMessages()

	s.emit(CompactionStartEvent{Type: "compaction_start", Reason: CompactionReasonManual})
	result, err := s.compactContext(compactionCtx, CompactionReasonManual, options)
	end := CompactionEndEvent{Type: "compaction_end", Reason: CompactionReasonManual, Result: result}
	if len(restored) > 0 {
		end.RestoredMessages = restored
	}
	if err != nil {
		end.Aborted = isCompactionAbort(compactionCtx, err)
		if !end.Aborted {
			end.ErrorMessage = fmt.Sprintf("Compaction failed: %v", err)
		}
	}
	s.emit(end)
	return result, err
}

// isCompactionAbort reports whether a compaction error comes from cancellation
// (AbortCompaction, a cancelled caller, or a hook cancelling). Deadlines are
// timeouts, so they are reported as failures.
func isCompactionAbort(compactionCtx context.Context, err error) bool {
	if ctxErr := compactionCtx.Err(); ctxErr != nil {
		return errors.Is(ctxErr, context.Canceled)
	}
	return errors.Is(err, ErrCompactionCancelled) || errors.Is(err, context.Canceled)
}

// compactContext requires operationMu and an idle agent.
func (s *AgentSession) compactContext(compactionCtx context.Context, reason CompactionReason, options *ManualCompactionOptions) (*CompactionResult, error) {
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
		return nil, ErrNothingToCompact
	}

	var headers map[string]string
	if model.Headers != nil {
		headers = make(map[string]string, len(*model.Headers))
		for key, value := range *model.Headers {
			headers[key] = value
		}
	}

	req := CompactionRequest{
		Reason:        reason,
		Preparation:   preparation,
		BranchEntries: pathEntries,
		Prompts:       DefaultSummaryPrompts(),
	}
	if options != nil {
		req.CustomInstructions = options.CustomInstructions
	}
	var usedDefault atomic.Bool
	next := func(ctx context.Context, req CompactionRequest) (*CompactionResult, error) {
		usedDefault.Store(true)
		return Compact(ctx, req.Preparation, model, *apiKey, &CompactOptions{
			Headers:            headers,
			CustomInstructions: req.CustomInstructions,
			ThinkingLevel:      state.ThinkingLevel,
			Prompts:            req.Prompts,
		})
	}

	s.mu.Lock()
	hook := s.compactionHook
	s.mu.Unlock()
	var result *CompactionResult
	var err error
	if hook != nil {
		result, err = hook(compactionCtx, req, next)
	} else {
		result, err = next(compactionCtx, req)
	}
	if err != nil {
		if ctxErr := compactionCtx.Err(); ctxErr != nil {
			return nil, compactionStopped(ctxErr)
		}
		return nil, err
	}
	if err := compactionCtx.Err(); err != nil {
		return nil, compactionStopped(err)
	}
	if result == nil {
		return nil, errors.New("compaction hook returned no result")
	}
	if !branchContains(pathEntries, result.FirstKeptEntryID) {
		return nil, fmt.Errorf("first kept entry %q is not on the current branch", result.FirstKeptEntryID)
	}

	// Store a JSON snapshot so later changes to the hook's value are not saved
	// in memory, and the entry reads the same as after a reload.
	var details any
	if result.Details != nil {
		raw, err := json.Marshal(result.Details)
		if err != nil {
			return nil, fmt.Errorf("compaction details: %w", err)
		}
		details = json.RawMessage(raw)
	}
	var fromHook *bool
	if hook != nil && !usedDefault.Load() {
		replaced := true
		fromHook = &replaced
	}

	// Past this point the summary is complete, so cancellation must not leave a
	// write half-done. Bound it like message writes so a stuck store fails.
	persistCtx, cancelPersist := context.WithTimeout(context.WithoutCancel(compactionCtx), messagePersistTimeout)
	_, err = s.SessionManager.AppendCompaction(
		persistCtx,
		result.Summary,
		result.FirstKeptEntryID,
		result.TokensBefore,
		details,
		fromHook,
	)
	cancelPersist()
	if err != nil {
		return nil, err
	}

	sessionContext := s.SessionManager.BuildSessionContext()
	if err := s.Agent.ReplaceMessages(sessionContext.Messages); err != nil {
		return nil, err
	}

	return result, nil
}

func compactionStopped(ctxErr error) error {
	if errors.Is(ctxErr, context.DeadlineExceeded) {
		return fmt.Errorf("compaction timed out: %w", ctxErr)
	}
	return fmt.Errorf("compaction cancelled: %w", ctxErr)
}

func branchContains(entries []SessionEntry, id string) bool {
	if id == "" {
		return false
	}
	for _, entry := range entries {
		if entry.GetBase().ID == id {
			return true
		}
	}
	return false
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
	if ctx == nil {
		ctx = context.Background()
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

	s.mu.Lock()
	// A manual compaction aborts the active run, so this prompt waits for it
	// below instead of joining that run's queue.
	manualCompaction := s.manualCompactionDone != nil
	busy := !manualCompaction && (s.managedRun || s.autoRunning || s.IsStreaming())
	automatic := s.autoRunning
	if busy && (options != nil && options.StreamingBehavior != "" || !automatic) {
		defer s.mu.Unlock()
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
	s.mu.Unlock()

	// Each pass waits out any manual compaction, then rechecks while holding
	// operationMu, since another Compact may have started in between.
	for {
		s.mu.Lock()
		done := s.manualCompactionDone
		s.mu.Unlock()
		if done != nil {
			select {
			case <-done:
			case <-ctx.Done():
				return ctx.Err()
			}
		}
		s.operationMu.Lock()
		s.mu.Lock()
		if s.manualCompactionDone == nil {
			break
		}
		s.mu.Unlock()
		s.operationMu.Unlock()
	}
	defer s.operationMu.Unlock()
	ctx, cancel := context.WithCancel(ctx)
	s.managedRun = true
	s.runCancel = cancel
	s.mu.Unlock()
	defer func() {
		cancel()
		s.mu.Lock()
		s.managedRun = false
		s.runCancel = nil
		s.mu.Unlock()
	}()
	if err := s.checkCompaction(ctx, false); err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	if model, err := s.GetModel(); err != nil {
		return err
	} else {
		if model.Name == "unknown" {
			return errors.New("model is undefined")
		}
	}

	runErr := s.Agent.PromptMessage(ctx, message)
	for {
		s.mu.Lock()
		pending := s.pendingEnd
		s.pendingEnd = false
		s.mu.Unlock()
		if pending && runErr == nil {
			if err := s.checkCompaction(ctx, true); err != nil {
				return err
			}
		}

		// Enqueueing and the transition to idle share mu, so a prompt accepted
		// after the loop's final queue poll cannot be stranded at this boundary.
		s.mu.Lock()
		messages := s.Agent.State().Messages
		canContinue := runErr == nil && ctx.Err() == nil && s.compactionCancel == nil
		if len(messages) > 0 {
			if last, ok := messages[len(messages)-1].(ai.AssistantMessage); ok {
				canContinue = canContinue && last.StopReason != ai.StopReasonError && last.StopReason != ai.StopReasonAborted
			}
		}
		if !canContinue || !s.Agent.HasQueuedMessages() {
			s.managedRun = false
			s.mu.Unlock()
			return runErr
		}
		s.mu.Unlock()
		runErr = s.Agent.Continue(ctx)
	}
}

// ============================================================================
// Automatic Compaction
// ============================================================================

// checkCompaction runs after agent_end or before a prompt, with operationMu held
// and the agent idle. pre-prompt checks include aborted responses.
func (s *AgentSession) checkCompaction(ctx context.Context, skipAbortedCheck bool) error {
	for {
		settings := s.GetCompactionSettings()
		if !settings.Enabled || s.IsCompacting() || ctx.Err() != nil || s.SessionManager == nil {
			return nil
		}

		// Track the latest assistant and usage source by branch order, not time.
		// Retained messages may have identical timestamps to the compaction, but
		// their usage still describes the old, larger context.
		var messages []agent.AgentMessage
		var last *ai.AssistantMessage
		for _, entry := range s.SessionManager.GetBranch() {
			if _, ok := entry.(*CompactionEntry); ok {
				messages = nil
				last = nil
				continue
			}
			if message := getMessageFromEntry(entry); message != nil {
				messages = append(messages, message)
				if assistant, ok := message.(ai.AssistantMessage); ok {
					last = &assistant
				}
			}
		}
		if last == nil || (skipAbortedCheck && last.StopReason == ai.StopReasonAborted) {
			return nil
		}
		model := s.Agent.State().Model
		sameModel := last.Provider == model.Provider && last.Model == model.ID
		overflow := sameModel && ai.IsContextOverflow(*last, model.ContextWindow)
		reason := CompactionReasonThreshold
		if overflow {
			s.mu.Lock()
			attempted := s.overflowAttempted
			s.overflowAttempted = true
			s.mu.Unlock()
			if attempted {
				if !skipAbortedCheck {
					// A new user message may still be sent after failed recovery.
					return nil
				}
				// Like any provider error, the overflow stays in the transcript.
				s.emit(CompactionEndEvent{
					Type: "compaction_end", Reason: CompactionReasonOverflow,
					ErrorMessage: "context overflow recovery failed after one compact-and-retry attempt",
				})
				return nil
			}
			reason = CompactionReasonOverflow
			// The error is already persisted, but must not be in the live retry context.
			liveMessages := s.Agent.State().Messages
			if len(liveMessages) > 0 {
				tail, ok := liveMessages[len(liveMessages)-1].(ai.AssistantMessage)
				if ok && tail.Provider == model.Provider && tail.Model == model.ID && ai.IsContextOverflow(tail, model.ContextWindow) {
					if err := s.Agent.ReplaceMessages(liveMessages[:len(liveMessages)-1]); err != nil {
						return err
					}
				}
			}
		} else {
			var tokens int64
			if last.StopReason == ai.StopReasonError {
				estimate := EstimateContextTokens(messages)
				if estimate.lastUsageIndex == nil {
					return nil
				}
				tokens = estimate.tokens
			} else {
				tokens = CalculateContextTokens(last.Usage)
			}
			if !ShouldTriggerCompaction(tokens, int64(model.ContextWindow), settings) {
				return nil
			}
		}

		continued, err := s.runAutoCompaction(ctx, reason, overflow)
		if err != nil || !continued {
			return err
		}
		// Continue is synchronous in Go. Check its terminal response only after
		// its listeners finish; this also bounds consecutive overflow recovery.
		skipAbortedCheck = true
	}
}

func (s *AgentSession) runAutoCompaction(ctx context.Context, reason CompactionReason, willRetry bool) (bool, error) {
	model := s.Agent.State().Model
	err := func() error {
		compactionCtx, cancel := context.WithCancel(ctx)
		s.mu.Lock()
		if s.compactionCancel != nil {
			s.mu.Unlock()
			cancel()
			return errors.New("compaction already in progress")
		}
		s.compactionCancel = cancel
		s.autoRunning = true
		s.mu.Unlock()
		defer func() {
			cancel()
			s.mu.Lock()
			s.compactionCancel = nil
			s.autoRunning = false
			s.mu.Unlock()
		}()
		s.emit(CompactionStartEvent{Type: "compaction_start", Reason: reason})
		result, err := s.compactContext(compactionCtx, reason, nil)
		end := CompactionEndEvent{Type: "compaction_end", Reason: reason, Result: result}
		if err != nil {
			end.Aborted = isCompactionAbort(compactionCtx, err)
			nothingToCompact := reason == CompactionReasonThreshold && errors.Is(err, ErrNothingToCompact)
			if !end.Aborted && !nothingToCompact {
				prefix := "Auto-compaction failed"
				if reason == CompactionReasonOverflow {
					prefix = "Context overflow recovery failed"
				}
				end.ErrorMessage = fmt.Sprintf("%s: %v", prefix, err)
			}
		} else {
			end.WillRetry = willRetry
		}
		// This reports compaction completion, not retry completion.
		s.emit(end)
		return err
	}()
	if err != nil {
		// compaction_end already reported the failure. Summarization errors and
		// AbortCompaction skip compaction; only the caller's cancellation stops Prompt.
		return false, ctx.Err()
	}
	if willRetry {
		// Reloading may retain the overflow's whole turn. Strip its tail again
		// without deleting the history entry.
		messages := s.Agent.State().Messages
		if len(messages) > 0 {
			last, ok := messages[len(messages)-1].(ai.AssistantMessage)
			if ok && last.Provider == model.Provider && last.Model == model.ID && ai.IsContextOverflow(last, model.ContextWindow) {
				if err := s.Agent.ReplaceMessages(messages[:len(messages)-1]); err != nil {
					return false, err
				}
			}
		}
	}
	if err := ctx.Err(); err != nil {
		return false, err
	}
	if willRetry || s.Agent.HasQueuedMessages() {
		return true, s.Agent.Continue(ctx)
	}
	return false, nil
}
