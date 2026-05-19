package agent

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/ColeBurch/burrow/internal/ai"
	"github.com/govalues/decimal"
)

func DefaultConvertToLLM(messages []AgentMessage) []ai.Message {
	result := make([]ai.Message, 0, len(messages))
	for _, msg := range messages {
		switch msg.(type) {
		case ai.UserMessage:
			result = append(result, ai.UserMessage(msg.(ai.UserMessage)))
		case ai.AssistantMessage:
			result = append(result, ai.AssistantMessage(msg.(ai.AssistantMessage)))
		case ai.ToolResultMessage:
			result = append(result, ai.ToolResultMessage(msg.(ai.ToolResultMessage)))
		}
	}
	return result
}

var EmptyUsage = ai.Usage{
	Input:      0,
	Output:     0,
	CacheRead:  0,
	CacheWrite: 0,
	Total:      0,
	Cost: ai.Cost{
		Input:      decimal.Zero,
		Output:     decimal.Zero,
		CacheRead:  decimal.Zero,
		CacheWrite: decimal.Zero,
		Total:      decimal.Zero,
	},
}

var DefaultModel = ai.Model[ai.API]{
	ID:            "unknown",
	Name:          "unknown",
	API:           ai.API("unknown"),
	Provider:      ai.Provider("unknown"),
	BaseURL:       "",
	Reasoning:     false,
	Input:         []string{},
	ContextWindow: 0,
	MaxTokens:     0,
	Cost: ai.Cost{
		Input:      decimal.Zero,
		Output:     decimal.Zero,
		CacheRead:  decimal.Zero,
		CacheWrite: decimal.Zero,
		Total:      decimal.Zero,
	},
}

type QueueMode string

const (
	QueueModeAll        QueueMode = "all"
	QueueModeOneAtATime QueueMode = "one-at-a-time"
)

type AgentState struct {
	SystemPrompt  string
	Model         ai.Model[ai.API]
	ThinkingLevel ai.ModelThinkingLevel

	Tools    []AgentTool[any, any]
	Messages []AgentMessage

	IsStreaming      bool
	StreamingMessage AgentMessage
	PendingToolCalls map[string]struct{}
	ErrorMessage     string
}

type InitialAgentState struct {
	SystemPrompt  *string
	Model         *ai.Model[ai.API]
	ThinkingLevel *ai.ModelThinkingLevel
	Tools         []AgentTool[any, any]
	Messages      []AgentMessage
}

func CreateMutableAgentState(initial *InitialAgentState) AgentState {
	state := AgentState{
		SystemPrompt:     "",
		Model:            DefaultModel,
		ThinkingLevel:    ai.ThinkingLevelOff,
		Tools:            []AgentTool[any, any]{},
		Messages:         []AgentMessage{},
		IsStreaming:      false,
		StreamingMessage: nil,
		PendingToolCalls: map[string]struct{}{},
		ErrorMessage:     "",
	}

	if initial == nil {
		return state
	}

	if initial.SystemPrompt != nil {
		state.SystemPrompt = *initial.SystemPrompt
	}

	if initial.Model != nil {
		state.Model = *initial.Model
	}

	if initial.ThinkingLevel != nil {
		state.ThinkingLevel = *initial.ThinkingLevel
	}

	if initial.Tools != nil {
		state.Tools = append([]AgentTool[any, any](nil), initial.Tools...)
	}

	if initial.Messages != nil {
		state.Messages = append([]AgentMessage(nil), initial.Messages...)
	}

	return state
}

type AgentOptions struct {
	InitialState *InitialAgentState

	ConvertToLLM     ConvertToLLMFunc
	TransformContext TransformContextFunc
	StreamFn         StreamFn

	GetAPIKey func(provider string) *string

	OnPayload  func(payload any, model ai.Model[ai.API]) (any, error)
	OnResponse func(response ai.ProviderResponse, model ai.Model[ai.API]) error

	BeforeToolCall func(
		ctx context.Context,
		call BeforeToolCallContext,
	) (*BeforeToolCallResult, error)

	AfterToolCall func(
		ctx context.Context,
		call AfterToolCallContext[any],
	) (*AfterToolCallResult, error)

	SteeringMode QueueMode
	FollowUpMode QueueMode

	SessionID       *string
	ThinkingBudgets *ai.ThinkingBudgets
	Transport       ai.Transport
	MaxRetryDelayMs *int64
	ToolExecution   ToolExecutionMode
}

type PendingMessageQueue struct {
	mu       sync.Mutex
	mode     QueueMode
	messages []AgentMessage
}

func NewPendingMessageQueue(mode QueueMode) *PendingMessageQueue {
	if mode == "" {
		mode = QueueModeOneAtATime
	}

	return &PendingMessageQueue{
		mode:     mode,
		messages: []AgentMessage{},
	}
}

func (q *PendingMessageQueue) Enqueue(message AgentMessage) {
	q.mu.Lock()
	defer q.mu.Unlock()

	q.messages = append(q.messages, message)
}

func (q *PendingMessageQueue) HasItems() bool {
	q.mu.Lock()
	defer q.mu.Unlock()

	return len(q.messages) > 0
}

func (q *PendingMessageQueue) Drain() []AgentMessage {
	q.mu.Lock()
	defer q.mu.Unlock()

	if q.mode == QueueModeAll {
		drained := append([]AgentMessage(nil), q.messages...)
		q.messages = []AgentMessage{}
		return drained
	}

	if len(q.messages) == 0 {
		return []AgentMessage{}
	}

	first := q.messages[0]
	q.messages = append([]AgentMessage(nil), q.messages[1:]...)

	return []AgentMessage{first}
}

func (q *PendingMessageQueue) Clear() {
	q.mu.Lock()
	defer q.mu.Unlock()

	q.messages = []AgentMessage{}
}

func (q *PendingMessageQueue) Mode() QueueMode {
	q.mu.Lock()
	defer q.mu.Unlock()

	return q.mode
}

func (q *PendingMessageQueue) SetMode(mode QueueMode) {
	q.mu.Lock()
	defer q.mu.Unlock()

	q.mode = mode
}

type ActiveRun struct {
	Ctx    context.Context
	Cancel context.CancelFunc
	Done   chan struct{}
}

type AgentListener func(event AgentEvent, ctx context.Context) error

type Agent struct {
	mu sync.Mutex

	state AgentState

	listeners []AgentListener

	steeringQueue *PendingMessageQueue
	followUpQueue *PendingMessageQueue

	ConvertToLLM     ConvertToLLMFunc
	TransformContext TransformContextFunc
	StreamFn         StreamFn
	GetAPIKey        func(provider string) *string

	OnPayload  func(payload any, model ai.Model[ai.API]) (any, error)
	OnResponse func(response ai.ProviderResponse, model ai.Model[ai.API]) error

	BeforeToolCall func(context.Context, BeforeToolCallContext) (*BeforeToolCallResult, error)
	AfterToolCall  func(context.Context, AfterToolCallContext[any]) (*AfterToolCallResult, error)

	activeRun *ActiveRun

	SessionID       *string
	ThinkingBudgets *ai.ThinkingBudgets
	Transport       ai.Transport
	MaxRetryDelayMs *int64
	ToolExecution   ToolExecutionMode
}

func NewAgent(options *AgentOptions) *Agent {
	if options == nil {
		options = &AgentOptions{}
	}

	state := CreateMutableAgentState(options.InitialState)

	convertToLLM := options.ConvertToLLM
	if convertToLLM == nil {
		convertToLLM = DefaultConvertToLLM
	}

	streamFn := options.StreamFn
	if streamFn == nil {
		streamFn = DefaultStreamFn
	}

	steeringMode := options.SteeringMode
	if steeringMode == "" {
		steeringMode = QueueModeOneAtATime
	}

	followUpMode := options.FollowUpMode
	if followUpMode == "" {
		followUpMode = QueueModeOneAtATime
	}

	transport := options.Transport
	if transport == "" {
		transport = ai.TransportAuto
	}

	toolExecution := options.ToolExecution
	if toolExecution == "" {
		toolExecution = ToolExecutionModeParallel
	}

	return &Agent{
		state:            state,
		listeners:        []AgentListener{},
		steeringQueue:    NewPendingMessageQueue(steeringMode),
		followUpQueue:    NewPendingMessageQueue(followUpMode),
		ConvertToLLM:     convertToLLM,
		TransformContext: options.TransformContext,
		StreamFn:         streamFn,
		GetAPIKey:        options.GetAPIKey,
		OnPayload:        options.OnPayload,
		OnResponse:       options.OnResponse,
		BeforeToolCall:   options.BeforeToolCall,
		AfterToolCall:    options.AfterToolCall,
		SessionID:        options.SessionID,
		ThinkingBudgets:  options.ThinkingBudgets,
		Transport:        transport,
		MaxRetryDelayMs:  options.MaxRetryDelayMs,
		ToolExecution:    toolExecution,
	}
}

func (a *Agent) Subscribe(listener AgentListener) func() {
	a.mu.Lock()
	defer a.mu.Unlock()

	a.listeners = append(a.listeners, listener)

	return func() {
		a.mu.Lock()
		defer a.mu.Unlock()

		for i, l := range a.listeners {
			if fmt.Sprintf("%p", l) == fmt.Sprintf("%p", listener) {
				a.listeners = append(a.listeners[:i], a.listeners[i+1:]...)
				return
			}
		}
	}
}

func (a *Agent) State() AgentState {
	a.mu.Lock()
	defer a.mu.Unlock()

	state := a.state
	state.Tools = append([]AgentTool[any, any](nil), a.state.Tools...)
	state.Messages = append([]AgentMessage(nil), a.state.Messages...)

	state.PendingToolCalls = map[string]struct{}{}
	for id := range a.state.PendingToolCalls {
		state.PendingToolCalls[id] = struct{}{}
	}

	return state
}

func (a *Agent) SteeringMode() QueueMode {
	return a.steeringQueue.Mode()
}

func (a *Agent) SetSteeringMode(mode QueueMode) {
	a.steeringQueue.SetMode(mode)
}

func (a *Agent) FollowUpMode() QueueMode {
	return a.followUpQueue.Mode()
}

func (a *Agent) SetFollowUpMode(mode QueueMode) {
	a.followUpQueue.SetMode(mode)
}

func (a *Agent) Steer(message AgentMessage) {
	a.steeringQueue.Enqueue(message)
}

func (a *Agent) FollowUp(message AgentMessage) {
	a.followUpQueue.Enqueue(message)
}

func (a *Agent) ClearSteeringQueue() {
	a.steeringQueue.Clear()
}

func (a *Agent) ClearFollowUpQueue() {
	a.followUpQueue.Clear()
}

func (a *Agent) ClearAllQueues() {
	a.ClearSteeringQueue()
	a.ClearFollowUpQueue()
}

func (a *Agent) HasQueuedMessages() bool {
	return a.steeringQueue.HasItems() || a.followUpQueue.HasItems()
}

func (a *Agent) Signal() context.Context {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.activeRun == nil {
		return nil
	}
	return a.activeRun.Ctx
}

func (a *Agent) Abort() {
	a.mu.Lock()
	active := a.activeRun
	a.mu.Unlock()

	if active != nil {
		active.Cancel()
	}
}

func (a *Agent) WaitForIdle() {
	a.mu.Lock()
	active := a.activeRun
	a.mu.Unlock()

	if active == nil {
		return
	}

	<-active.Done
}

func (a *Agent) Reset() {
	a.mu.Lock()
	defer a.mu.Unlock()

	a.state.Messages = []AgentMessage{}
	a.state.IsStreaming = false
	a.state.StreamingMessage = nil
	a.state.PendingToolCalls = map[string]struct{}{}
	a.state.ErrorMessage = ""

	a.followUpQueue.Clear()
	a.steeringQueue.Clear()
}

func (a *Agent) PromptText(ctx context.Context, text string, images ...ai.ImageContent) error {
	content := make([]ai.UserContent, 0, 1+len(images))
	content = append(content, ai.TextContent{
		Type: ai.ContentTypeText,
		Text: text,
	})
	for _, image := range images {
		content = append(content, image)
	}

	msg := ai.UserMessage{
		Role:      ai.RoleUser,
		Content:   content,
		Timestamp: time.Now().UnixMilli(),
	}

	return a.PromptMessages(ctx, []AgentMessage{msg})
}

func (a *Agent) PromptMessage(ctx context.Context, message AgentMessage) error {
	return a.PromptMessages(ctx, []AgentMessage{message})
}

func (a *Agent) PromptMessages(ctx context.Context, messages []AgentMessage) error {
	a.mu.Lock()
	if a.activeRun != nil {
		a.mu.Unlock()
		return fmt.Errorf("agent is already processing a prompt; use Steer or FollowUp, or wait for completion")
	}
	a.mu.Unlock()

	return a.runPromptMessages(ctx, messages, false)
}

func (a *Agent) Continue(ctx context.Context) error {
	a.mu.Lock()
	if a.activeRun != nil {
		a.mu.Unlock()
		return fmt.Errorf("agent is already processing; wait for completion before continuing")
	}

	messages := append([]AgentMessage(nil), a.state.Messages...)
	a.mu.Unlock()

	if len(messages) == 0 {
		return fmt.Errorf("no messages to continue from")
	}

	last := messages[len(messages)-1]

	if last.AgentMessageType() == string(ai.RoleAssistant) {
		queuedSteering := a.steeringQueue.Drain()
		if len(queuedSteering) > 0 {
			return a.runPromptMessages(ctx, queuedSteering, true)
		}

		queuedFollowUps := a.followUpQueue.Drain()
		if len(queuedFollowUps) > 0 {
			return a.runPromptMessages(ctx, queuedFollowUps, false)
		}

		return fmt.Errorf("cannot continue from message role: assistant")
	}

	return a.runContinuation(ctx)
}

func (a *Agent) runPromptMessages(ctx context.Context, messages []AgentMessage, skipInitialSteeringPoll bool) error {
	return a.runWithLifecycle(ctx, func(runCtx context.Context) error {
		_, err := RunAgentLoop(
			messages,
			a.createContextSnapshot(),
			a.createLoopConfig(skipInitialSteeringPoll),
			runCtx,
			func(event AgentEvent) error {
				return a.processEvent(event)
			},
			&a.StreamFn,
		)
		return err
	})
}

func (a *Agent) runContinuation(ctx context.Context) error {
	return a.runWithLifecycle(ctx, func(runCtx context.Context) error {
		_, err := RunAgentLoopContinue(
			a.createContextSnapshot(),
			a.createLoopConfig(false),
			runCtx,
			func(event AgentEvent) error {
				return a.processEvent(event)
			},
			&a.StreamFn,
		)
		return err
	})
}

func (a *Agent) createContextSnapshot() AgentContext {
	a.mu.Lock()
	defer a.mu.Unlock()

	return AgentContext{
		SystemPrompt: a.state.SystemPrompt,
		Messages:     append([]AgentMessage(nil), a.state.Messages...),
		Tools:        append([]AgentTool[any, any](nil), a.state.Tools...),
	}
}

func (a *Agent) createLoopConfig(skipInitialSteeringPoll bool) AgentLoopConfig {
	a.mu.Lock()
	model := a.state.Model
	thinkingLevel := a.state.ThinkingLevel
	a.mu.Unlock()

	var reasoning *ai.ModelThinkingLevel
	if thinkingLevel != "" && thinkingLevel != ai.ThinkingLevelOff {
		level := thinkingLevel
		reasoning = &level
	}

	transport := a.Transport
	toolExecution := a.ToolExecution
	skip := skipInitialSteeringPoll

	return AgentLoopConfig{
		SimpleStreamOptions: ai.SimpleStreamOptions{
			StreamOptions: ai.StreamOptions{
				SessionId:         a.SessionID,
				Transport:         &transport,
				OnPayload:         a.OnPayload,
				OnResponse:        a.OnResponse,
				MaxRetriesDelayMs: a.MaxRetryDelayMs,
			},
			Reasoning:       reasoning,
			ThinkingBudgets: a.ThinkingBudgets,
		},
		Model:            model,
		Stream:           a.StreamFn,
		ConvertToLLM:     a.ConvertToLLM,
		TransformContext: a.TransformContext,
		GetAPIKey:        a.GetAPIKey,
		ToolExecution:    &toolExecution,
		BeforeToolCall:   a.BeforeToolCall,
		AfterToolCall:    a.AfterToolCall,
		GetSteeringMessages: func(ctx context.Context) []AgentMessage {
			if skip {
				skip = false
				return []AgentMessage{}
			}
			return a.steeringQueue.Drain()
		},
		GetFollowUpMessages: func(ctx context.Context) []AgentMessage {
			return a.followUpQueue.Drain()
		},
	}
}

func (a *Agent) runWithLifecycle(ctx context.Context, executor func(context.Context) error) error {
	a.mu.Lock()
	if a.activeRun != nil {
		a.mu.Unlock()
		return fmt.Errorf("agent is already processing")
	}

	if ctx == nil {
		ctx = context.Background()
	}

	runCtx, cancel := context.WithCancel(ctx)
	done := make(chan struct{})

	a.activeRun = &ActiveRun{
		Ctx:    runCtx,
		Cancel: cancel,
		Done:   done,
	}

	a.state.IsStreaming = true
	a.state.StreamingMessage = nil
	a.state.ErrorMessage = ""
	a.mu.Unlock()

	var runErr error

	defer func() {
		a.finishRun()
		close(done)
		cancel()
	}()

	if err := executor(runCtx); err != nil {
		runErr = err
		_ = a.handleRunFailure(err, runCtx.Err() != nil)
	}

	return runErr
}

func (a *Agent) handleRunFailure(err error, aborted bool) error {
	a.mu.Lock()
	model := a.state.Model
	a.mu.Unlock()

	stopReason := ai.StopReasonError
	if aborted {
		stopReason = ai.StopReasonAborted
	}

	failureMessage := ai.AssistantMessage{
		Role:         ai.RoleAssistant,
		Content:      []ai.AssistantContent{ai.TextContent{Type: ai.ContentTypeText, Text: ""}},
		Api:          model.API,
		Provider:     model.Provider,
		Model:        model.ID,
		Usage:        EmptyUsage,
		StopReason:   stopReason,
		ErrorMessage: err.Error(),
		Timestamp:    time.Now().UnixMilli(),
	}

	a.mu.Lock()
	a.state.Messages = append(a.state.Messages, failureMessage)
	a.state.ErrorMessage = failureMessage.ErrorMessage
	a.mu.Unlock()

	return a.processEvent(EndEvent{
		Type:     "agent_end",
		Messages: []AgentMessage{failureMessage},
	})
}

func (a *Agent) finishRun() {
	a.mu.Lock()
	defer a.mu.Unlock()

	a.state.IsStreaming = false
	a.state.StreamingMessage = nil
	a.state.PendingToolCalls = map[string]struct{}{}
	a.activeRun = nil
}

func (a *Agent) processEvent(event AgentEvent) error {
	a.mu.Lock()

	switch e := event.(type) {
	case MessageStartEvent:
		a.state.StreamingMessage = e.Message

	case MessageUpdateEvent:
		a.state.StreamingMessage = e.Message

	case MessageEndEvent:
		a.state.StreamingMessage = nil
		a.state.Messages = append(a.state.Messages, e.Message)

	case ToolExecutionStartEvent:
		next := map[string]struct{}{}
		for id := range a.state.PendingToolCalls {
			next[id] = struct{}{}
		}
		next[e.ToolCallID] = struct{}{}
		a.state.PendingToolCalls = next

	case ToolExecutionEndEvent:
		next := map[string]struct{}{}
		for id := range a.state.PendingToolCalls {
			if id != e.ToolCallID {
				next[id] = struct{}{}
			}
		}
		a.state.PendingToolCalls = next

	case TurnEndEvent:
		if msg, ok := e.Message.(ai.AssistantMessage); ok && msg.ErrorMessage != "" {
			a.state.ErrorMessage = msg.ErrorMessage
		}

	case EndEvent:
		a.state.StreamingMessage = nil
	}

	active := a.activeRun
	listeners := append([]AgentListener(nil), a.listeners...)
	a.mu.Unlock()

	if active == nil {
		return fmt.Errorf("agent listener invoked outside active run")
	}

	for _, listener := range listeners {
		if err := listener(event, active.Ctx); err != nil {
			return err
		}
	}

	return nil
}
