package agent

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sync"
	"time"

	"github.com/ColeBurch/burrow/ai"
)

const agentStreamBuffer = 64

type AgentEventSink func(event AgentEvent) error

func AgentLoop(
	prompts []AgentMessage,
	context AgentContext,
	config AgentLoopConfig,
	signal context.Context,
	streamFn *StreamFn,
) (*ai.EventStream[AgentEvent, []AgentMessage], error) {
	stream := CreateAgentStream()
	go func() {
		messages, err := RunAgentLoop(
			prompts,
			context,
			config,
			signal,
			func(event AgentEvent) error {
				stream.Push(event)
				return nil
			},
			streamFn,
		)
		if err != nil {
			stream.Push(EndEvent{
				Type:     "agent_end",
				Messages: messages,
			})
			return
		}
		stream.End(messages)
	}()
	return stream, nil
}

func AgentLoopContinue(
	context AgentContext,
	config AgentLoopConfig,
	signal context.Context,
	streamFn *StreamFn,
) (*ai.EventStream[AgentEvent, []AgentMessage], error) {
	if len(context.Messages) == 0 {
		return nil, fmt.Errorf("Cannot continue: no messages in context")
	}

	if context.Messages[len(context.Messages)-1].AgentMessageType() == "assistant" {
		return nil, fmt.Errorf("Cannot continue from message type: assistant")
	}

	stream := CreateAgentStream()

	go func() {
		messages, err := RunAgentLoopContinue(
			context,
			config,
			signal,
			func(event AgentEvent) error {
				stream.Push(event)
				return nil
			},
			streamFn,
		)
		if err != nil {
			stream.Push(EndEvent{
				Type:     "agent_end",
				Messages: messages,
			})
			return
		}
		stream.End(messages)

	}()

	return stream, nil
}

func RunAgentLoop(
	prompts []AgentMessage,
	context AgentContext,
	config AgentLoopConfig,
	signal context.Context,
	emit AgentEventSink,
	streamFn *StreamFn,
) ([]AgentMessage, error) {
	newMessages := append([]AgentMessage{}, prompts...)
	currentContext := AgentContext{
		SystemPrompt: context.SystemPrompt,
		Messages:     append(append([]AgentMessage{}, context.Messages...), prompts...),
		Tools:        context.Tools,
	}

	if err := emit(StartEvent{Type: "agent_start"}); err != nil {
		return newMessages, err
	}
	if err := emit(TurnStartEvent{Type: "turn_start"}); err != nil {
		return newMessages, err
	}
	for _, prompt := range prompts {
		if err := emit(MessageStartEvent{Type: "message_start", Message: prompt}); err != nil {
			return newMessages, err
		}
		if err := emit(MessageEndEvent{Type: "message_end", Message: prompt}); err != nil {
			return newMessages, err
		}
	}

	if err := RunLoop(currentContext, &newMessages, config, signal, emit, streamFn); err != nil {
		return newMessages, err
	}

	return newMessages, nil
}

func RunAgentLoopContinue(
	context AgentContext,
	config AgentLoopConfig,
	signal context.Context,
	emit AgentEventSink,
	streamFn *StreamFn,
) ([]AgentMessage, error) {
	if len(context.Messages) == 0 {
		return nil, fmt.Errorf("Cannot continue: no messages in context")
	}

	if context.Messages[len(context.Messages)-1].AgentMessageType() == "assistant" {
		return nil, fmt.Errorf("Cannot continue from message type: assistant")
	}

	newMessages := []AgentMessage{}
	currentContext := AgentContext{
		SystemPrompt: context.SystemPrompt,
		Messages:     slices.Clone(context.Messages),
		Tools:        context.Tools,
	}

	if err := emit(StartEvent{Type: "agent_start"}); err != nil {
		return newMessages, err
	}
	if err := emit(TurnStartEvent{Type: "turn_start"}); err != nil {
		return newMessages, err
	}

	if err := RunLoop(currentContext, &newMessages, config, signal, emit, streamFn); err != nil {
		return newMessages, err
	}
	return newMessages, nil
}

func CreateAgentStream() *ai.EventStream[AgentEvent, []AgentMessage] {
	return ai.NewEventStream[AgentEvent, []AgentMessage](
		agentStreamBuffer,
		func(event AgentEvent) bool {
			return event.EventType() == "agent_end"
		},
		func(event AgentEvent) []AgentMessage {
			if event, ok := event.(EndEvent); ok {
				return event.Messages
			}
			return []AgentMessage{}
		},
	)
}

func RunLoop(
	currentContext AgentContext,
	newMessages *[]AgentMessage,
	config AgentLoopConfig,
	signal context.Context,
	emit AgentEventSink,
	streamFn *StreamFn,
) error {
	firstTurn := true
	pendingMessages := config.GetSteeringMessages(signal)

	for {
		hasMoreToolCalls := true

		for hasMoreToolCalls || len(pendingMessages) > 0 {
			if !firstTurn {
				if err := emit(TurnStartEvent{Type: "turn_start"}); err != nil {
					return err
				}
			} else {
				firstTurn = false
			}

			if len(pendingMessages) > 0 {
				for _, msg := range pendingMessages {
					if err := emit(MessageStartEvent{Type: "message_start", Message: msg}); err != nil {
						return err
					}
					if err := emit(MessageEndEvent{Type: "message_end", Message: msg}); err != nil {
						return err
					}
					currentContext.Messages = append(currentContext.Messages, msg)
					*newMessages = append(*newMessages, msg)
				}
				pendingMessages = []AgentMessage{}
			}

			var selectedStreamFn StreamFn
			if streamFn != nil {
				selectedStreamFn = *streamFn
			}
			message, err := streamAssistantResponse(&currentContext, config, signal, emit, selectedStreamFn)
			if err != nil {
				*newMessages = append(*newMessages, message)
				if emitErr := emit(TurnEndEvent{Type: "turn_end", Message: message, ToolResults: []ai.ToolResultMessage{}}); emitErr != nil {
					return emitErr
				}
				if emitErr := emit(EndEvent{Type: "agent_end", Messages: *newMessages}); emitErr != nil {
					return emitErr
				}
				return err
			}
			*newMessages = append(*newMessages, message)

			if message.StopReason == "error" || message.StopReason == "aborted" {
				if err := emit(TurnEndEvent{Type: "turn_end", Message: message, ToolResults: []ai.ToolResultMessage{}}); err != nil {
					return err
				}
				if err := emit(EndEvent{Type: "agent_end", Messages: *newMessages}); err != nil {
					return err
				}
				return nil
			}

			var toolCalls []ai.ToolCall
			for _, content := range message.Content {
				switch content.(type) {
				case ai.ToolCall:
					toolCalls = append(toolCalls, content.(ai.ToolCall))
				}
			}

			toolResults := []ai.ToolResultMessage{}
			hasMoreToolCalls = false
			if len(toolCalls) > 0 {
				executedToolBatch, err := ExecuteToolCalls(currentContext, message, config, signal, emit)
				if err != nil {
					return err
				}
				for _, result := range executedToolBatch.message {
					toolResults = append(toolResults, result)
				}
				hasMoreToolCalls = !executedToolBatch.terminate

				for _, result := range toolResults {
					currentContext.Messages = append(currentContext.Messages, result)
					*newMessages = append(*newMessages, result)
				}
			}

			if err := emit(TurnEndEvent{
				Type:        "turn_end",
				Message:     message,
				ToolResults: toolResults,
			}); err != nil {
				return err
			}

			if config.ShouldStopAfterTurn != nil && config.ShouldStopAfterTurn(ShouldStopAfterTurnContext{
				Message:     message,
				ToolResults: toolResults,
				Context:     currentContext,
				NewMessages: *newMessages,
			}) {
				if err := emit(EndEvent{
					Type:     "agent_end",
					Messages: *newMessages,
				}); err != nil {
					return err
				}
				return nil
			}

			pendingMessages = config.GetSteeringMessages(signal)
		}

		followUpMessages := config.GetFollowUpMessages(signal)
		if len(followUpMessages) > 0 {
			pendingMessages = append(pendingMessages, followUpMessages...)
			continue
		}

		break
	}

	if err := emit(EndEvent{
		Type:     "agent_end",
		Messages: *newMessages,
	}); err != nil {
		return err
	}
	return nil
}

func streamAssistantResponse(
	agentContext *AgentContext,
	config AgentLoopConfig,
	signal context.Context,
	emit AgentEventSink,
	streamFn StreamFn,
) (ai.AssistantMessage, error) {
	messages := agentContext.Messages
	if config.TransformContext != nil {
		messages = config.TransformContext(signal, messages)
	}

	llmMessages := config.ConvertToLLM(messages)
	var tools []*ai.Tool
	if len(agentContext.Tools) > 0 {
		tools = make([]*ai.Tool, 0, len(agentContext.Tools))
		for i := range agentContext.Tools {
			tool := agentContext.Tools[i].Tool
			tools = append(tools, &tool)
		}
	}
	llmContext := ai.ModelContext{
		SystemPrompt: &agentContext.SystemPrompt,
		Messages:     llmMessages,
		Tools:        tools,
	}

	if streamFn == nil {
		streamFn = config.Stream
	}
	if streamFn == nil {
		streamFn = DefaultStreamFn
	}

	options := config.SimpleStreamOptions
	if config.GetAPIKey != nil {
		if apiKey := config.GetAPIKey(string(config.Model.Provider)); apiKey != nil {
			options.ApiKey = apiKey
		}
	}

	options.Signal = signal

	response, err := streamFn(signal, config.Model, llmContext, &options)
	if err != nil {
		message := AssistantErrorMessage(signal, config.Model, err)
		agentContext.Messages = append(agentContext.Messages, message)
		if err := emit(MessageStartEvent{Type: "message_start", Message: message}); err != nil {
			return message, err
		}
		if err := emit(MessageEndEvent{Type: "message_end", Message: message}); err != nil {
			return message, err
		}
		return message, nil
	}
	if response == nil {
		message := AssistantErrorMessage(signal, config.Model, errors.New("stream function returned nil stream"))
		agentContext.Messages = append(agentContext.Messages, message)
		if err := emit(MessageStartEvent{Type: "message_start", Message: message}); err != nil {
			return message, err
		}
		if err := emit(MessageEndEvent{Type: "message_end", Message: message}); err != nil {
			return message, err
		}
		return message, nil
	}

	addedPartial := false
	for event := range response.Events() {
		switch event.EventType() {
		case "start":
			var partial ai.AssistantMessage
			switch event := event.(type) {
			case ai.StartEvent:
				partial = event.Partial
			default:
				continue
			}
			agentContext.Messages = append(agentContext.Messages, partial)
			addedPartial = true
			if err := emit(MessageStartEvent{Type: "message_start", Message: partial}); err != nil {
				return partial, err
			}

		case "text_start", "text_delta", "text_end",
			"thinking_start", "thinking_delta", "thinking_end",
			"toolcall_start", "toolcall_delta", "toolcall_end":
			var partial ai.AssistantMessage
			switch event := event.(type) {
			case ai.TextStartEvent:
				partial = event.Partial
			case ai.TextDeltaEvent:
				partial = event.Partial
			case ai.TextEndEvent:
				partial = event.Partial
			case ai.ThinkingStartEvent:
				partial = event.Partial
			case ai.ThinkingDeltaEvent:
				partial = event.Partial
			case ai.ThinkingEndEvent:
				partial = event.Partial
			case ai.ToolCallStartEvent:
				partial = event.Partial
			case ai.ToolCallDeltaEvent:
				partial = event.Partial
			case ai.ToolCallEndEvent:
				partial = event.Partial
			default:
				continue
			}
			if !addedPartial || len(agentContext.Messages) == 0 {
				continue
			}
			agentContext.Messages[len(agentContext.Messages)-1] = partial
			if err := emit(MessageUpdateEvent{
				Type:                  "message_update",
				Message:               partial,
				AssistantMessageEvent: event,
			}); err != nil {
				return partial, err
			}

		case "done", "error":
			finalMessage, err := response.Result()
			if err != nil && finalMessage.Role != ai.RoleAssistant {
				finalMessage = AssistantErrorMessage(signal, config.Model, err)
			}
			if addedPartial && len(agentContext.Messages) > 0 {
				agentContext.Messages[len(agentContext.Messages)-1] = finalMessage
			} else {
				agentContext.Messages = append(agentContext.Messages, finalMessage)
				if err := emit(MessageStartEvent{Type: "message_start", Message: finalMessage}); err != nil {
					return finalMessage, err
				}
			}
			if err := emit(MessageEndEvent{Type: "message_end", Message: finalMessage}); err != nil {
				return finalMessage, err
			}
			return finalMessage, nil
		}
	}

	finalMessage, err := response.Result()
	if err != nil && finalMessage.Role != ai.RoleAssistant {
		finalMessage = AssistantErrorMessage(signal, config.Model, err)
	}
	if addedPartial && len(agentContext.Messages) > 0 {
		agentContext.Messages[len(agentContext.Messages)-1] = finalMessage
	} else {
		agentContext.Messages = append(agentContext.Messages, finalMessage)
		if err := emit(MessageStartEvent{Type: "message_start", Message: finalMessage}); err != nil {
			return finalMessage, err
		}
	}
	if err := emit(MessageEndEvent{Type: "message_end", Message: finalMessage}); err != nil {
		return finalMessage, err
	}
	return finalMessage, nil
}

type ExecutedToolCallBatch struct {
	message   []ai.ToolResultMessage
	terminate bool
}

type PreparedToolCallResult interface {
	EventType() string
}

type PreparedToolCall struct {
	toolCall ai.ToolCall
	tool     AgentTool[any, any]
	args     any
}

func (PreparedToolCall) EventType() string { return "prepared" }

type ImmediateToolOutcome struct {
	result  AgentToolResult[any]
	isError bool
}

func (ImmediateToolOutcome) EventType() string { return "immediate" }

type ExecutedToolCallOutcome struct {
	result  AgentToolResult[any]
	isError bool
}

type FinalizedToolCallOutcome struct {
	toolCall ai.ToolCall
	result   AgentToolResult[any]
	isError  bool
}

type FinalizedToolCallEntryFunc func() (FinalizedToolCallOutcome, error)

func (FinalizedToolCallEntryFunc) isFinalizedToolCallEntry() {}

func ExecuteToolCalls(
	currentContext AgentContext,
	assistantMessage ai.AssistantMessage,
	config AgentLoopConfig,
	signal context.Context,
	emit AgentEventSink,
) (ExecutedToolCallBatch, error) {
	var toolCalls []ai.ToolCall
	for _, content := range assistantMessage.Content {
		if toolCall, ok := content.(ai.ToolCall); ok {
			toolCalls = append(toolCalls, toolCall)
		}
	}

	var hasSequentialToolCalls bool
	for _, toolCall := range toolCalls {
		for _, toolLibItem := range currentContext.Tools {
			if toolLibItem.Name == toolCall.Name && toolLibItem.ToolExecutionMode != nil && *toolLibItem.ToolExecutionMode == ToolExecutionModeSequential {
				hasSequentialToolCalls = true
				break
			}
		}
	}
	if config.ToolExecution != nil && *config.ToolExecution == ToolExecutionModeSequential || hasSequentialToolCalls {
		return ExecuteToolCallsSequential(currentContext, assistantMessage, toolCalls, config, signal, emit)
	}

	return ExecuteToolCallsParallel(currentContext, assistantMessage, toolCalls, config, signal, emit)
}

func ExecuteToolCallsParallel(
	currentContext AgentContext,
	assistantMessage ai.AssistantMessage,
	toolCalls []ai.ToolCall,
	config AgentLoopConfig,
	signal context.Context,
	emit AgentEventSink,
) (ExecutedToolCallBatch, error) {
	type finalizedEntry struct {
		finalized *FinalizedToolCallOutcome
		run       func() (FinalizedToolCallOutcome, error)
	}

	finalizedCalls := make([]finalizedEntry, 0, len(toolCalls))

	for _, toolCall := range toolCalls {
		if err := emit(ToolExecutionStartEvent{
			Type:       "tool_execution_start",
			ToolCallID: toolCall.Id,
			ToolName:   toolCall.Name,
			ToolArgs:   toolCall.Args,
		}); err != nil {
			return ExecutedToolCallBatch{}, err
		}

		preparation := PrepareToolCall(currentContext, assistantMessage, toolCall, config, signal)
		switch preparation := preparation.(type) {
		case ImmediateToolOutcome:
			finalized := FinalizedToolCallOutcome{
				toolCall: toolCall,
				result:   preparation.result,
				isError:  preparation.isError,
			}
			if err := EmitToolExecutionEnd(finalized, emit); err != nil {
				return ExecutedToolCallBatch{}, err
			}
			finalizedCalls = append(finalizedCalls, finalizedEntry{finalized: &finalized})

		case PreparedToolCall:
			prepared := preparation
			finalizedCalls = append(finalizedCalls, finalizedEntry{
				run: func() (FinalizedToolCallOutcome, error) {
					executed, err := ExecutePreparedToolCall(prepared, signal, emit)
					if err != nil {
						finalized := FinalizedToolCallOutcome{
							toolCall: prepared.toolCall,
							result:   CreateErrorToolResult(err.Error()),
							isError:  true,
						}
						return finalized, EmitToolExecutionEnd(finalized, emit)
					}

					finalized := FinalizeExecutedToolCall(
						currentContext,
						assistantMessage,
						prepared,
						executed,
						&config,
						signal,
					)
					return finalized, EmitToolExecutionEnd(finalized, emit)
				},
			})
		}
	}

	orderedFinalizedCalls := make([]FinalizedToolCallOutcome, len(finalizedCalls))
	errs := make([]error, len(finalizedCalls))
	var wg sync.WaitGroup
	for i, entry := range finalizedCalls {
		if entry.finalized != nil {
			orderedFinalizedCalls[i] = *entry.finalized
			continue
		}

		wg.Add(1)
		go func(i int, run func() (FinalizedToolCallOutcome, error)) {
			defer wg.Done()
			orderedFinalizedCalls[i], errs[i] = run()
		}(i, entry.run)
	}
	wg.Wait()
	if err := errors.Join(errs...); err != nil {
		return ExecutedToolCallBatch{}, err
	}

	messages := make([]ai.ToolResultMessage, 0, len(orderedFinalizedCalls))
	for _, finalized := range orderedFinalizedCalls {
		toolResultMessage := CreateToolResultMessage(finalized)
		if err := EmitToolResultMessage(*toolResultMessage, emit); err != nil {
			return ExecutedToolCallBatch{}, err
		}
		messages = append(messages, *toolResultMessage)
	}

	return ExecutedToolCallBatch{
		message:   messages,
		terminate: ShouldTerminateToolBatch(orderedFinalizedCalls),
	}, nil
}

func ExecuteToolCallsSequential(
	currentContext AgentContext,
	assistantMessage ai.AssistantMessage,
	toolCalls []ai.ToolCall,
	config AgentLoopConfig,
	signal context.Context,
	emit AgentEventSink,
) (ExecutedToolCallBatch, error) {
	var finalizedCalls []FinalizedToolCallOutcome
	var messages []ai.ToolResultMessage

	for _, toolCall := range toolCalls {
		if err := emit(ToolExecutionStartEvent{
			Type:       "tool_execution_start",
			ToolCallID: toolCall.Id,
			ToolName:   toolCall.Name,
			ToolArgs:   toolCall.Args,
		}); err != nil {
			return ExecutedToolCallBatch{}, err
		}

		preparation := PrepareToolCall(currentContext, assistantMessage, toolCall, config, signal)
		var finalized FinalizedToolCallOutcome
		switch preparation.(type) {
		case ImmediateToolOutcome:
			finalized = FinalizedToolCallOutcome{
				toolCall: toolCall,
				result:   preparation.(ImmediateToolOutcome).result,
				isError:  preparation.(ImmediateToolOutcome).isError,
			}
		case PreparedToolCall:
			executed, err := ExecutePreparedToolCall(preparation.(PreparedToolCall), signal, emit)
			if err != nil {
				finalized = FinalizedToolCallOutcome{
					toolCall: toolCall,
					result:   CreateErrorToolResult(err.Error()),
					isError:  true,
				}
				continue
			}
			finalized = FinalizeExecutedToolCall(
				currentContext,
				assistantMessage,
				preparation.(PreparedToolCall),
				executed,
				&config,
				signal,
			)
		}
		if err := EmitToolExecutionEnd(finalized, emit); err != nil {
			return ExecutedToolCallBatch{}, err
		}
		toolResultMessage := CreateToolResultMessage(finalized)
		if err := EmitToolResultMessage(*toolResultMessage, emit); err != nil {
			return ExecutedToolCallBatch{}, err
		}
		finalizedCalls = append(finalizedCalls, finalized)
		messages = append(messages, *toolResultMessage)
	}

	return ExecutedToolCallBatch{
		message:   messages,
		terminate: ShouldTerminateToolBatch(finalizedCalls),
	}, nil
}

func ShouldTerminateToolBatch(finalizedCalls []FinalizedToolCallOutcome) bool {
	if len(finalizedCalls) == 0 {
		return false
	}

	for _, call := range finalizedCalls {
		if call.result.Terminate == nil || *call.result.Terminate != true {
			return false
		}
	}

	return true
}

func PrepareToolCallArguments(tool AgentTool[any, any], toolCall ai.ToolCall) ai.ToolCall {
	if tool.PrepareArguments == nil {
		return toolCall
	}
	args, err := tool.PrepareArguments(toolCall.Args)
	if err != nil {
		toolCall.Args = nil
	} else {
		toolCall.Args = args
	}
	return toolCall
}

func CreateErrorToolResult(message string) AgentToolResult[any] {
	return AgentToolResult[any]{
		Content: []ai.ToolResultContent{
			ai.TextContent{
				Type: "text",
				Text: message,
			},
		},
	}
}

func PrepareToolCall(
	currentContext AgentContext,
	assistantMessage ai.AssistantMessage,
	toolCall ai.ToolCall,
	config AgentLoopConfig,
	signal context.Context,
) PreparedToolCallResult {
	toolIndex := slices.IndexFunc(currentContext.Tools, func(tool AgentTool[any, any]) bool {
		return tool.Name == toolCall.Name
	})

	if toolIndex == -1 {
		return ImmediateToolOutcome{
			result: CreateErrorToolResult(fmt.Sprintf("Tool %s not found", toolCall.Name)),
		}
	}

	preparedToolCall := PrepareToolCallArguments(currentContext.Tools[toolIndex], toolCall)
	validatedArgs, err := ai.ValidateToolArguments(currentContext.Tools[toolIndex].Tool, preparedToolCall)
	if err != nil {
		return ImmediateToolOutcome{
			result: CreateErrorToolResult(err.Error()),
		}
	}
	if config.BeforeToolCall != nil {
		result, err := config.BeforeToolCall(signal, BeforeToolCallContext{
			AssistantMessage: assistantMessage,
			ToolCall:         preparedToolCall,
			args:             validatedArgs,
			Context:          currentContext,
		})
		if err != nil {
			return ImmediateToolOutcome{
				result:  CreateErrorToolResult(err.Error()),
				isError: true,
			}
		}
		if result != nil && result.Block != nil && *result.Block {
			var reason string
			if result.Reason != nil {
				reason = *result.Reason
			} else {
				reason = "Tool execution was blocked"
			}
			return ImmediateToolOutcome{
				result:  CreateErrorToolResult(reason),
				isError: true,
			}
		}
	}

	agentTool := AgentTool[any, any]{
		Tool:              currentContext.Tools[toolIndex].Tool,
		Label:             currentContext.Tools[toolIndex].Label,
		Execute:           currentContext.Tools[toolIndex].Execute,
		ToolExecutionMode: currentContext.Tools[toolIndex].ToolExecutionMode,
	}

	return PreparedToolCall{
		toolCall: preparedToolCall,
		tool:     agentTool,
		args:     validatedArgs,
	}
}

func ExecutePreparedToolCall(
	prepared PreparedToolCall,
	signal context.Context,
	emit AgentEventSink,
) (ExecutedToolCallOutcome, error) {
	var updateErrs []error
	var updateErrsMu sync.Mutex

	result, err := prepared.tool.Execute(signal, prepared.toolCall.Id, prepared.args, func(partialResult AgentToolResult[any]) {
		if emit == nil {
			return
		}

		if err := emit(ToolExecutionUpdateEvent{
			Type:          "tool_execution_update",
			ToolCallID:    prepared.toolCall.Id,
			ToolName:      prepared.toolCall.Name,
			ToolArgs:      prepared.toolCall.Args,
			PartialResult: partialResult,
		}); err != nil {
			updateErrsMu.Lock()
			updateErrs = append(updateErrs, err)
			updateErrsMu.Unlock()
		}
	})

	updateErrsMu.Lock()
	updateErr := errors.Join(updateErrs...)
	updateErrsMu.Unlock()
	if updateErr != nil {
		return ExecutedToolCallOutcome{}, updateErr
	}

	if err != nil {
		return ExecutedToolCallOutcome{
			result:  CreateErrorToolResult(err.Error()),
			isError: true,
		}, nil
	}
	return ExecutedToolCallOutcome{
		result:  result,
		isError: false,
	}, nil
}

func FinalizeExecutedToolCall(
	currentContext AgentContext,
	assistantMessage ai.AssistantMessage,
	prepared PreparedToolCall,
	executed ExecutedToolCallOutcome,
	config *AgentLoopConfig,
	signal context.Context,
) FinalizedToolCallOutcome {
	result := executed.result
	isError := executed.isError

	if config.AfterToolCall != nil {
		afterResult, err := config.AfterToolCall(signal, AfterToolCallContext[any]{
			AssistantMessage: assistantMessage,
			ToolCall:         prepared.toolCall,
			args:             prepared.args,
			Result:           result,
			IsError:          isError,
			Context:          currentContext,
		})
		if err != nil {
			return FinalizedToolCallOutcome{
				toolCall: prepared.toolCall,
				result:   CreateErrorToolResult(err.Error()),
				isError:  true,
			}
		}
		if afterResult != nil {
			if afterResult.Content != nil {
				result.Content = *afterResult.Content
			}
			if afterResult.Details != nil {
				result.Details = *afterResult.Details
			}
			if afterResult.IsError != nil {
				isError = *afterResult.IsError
			}
			if afterResult.Terminate != nil {
				result.Terminate = afterResult.Terminate
			}
		}
	}

	return FinalizedToolCallOutcome{
		toolCall: prepared.toolCall,
		result:   result,
		isError:  isError,
	}
}

func EmitToolExecutionEnd(
	finalized FinalizedToolCallOutcome,
	emit AgentEventSink,
) error {
	return emit(ToolExecutionEndEvent{
		Type:       "tool_execution_end",
		ToolCallID: finalized.toolCall.Id,
		ToolName:   finalized.toolCall.Name,
		Result:     finalized.result,
		IsError:    finalized.isError,
	})
}

func CreateToolResultMessage(
	finalized FinalizedToolCallOutcome,
) *ai.ToolResultMessage {
	return &ai.ToolResultMessage{
		ToolCallID: finalized.toolCall.Id,
		ToolName:   finalized.toolCall.Name,
		Content:    finalized.result.Content,
		Details:    finalized.result.Details,
		IsError:    finalized.isError,
		Timestamp:  time.Now().UnixMilli(),
	}
}

func EmitToolResultMessage(
	toolResultMessage ai.ToolResultMessage,
	emit AgentEventSink,
) error {
	if err := emit(MessageStartEvent{
		Type:    "message_start",
		Message: toolResultMessage,
	}); err != nil {
		return err
	}
	return emit(MessageEndEvent{
		Type:    "message_end",
		Message: toolResultMessage,
	})
}
