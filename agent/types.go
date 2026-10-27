// Ported from pi's packages/agent/src/types.ts (MIT, see LICENSE).

package agent

import (
	"context"

	"github.com/ColeBurch/burrow/ai"
)

type ToolExecutionMode string

const (
	ToolExecutionModeSequential ToolExecutionMode = "sequential"
	ToolExecutionModeParallel   ToolExecutionMode = "parallel"
)

type ConvertToLLMFunc func(messages []AgentMessage) []ai.Message

type TransformContextFunc func(ctx context.Context, messages []AgentMessage) []AgentMessage

// StreamFn starts an assistant message stream.
//
// It may return an error when no stream could be created, such as invalid
// configuration, an unsupported model/provider, missing credentials, or request
// construction failure. If a non-nil stream is returned, callers should consume
// it; runtime/provider/transport failures after stream creation are reported by
// the stream events and Result method.
type StreamFn func(
	ctx context.Context,
	model ai.Model[ai.API],
	modelContext ai.ModelContext,
	options *ai.SimpleStreamOptions,
) (*ai.AssistantMessageEventStream, error)

// Result returned from `beforeToolCall`.
//
// Returning `{ block: true }` prevents the tool from executing. The loop emits an error tool result instead.
// `reason` becomes the text shown in that error result. If omitted, a default blocked message is used.
type BeforeToolCallResult struct {
	Block  *bool
	Reason *string
}

/**
 * Partial override returned from `afterToolCall`.
 *
 * Merge semantics are field-by-field:
 * - `content`: if provided, replaces the tool result content array in full
 * - `details`: if provided, replaces the tool result details value in full
 * - `isError`: if provided, replaces the tool result error flag
 * - `terminate`: if provided, replaces the early-termination hint
 *
 * Omitted fields keep the original executed tool result values.
 * There is no deep merge for `content` or `details`.
 */
type AfterToolCallResult struct {
	Content   *[]ai.ToolResultContent
	Details   *any
	IsError   *bool
	Terminate *bool
}

type AgentToolResult[TDetails any] struct {
	Content []ai.ToolResultContent
	Details TDetails
	// IsError reports a failure without returning an error. The model sees
	// Content as an error result, like a returned error, but Content keeps
	// its images and Details are kept for the UI.
	IsError   bool
	Terminate *bool
}

/** Context snapshot passed into the low-level agent loop. */
type AgentContext struct {
	/** System prompt included with the request. */
	SystemPrompt string
	/** Transcript visible to the model. */
	Messages []AgentMessage
	/** Tools available for this run. */
	Tools []AgentTool[any, any]
}

type BeforeToolCallContext struct {
	/** The assistant message that requested the tool call. */
	AssistantMessage ai.AssistantMessage
	/** The raw tool call block from `assistantMessage.content`. */
	ToolCall ai.ToolCall
	/** Validated tool arguments for the target tool schema. */
	Args any
	/** Current agent context at the time the tool call is prepared. */
	Context AgentContext
}

type AfterToolCallContext[TDetails any] struct {
	/** The assistant message that requested the tool call. */
	AssistantMessage ai.AssistantMessage
	/** The raw tool call block from `assistantMessage.content`. */
	ToolCall ai.ToolCall
	/** Validated tool arguments for the target tool schema. */
	Args any
	/** The executed tool result before any `afterToolCall` overrides are applied. */
	Result AgentToolResult[TDetails]
	/** Whether the executed tool result is currently treated as an error. */
	IsError bool
	/** Current agent context at the time the tool call is prepared. */
	Context AgentContext
}

type AgentToolUpdateCallback[TDetails any] func(partialResult AgentToolResult[TDetails])

type ShouldStopAfterTurnContext struct {
	Message     ai.AssistantMessage
	ToolResults []ai.ToolResultMessage
	Context     AgentContext
	NewMessages []AgentMessage
}

// PrepareNextTurnContext is the turn that just completed, passed to
// PrepareNextTurn before the next one starts.
type PrepareNextTurnContext = ShouldStopAfterTurnContext

// AgentLoopTurnUpdate changes the next turn of a run. Nil fields keep the
// current value.
type AgentLoopTurnUpdate struct {
	// Context for the next provider request, such as one with other tools.
	Context *AgentContext
	// Messages to append before the next provider request, with normal
	// lifecycle events.
	Messages []AgentMessage
	// Model for the next provider request.
	Model *ai.Model[ai.API]
	// ThinkingLevel for the next provider request.
	ThinkingLevel *ai.ModelThinkingLevel
}

// ToolAnnotations are hints about a tool's behavior, such as those of MCP
// tools. Nil means unknown. They are hints: do not rely on them for
// security.
type ToolAnnotations struct {
	// ReadOnlyHint: the tool does not change its environment.
	ReadOnlyHint *bool `json:"readOnlyHint,omitempty"`
	// DestructiveHint: the tool may make destructive changes, not only
	// additive ones.
	DestructiveHint *bool `json:"destructiveHint,omitempty"`
	// IdempotentHint: calling the tool again with the same arguments has no
	// further effect.
	IdempotentHint *bool `json:"idempotentHint,omitempty"`
	// OpenWorldHint: the tool interacts with an open world of external
	// entities, such as the web.
	OpenWorldHint *bool `json:"openWorldHint,omitempty"`
}

type AgentTool[TParameters any, TDetails any] struct {
	ai.Tool
	Label string
	// Optional compatibility shim for raw tool-call arguments before validation.
	PrepareArguments func(args any) (TParameters, error)
	/**
	 * Execute the tool call. Return an error on failure, or a result with
	 * IsError set; do not only describe the failure in `content`.
	 */
	Execute func(
		ctx context.Context,
		toolCallID string,
		params TParameters,
		onUpdate AgentToolUpdateCallback[TDetails],
	) (AgentToolResult[TDetails], error)
	ToolExecutionMode *ToolExecutionMode
	// Annotations are optional hints about the tool's behavior, for apps
	// that ask before running some tools. The model does not see them.
	Annotations *ToolAnnotations
}

// AgentMessage is the union of LLM message types (user, assistant, system)
// and custom messages. This allows custom messages to be added while still
// maintaining type safety and compatibility with the LLM message types.
type AgentMessage interface {
	AgentMessageType() string
}

type AgentEvent interface {
	EventType() string
}

type AgentLoopConfig struct {
	ai.SimpleStreamOptions
	Model               ai.Model[ai.API]
	Stream              StreamFn             `json:"-"`
	ConvertToLLM        ConvertToLLMFunc     `json:"convert_to_llm"`
	TransformContext    TransformContextFunc `json:"transform_context"`
	ResolveAuth         func(ctx context.Context, model ai.Model[ai.API]) (ai.ModelAuth, error)
	ShouldStopAfterTurn func(ctx ShouldStopAfterTurnContext) bool
	// PrepareNextTurn runs after a turn completes and before the next one
	// starts, and may change the next turn, for example its tools. An error
	// ends the run.
	PrepareNextTurn     func(ctx context.Context, turn PrepareNextTurnContext) (*AgentLoopTurnUpdate, error)
	GetSteeringMessages func(ctx context.Context) []AgentMessage
	GetFollowUpMessages func(ctx context.Context) []AgentMessage
	ToolExecution       *ToolExecutionMode
	BeforeToolCall      func(ctx context.Context, call BeforeToolCallContext) (*BeforeToolCallResult, error)
	AfterToolCall       func(ctx context.Context, call AfterToolCallContext[any]) (*AfterToolCallResult, error)
}

type StartEvent struct {
	Type string `json:"type"`
}

func (StartEvent) EventType() string { return "agent_start" }

type EndEvent struct {
	Type     string         `json:"type"`
	Messages []AgentMessage `json:"messages"`
}

func (EndEvent) EventType() string { return "agent_end" }

type TurnStartEvent struct {
	Type string `json:"type"`
}

func (TurnStartEvent) EventType() string { return "turn_start" }

type TurnEndEvent struct {
	Type        string                 `json:"type"`
	Message     AgentMessage           `json:"message"`
	ToolResults []ai.ToolResultMessage `json:"tool_results"`
}

func (TurnEndEvent) EventType() string { return "turn_end" }

type MessageStartEvent struct {
	Type      string       `json:"type"`
	MessageID string       `json:"messageId"`
	Message   AgentMessage `json:"message"`
}

func (MessageStartEvent) EventType() string { return "message_start" }

type MessageUpdateEvent struct {
	Type                  string       `json:"type"`
	MessageID             string       `json:"messageId"`
	Message               AgentMessage `json:"message"`
	AssistantMessageEvent ai.AssistantMessageEvent
}

func (MessageUpdateEvent) EventType() string { return "message_update" }

type MessageEndEvent struct {
	Type      string       `json:"type"`
	MessageID string       `json:"messageId"`
	Message   AgentMessage `json:"message"`
}

func (MessageEndEvent) EventType() string { return "message_end" }

type ToolExecutionStartEvent struct {
	Type       string `json:"type"`
	ToolCallID string `json:"tool_call_id"`
	ToolName   string `json:"tool_name"`
	ToolArgs   any    `json:"tool_args"`
}

func (ToolExecutionStartEvent) EventType() string { return "tool_execution_start" }

type ToolExecutionUpdateEvent struct {
	Type          string `json:"type"`
	ToolCallID    string `json:"tool_call_id"`
	ToolName      string `json:"tool_name"`
	ToolArgs      any    `json:"tool_args"`
	PartialResult any    `json:"partial_result"`
}

func (ToolExecutionUpdateEvent) EventType() string { return "tool_execution_update" }

type ToolExecutionEndEvent struct {
	Type       string `json:"type"`
	ToolCallID string `json:"tool_call_id"`
	ToolName   string `json:"tool_name"`
	Result     any    `json:"result"`
	IsError    bool   `json:"is_error"`
}

func (ToolExecutionEndEvent) EventType() string { return "tool_execution_end" }
