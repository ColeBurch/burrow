package ai

import (
	"context"

	"github.com/govalues/decimal"
)

type Role string

const (
	RoleUser       Role = "user"
	RoleAssistant  Role = "assistant"
	RoleToolResult Role = "toolResult"
)

type ContentType string

const (
	ContentTypeText     ContentType = "text"
	ContentTypeThinking ContentType = "thinking"
	ContentTypeImage    ContentType = "image"
	ContentTypeToolCall ContentType = "toolCall"
)

type TextContent struct {
	Type          ContentType `json:"type"`
	Text          string      `json:"text"`
	TextSignature *string     `json:"textSignature,omitempty"`
}

type ThinkingContent struct {
	Type              ContentType `json:"type"`
	Thinking          string      `json:"thinking"`
	ThinkingSignature *string     `json:"thinkingSignature,omitempty"`
	Redacted          *bool       `json:"redacted,omitempty"`
}

type ImageContent struct {
	Type     ContentType `json:"type"`
	Image    string      `json:"image"`
	MimeType string      `json:"mimeType"`
}

type ToolCall struct {
	Type             ContentType    `json:"type"`
	Id               string         `json:"id"`
	Name             string         `json:"name"`
	Args             map[string]any `json:"args"`
	ThoughtSignature *string        `json:"thoughtSignature,omitempty"`
	PartialJson      *string        `json:"partialJson,omitempty"`
}

type UserContent interface {
	isUserContent()
}

func (TextContent) isUserContent()  {}
func (ImageContent) isUserContent() {}

type UserMessage struct {
	Role      Role          `json:"role"`
	Content   []UserContent `json:"content"`
	Timestamp int64         `json:"timestamp"` // Unix timestamp in milliseconds
}

type Usage struct {
	Input      int64
	Output     int64
	CacheRead  int64
	CacheWrite int64
	Total      int64
	Cost       Cost
}

type Cost struct {
	Input      decimal.Decimal
	Output     decimal.Decimal
	CacheRead  decimal.Decimal
	CacheWrite decimal.Decimal
	Total      decimal.Decimal
}

type AssistantContent interface {
	isAssistantContent()
}

func (TextContent) isAssistantContent()     {}
func (ThinkingContent) isAssistantContent() {}
func (ToolCall) isAssistantContent()        {}

type API string

const (
	APIOpenAIResponses API = "openai-responses"
)

func (a API) IsKnown() bool {
	switch a {
	case APIOpenAIResponses:
		return true
	default:
		return false
	}
}

type Provider string

const (
	ProviderOpenAI Provider = "openai"
)

func (p Provider) IsKnown() bool {
	switch p {
	case ProviderOpenAI:
		return true
	default:
		return false
	}
}

type StopReason string

const (
	StopReasonStop    StopReason = "stop"
	StopReasonLength  StopReason = "length"
	StopReasonToolUse StopReason = "toolUse"
	StopReasonError   StopReason = "error"
	StopReasonAborted StopReason = "aborted"
)

type AssistantMessage struct {
	Role          Role                         `json:"role"`
	Content       []AssistantContent           `json:"content"`
	Api           API                          `json:"api"`
	Provider      Provider                     `json:"provider"`
	Model         string                       `json:"model"`
	ResponseModel *string                      `json:"responseModel,omitempty"`
	ResponseID    *string                      `json:"responseId,omitempty"`
	Diagnostics   []AssistantMessageDiagnostic `json:"diagnostics,omitempty"`
	Usage         Usage                        `json:"usage"`
	StopReason    StopReason                   `json:"stopReason"`
	ErrorMessage  string                       `json:"errorMessage,omitempty"`
	Timestamp     int64                        `json:"timestamp"` // Unix timestamp in milliseconds
}

type ToolResultContent interface {
	isToolResultContent()
}

func (TextContent) isToolResultContent()  {}
func (ImageContent) isToolResultContent() {}

type ToolResultMessage struct {
	Role       Role                `json:"role"`
	ToolCallID string              `json:"toolCallId"`
	ToolName   string              `json:"toolName"`
	Content    []ToolResultContent `json:"content"`
	Details    any                 `json:"details,omitempty"`
	IsError    bool                `json:"isError"`
	Timestamp  int64               `json:"timestamp"` // Unix timestamp in milliseconds
}

type Message interface {
	isMessage()
}

func (UserMessage) isMessage()       {}
func (AssistantMessage) isMessage()  {}
func (ToolResultMessage) isMessage() {}

// AgentMessageType returns the message type as a string for use in the agent loop.
func (UserMessage) AgentMessageType() string {
	return string(RoleUser)
}

func (AssistantMessage) AgentMessageType() string {
	return string(RoleAssistant)
}

func (ToolResultMessage) AgentMessageType() string {
	return string(RoleToolResult)
}

type Tool struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"`
}

type ModelContext struct {
	SystemPrompt *string   `json:"systemPrompt,omitempty"`
	Messages     []Message `json:"messages,omitempty"`
	Tools        []*Tool   `json:"tools,omitempty"`
}

type AssistantMessageEvent interface {
	EventType() string
}

type StartEvent struct {
	Type    string           `json:"type"`
	Partial AssistantMessage `json:"partial"`
}

func (StartEvent) EventType() string { return "start" }

type TextStartEvent struct {
	Type         string           `json:"type"`
	ContentIndex int64            `json:"contentIndex"`
	Partial      AssistantMessage `json:"partial"`
}

func (TextStartEvent) EventType() string { return "text_start" }

type TextDeltaEvent struct {
	Type         string           `json:"type"`
	ContentIndex int64            `json:"contentIndex"`
	Delta        string           `json:"delta"`
	Partial      AssistantMessage `json:"partial"`
}

func (TextDeltaEvent) EventType() string { return "text_delta" }

type TextEndEvent struct {
	Type         string           `json:"type"`
	ContentIndex int64            `json:"contentIndex"`
	Content      string           `json:"content"`
	Partial      AssistantMessage `json:"partial"`
}

func (TextEndEvent) EventType() string { return "text_end" }

type ThinkingStartEvent struct {
	Type         string           `json:"type"`
	ContentIndex int64            `json:"contentIndex"`
	Partial      AssistantMessage `json:"partial"`
}

func (ThinkingStartEvent) EventType() string { return "thinking_start" }

type ThinkingDeltaEvent struct {
	Type         string           `json:"type"`
	ContentIndex int64            `json:"contentIndex"`
	Delta        string           `json:"delta"`
	Partial      AssistantMessage `json:"partial"`
}

func (ThinkingDeltaEvent) EventType() string { return "thinking_delta" }

type ThinkingEndEvent struct {
	Type         string           `json:"type"`
	ContentIndex int64            `json:"contentIndex"`
	Content      string           `json:"content"`
	Partial      AssistantMessage `json:"partial"`
}

func (ThinkingEndEvent) EventType() string { return "thinking_end" }

type ToolCallStartEvent struct {
	Type         string           `json:"type"`
	ContentIndex int64            `json:"contentIndex"`
	Partial      AssistantMessage `json:"partial"`
}

func (ToolCallStartEvent) EventType() string { return "toolcall_start" }

type ToolCallDeltaEvent struct {
	Type         string           `json:"type"`
	ContentIndex int64            `json:"contentIndex"`
	Delta        string           `json:"delta"`
	Partial      AssistantMessage `json:"partial"`
}

func (ToolCallDeltaEvent) EventType() string { return "toolcall_delta" }

type ToolCallEndEvent struct {
	Type         string           `json:"type"`
	ContentIndex int64            `json:"contentIndex"`
	ToolCall     ToolCall         `json:"toolCall"`
	Partial      AssistantMessage `json:"partial"`
}

func (ToolCallEndEvent) EventType() string { return "toolcall_end" }

type DoneEvent struct {
	Type    string           `json:"type"`
	Reason  StopReason       `json:"reason"`
	Message AssistantMessage `json:"message"`
}

func (DoneEvent) EventType() string { return "done" }

type ErrorEvent struct {
	Type    string           `json:"type"`
	Reason  StopReason       `json:"reason"`
	Message AssistantMessage `json:"message"`
}

func (ErrorEvent) EventType() string { return "error" }

type ModelThinkingLevel string

const (
	ThinkingLevelOff     ModelThinkingLevel = "off"
	ThinkingLevelMinimal ModelThinkingLevel = "minimal"
	ThinkingLevelLow     ModelThinkingLevel = "low"
	ThinkingLevelMedium  ModelThinkingLevel = "medium"
	ThinkingLevelHigh    ModelThinkingLevel = "high"
	ThinkingLevelXHigh   ModelThinkingLevel = "xhigh"
)

type ThinkingLevelMap map[ModelThinkingLevel]*string

type ThinkingBudgets struct {
	Minimal *int64 `json:"minimal,omitempty"`
	Low     *int64 `json:"low,omitempty"`
	Medium  *int64 `json:"medium,omitempty"`
	High    *int64 `json:"high,omitempty"`
}

type CacheRetention string

const (
	CacheRetentionNone  CacheRetention = "none"
	CacheRetentionShort CacheRetention = "short"
	CacheRetentionLong  CacheRetention = "long"
)

type Transport string

const (
	TransportSSE             Transport = "sse"
	TransportWebsocket       Transport = "websocket"
	TransportWebsocketCached Transport = "websocket-cached"
	TransportAuto            Transport = "auto"
)

type ProviderResponse struct {
	Status  int               `json:"status"`
	Headers map[string]string `json:"headers"`
}

type StreamOptions struct {
	Temperature       *decimal.Decimal                                        `json:"temperature,omitempty"`
	MaxTokens         *int64                                                  `json:"max_tokens,omitempty"`
	Signal            *context.Context                                        `json:"signal,omitempty"`
	ApiKey            *string                                                 `json:"api_key,omitempty"`
	Transport         *Transport                                              `json:"transport,omitempty"`
	Cache             *CacheRetention                                         `json:"cache,omitempty"`
	SessionId         *string                                                 `json:"session_id,omitempty"`
	OnPayload         func(payload any, model Model[API]) (any, error)        `json:"on_payload,omitempty"`
	OnResponse        func(response ProviderResponse, model Model[API]) error `json:"on_response,omitempty"`
	Headers           map[string]string                                       `json:"headers,omitempty"`
	TimeoutMs         *int64                                                  `json:"timeout_ms,omitempty"`
	MaxRetries        *int64                                                  `json:"max_retries,omitempty"`
	MaxRetriesDelayMs *int64                                                  `json:"max_retries_delay_ms,omitempty"`
	Metadata          map[string]any                                          `json:"metadata,omitempty"`
}

type SimpleStreamOptions struct {
	StreamOptions
	Reasoning       *ModelThinkingLevel
	ThinkingBudgets *ThinkingBudgets
}

type OpenAIResponsesCompat struct {
	SendSessionIdHeader        *bool `json:"send_session_id_header,omitempty"`
	SupportsLongCacheRetention *bool `json:"supports_long_cache_retention,omitempty"`
}

type Model[tApi API] struct {
	ID               string                 `json:"id"`
	Name             string                 `json:"name"`
	API              tApi                   `json:"api"`
	Provider         Provider               `json:"provider"`
	BaseURL          string                 `json:"baseUrl"`
	Reasoning        bool                   `json:"reasoning"`
	ThinkingLevelMap ThinkingLevelMap       `json:"thinkingLevelMap"`
	Input            []string               `json:"input"`
	Cost             Cost                   `json:"cost"` // $/million tokens
	ContextWindow    int                    `json:"contextWindow"`
	MaxTokens        int                    `json:"maxTokens"`
	Headers          *map[string]string     `json:"headers"`
	Compat           *OpenAIResponsesCompat `json:"compat"`
}

func modelWithAPI[tApi API](model Model[API], api tApi) Model[tApi] {
	return Model[tApi]{
		ID:               model.ID,
		Name:             model.Name,
		API:              api,
		Provider:         model.Provider,
		BaseURL:          model.BaseURL,
		Reasoning:        model.Reasoning,
		ThinkingLevelMap: model.ThinkingLevelMap,
		Input:            model.Input,
		Cost:             model.Cost,
		ContextWindow:    model.ContextWindow,
		MaxTokens:        model.MaxTokens,
		Headers:          model.Headers,
		Compat:           model.Compat,
	}
}
