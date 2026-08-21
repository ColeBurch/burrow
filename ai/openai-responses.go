package ai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"hash/fnv"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/govalues/decimal"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"
)

type OpenAIResponseOptions struct {
	StreamOptions
	ReasoningEffort  *string `json:"reasoningEffort,omitempty"`
	ReasoningSummary *string `json:"reasoningSummary,omitempty"`
	ServiceTier      *string `json:"serviceTier,omitempty"`
}

var OpenAIToolCallProviders = map[Provider]struct{}{
	ProviderOpenAI: {},
}

func ResolveCacheRetention(cacheRetention CacheRetention) CacheRetention {
	if cacheRetention != "" && cacheRetention != CacheRetentionNone {
		return cacheRetention
	}
	return CacheRetentionShort
}

type ResolvedOpenAIResponsesCompat struct {
	SendSessionIDHeader        bool
	SupportsLongCacheRetention bool
}

func getCompat(model Model[API]) ResolvedOpenAIResponsesCompat {
	sendSessionIDHeader := true
	supportsLongCacheRetention := true

	if model.Compat != nil {
		if model.Compat.SendSessionIdHeader != nil {
			sendSessionIDHeader = *model.Compat.SendSessionIdHeader
		}

		if model.Compat.SupportsLongCacheRetention != nil {
			supportsLongCacheRetention = *model.Compat.SupportsLongCacheRetention
		}
	}

	return ResolvedOpenAIResponsesCompat{
		SendSessionIDHeader:        sendSessionIDHeader,
		SupportsLongCacheRetention: supportsLongCacheRetention,
	}
}

func GetPromptCacheRetention(
	compat ResolvedOpenAIResponsesCompat,
	cacheRetention CacheRetention,
) responses.ResponseNewParamsPromptCacheRetention {
	if cacheRetention == CacheRetentionLong && compat.SupportsLongCacheRetention {
		return responses.ResponseNewParamsPromptCacheRetention24h
	}

	return ""
}

type ConvertResponseMessagesOptions struct {
	IncludeSystemPrompt *bool `json:"includeSystemPrompt,omitempty"`
}

type responsesStreamProcessor struct {
	output *AssistantMessage
	stream *AssistantMessageEventStream

	currentContentIndex int
	currentItemType     string

	currentParts []responses.ResponseStreamEventUnionPart
}

func StreamOpenAIResponses(model Model[API], modelContext ModelContext, options *StreamOptions) (*AssistantMessageEventStream, error) {
	var simpleOptions *SimpleStreamOptions
	if options != nil {
		simpleOptions = &SimpleStreamOptions{StreamOptions: *options}
	}

	return StreamSimpleOpenAIResponses(model, modelContext, simpleOptions)
}

func StreamSimpleOpenAIResponses(model Model[API], modelContext ModelContext, options *SimpleStreamOptions) (*AssistantMessageEventStream, error) {
	apiKey := ""
	if options != nil && options.ApiKey != nil {
		apiKey = *options.ApiKey
	} else {
		apiKey = getEnvAPIKey(model.Provider)
	}
	if apiKey == "" {
		return nil, fmt.Errorf("No API key for provider: %s", model.Provider)
	}

	base := buildBaseOpenAIResponseOptions(options, apiKey)

	if options != nil && options.Reasoning != nil {
		clampedReasoning := ClampThinkingLevel(model, *options.Reasoning)
		if clampedReasoning != ThinkingLevelOff {
			reasoningEffort := string(clampedReasoning)
			base.ReasoningEffort = &reasoningEffort
		}
	}

	ctx := context.Background()
	if options != nil && options.Signal != nil {
		ctx = options.Signal
	}

	return StreamOpenAIResponse(ctx, model, modelContext, base)
}

func getEnvAPIKey(provider Provider) string {
	switch provider {
	case ProviderOpenAI:
		return os.Getenv("OPENAI_API_KEY")
	default:
		return ""
	}
}

func buildBaseOpenAIResponseOptions(options *SimpleStreamOptions, apiKey string) *OpenAIResponseOptions {
	base := &OpenAIResponseOptions{}
	if options != nil {
		base.StreamOptions = options.StreamOptions
	}

	base.ApiKey = &apiKey
	if base.Cache == nil {
		cache := CacheRetentionShort
		base.Cache = &cache
	}

	return base
}

func httpHeadersToMap(headers http.Header) map[string]string {
	result := make(map[string]string, len(headers))
	for key, values := range headers {
		if len(values) > 0 {
			result[key] = values[0]
		}
	}
	return result
}

func StreamOpenAIResponse(ctx context.Context, model Model[API], modelContext ModelContext, options *OpenAIResponseOptions) (*AssistantMessageEventStream, error) {
	stream := NewAssistantMessageEventStream(64)

	output := AssistantMessage{
		Role:     RoleAssistant,
		Content:  []AssistantContent{},
		Api:      model.API,
		Provider: model.Provider,
		Model:    model.ID,
		Usage: Usage{
			Input:      0,
			Output:     0,
			CacheRead:  0,
			CacheWrite: 0,
			Total:      0,
			Cost: Cost{
				Input:      decimal.Zero,
				Output:     decimal.Zero,
				CacheRead:  decimal.Zero,
				CacheWrite: decimal.Zero,
				Total:      decimal.Zero,
			},
		},
		StopReason: StopReasonStop,
		Timestamp:  time.Now().UnixMilli(),
	}

	if options == nil {
		return nil, fmt.Errorf("options are required")
	}
	apiKey := options.ApiKey
	if apiKey == nil {
		return nil, fmt.Errorf("apiKey is required")
	}
	cache := CacheRetentionShort
	if options.Cache != nil {
		cache = *options.Cache
	}
	var cacheSessionId *string
	cacheRetention := ResolveCacheRetention(cache)
	if cacheRetention == CacheRetentionNone {
		cacheSessionId = nil
	} else {
		cacheSessionId = options.SessionId
	}
	client, err := CreateOpenAIResponsesClient(model, modelContext, apiKey, &options.Headers, cacheSessionId)
	if err != nil {
		return nil, err
	}
	params, err := BuildOpenAIResponsesParams(model, modelContext, options)
	if err != nil {
		return nil, err
	}

	if options.OnPayload != nil {
		nextPayload, err := options.OnPayload(params, model)
		if err != nil {
			return nil, err
		}

		if nextPayload != nil {
			nextParams, ok := nextPayload.(responses.ResponseNewParams)
			if !ok {
				return nil, fmt.Errorf("OnPayload returned %T, want responses.ResponseNewParams", nextPayload)
			}
			params = nextParams
		}
	}

	requestOptions := []option.RequestOption{}

	if options.OnResponse != nil {
		requestOptions = append(requestOptions, option.WithMiddleware(
			func(req *http.Request, next option.MiddlewareNext) (*http.Response, error) {
				resp, err := next(req)
				if err != nil {
					return resp, err
				}

				if resp != nil {
					if hookErr := options.OnResponse(ProviderResponse{
						Status:  resp.StatusCode,
						Headers: httpHeadersToMap(resp.Header),
					}, model); hookErr != nil {
						return resp, hookErr
					}
				}

				return resp, nil
			},
		))
	}

	openaiStream := client.Responses.NewStreaming(ctx, params, requestOptions...)

	processor := &responsesStreamProcessor{
		output: &output,
		stream: stream,
	}

	processor.currentContentIndex = -1
	processor.stream.Push(StartEvent{Type: "start", Partial: output})

	go func() {
		sentTerminalEvent := false
		for openaiStream.Next() {
			event := openaiStream.Current()
			processor.ProcessResponsesStream(event, model, options)
			if isOpenAIResponsesTerminalEvent(event) {
				sentTerminalEvent = true
			}
		}

		if err := openaiStream.Err(); err != nil {
			pushOpenAIResponsesStreamError(ctx, stream, &output, err)
			return
		}

		if !sentTerminalEvent {
			pushAssistantStreamError(stream, &output, StopReasonError, "OpenAI stream ended without terminal event")
		}
	}()

	return stream, nil
}

func isOpenAIResponsesTerminalEvent(event responses.ResponseStreamEventUnion) bool {
	switch event.Type {
	case "response.completed", "response.failed", "error":
		return true
	default:
		return false
	}
}

func pushOpenAIResponsesStreamError(ctx context.Context, stream *AssistantMessageEventStream, output *AssistantMessage, err error) {
	reason := StopReasonError
	if ctx.Err() != nil || errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		reason = StopReasonAborted
	}
	pushAssistantStreamError(stream, output, reason, err.Error())
}

func pushAssistantStreamError(stream *AssistantMessageEventStream, output *AssistantMessage, reason StopReason, message string) {
	output.StopReason = reason
	output.ErrorMessage = message

	stream.Push(ErrorEvent{
		Type:    "error",
		Reason:  reason,
		Message: *output,
	})
}

func CreateOpenAIResponsesClient(
	model Model[API],
	modelContext ModelContext,
	apiKey *string,
	optionHeaders *map[string]string,
	sessionID *string,
) (openai.Client, error) {
	_ = modelContext // Used by provider-specific dynamic headers when those providers are implemented.

	resolvedAPIKey := ""
	if apiKey != nil {
		resolvedAPIKey = *apiKey
	} else {
		resolvedAPIKey = os.Getenv("OPENAI_API_KEY")
	}
	if resolvedAPIKey == "" {
		return openai.Client{}, fmt.Errorf("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass it as an argument")
	}

	compat := getCompat(model)
	headers := make(map[string]string)
	if model.Headers != nil {
		for k, v := range *model.Headers {
			headers[k] = v
		}
	}

	if sessionID != nil && *sessionID != "" {
		if compat.SendSessionIDHeader {
			headers["session_id"] = *sessionID
		}
		headers["x-client-request-id"] = *sessionID
	}

	// Merge option headers last so they can override defaults.
	if optionHeaders != nil {
		for k, v := range *optionHeaders {
			headers[k] = v
		}
	}

	clientOptions := []option.RequestOption{option.WithAPIKey(resolvedAPIKey)}
	if model.BaseURL != "" {
		clientOptions = append(clientOptions, option.WithBaseURL(model.BaseURL))
	}
	for k, v := range headers {
		clientOptions = append(clientOptions, option.WithHeader(k, v))
	}

	return openai.NewClient(clientOptions...), nil
}

func BuildOpenAIResponsesParams(
	model Model[API],
	modelContext ModelContext,
	options *OpenAIResponseOptions,
) (responses.ResponseNewParams, error) {
	messages, err := ConvertResponsesMessages(model, modelContext, OpenAIToolCallProviders, ConvertResponseMessagesOptions{})
	if err != nil {
		return responses.ResponseNewParams{}, err
	}

	cacheRetention := CacheRetentionShort
	if options != nil && options.Cache != nil {
		cacheRetention = ResolveCacheRetention(*options.Cache)
	}

	compat := getCompat(model)

	params := responses.ResponseNewParams{
		Model: shared.ResponsesModel(model.ID),
		Input: responses.ResponseNewParamsInputUnion{
			OfInputItemList: messages,
		},
		Store: param.NewOpt(false),
	}

	if cacheRetention != CacheRetentionNone && options != nil && options.SessionId != nil {
		params.PromptCacheKey = param.NewOpt(*options.SessionId)
	}

	if retention := GetPromptCacheRetention(compat, cacheRetention); retention != "" {
		params.PromptCacheRetention = retention
	}

	if options != nil {
		if options.MaxTokens != nil {
			params.MaxOutputTokens = param.NewOpt(*options.MaxTokens)
		}

		if options.Temperature != nil {
			f, _ := options.Temperature.Float64()
			params.Temperature = param.NewOpt(f)
		}

		if options.ServiceTier != nil {
			params.ServiceTier = responses.ResponseNewParamsServiceTier(*options.ServiceTier)
		}
	}

	if len(modelContext.Tools) > 0 {
		tools, err := ConvertResponsesTools(modelContext.Tools, nil)
		if err != nil {
			return responses.ResponseNewParams{}, err
		}
		params.Tools = tools
	}

	if model.Reasoning {
		if options != nil && (options.ReasoningEffort != nil || options.ReasoningSummary != nil) {
			effort := "medium"
			if options.ReasoningEffort != nil {
				effort = *options.ReasoningEffort
				if model.ThinkingLevelMap != nil {
					if mapped := model.ThinkingLevelMap[ModelThinkingLevel(effort)]; mapped != nil {
						effort = *mapped
					}
				}
			}

			summary := "auto"
			if options.ReasoningSummary != nil {
				summary = *options.ReasoningSummary
			}

			params.Reasoning = shared.ReasoningParam{
				Effort:  shared.ReasoningEffort(effort),
				Summary: shared.ReasoningSummary(summary),
			}

			params.Include = []responses.ResponseIncludable{
				responses.ResponseIncludableReasoningEncryptedContent,
			}
		} else if model.Provider != "github-copilot" {
			effort := "none"
			if model.ThinkingLevelMap != nil {
				if mapped := model.ThinkingLevelMap[ThinkingLevelOff]; mapped != nil {
					effort = *mapped
				}
			}

			params.Reasoning = shared.ReasoningParam{
				Effort: shared.ReasoningEffort(effort),
			}
		}
	}

	return params, nil
}

func ConvertResponsesMessages(model Model[API], modelContext ModelContext, providers map[Provider]struct{}, options ConvertResponseMessagesOptions) (responses.ResponseInputParam, error) {
	messages := make(responses.ResponseInputParam, 0, len(modelContext.Messages))

	transformedMessages := TransformMessages(modelContext.Messages, model, OpenAIToolCallProviders)
	includeSystemPromt := true
	if options.IncludeSystemPrompt != nil {
		includeSystemPromt = *options.IncludeSystemPrompt
	}

	if modelContext.SystemPrompt != nil && includeSystemPromt {
		if model.Reasoning {
			messages = append(messages, responses.ResponseInputItemParamOfMessage(*modelContext.SystemPrompt, responses.EasyInputMessageRoleDeveloper))
		} else {
			messages = append(messages, responses.ResponseInputItemParamOfMessage(*modelContext.SystemPrompt, responses.EasyInputMessageRoleSystem))
		}
	}

	for i, msg := range transformedMessages {
		switch m := msg.(type) {
		case UserMessage:
			if len(m.Content) == 0 {
				continue
			}
			content := make(responses.ResponseInputMessageContentListParam, 0, len(m.Content))
			for _, part := range m.Content {
				switch p := part.(type) {
				case TextContent:
					content = append(content, responses.ResponseInputContentParamOfInputText(p.Text))
				case ImageContent:
					image := responses.ResponseInputContentParamOfInputImage(responses.ResponseInputImageDetailAuto)
					if image.OfInputImage != nil {
						image.OfInputImage.ImageURL = param.NewOpt("data:" + p.MimeType + ";base64," + p.Image)
					}
					content = append(content, image)
				}
			}
			messages = append(messages, responses.ResponseInputItemParamOfMessage(content, responses.EasyInputMessageRoleUser))
		case AssistantMessage:
			isDifferentModel := m.Model != model.ID && m.Provider == model.Provider && m.Api == model.API
			for _, part := range m.Content {
				switch p := part.(type) {
				case TextContent:
					parsedSignature := ParseTextSignature(p.TextSignature)
					msgID := ""
					if parsedSignature != nil {
						msgID = parsedSignature.ID
					}
					if msgID == "" {
						msgID = fmt.Sprintf("msg_%d", i)
					} else if len(msgID) > 64 {
						msgID = fmt.Sprintf("msg_%s", shortHash(msgID))
					}

					content := []responses.ResponseOutputMessageContentUnionParam{{
						OfOutputText: &responses.ResponseOutputTextParam{Text: p.Text},
					}}
					message := responses.ResponseInputItemParamOfOutputMessage(content, msgID, responses.ResponseOutputMessageStatusCompleted)
					if parsedSignature != nil && parsedSignature.Phase != nil && message.OfOutputMessage != nil {
						message.OfOutputMessage.Phase = responses.ResponseOutputMessagePhase(*parsedSignature.Phase)
					}
					messages = append(messages, message)
				case ThinkingContent:
					if p.ThinkingSignature != nil && *p.ThinkingSignature != "" {
						var reasoning responses.ResponseReasoningItemParam
						if err := json.Unmarshal([]byte(*p.ThinkingSignature), &reasoning); err != nil {
							return nil, err
						}

						messages = append(messages, responses.ResponseInputItemUnionParam{
							OfReasoning: &reasoning,
						})
					}
				case ToolCall:
					args, err := json.Marshal(p.Args)
					if err != nil {
						return nil, err
					}

					callID := p.Id
					itemID := ""
					if parts := strings.SplitN(p.Id, "|", 2); len(parts) == 2 {
						callID = parts[0]
						itemID = parts[1]
					}

					functionCall := responses.ResponseFunctionToolCallParam{
						Arguments: string(args),
						CallID:    callID,
						Name:      p.Name,
					}
					if itemID != "" && !(isDifferentModel && strings.HasPrefix(itemID, "fc_")) {
						functionCall.ID = param.NewOpt(itemID)
					}
					messages = append(messages, responses.ResponseInputItemUnionParam{OfFunctionCall: &functionCall})
				}
			}
		case ToolResultMessage:
			var output strings.Builder
			for _, part := range m.Content {
				if p, ok := part.(TextContent); ok {
					output.WriteString(p.Text)
				}
			}
			callID := m.ToolCallID
			if parts := strings.SplitN(callID, "|", 2); len(parts) == 2 {
				callID = parts[0]
			}
			messages = append(messages, responses.ResponseInputItemParamOfFunctionCallOutput(callID, output.String()))
		}
	}
	return messages, nil
}

func normalizeIDPart(part string) string {
	var b strings.Builder

	for _, r := range part {
		if (r >= 'a' && r <= 'z') ||
			(r >= 'A' && r <= 'Z') ||
			(r >= '0' && r <= '9') ||
			r == '_' ||
			r == '-' {
			b.WriteRune(r)
		} else {
			b.WriteByte('_')
		}
	}

	normalized := b.String()
	if len(normalized) > 64 {
		normalized = normalized[:64]
	}

	return strings.TrimRight(normalized, "_")
}

func shortHash(s string) string {
	h := fnv.New64a()
	_, _ = h.Write([]byte(s))
	return strconv.FormatUint(h.Sum64(), 36)
}

func buildForeignResponsesItemID(itemID string) string {
	normalized := "fc_" + shortHash(itemID)
	if len(normalized) > 64 {
		return normalized[:64]
	}
	return normalized
}

func normalizeToolCallID(
	id string,
	targetModel Model[API],
	source AssistantMessage,
	allowedToolCallProviders map[Provider]struct{},
) string {
	if _, ok := allowedToolCallProviders[targetModel.Provider]; !ok {
		return normalizeIDPart(id)
	}

	if !strings.Contains(id, "|") {
		return normalizeIDPart(id)
	}

	parts := strings.SplitN(id, "|", 2)
	callID := parts[0]
	itemID := parts[1]

	normalizedCallID := normalizeIDPart(callID)

	isForeignToolCall :=
		source.Provider != targetModel.Provider ||
			source.Api != targetModel.API

	var normalizedItemID string
	if isForeignToolCall {
		normalizedItemID = buildForeignResponsesItemID(itemID)
	} else {
		normalizedItemID = normalizeIDPart(itemID)
	}

	// OpenAI Responses API requires item ID to start with "fc_".
	if !strings.HasPrefix(normalizedItemID, "fc_") {
		normalizedItemID = normalizeIDPart("fc_" + normalizedItemID)
	}

	return normalizedCallID + "|" + normalizedItemID
}

type TextSignaturePhase string

const (
	PhaseCommentary  TextSignaturePhase = "commentary"
	PhaseFinalAnswer TextSignaturePhase = "final_answer"
)

type ParsedTextSignature struct {
	ID    string
	Phase *TextSignaturePhase
}

type textSignatureV1 struct {
	V     int                 `json:"v"`
	ID    string              `json:"id"`
	Phase *TextSignaturePhase `json:"phase,omitempty"`
}

func ParseTextSignature(signature *string) *ParsedTextSignature {
	if signature == nil || *signature == "" {
		return nil
	}

	if (*signature)[0] == '{' {
		var parsed textSignatureV1

		if err := json.Unmarshal([]byte(*signature), &parsed); err == nil {
			if parsed.V == 1 && parsed.ID != "" {
				result := &ParsedTextSignature{
					ID: parsed.ID,
				}

				if parsed.Phase != nil &&
					(*parsed.Phase == PhaseCommentary ||
						*parsed.Phase == PhaseFinalAnswer) {
					result.Phase = parsed.Phase
				}

				return result
			}
		}
	}

	return &ParsedTextSignature{
		ID: *signature,
	}
}

func EncodeTextSignatureV1(id string, phase *TextSignaturePhase) string {
	payload := textSignatureV1{
		V:     1,
		ID:    id,
		Phase: phase,
	}

	data, err := json.Marshal(payload)
	if err != nil {
		return id
	}

	return string(data)
}

func MapOpenAIResponseStopReason(status responses.ResponseStatus) (StopReason, error) {
	switch string(status) {
	case "completed":
		return StopReasonStop, nil
	case "incomplete":
		return StopReasonLength, nil
	case "failed":
		return StopReasonError, nil
	case "cancelled":
		return StopReasonAborted, nil
	case "in_progress":
		return StopReasonStop, nil
	case "queued":
		return StopReasonStop, nil
	default:
		return StopReasonError, fmt.Errorf("Unhandled stop reason: %s", string(status))
	}
}

func (p *responsesStreamProcessor) ProcessResponsesStream(
	event responses.ResponseStreamEventUnion,
	model Model[API],
	options *OpenAIResponseOptions,
) {
	switch event.Type {
	case "response.created":
		p.output.ResponseID = &event.Response.ID

	case "response.output_item.added":
		item := event.Item

		switch item.Type {
		case "reasoning":
			block := ThinkingContent{
				Type:     ContentTypeThinking,
				Thinking: "",
			}

			p.output.Content = append(p.output.Content, block)
			p.currentContentIndex = len(p.output.Content) - 1
			p.currentItemType = "reasoning"
			p.currentParts = nil

			partial := *p.output
			partial.Content = append([]AssistantContent(nil), p.output.Content...)
			p.stream.Push(ThinkingStartEvent{
				Type:         "thinking_start",
				ContentIndex: int64(p.currentContentIndex),
				Partial:      partial,
			})

		case "message":
			block := TextContent{
				Type: ContentTypeText,
				Text: "",
			}

			p.output.Content = append(p.output.Content, block)
			p.currentContentIndex = len(p.output.Content) - 1
			p.currentItemType = "message"
			p.currentParts = nil

			partial := *p.output
			partial.Content = append([]AssistantContent(nil), p.output.Content...)
			p.stream.Push(TextStartEvent{
				Type:         "text_start",
				ContentIndex: int64(p.currentContentIndex),
				Partial:      partial,
			})

		case "function_call":
			partialJSON := item.Arguments.OfString

			block := ToolCall{
				Type:        ContentTypeToolCall,
				Id:          fmt.Sprintf("%s|%s", item.CallID, item.ID),
				Name:        item.Name,
				Args:        map[string]any{},
				PartialJson: &partialJSON,
			}

			p.output.Content = append(p.output.Content, block)
			p.currentContentIndex = len(p.output.Content) - 1
			p.currentItemType = "function_call"

			partial := *p.output
			partial.Content = append([]AssistantContent(nil), p.output.Content...)
			p.stream.Push(ToolCallStartEvent{
				Type:         "toolcall_start",
				ContentIndex: int64(p.currentContentIndex),
				Partial:      partial,
			})
		}
	case "response.reasoning_summary_part.added":
		if p.currentItemType == "reasoning" {
			p.currentParts = append(p.currentParts, event.Part)
		}
	case "response.reasoning_summary_text.delta":
		if p.currentItemType != "reasoning" || p.currentContentIndex < 0 {
			return
		}

		if len(p.currentParts) == 0 {
			return
		}

		block, ok := p.output.Content[p.currentContentIndex].(ThinkingContent)
		if !ok {
			return
		}

		block.Thinking += event.Delta
		p.output.Content[p.currentContentIndex] = block

		last := &p.currentParts[len(p.currentParts)-1]
		last.Text += event.Delta

		partial := *p.output
		partial.Content = append([]AssistantContent(nil), p.output.Content...)
		p.stream.Push(ThinkingDeltaEvent{
			Type:         "thinking_delta",
			ContentIndex: int64(p.currentContentIndex),
			Delta:        event.Delta,
			Partial:      partial,
		})
	case "response.reasoning_summary_part.done":
		if p.currentItemType != "reasoning" || p.currentContentIndex < 0 {
			return
		}

		if len(p.currentParts) == 0 {
			return
		}

		block, ok := p.output.Content[p.currentContentIndex].(ThinkingContent)
		if !ok {
			return
		}

		delta := "\n\n"

		block.Thinking += delta
		p.output.Content[p.currentContentIndex] = block

		last := &p.currentParts[len(p.currentParts)-1]
		last.Text += delta

		partial := *p.output
		partial.Content = append([]AssistantContent(nil), p.output.Content...)
		p.stream.Push(ThinkingDeltaEvent{
			Type:         "thinking_delta",
			ContentIndex: int64(p.currentContentIndex),
			Delta:        delta,
			Partial:      partial,
		})
	case "response.content_part.added":
		if p.currentItemType == "message" {
			p.currentParts = append(p.currentParts, event.Part)
		}
	case "response.output_text.delta":
		if p.currentItemType != "message" || p.currentContentIndex < 0 {
			return
		}

		if len(p.currentParts) == 0 {
			return
		}

		lastPart := &p.currentParts[len(p.currentParts)-1]
		if lastPart.Type != "output_text" {
			return
		}

		block, ok := p.output.Content[p.currentContentIndex].(TextContent)
		if !ok {
			return
		}

		block.Text += event.Delta
		p.output.Content[p.currentContentIndex] = block

		lastPart.Text += event.Delta

		partial := *p.output
		partial.Content = append([]AssistantContent(nil), p.output.Content...)
		p.stream.Push(TextDeltaEvent{
			Type:         "text_delta",
			ContentIndex: int64(p.currentContentIndex),
			Delta:        event.Delta,
			Partial:      partial,
		})
	case "response.refusal.delta":
		if p.currentItemType != "message" || p.currentContentIndex < 0 {
			return
		}

		if len(p.currentParts) == 0 {
			return
		}

		lastPart := &p.currentParts[len(p.currentParts)-1]
		if lastPart.Type != "refusal" {
			return
		}

		block, ok := p.output.Content[p.currentContentIndex].(TextContent)
		if !ok {
			return
		}

		block.Text += event.Delta
		p.output.Content[p.currentContentIndex] = block

		lastPart.Refusal += event.Delta

		partial := *p.output
		partial.Content = append([]AssistantContent(nil), p.output.Content...)
		p.stream.Push(TextDeltaEvent{
			Type:         "text_delta",
			ContentIndex: int64(p.currentContentIndex),
			Delta:        event.Delta,
			Partial:      partial,
		})
	case "response.function_call_arguments.delta":
		if p.currentItemType != "function_call" || p.currentContentIndex < 0 {
			return
		}

		block, ok := p.output.Content[p.currentContentIndex].(ToolCall)
		if !ok {
			return
		}

		partialJSON := ""
		if block.PartialJson != nil {
			partialJSON = *block.PartialJson
		}

		partialJSON += event.Delta
		block.PartialJson = &partialJSON

		//Will fail until full valid JSON is received
		var args map[string]any
		if err := json.Unmarshal([]byte(partialJSON), &args); err == nil {
			block.Args = args
		}

		p.output.Content[p.currentContentIndex] = block

		partial := *p.output
		partial.Content = append([]AssistantContent(nil), p.output.Content...)
		p.stream.Push(ToolCallDeltaEvent{
			Type:         "toolcall_delta",
			ContentIndex: int64(p.currentContentIndex),
			Delta:        event.Delta,
			Partial:      partial,
		})
	case "response.function_call_arguments.done":
		if p.currentItemType != "function_call" || p.currentContentIndex < 0 {
			return
		}

		block, ok := p.output.Content[p.currentContentIndex].(ToolCall)
		if !ok {
			return
		}

		previousPartialJSON := ""
		if block.PartialJson != nil {
			previousPartialJSON = *block.PartialJson
		}

		block.PartialJson = &event.Arguments

		var args map[string]any
		if err := json.Unmarshal([]byte(event.Arguments), &args); err == nil {
			block.Args = args
		}

		p.output.Content[p.currentContentIndex] = block

		if strings.HasPrefix(event.Arguments, previousPartialJSON) {
			delta := event.Arguments[len(previousPartialJSON):]
			if delta != "" {
				partial := *p.output
				partial.Content = append([]AssistantContent(nil), p.output.Content...)
				p.stream.Push(ToolCallDeltaEvent{
					Type:         "toolcall_delta",
					ContentIndex: int64(p.currentContentIndex),
					Delta:        delta,
					Partial:      partial,
				})
			}
		}
	case "response.output_item.done":
		item := event.Item

		switch item.Type {
		case "reasoning":
			if p.currentContentIndex < 0 {
				return
			}

			block, ok := p.output.Content[p.currentContentIndex].(ThinkingContent)
			if !ok {
				return
			}

			parts := make([]string, 0, len(item.Summary))
			for _, summary := range item.Summary {
				parts = append(parts, summary.Text)
			}

			block.Thinking = strings.Join(parts, "\n\n")

			signatureBytes, err := json.Marshal(item)
			if err == nil {
				signature := string(signatureBytes)
				block.ThinkingSignature = &signature
			}

			p.output.Content[p.currentContentIndex] = block

			partial := *p.output
			partial.Content = append([]AssistantContent(nil), p.output.Content...)
			p.stream.Push(ThinkingEndEvent{
				Type:         "thinking_end",
				ContentIndex: int64(p.currentContentIndex),
				Content:      block.Thinking,
				Partial:      partial,
			})

			p.currentContentIndex = -1
			p.currentItemType = ""
			p.currentParts = nil
		case "message":
			if p.currentContentIndex < 0 {
				return
			}

			block, ok := p.output.Content[p.currentContentIndex].(TextContent)
			if !ok {
				return
			}

			var text strings.Builder

			for _, content := range item.Content {
				if content.Type == "output_text" {
					text.WriteString(content.Text)
				} else {
					text.WriteString(content.Refusal)
				}
			}

			block.Text = text.String()

			var phase *TextSignaturePhase
			if item.Phase != "" {
				p := TextSignaturePhase(item.Phase)
				phase = &p
			}

			signature := EncodeTextSignatureV1(item.ID, phase)
			block.TextSignature = &signature

			p.output.Content[p.currentContentIndex] = block

			partial := *p.output
			partial.Content = append([]AssistantContent(nil), p.output.Content...)
			p.stream.Push(TextEndEvent{
				Type:         "text_end",
				ContentIndex: int64(p.currentContentIndex),
				Content:      block.Text,
				Partial:      partial,
			})

			p.currentContentIndex = -1
			p.currentItemType = ""
			p.currentParts = nil
		case "function_call":
			var toolCall ToolCall

			if p.currentContentIndex >= 0 {
				if block, ok := p.output.Content[p.currentContentIndex].(ToolCall); ok {
					argsJSON := item.Arguments.OfString
					if block.PartialJson != nil && *block.PartialJson != "" {
						argsJSON = *block.PartialJson
					}

					args := map[string]any{}
					if argsJSON == "" {
						argsJSON = "{}"
					}
					_ = json.Unmarshal([]byte(argsJSON), &args)

					block.Args = args
					block.PartialJson = nil // strip scratch buffer

					p.output.Content[p.currentContentIndex] = block
					toolCall = block
				}
			}

			if toolCall.Type == "" {
				argsJSON := item.Arguments.OfString
				if argsJSON == "" {
					argsJSON = "{}"
				}

				args := map[string]any{}
				_ = json.Unmarshal([]byte(argsJSON), &args)

				toolCall = ToolCall{
					Type: ContentTypeToolCall,
					Id:   fmt.Sprintf("%s|%s", item.CallID, item.ID),
					Name: item.Name,
					Args: args,
				}

				p.output.Content = append(p.output.Content, toolCall)
				p.currentContentIndex = len(p.output.Content) - 1
			}

			partial := *p.output
			partial.Content = append([]AssistantContent(nil), p.output.Content...)
			p.stream.Push(ToolCallEndEvent{
				Type:         "toolcall_end",
				ContentIndex: int64(p.currentContentIndex),
				ToolCall:     toolCall,
				Partial:      partial,
			})

			p.currentContentIndex = -1
			p.currentItemType = ""
			p.currentParts = nil
		}
	case "response.completed":
		response := event.Response

		if response.ID != "" {
			p.output.ResponseID = &response.ID
		}

		cachedTokens := response.Usage.InputTokensDetails.CachedTokens

		p.output.Usage = Usage{
			// OpenAI includes cached tokens in input_tokens, so subtract to get non-cached input.
			Input:      response.Usage.InputTokens - cachedTokens,
			Output:     response.Usage.OutputTokens,
			CacheRead:  cachedTokens,
			CacheWrite: 0,
			Total:      response.Usage.TotalTokens,
			Cost: Cost{
				Input:      decimal.Zero,
				Output:     decimal.Zero,
				CacheRead:  decimal.Zero,
				CacheWrite: decimal.Zero,
				Total:      decimal.Zero,
			},
		}

		var err error
		p.output.Usage, err = CalculateCost(model, p.output.Usage)
		if err != nil {
			p.stream.End(err)
			return
		}

		p.output.StopReason, err = MapOpenAIResponseStopReason(response.Status)
		if err != nil {
			p.output.StopReason = StopReasonError
		}

		for _, content := range p.output.Content {
			if _, ok := content.(ToolCall); ok && p.output.StopReason == StopReasonStop {
				p.output.StopReason = StopReasonToolUse
				break
			}
		}

		p.stream.Push(DoneEvent{
			Type:    "done",
			Message: *p.output,
		})
	case "error":
		message := fmt.Sprintf("Error Code %s: %s", event.Code, event.Message)
		if message == "Error Code : " {
			message = "Unknown error"
		}

		pushAssistantStreamError(p.stream, p.output, StopReasonError, message)
	case "response.failed":
		response := event.Response

		msg := "Unknown error (no error details in response)"

		if response.Error.Code != "" || response.Error.Message != "" {
			code := string(response.Error.Code)
			if code == "" {
				code = "unknown"
			}

			message := response.Error.Message
			if message == "" {
				message = "no message"
			}

			msg = fmt.Sprintf("%s: %s", code, message)
		} else if response.IncompleteDetails.Reason != "" {
			msg = fmt.Sprintf("incomplete: %s", response.IncompleteDetails.Reason)
		}

		p.output.StopReason = StopReasonError
		p.output.ErrorMessage = msg

		p.stream.Push(ErrorEvent{
			Type:    "error",
			Reason:  StopReasonError,
			Message: *p.output,
		})
	}
}
