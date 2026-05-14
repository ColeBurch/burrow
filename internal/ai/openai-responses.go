package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"hash/fnv"
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

func StreamOpenAIResponse(ctx context.Context, model Model[API], modelContext ModelContext, options *OpenAIResponseOptions) (*AssistantMessageEventStream, error) {
	stream := NewAssistantMessageEventStream(16)

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

	apiKey := options.ApiKey
	if apiKey == nil {
		return nil, fmt.Errorf("apiKey is required")
	}
	var cacheSessionId *string
	cacheRetention := ResolveCacheRetention(*options.Cache)
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

	openaiStream := client.Responses.NewStreaming(ctx, params)
	stream.Push(StartEvent{Type: "start", Partial: output})

	for openaiStream.Next() {
		_ = openaiStream.Current()

	}

	if openaiStream.Err() != nil {
		panic(openaiStream.Err())
	}

	return stream, nil
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

func ProcessResponsesStream(
	openAIStream responses.ResponseStreamEventUnion,
	output AssistantMessage,
	stream AssistantMessageEventStream,
	model Model[API],
	options *OpenAIResponseOptions,
) {

}
