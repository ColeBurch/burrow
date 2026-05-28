package ai

import (
	"slices"
	"strings"
	"time"

	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
)

const NonVisionUserImagePlaceholder = "(image omitted: model does not support images)"
const NonVisionToolImagePlaceholder = "(tool image omitted: model does not support images)"

func ReplaceUserImagesWithPlaceholder(content []UserContent, placeholder string) []UserContent {
	result := make([]UserContent, 0, len(content))
	previousWasPlaceholder := false

	for _, item := range content {
		switch b := item.(type) {
		case ImageContent:
			if !previousWasPlaceholder {
				result = append(result, TextContent{Type: ContentTypeText, Text: placeholder})
			}
			previousWasPlaceholder = true
		case TextContent:
			result = append(result, item)
			previousWasPlaceholder = b.Text == placeholder
		default:
			result = append(result, item)
			previousWasPlaceholder = false
		}
	}
	return result
}

func ReplaceToolResultImagesWithPlaceholder(content []ToolResultContent, placeholder string) []ToolResultContent {
	result := make([]ToolResultContent, 0, len(content))
	previousWasPlaceholder := false

	for _, item := range content {
		switch b := item.(type) {
		case ImageContent:
			if !previousWasPlaceholder {
				result = append(result, TextContent{Type: ContentTypeText, Text: placeholder})
			}
			previousWasPlaceholder = true
		case TextContent:
			result = append(result, item)
			previousWasPlaceholder = b.Text == placeholder
		default:
			result = append(result, item)
			previousWasPlaceholder = false
		}
	}
	return result
}

func downgradeUnsupportedImages(messages []Message, model Model[API]) []Message {
	if slices.Contains(model.Input, "image") {
		return messages
	}

	result := make([]Message, len(messages))

	for i, msg := range messages {
		switch m := msg.(type) {
		case UserMessage:
			if m.Content != nil {
				m.Content = ReplaceUserImagesWithPlaceholder(
					m.Content,
					NonVisionUserImagePlaceholder,
				)
			}
			result[i] = m

		case ToolResultMessage:
			m.Content = ReplaceToolResultImagesWithPlaceholder(
				m.Content,
				NonVisionToolImagePlaceholder,
			)
			result[i] = m

		default:
			result[i] = msg
		}
	}

	return result
}

func TransformMessages(messages []Message, model Model[API], allowedToolCallProviders map[Provider]struct{}) []Message {
	var toolCallIdMap = make(map[string]string)
	imageAwareMessages := downgradeUnsupportedImages(messages, model)

	transformedMessages := make([]Message, 0, len(imageAwareMessages))

	// First pass: transform messages (unsupported image downgrade, thinking blocks, tool call ID normalization)
	for _, msg := range imageAwareMessages {
		switch r := msg.(type) {
		case UserMessage:
			transformedMessages = append(transformedMessages, msg)
		case ToolResultMessage:
			if normalizedID, ok := toolCallIdMap[r.ToolCallID]; ok && normalizedID != r.ToolCallID {
				r.ToolCallID = normalizedID
			}
			transformedMessages = append(transformedMessages, r)
		case AssistantMessage:
			assistantMsg := r

			isSameModel :=
				assistantMsg.Provider == model.Provider &&
					assistantMsg.Api == model.API &&
					assistantMsg.Model == model.ID

			newContent := make([]AssistantContent, 0, len(assistantMsg.Content))

			for _, block := range assistantMsg.Content {
				switch block.(type) {
				case ThinkingContent:
					if block.(ThinkingContent).Redacted != nil && *block.(ThinkingContent).Redacted {
						if isSameModel {
							newContent = append(newContent, block)
						}
						continue
					}

					if isSameModel && block.(ThinkingContent).ThinkingSignature != nil && *block.(ThinkingContent).ThinkingSignature != "" {
						newContent = append(newContent, block)
						continue
					}

					if strings.TrimSpace(block.(ThinkingContent).Thinking) == "" {
						continue
					}

					if isSameModel {
						newContent = append(newContent, block)
					} else {
						newContent = append(newContent, ThinkingContent{
							Type:     ContentTypeThinking,
							Thinking: block.(ThinkingContent).Thinking,
						})
					}

				case TextContent:
					newContent = append(newContent, TextContent{
						Type: ContentTypeText,
						Text: block.(TextContent).Text,
					})

				case ToolCall:
					toolCall := block.(ToolCall)

					if !isSameModel && toolCall.ThoughtSignature != nil && *toolCall.ThoughtSignature != "" {
						toolCall.ThoughtSignature = nil
					}

					if !isSameModel {
						normalizedID := normalizeToolCallID(toolCall.Id, model, assistantMsg, allowedToolCallProviders)
						if normalizedID != toolCall.Id {
							toolCallIdMap[toolCall.Id] = normalizedID
							toolCall.Id = normalizedID
						}
					}

					newContent = append(newContent, toolCall)

				default:
					newContent = append(newContent, block)
				}
			}

			assistantMsg.Content = newContent
			transformedMessages = append(transformedMessages, assistantMsg)
		}
	}

	// Second pass: insert synthetic empty tool results for orphaned tool calls
	// This preserves thinking signatures and satisfies API requirements
	result := []Message{}
	pendingToolCalls := []ToolCall{}
	existingToolResultIDs := map[string]bool{}

	insertSyntheticToolResults := func() {
		if len(pendingToolCalls) == 0 {
			return
		}

		for _, tc := range pendingToolCalls {
			if !existingToolResultIDs[tc.Id] {
				result = append(result, ToolResultMessage{
					Role:       RoleToolResult,
					ToolCallID: tc.Id,
					ToolName:   tc.Name,
					Content: []ToolResultContent{
						TextContent{
							Type: ContentTypeText,
							Text: "No result provided",
						},
					},
					IsError:   true,
					Timestamp: time.Now().UnixMilli(),
				})
			}
		}

		pendingToolCalls = nil
		existingToolResultIDs = map[string]bool{}
	}

	for _, msg := range transformedMessages {
		switch msg.(type) {
		case AssistantMessage:
			insertSyntheticToolResults()

			assistantMsg := msg.(AssistantMessage)
			if assistantMsg.StopReason == "error" || assistantMsg.StopReason == "aborted" {
				continue
			}

			toolCalls := []ToolCall{}
			for _, block := range assistantMsg.Content {
				if toolCall, ok := block.(ToolCall); ok && toolCall.Type == "toolCall" {
					toolCalls = append(toolCalls, toolCall)
				}
			}

			if len(toolCalls) > 0 {
				pendingToolCalls = toolCalls
				existingToolResultIDs = map[string]bool{}
			}

			result = append(result, msg)
		case ToolResultMessage:
			existingToolResultIDs[msg.(ToolResultMessage).ToolCallID] = true
			result = append(result, msg)
		case UserMessage:
			insertSyntheticToolResults()
			result = append(result, msg)
		default:
			result = append(result, msg)
		}
	}

	insertSyntheticToolResults()

	return result
}

type ConvertResponsesToolsOptions struct {
	Strict *bool
}

func ConvertResponsesTools(
	tools []*Tool,
	options *ConvertResponsesToolsOptions,
) ([]responses.ToolUnionParam, error) {
	strict := false
	if options != nil && options.Strict != nil {
		strict = *options.Strict
	}

	result := make([]responses.ToolUnionParam, 0, len(tools))

	for _, tool := range tools {
		result = append(result, responses.ToolUnionParam{
			OfFunction: &responses.FunctionToolParam{
				Name:        tool.Name,
				Description: param.NewOpt(tool.Description),
				Parameters:  tool.Parameters,
				Strict:      param.NewOpt(strict),
			},
		})
	}

	return result, nil
}
