package ai

import "testing"

func testTransformModel() Model[API] {
	return Model[API]{
		ID:       "gpt-test",
		API:      APIOpenAIResponses,
		Provider: ProviderOpenAI,
		Input:    []string{"text"},
	}
}

func TestTransformMessagesPreservesAssistantMessages(t *testing.T) {
	model := testTransformModel()
	messages := []Message{
		UserMessage{
			Role: RoleUser,
			Content: []UserContent{
				TextContent{Type: ContentTypeText, Text: "hello"},
			},
		},
		AssistantMessage{
			Role:       RoleAssistant,
			Api:        model.API,
			Provider:   model.Provider,
			Model:      model.ID,
			Content:    []AssistantContent{TextContent{Type: ContentTypeText, Text: "hi there"}},
			StopReason: StopReasonStop,
		},
	}

	got := TransformMessages(messages, model, nil)
	if len(got) != 2 {
		t.Fatalf("TransformMessages returned %d messages, want 2: %#v", len(got), got)
	}

	assistantMsg, ok := got[1].(AssistantMessage)
	if !ok {
		t.Fatalf("got[1] = %T, want AssistantMessage", got[1])
	}
	if len(assistantMsg.Content) != 1 {
		t.Fatalf("assistant content length = %d, want 1", len(assistantMsg.Content))
	}
	text, ok := assistantMsg.Content[0].(TextContent)
	if !ok {
		t.Fatalf("assistant content[0] = %T, want TextContent", assistantMsg.Content[0])
	}
	if text.Text != "hi there" {
		t.Fatalf("assistant text = %q, want %q", text.Text, "hi there")
	}
}

func TestTransformMessagesDoesNotPanicOnNonToolAssistantContentWhenFindingToolCalls(t *testing.T) {
	model := testTransformModel()
	messages := []Message{
		AssistantMessage{
			Role:     RoleAssistant,
			Api:      model.API,
			Provider: model.Provider,
			Model:    model.ID,
			Content: []AssistantContent{
				TextContent{Type: ContentTypeText, Text: "I'll call a tool."},
				ToolCall{Type: ContentTypeToolCall, Id: "call-1", Name: "lookup", Args: map[string]any{}},
			},
			StopReason: StopReasonToolUse,
		},
		UserMessage{
			Role: RoleUser,
			Content: []UserContent{
				TextContent{Type: ContentTypeText, Text: "next"},
			},
		},
	}

	got := TransformMessages(messages, model, nil)
	if len(got) != 3 {
		t.Fatalf("TransformMessages returned %d messages, want assistant, synthetic tool result, user: %#v", len(got), got)
	}
	if _, ok := got[0].(AssistantMessage); !ok {
		t.Fatalf("got[0] = %T, want AssistantMessage", got[0])
	}
	toolResult, ok := got[1].(ToolResultMessage)
	if !ok {
		t.Fatalf("got[1] = %T, want synthetic ToolResultMessage", got[1])
	}
	if toolResult.Role != RoleToolResult {
		t.Fatalf("synthetic tool result role = %q, want %q", toolResult.Role, RoleToolResult)
	}
	if toolResult.ToolCallID != "call-1" || toolResult.ToolName != "lookup" || !toolResult.IsError {
		t.Fatalf("synthetic tool result = %#v, want call-1 lookup error result", toolResult)
	}
	if _, ok := got[2].(UserMessage); !ok {
		t.Fatalf("got[2] = %T, want UserMessage", got[2])
	}
}
