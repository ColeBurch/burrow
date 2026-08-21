package ai

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"testing"
	"time"

	"github.com/govalues/decimal"
	"github.com/joho/godotenv"
	"github.com/openai/openai-go/v3/responses"
)

func testResponsesModel() Model[API] {
	return Model[API]{
		ID:       "gpt-4.1-mini",
		API:      APIOpenAIResponses,
		Provider: ProviderOpenAI,
	}
}

func newTestResponsesStreamProcessor(model Model[API]) (*responsesStreamProcessor, *AssistantMessageEventStream) {
	output := AssistantMessage{
		Role:       RoleAssistant,
		Content:    []AssistantContent{},
		Api:        model.API,
		Provider:   model.Provider,
		Model:      model.ID,
		StopReason: StopReasonStop,
		Timestamp:  time.Now().UnixMilli(),
	}

	stream := NewAssistantMessageEventStream(64)
	return &responsesStreamProcessor{
		output:              &output,
		stream:              stream,
		currentContentIndex: -1,
	}, stream
}

func collectAvailableEventTypes(stream *AssistantMessageEventStream) []string {
	var eventTypes []string
	for {
		select {
		case event, ok := <-stream.Events():
			if !ok {
				return eventTypes
			}
			eventTypes = append(eventTypes, event.EventType())
		default:
			return eventTypes
		}
	}
}

func assertEventTypes(t *testing.T, stream *AssistantMessageEventStream, wantTypes ...string) {
	t.Helper()
	eventTypes := collectAvailableEventTypes(stream)
	if len(eventTypes) != len(wantTypes) {
		t.Fatalf("event types length = %d, want %d: %#v", len(eventTypes), len(wantTypes), eventTypes)
	}
	for i := range wantTypes {
		if eventTypes[i] != wantTypes[i] {
			t.Fatalf("eventTypes[%d] = %q, want %q; all events: %#v", i, eventTypes[i], wantTypes[i], eventTypes)
		}
	}
}

func requireToolArgsMap(t *testing.T, args any) map[string]any {
	t.Helper()
	argsMap, ok := args.(map[string]any)
	if !ok {
		t.Fatalf("tool args = %T, want map[string]any", args)
	}
	return argsMap
}

func TestResponsesStreamProcessorPartialEventsAreSafeForConcurrentReaders(t *testing.T) {
	model := testResponsesModel()
	processor, stream := newTestResponsesStreamProcessor(model)
	readerStarted := make(chan struct{})
	producerDone := make(chan struct{})
	readsDone := make(chan int64, 1)

	go func() {
		for event := range stream.Events() {
			delta, ok := event.(TextDeltaEvent)
			if !ok {
				continue
			}

			partial := delta.Partial
			close(readerStarted)

			var reads int64
			for {
				select {
				case <-producerDone:
					readsDone <- reads
					return
				default:
					if len(partial.Content) > 0 {
						if text, ok := partial.Content[0].(TextContent); ok {
							reads += int64(len(text.Text))
						}
					}
					runtime.Gosched()
				}
			}
		}
	}()

	processor.ProcessResponsesStream(responses.ResponseStreamEventUnion{
		Type: "response.output_item.added",
		Item: responses.ResponseOutputItemUnion{Type: "message"},
	}, model, nil)
	processor.ProcessResponsesStream(responses.ResponseStreamEventUnion{
		Type: "response.content_part.added",
		Part: responses.ResponseStreamEventUnionPart{Type: "output_text"},
	}, model, nil)
	processor.ProcessResponsesStream(responses.ResponseStreamEventUnion{
		Type:  "response.output_text.delta",
		Delta: "a",
	}, model, nil)

	select {
	case <-readerStarted:
	case <-time.After(time.Second):
		t.Fatal("the partial event reader did not start")
	}

	for range 32 {
		processor.ProcessResponsesStream(responses.ResponseStreamEventUnion{
			Type:  "response.output_text.delta",
			Delta: "b",
		}, model, nil)
	}

	close(producerDone)
	stream.End(nil)

	select {
	case reads := <-readsDone:
		if reads == 0 {
			t.Fatal("the consumer did not read the partial message")
		}
	case <-time.After(time.Second):
		t.Fatal("the partial event reader did not stop")
	}
}

func TestResponsesStreamProcessorTextEvents(t *testing.T) {
	model := Model[API]{
		ID:       "gpt-4.1-mini",
		API:      APIOpenAIResponses,
		Provider: ProviderOpenAI,
	}

	output := AssistantMessage{
		Role:       RoleAssistant,
		Content:    []AssistantContent{},
		Api:        model.API,
		Provider:   model.Provider,
		Model:      model.ID,
		StopReason: StopReasonStop,
		Timestamp:  time.Now().UnixMilli(),
	}

	stream := NewAssistantMessageEventStream(16)
	processor := &responsesStreamProcessor{
		output:              &output,
		stream:              stream,
		currentContentIndex: -1,
	}

	processor.ProcessResponsesStream(responses.ResponseStreamEventUnion{
		Type: "response.output_item.added",
		Item: responses.ResponseOutputItemUnion{
			ID:   "msg_1",
			Type: "message",
		},
	}, model, nil)
	processor.ProcessResponsesStream(responses.ResponseStreamEventUnion{
		Type: "response.content_part.added",
		Part: responses.ResponseStreamEventUnionPart{
			Type: "output_text",
		},
	}, model, nil)
	processor.ProcessResponsesStream(responses.ResponseStreamEventUnion{
		Type:  "response.output_text.delta",
		Delta: "hel",
	}, model, nil)
	processor.ProcessResponsesStream(responses.ResponseStreamEventUnion{
		Type:  "response.output_text.delta",
		Delta: "lo",
	}, model, nil)
	processor.ProcessResponsesStream(responses.ResponseStreamEventUnion{
		Type: "response.output_item.done",
		Item: responses.ResponseOutputItemUnion{
			ID:   "msg_1",
			Type: "message",
			Content: []responses.ResponseOutputMessageContentUnion{
				{Type: "output_text", Text: "hello"},
			},
		},
	}, model, nil)
	processor.ProcessResponsesStream(responses.ResponseStreamEventUnion{
		Type: "response.completed",
		Response: responses.Response{
			ID:     "resp_1",
			Status: responses.ResponseStatusCompleted,
			Usage: responses.ResponseUsage{
				InputTokens:  3,
				OutputTokens: 2,
				TotalTokens:  5,
			},
		},
	}, model, nil)

	var eventTypes []string
	for event := range stream.Events() {
		eventTypes = append(eventTypes, event.EventType())
	}

	wantTypes := []string{"text_start", "text_delta", "text_delta", "text_end", "done"}
	if len(eventTypes) != len(wantTypes) {
		t.Fatalf("event types length = %d, want %d: %#v", len(eventTypes), len(wantTypes), eventTypes)
	}
	for i := range wantTypes {
		if eventTypes[i] != wantTypes[i] {
			t.Fatalf("eventTypes[%d] = %q, want %q; all events: %#v", i, eventTypes[i], wantTypes[i], eventTypes)
		}
	}

	msg, err := stream.Result()
	if err != nil {
		t.Fatal(err)
	}
	if msg.ResponseID == nil || *msg.ResponseID != "resp_1" {
		t.Fatalf("ResponseID = %v, want resp_1", msg.ResponseID)
	}
	if msg.StopReason != StopReasonStop {
		t.Fatalf("StopReason = %q, want %q", msg.StopReason, StopReasonStop)
	}
	if len(msg.Content) != 1 {
		t.Fatalf("len(Content) = %d, want 1", len(msg.Content))
	}
	text, ok := msg.Content[0].(TextContent)
	if !ok {
		t.Fatalf("Content[0] = %T, want TextContent", msg.Content[0])
	}
	if text.Text != "hello" {
		t.Fatalf("text = %q, want hello", text.Text)
	}
}

func TestStreamOpenAIResponseNilOptionsReturnsSetupError(t *testing.T) {
	_, err := StreamOpenAIResponse(context.Background(), testResponsesModel(), ModelContext{}, nil)
	if err == nil || err.Error() != "options are required" {
		t.Fatalf("err = %v, want options are required", err)
	}
}

func TestResponsesStreamProcessorCreatedEventSetsResponseID(t *testing.T) {
	model := testResponsesModel()
	processor, stream := newTestResponsesStreamProcessor(model)

	processor.ProcessResponsesStream(responses.ResponseStreamEventUnion{
		Type:     "response.created",
		Response: responses.Response{ID: "resp_created"},
	}, model, nil)

	assertEventTypes(t, stream)
	if processor.output.ResponseID == nil || *processor.output.ResponseID != "resp_created" {
		t.Fatalf("ResponseID = %v, want resp_created", processor.output.ResponseID)
	}
}

func TestResponsesStreamProcessorReasoningEvents(t *testing.T) {
	model := testResponsesModel()
	processor, stream := newTestResponsesStreamProcessor(model)

	processor.ProcessResponsesStream(responses.ResponseStreamEventUnion{
		Type: "response.output_item.added",
		Item: responses.ResponseOutputItemUnion{ID: "rs_1", Type: "reasoning"},
	}, model, nil)
	processor.ProcessResponsesStream(responses.ResponseStreamEventUnion{
		Type: "response.reasoning_summary_part.added",
		Part: responses.ResponseStreamEventUnionPart{Type: "summary_text"},
	}, model, nil)
	processor.ProcessResponsesStream(responses.ResponseStreamEventUnion{
		Type:  "response.reasoning_summary_text.delta",
		Delta: "think",
	}, model, nil)
	processor.ProcessResponsesStream(responses.ResponseStreamEventUnion{
		Type: "response.reasoning_summary_part.done",
	}, model, nil)
	processor.ProcessResponsesStream(responses.ResponseStreamEventUnion{
		Type: "response.output_item.done",
		Item: responses.ResponseOutputItemUnion{
			ID:      "rs_1",
			Type:    "reasoning",
			Summary: []responses.ResponseReasoningItemSummary{{Text: "final one"}, {Text: "final two"}},
		},
	}, model, nil)

	assertEventTypes(t, stream, "thinking_start", "thinking_delta", "thinking_delta", "thinking_end")
	thinking, ok := processor.output.Content[0].(ThinkingContent)
	if !ok {
		t.Fatalf("Content[0] = %T, want ThinkingContent", processor.output.Content[0])
	}
	if thinking.Thinking != "final one\n\nfinal two" {
		t.Fatalf("thinking = %q", thinking.Thinking)
	}
	if thinking.ThinkingSignature == nil || *thinking.ThinkingSignature == "" {
		t.Fatal("expected thinking signature")
	}
}

func TestResponsesStreamProcessorRefusalEvents(t *testing.T) {
	model := testResponsesModel()
	processor, stream := newTestResponsesStreamProcessor(model)

	processor.ProcessResponsesStream(responses.ResponseStreamEventUnion{
		Type: "response.output_item.added",
		Item: responses.ResponseOutputItemUnion{ID: "msg_1", Type: "message"},
	}, model, nil)
	processor.ProcessResponsesStream(responses.ResponseStreamEventUnion{
		Type: "response.content_part.added",
		Part: responses.ResponseStreamEventUnionPart{Type: "refusal"},
	}, model, nil)
	processor.ProcessResponsesStream(responses.ResponseStreamEventUnion{
		Type:  "response.refusal.delta",
		Delta: "no",
	}, model, nil)
	processor.ProcessResponsesStream(responses.ResponseStreamEventUnion{
		Type: "response.output_item.done",
		Item: responses.ResponseOutputItemUnion{
			ID:      "msg_1",
			Type:    "message",
			Content: []responses.ResponseOutputMessageContentUnion{{Type: "refusal", Refusal: "no"}},
		},
	}, model, nil)

	assertEventTypes(t, stream, "text_start", "text_delta", "text_end")
	text := processor.output.Content[0].(TextContent)
	if text.Text != "no" {
		t.Fatalf("text = %q, want no", text.Text)
	}
}

func TestResponsesStreamProcessorToolCallEvents(t *testing.T) {
	model := testResponsesModel()
	processor, stream := newTestResponsesStreamProcessor(model)

	processor.ProcessResponsesStream(responses.ResponseStreamEventUnion{
		Type: "response.output_item.added",
		Item: responses.ResponseOutputItemUnion{ID: "fc_1", Type: "function_call", CallID: "call_1", Name: "lookup"},
	}, model, nil)
	processor.ProcessResponsesStream(responses.ResponseStreamEventUnion{
		Type:  "response.function_call_arguments.delta",
		Delta: "{\"city\"",
	}, model, nil)
	processor.ProcessResponsesStream(responses.ResponseStreamEventUnion{
		Type:      "response.function_call_arguments.done",
		Arguments: "{\"city\":\"Paris\"}",
	}, model, nil)
	processor.ProcessResponsesStream(responses.ResponseStreamEventUnion{
		Type: "response.output_item.done",
		Item: responses.ResponseOutputItemUnion{ID: "fc_1", Type: "function_call", CallID: "call_1", Name: "lookup"},
	}, model, nil)

	assertEventTypes(t, stream, "toolcall_start", "toolcall_delta", "toolcall_delta", "toolcall_end")
	tool, ok := processor.output.Content[0].(ToolCall)
	if !ok {
		t.Fatalf("Content[0] = %T, want ToolCall", processor.output.Content[0])
	}
	args := requireToolArgsMap(t, tool.Args)
	if tool.Id != "call_1|fc_1" || tool.Name != "lookup" || tool.PartialJson != nil || args["city"] != "Paris" {
		t.Fatalf("tool call = %+v", tool)
	}
}

func TestResponsesStreamProcessorToolCallDoneFallbackAndNoDuplicateDelta(t *testing.T) {
	model := testResponsesModel()
	processor, stream := newTestResponsesStreamProcessor(model)

	processor.ProcessResponsesStream(responses.ResponseStreamEventUnion{
		Type: "response.output_item.done",
		Item: responses.ResponseOutputItemUnion{ID: "fc_2", Type: "function_call", CallID: "call_2", Name: "sum", Arguments: responses.ResponseOutputItemUnionArguments{OfString: "{\"n\":2}"}},
	}, model, nil)

	assertEventTypes(t, stream, "toolcall_end")
	tool := processor.output.Content[0].(ToolCall)
	args := requireToolArgsMap(t, tool.Args)
	if tool.Id != "call_2|fc_2" || args["n"] != float64(2) {
		t.Fatalf("tool call = %+v", tool)
	}

	processor, stream = newTestResponsesStreamProcessor(model)
	processor.ProcessResponsesStream(responses.ResponseStreamEventUnion{
		Type: "response.output_item.added",
		Item: responses.ResponseOutputItemUnion{ID: "fc_3", Type: "function_call", CallID: "call_3", Name: "noop"},
	}, model, nil)
	processor.ProcessResponsesStream(responses.ResponseStreamEventUnion{
		Type:      "response.function_call_arguments.done",
		Arguments: "{}",
	}, model, nil)
	assertEventTypes(t, stream, "toolcall_start", "toolcall_delta")
}

func TestResponsesStreamProcessorCompletedEvent(t *testing.T) {
	model := testResponsesModel()
	model.Cost.Input = decimal.MustParse("1")
	model.Cost.Output = decimal.MustParse("2")
	model.Cost.CacheRead = decimal.MustParse("0.5")
	processor, stream := newTestResponsesStreamProcessor(model)
	processor.output.Content = append(processor.output.Content, ToolCall{Type: ContentTypeToolCall, Id: "call|item", Name: "tool", Args: map[string]any{}})

	processor.ProcessResponsesStream(responses.ResponseStreamEventUnion{
		Type: "response.completed",
		Response: responses.Response{
			ID:     "resp_done",
			Status: responses.ResponseStatusCompleted,
			Usage: responses.ResponseUsage{
				InputTokens:        10,
				OutputTokens:       4,
				TotalTokens:        14,
				InputTokensDetails: responses.ResponseUsageInputTokensDetails{CachedTokens: 3},
			},
		},
	}, model, nil)

	assertEventTypes(t, stream, "done")
	msg, err := stream.Result()
	if err != nil {
		t.Fatal(err)
	}
	if msg.ResponseID == nil || *msg.ResponseID != "resp_done" {
		t.Fatalf("ResponseID = %v, want resp_done", msg.ResponseID)
	}
	if msg.StopReason != StopReasonToolUse {
		t.Fatalf("StopReason = %q, want tool use", msg.StopReason)
	}
	if msg.Usage.Input != 7 || msg.Usage.CacheRead != 3 || msg.Usage.Output != 4 || msg.Usage.Total != 14 {
		t.Fatalf("Usage = %+v", msg.Usage)
	}
}

func TestResponsesStreamProcessorCompletedStopReasons(t *testing.T) {
	model := testResponsesModel()
	tests := []struct {
		status responses.ResponseStatus
		want   StopReason
	}{
		{responses.ResponseStatusIncomplete, StopReasonLength},
		{responses.ResponseStatusFailed, StopReasonError},
		{responses.ResponseStatusCancelled, StopReasonAborted},
		{responses.ResponseStatusInProgress, StopReasonStop},
		{responses.ResponseStatusQueued, StopReasonStop},
		{responses.ResponseStatus("mystery"), StopReasonError},
	}

	for _, tt := range tests {
		t.Run(string(tt.status), func(t *testing.T) {
			processor, stream := newTestResponsesStreamProcessor(model)
			processor.ProcessResponsesStream(responses.ResponseStreamEventUnion{
				Type:     "response.completed",
				Response: responses.Response{Status: tt.status},
			}, model, nil)
			assertEventTypes(t, stream, "done")
			msg, err := stream.Result()
			if err != nil {
				t.Fatal(err)
			}
			if msg.StopReason != tt.want {
				t.Fatalf("StopReason = %q, want %q", msg.StopReason, tt.want)
			}
		})
	}
}

func TestResponsesStreamProcessorErrorEvents(t *testing.T) {
	model := testResponsesModel()
	tests := []struct {
		name  string
		event responses.ResponseStreamEventUnion
		want  string
	}{
		{name: "error with details", event: responses.ResponseStreamEventUnion{Type: "error", Code: "bad", Message: "broken"}, want: "Error Code bad: broken"},
		{name: "unknown error", event: responses.ResponseStreamEventUnion{Type: "error"}, want: "Unknown error"},
		{name: "failed details", event: responses.ResponseStreamEventUnion{Type: "response.failed", Response: responses.Response{Error: responses.ResponseError{Code: "server_error", Message: "down"}}}, want: "server_error: down"},
		{name: "failed incomplete", event: responses.ResponseStreamEventUnion{Type: "response.failed", Response: responses.Response{IncompleteDetails: responses.ResponseIncompleteDetails{Reason: "max_output_tokens"}}}, want: "incomplete: max_output_tokens"},
		{name: "failed unknown", event: responses.ResponseStreamEventUnion{Type: "response.failed"}, want: "Unknown error (no error details in response)"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			processor, stream := newTestResponsesStreamProcessor(model)
			processor.ProcessResponsesStream(tt.event, model, nil)
			assertEventTypes(t, stream, "error")
			msg, err := stream.Result()
			if err == nil || err.Error() != tt.want {
				t.Fatalf("err = %v, want %q", err, tt.want)
			}
			if msg.StopReason != StopReasonError || msg.ErrorMessage != tt.want {
				t.Fatalf("Result message = %+v, want error final message", msg)
			}
			if processor.output.StopReason != StopReasonError || processor.output.ErrorMessage != tt.want {
				t.Fatalf("output = %+v", processor.output)
			}
		})
	}
}

func TestResponsesStreamProcessorIgnoresInvalidStateEvents(t *testing.T) {
	model := testResponsesModel()
	processor, stream := newTestResponsesStreamProcessor(model)

	ignoredEvents := []responses.ResponseStreamEventUnion{
		{Type: "response.reasoning_summary_part.added", Part: responses.ResponseStreamEventUnionPart{Type: "summary_text"}},
		{Type: "response.reasoning_summary_text.delta", Delta: "ignored"},
		{Type: "response.reasoning_summary_part.done"},
		{Type: "response.content_part.added", Part: responses.ResponseStreamEventUnionPart{Type: "output_text"}},
		{Type: "response.output_text.delta", Delta: "ignored"},
		{Type: "response.refusal.delta", Delta: "ignored"},
		{Type: "response.function_call_arguments.delta", Delta: "ignored"},
		{Type: "response.function_call_arguments.done", Arguments: "{}"},
		{Type: "response.output_item.done", Item: responses.ResponseOutputItemUnion{Type: "reasoning"}},
		{Type: "response.output_item.done", Item: responses.ResponseOutputItemUnion{Type: "message"}},
	}
	for _, event := range ignoredEvents {
		processor.ProcessResponsesStream(event, model, nil)
	}

	assertEventTypes(t, stream)
	if len(processor.output.Content) != 0 {
		t.Fatalf("Content = %#v, want empty", processor.output.Content)
	}

	processor.ProcessResponsesStream(responses.ResponseStreamEventUnion{Type: "response.output_item.added", Item: responses.ResponseOutputItemUnion{Type: "message"}}, model, nil)
	processor.ProcessResponsesStream(responses.ResponseStreamEventUnion{Type: "response.output_text.delta", Delta: "ignored without part"}, model, nil)
	processor.ProcessResponsesStream(responses.ResponseStreamEventUnion{Type: "response.content_part.added", Part: responses.ResponseStreamEventUnionPart{Type: "refusal"}}, model, nil)
	processor.ProcessResponsesStream(responses.ResponseStreamEventUnion{Type: "response.output_text.delta", Delta: "ignored wrong part"}, model, nil)
	assertEventTypes(t, stream, "text_start")
	if processor.output.Content[0].(TextContent).Text != "" {
		t.Fatalf("text = %q, want empty", processor.output.Content[0].(TextContent).Text)
	}
}

func TestStreamOpenAIResponseToolCallLive(t *testing.T) {
	err := godotenv.Load("../../.env")
	if err != nil {
		fmt.Println("Error loading .env file", err)
	}
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Skip("OPENAI_API_KEY not set")
	}

	maxTokens := int64(1000)
	cache := CacheRetentionNone
	xhigh := "xhigh"

	model := Model[API]{
		ID:               "gpt-5.4-nano",
		API:              APIOpenAIResponses,
		Provider:         ProviderOpenAI,
		BaseURL:          "https://api.openai.com/v1",
		Reasoning:        true,
		ThinkingLevelMap: ThinkingLevelMap{ThinkingLevelOff: nil, ThinkingLevelXHigh: &xhigh},
		Input:            []string{"text", "image"},
		Cost: Cost{
			Input:      decimal.MustParse("0.2"),
			Output:     decimal.MustParse("1.25"),
			CacheRead:  decimal.MustParse("0.02"),
			CacheWrite: decimal.Zero,
		},
		ContextWindow: 400000,
		MaxTokens:     128000,
	}

	modelContext := ModelContext{
		Messages: []Message{
			UserMessage{
				Role: RoleUser,
				Content: []UserContent{
					TextContent{Type: ContentTypeText, Text: "Use the getweather tool to get the weather in Cupertino, California. Do not answer from memory; call the tool."},
				},
			},
		},
		Tools: []*Tool{
			{
				Name:        "getweather",
				Description: "Get the current weather for a location.",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"city": map[string]any{
							"type":        "string",
							"description": "The city to get weather for.",
						},
						"state": map[string]any{
							"type":        "string",
							"description": "The US state to get weather for.",
						},
					},
					"required":             []string{"city", "state"},
					"additionalProperties": false,
				},
			},
		},
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	stream, err := StreamOpenAIResponse(ctx, model, modelContext, &OpenAIResponseOptions{
		StreamOptions: StreamOptions{
			ApiKey:    &apiKey,
			Cache:     &cache,
			MaxTokens: &maxTokens,
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	for event := range stream.Events() {
		t.Logf("event: %T %s", event, event.EventType())
	}

	msg, err := stream.Result()
	if err != nil {
		t.Fatal(err)
	}
	if msg.StopReason != StopReasonToolUse {
		t.Fatalf("StopReason = %q, want %q; message: %+v", msg.StopReason, StopReasonToolUse, msg)
	}

	var weatherToolCall *ToolCall
	for _, content := range msg.Content {
		if toolCall, ok := content.(ToolCall); ok && toolCall.Name == "getweather" {
			weatherToolCall = &toolCall
			break
		}
	}
	if weatherToolCall == nil {
		t.Fatalf("expected getweather tool call in final message: %+v", msg)
	}
	args := requireToolArgsMap(t, weatherToolCall.Args)
	if args["city"] != "Cupertino" || args["state"] != "California" {
		t.Fatalf("unexpected getweather args: %+v", weatherToolCall.Args)
	}
	t.Logf("final message: %+v", msg)
}

func TestStreamOpenAIResponseLive(t *testing.T) {
	err := godotenv.Load("../../.env")
	if err != nil {
		fmt.Println("Error loading .env file", err)
	}
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Skip("OPENAI_API_KEY not set")
	}

	maxTokens := int64(1000)
	cache := CacheRetentionNone
	xhigh := "xhigh"

	model := Model[API]{
		ID:               "gpt-5.4-nano",
		API:              APIOpenAIResponses,
		Provider:         ProviderOpenAI,
		BaseURL:          "https://api.openai.com/v1",
		Reasoning:        true,
		ThinkingLevelMap: ThinkingLevelMap{ThinkingLevelOff: nil, ThinkingLevelXHigh: &xhigh},
		Input:            []string{"text", "image"},
		Cost: Cost{
			Input:      decimal.MustParse("0.2"),
			Output:     decimal.MustParse("1.25"),
			CacheRead:  decimal.MustParse("0.02"),
			CacheWrite: decimal.Zero,
		},
		ContextWindow: 400000,
		MaxTokens:     128000,
	}

	modelContext := ModelContext{
		Messages: []Message{
			UserMessage{
				Role: RoleUser,
				Content: []UserContent{
					TextContent{Type: ContentTypeText, Text: "Say exactly: hello I am a chatbot"},
				},
			},
		},
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	stream, err := StreamOpenAIResponse(ctx, model, modelContext, &OpenAIResponseOptions{
		StreamOptions: StreamOptions{
			ApiKey:    &apiKey,
			Cache:     &cache,
			MaxTokens: &maxTokens,
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	for event := range stream.Events() {
		t.Logf("event: %T %s", event, event.EventType())
	}

	msg, err := stream.Result()
	if err != nil {
		t.Fatal(err)
	}
	if len(msg.Content) == 0 {
		t.Fatal("expected assistant content")
	}
	t.Logf("final message: %+v", msg)
}
