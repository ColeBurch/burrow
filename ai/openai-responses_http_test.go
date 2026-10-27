package ai_test

import (
	"bytes"
	"context"
	"errors"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"testing"
	"testing/synctest"
	"time"

	"github.com/ColeBurch/burrow/ai"
	"github.com/ColeBurch/burrow/ai/providertest"
	oair "github.com/ColeBurch/burrow/ai/providertest/openairesponses"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
)

var weatherTool = &ai.Tool{
	Name:        "get_weather",
	Description: "Get the current weather for a city.",
	Parameters: map[string]any{
		"type":                 "object",
		"properties":           map[string]any{"city": map[string]any{"type": "string"}},
		"required":             []string{"city"},
		"additionalProperties": false,
	},
}

func TestStreamOpenAIResponseOverHTTPStreamsText(t *testing.T) {
	server := providertest.NewServer(t, oair.Reply(
		oair.Text("msg_1", "hello"),
		oair.Completed("resp_1", oair.Usage{Input: 3, Output: 2}),
	))
	model := oair.Model()
	sessionID := "session-1"

	eventTypes, msg, err := streamFromServer(t, t.Context(), server, model, ai.ModelContext{
		Messages: []ai.Message{userText("hi")},
	}, &ai.OpenAIResponseOptions{
		StreamOptions: ai.StreamOptions{
			SessionId: &sessionID,
			Headers:   map[string]string{"x-extra": "1"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	wantTypes := []string{"start", "text_start", "text_delta", "text_end", "done"}
	if !slices.Equal(eventTypes, wantTypes) {
		t.Fatalf("event types = %#v, want %#v", eventTypes, wantTypes)
	}
	if msg.ResponseID == nil || *msg.ResponseID != "resp_1" {
		t.Fatalf("ResponseID = %v, want resp_1", msg.ResponseID)
	}
	if len(msg.Content) != 1 {
		t.Fatalf("len(Content) = %d, want 1", len(msg.Content))
	}
	if text, ok := msg.Content[0].(ai.TextContent); !ok || text.Text != "hello" {
		t.Fatalf("Content[0] = %#v, want TextContent hello", msg.Content[0])
	}
	if msg.Usage.Input != 3 || msg.Usage.Output != 2 || msg.Usage.Total != 5 {
		t.Fatalf("Usage = %+v", msg.Usage)
	}

	requests := server.Requests()
	if len(requests) != 1 {
		t.Fatalf("got %d requests, want 1", len(requests))
	}
	req := requests[0]
	if req.Method != http.MethodPost || req.Path != "/v1/responses" {
		t.Fatalf("request = %s %s, want POST /v1/responses", req.Method, req.Path)
	}
	for header, want := range map[string]string{
		"Authorization":       "Bearer test-key",
		"X-Client-Request-Id": sessionID,
		"X-Extra":             "1",
	} {
		if got := req.Header.Get(header); got != want {
			t.Errorf("header %s = %q, want %q", header, got, want)
		}
	}
	body := req.JSON(t)
	for field, want := range map[string]any{
		"model":            model.ID,
		"stream":           true,
		"store":            false,
		"prompt_cache_key": sessionID,
	} {
		if got := body[field]; got != want {
			t.Errorf("body %s = %#v, want %#v", field, got, want)
		}
	}
}

func TestStreamOpenAIResponseOverHTTPStreamsToolCall(t *testing.T) {
	server := providertest.NewServer(t, oair.Reply(
		oair.ToolCall("call_1", "fc_1", "get_weather", `{"city":"Paris"}`),
		oair.Completed("resp_1", oair.Usage{Input: 5, Output: 3}),
	))

	eventTypes, msg, err := streamFromServer(t, t.Context(), server, oair.Model(), ai.ModelContext{
		Messages: []ai.Message{userText("Weather in Paris?")},
		Tools:    []*ai.Tool{weatherTool},
	}, nil)
	if err != nil {
		t.Fatal(err)
	}

	wantTypes := []string{"start", "toolcall_start", "toolcall_delta", "toolcall_end", "done"}
	if !slices.Equal(eventTypes, wantTypes) {
		t.Fatalf("event types = %#v, want %#v", eventTypes, wantTypes)
	}
	if msg.StopReason != ai.StopReasonToolUse {
		t.Errorf("StopReason = %q, want %q", msg.StopReason, ai.StopReasonToolUse)
	}
	if len(msg.Content) != 1 {
		t.Fatalf("len(Content) = %d, want 1", len(msg.Content))
	}
	call, ok := msg.Content[0].(ai.ToolCall)
	if !ok {
		t.Fatalf("Content[0] = %T, want ToolCall", msg.Content[0])
	}
	if call.Id != "call_1|fc_1" || call.Name != "get_weather" {
		t.Errorf("tool call = %s %q, want get_weather %q", call.Name, call.Id, "call_1|fc_1")
	}
	if args, ok := call.Args.(map[string]any); !ok || args["city"] != "Paris" {
		t.Errorf("tool call args = %#v, want city Paris", call.Args)
	}

	tools, ok := server.Requests()[0].JSON(t)["tools"].([]any)
	if !ok || len(tools) != 1 {
		t.Fatalf("body tools = %#v, want one tool", tools)
	}
	tool, _ := tools[0].(map[string]any)
	if tool["type"] != "function" || tool["name"] != "get_weather" {
		t.Errorf("body tool = %#v, want function get_weather", tool)
	}
	if params, _ := tool["parameters"].(map[string]any); params["required"] == nil {
		t.Errorf("body tool parameters = %#v, want the tool's schema", tool["parameters"])
	}
}

// The reply to a tool call must name it by call ID, and the call itself must
// be replayed with the item ID the API assigned.
func TestStreamOpenAIResponseOverHTTPRoundTripsToolCall(t *testing.T) {
	server := providertest.NewServer(t, providertest.Sequence(t,
		oair.Reply(
			oair.ToolCall("call_1", "fc_1", "get_weather", `{"city":"Paris"}`),
			oair.Completed("resp_1", oair.Usage{Input: 5, Output: 3}),
		),
		oair.Reply(
			oair.Text("msg_2", "It is sunny in Paris."),
			oair.Completed("resp_2", oair.Usage{Input: 9, Output: 6}),
		),
	))
	model := oair.Model()
	modelContext := ai.ModelContext{
		Messages: []ai.Message{userText("Weather in Paris?")},
		Tools:    []*ai.Tool{weatherTool},
	}

	_, first, err := streamFromServer(t, t.Context(), server, model, modelContext, nil)
	if err != nil {
		t.Fatal(err)
	}
	call, ok := first.Content[0].(ai.ToolCall)
	if !ok {
		t.Fatalf("first reply Content[0] = %T, want ToolCall", first.Content[0])
	}
	modelContext.Messages = append(modelContext.Messages, first, ai.ToolResultMessage{
		Role:       ai.RoleToolResult,
		ToolCallID: call.Id,
		ToolName:   call.Name,
		Content:    []ai.ToolResultContent{ai.TextContent{Type: ai.ContentTypeText, Text: "sunny"}},
	})
	if _, _, err := streamFromServer(t, t.Context(), server, model, modelContext, nil); err != nil {
		t.Fatal(err)
	}

	second := server.Requests()[1]
	calls := oair.InputItems(t, second, "function_call")
	if len(calls) != 1 {
		t.Fatalf("function_call items = %#v, want one", calls)
	}
	if calls[0]["call_id"] != "call_1" || calls[0]["id"] != "fc_1" || calls[0]["name"] != "get_weather" {
		t.Errorf("function_call = %#v, want call_1/fc_1 get_weather", calls[0])
	}
	if calls[0]["arguments"] != `{"city":"Paris"}` {
		t.Errorf("function_call arguments = %#v", calls[0]["arguments"])
	}
	outputs := oair.InputItems(t, second, "function_call_output")
	if len(outputs) != 1 || outputs[0]["call_id"] != "call_1" || outputs[0]["output"] != "sunny" {
		t.Errorf("function_call_output items = %#v, want call_1 sunny", outputs)
	}
}

// A tool result with an image reaches a vision model as input parts.
func TestStreamOpenAIResponseOverHTTPSendsToolResultImages(t *testing.T) {
	server := providertest.NewServer(t, providertest.Sequence(t,
		oair.Reply(oair.Text("msg_1", "A cat."), oair.Completed("resp_1", oair.Usage{Input: 9, Output: 2})),
	))
	model := oair.Model()
	model.Input = []string{"text", "image"}
	call := ai.ToolCall{Type: ai.ContentTypeToolCall, Id: "call_1|fc_1", Name: "screenshot", Args: map[string]any{}}
	modelContext := ai.ModelContext{Messages: []ai.Message{
		userText("What is on screen?"),
		ai.AssistantMessage{
			Role: ai.RoleAssistant, Api: model.API, Provider: model.Provider, Model: model.ID,
			Content: []ai.AssistantContent{call}, StopReason: ai.StopReasonToolUse,
		},
		ai.ToolResultMessage{
			Role: ai.RoleToolResult, ToolCallID: call.Id, ToolName: call.Name,
			Content: []ai.ToolResultContent{
				ai.TextContent{Type: ai.ContentTypeText, Text: "Captured"},
				ai.ImageContent{Type: ai.ContentTypeImage, Image: "aW1n", MimeType: "image/png"},
			},
		},
	}}
	if _, _, err := streamFromServer(t, t.Context(), server, model, modelContext, nil); err != nil {
		t.Fatal(err)
	}

	outputs := oair.InputItems(t, server.Requests()[0], "function_call_output")
	if len(outputs) != 1 || outputs[0]["call_id"] != "call_1" {
		t.Fatalf("function_call_output items = %#v", outputs)
	}
	parts, ok := outputs[0]["output"].([]any)
	if !ok || len(parts) != 2 {
		t.Fatalf("output = %#v, want text and image parts", outputs[0]["output"])
	}
	textPart, _ := parts[0].(map[string]any)
	imagePart, _ := parts[1].(map[string]any)
	if textPart["type"] != "input_text" || textPart["text"] != "Captured" {
		t.Errorf("text part = %#v", textPart)
	}
	if imagePart["type"] != "input_image" || imagePart["image_url"] != "data:image/png;base64,aW1n" || imagePart["detail"] != "auto" {
		t.Errorf("image part = %#v", imagePart)
	}
}

// With store=false the API keeps no reasoning state, so the encrypted reasoning
// from one turn must be sent back on the next.
func TestStreamOpenAIResponseOverHTTPRoundTripsReasoning(t *testing.T) {
	server := providertest.NewServer(t, providertest.Sequence(t,
		oair.Reply(
			oair.Reasoning("rs_1", "Thinking it over.", "encrypted-1"),
			oair.Text("msg_1", "4"),
			oair.Completed("resp_1", oair.Usage{Input: 5, Output: 9}),
		),
		oair.Reply(
			oair.Text("msg_2", "5"),
			oair.Completed("resp_2", oair.Usage{Input: 12, Output: 1}),
		),
	))
	model := oair.Model()
	model.Reasoning = true
	effort := "high"
	options := func() *ai.OpenAIResponseOptions {
		return &ai.OpenAIResponseOptions{ReasoningEffort: &effort}
	}
	modelContext := ai.ModelContext{Messages: []ai.Message{userText("2+2?")}}

	eventTypes, first, err := streamFromServer(t, t.Context(), server, model, modelContext, options())
	if err != nil {
		t.Fatal(err)
	}
	wantTypes := []string{
		"start",
		"thinking_start", "thinking_delta", "thinking_delta", "thinking_end",
		"text_start", "text_delta", "text_end",
		"done",
	}
	if !slices.Equal(eventTypes, wantTypes) {
		t.Fatalf("event types = %#v, want %#v", eventTypes, wantTypes)
	}
	thinking, ok := first.Content[0].(ai.ThinkingContent)
	if !ok {
		t.Fatalf("Content[0] = %T, want ThinkingContent", first.Content[0])
	}
	if thinking.Thinking != "Thinking it over." || thinking.ThinkingSignature == nil {
		t.Errorf("thinking = %q with signature %v, want the summary and a signature", thinking.Thinking, thinking.ThinkingSignature)
	}

	firstBody := server.Requests()[0].JSON(t)
	reasoning, _ := firstBody["reasoning"].(map[string]any)
	if reasoning["effort"] != "high" || reasoning["summary"] != "auto" {
		t.Errorf("body reasoning = %#v, want effort high and summary auto", firstBody["reasoning"])
	}
	if include, _ := firstBody["include"].([]any); !slices.Contains(include, any("reasoning.encrypted_content")) {
		t.Errorf("body include = %#v, want reasoning.encrypted_content", firstBody["include"])
	}

	modelContext.Messages = append(modelContext.Messages, first, userText("And 2+3?"))
	if _, _, err := streamFromServer(t, t.Context(), server, model, modelContext, options()); err != nil {
		t.Fatal(err)
	}
	items := oair.InputItems(t, server.Requests()[1], "reasoning")
	if len(items) != 1 {
		t.Fatalf("reasoning items = %#v, want one", items)
	}
	if items[0]["id"] != "rs_1" || items[0]["encrypted_content"] != "encrypted-1" {
		t.Errorf("reasoning item = %#v, want rs_1 with encrypted-1", items[0])
	}
}

func TestStreamOpenAIResponseOverHTTPReportsRequestErrors(t *testing.T) {
	tests := []struct {
		name         string
		handler      http.Handler
		wantInError  string
		wantOverflow bool
	}{
		{
			name:        "invalid api key",
			handler:     oair.ErrorResponse(http.StatusUnauthorized, "invalid_api_key", "Incorrect API key provided: sk-test."),
			wantInError: "Incorrect API key provided",
		},
		{
			// Rate-limit text mentions tokens, but must not trigger compaction.
			name: "rate limit",
			handler: oair.ErrorResponse(http.StatusTooManyRequests, "rate_limit_exceeded",
				"Rate limit reached for gpt-4.1-mini in organization org-test on tokens per min (TPM): Limit 30000, Used 29000, Requested 2000. Please try again in 2s."),
			wantInError: "Rate limit reached",
		},
		{
			name:         "context overflow",
			handler:      oair.ContextOverflow(),
			wantInError:  oair.ContextOverflowMessage,
			wantOverflow: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := providertest.NewServer(t, tt.handler)
			model := oair.Model()
			maxRetries := int64(0)

			eventTypes, msg, err := streamFromServer(t, t.Context(), server, model, ai.ModelContext{}, &ai.OpenAIResponseOptions{
				StreamOptions: ai.StreamOptions{MaxRetries: &maxRetries},
			})
			if err == nil {
				t.Fatal("stream succeeded, want an error")
			}
			if !slices.Equal(eventTypes, []string{"start", "error"}) {
				t.Errorf("event types = %#v, want start then error", eventTypes)
			}
			if msg.StopReason != ai.StopReasonError {
				t.Errorf("StopReason = %q, want %q", msg.StopReason, ai.StopReasonError)
			}
			if !strings.Contains(msg.ErrorMessage, tt.wantInError) {
				t.Errorf("ErrorMessage = %q, want it to contain %q", msg.ErrorMessage, tt.wantInError)
			}
			if got := ai.IsContextOverflow(msg, model.ContextWindow); got != tt.wantOverflow {
				t.Errorf("IsContextOverflow = %v, want %v", got, tt.wantOverflow)
			}
		})
	}
}

// OpenAI can also report context overflow inside the stream. Each form must
// reach IsContextOverflow, which drives compaction in baseAgent.
func TestStreamOpenAIResponseOverHTTPReportsStreamedContextOverflow(t *testing.T) {
	tests := map[string][]string{
		"response.failed event": oair.Failed("resp_1", "context_length_exceeded", oair.ContextOverflowMessage),
		"error event":           oair.Error("context_length_exceeded", oair.ContextOverflowMessage),
	}
	for name, events := range tests {
		t.Run(name, func(t *testing.T) {
			server := providertest.NewServer(t, oair.Reply(events))
			model := oair.Model()

			_, msg, _ := streamFromServer(t, t.Context(), server, model, ai.ModelContext{}, nil)
			if msg.StopReason != ai.StopReasonError {
				t.Errorf("StopReason = %q, want %q", msg.StopReason, ai.StopReasonError)
			}
			if !ai.IsContextOverflow(msg, model.ContextWindow) {
				t.Errorf("IsContextOverflow = false for error message %q", msg.ErrorMessage)
			}
		})
	}
}

func TestStreamOpenAIResponseOverHTTPReportsTruncatedStream(t *testing.T) {
	server := providertest.NewServer(t, oair.Reply(oair.Text("msg_1", "partial")))

	_, msg, err := streamFromServer(t, t.Context(), server, oair.Model(), ai.ModelContext{}, nil)
	if err == nil {
		t.Fatal("stream succeeded, want an error")
	}
	if msg.StopReason != ai.StopReasonError || msg.ErrorMessage != "OpenAI stream ended without terminal event" {
		t.Errorf("result = %q %q, want error for a missing terminal event", msg.StopReason, msg.ErrorMessage)
	}
	if len(msg.Content) != 1 {
		t.Fatalf("len(Content) = %d, want the partial text kept", len(msg.Content))
	}
	if text, ok := msg.Content[0].(ai.TextContent); !ok || text.Text != "partial" {
		t.Errorf("Content[0] = %#v, want TextContent partial", msg.Content[0])
	}
}

// This test uses a loopback listener rather than the in-memory network. When a
// request is canceled mid-stream, the in-memory network usually ends the
// response body with a clean EOF instead of the context's error, which real
// connections never do.
func TestStreamOpenAIResponseOverHTTPCancelsRequestMidStream(t *testing.T) {
	requestCanceled := make(chan struct{})
	server := providertest.NewServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		oair.WriteEvents(w, oair.Text("msg_1", "partial")[:3])
		select {
		case <-r.Context().Done():
			close(requestCanceled)
		case <-time.After(10 * time.Second):
		}
	}))
	server.Start()
	model := oair.Model()
	model.BaseURL = server.URL + "/v1"
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()

	apiKey := "test-key"
	stream, err := ai.StreamOpenAIResponse(ctx, model, ai.ModelContext{}, &ai.OpenAIResponseOptions{
		StreamOptions: ai.StreamOptions{ApiKey: &apiKey, HTTPClient: server.Client()},
	})
	if err != nil {
		t.Fatal(err)
	}
	for event := range stream.Events() {
		if event.EventType() == "text_delta" {
			cancel()
		}
	}
	msg, _ := stream.Result()
	select {
	case <-requestCanceled:
	case <-time.After(10 * time.Second):
		t.Error("the HTTP request was not canceled")
	}

	if msg.StopReason != ai.StopReasonAborted {
		t.Errorf("StopReason = %q, want %q (error %q)", msg.StopReason, ai.StopReasonAborted, msg.ErrorMessage)
	}
	if len(msg.Content) != 1 {
		t.Fatalf("len(Content) = %d, want the partial text kept", len(msg.Content))
	}
	if text, ok := msg.Content[0].(ai.TextContent); !ok || text.Text != "partial" {
		t.Errorf("Content[0] = %#v, want TextContent partial", msg.Content[0])
	}
}

func TestStreamOpenAIResponseOverHTTPCallsOnResponse(t *testing.T) {
	reply := oair.Reply(oair.Text("msg_1", "hi"), oair.Completed("resp_1", oair.Usage{}))
	server := providertest.NewServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Request-Id", "req_123")
		reply.ServeHTTP(w, r)
	}))

	var got []ai.ProviderResponse
	_, _, err := streamFromServer(t, t.Context(), server, oair.Model(), ai.ModelContext{}, &ai.OpenAIResponseOptions{
		StreamOptions: ai.StreamOptions{
			OnResponse: func(response ai.ProviderResponse, _ ai.Model[ai.API]) error {
				got = append(got, response)
				return nil
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 {
		t.Fatalf("OnResponse called %d times, want 1", len(got))
	}
	if got[0].Status != http.StatusOK || got[0].Headers["X-Request-Id"] != "req_123" {
		t.Errorf("OnResponse got status %d headers %v, want 200 with X-Request-Id req_123", got[0].Status, got[0].Headers)
	}
}

func TestStreamOpenAIResponseOverHTTPOnResponseErrorEndsStream(t *testing.T) {
	server := providertest.NewServer(t, oair.Reply(oair.Text("msg_1", "hi"), oair.Completed("resp_1", oair.Usage{})))
	hookErr := errors.New("response rejected by hook")

	_, msg, err := streamFromServer(t, t.Context(), server, oair.Model(), ai.ModelContext{}, &ai.OpenAIResponseOptions{
		StreamOptions: ai.StreamOptions{
			OnResponse: func(ai.ProviderResponse, ai.Model[ai.API]) error { return hookErr },
		},
	})
	if err == nil {
		t.Fatal("stream succeeded, want the hook's error")
	}
	if msg.StopReason != ai.StopReasonError || !strings.Contains(msg.ErrorMessage, hookErr.Error()) {
		t.Errorf("result = %q %q, want an error containing %q", msg.StopReason, msg.ErrorMessage, hookErr)
	}
}

func TestStreamOpenAIResponseOverHTTPSendsOnPayloadReplacement(t *testing.T) {
	server := providertest.NewServer(t, oair.Reply(oair.Text("msg_1", "hi"), oair.Completed("resp_1", oair.Usage{})))

	_, _, err := streamFromServer(t, t.Context(), server, oair.Model(), ai.ModelContext{}, &ai.OpenAIResponseOptions{
		StreamOptions: ai.StreamOptions{
			OnPayload: func(payload any, _ ai.Model[ai.API]) (any, error) {
				params := payload.(responses.ResponseNewParams)
				params.MaxOutputTokens = param.NewOpt[int64](42)
				return params, nil
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got := server.Requests()[0].JSON(t)["max_output_tokens"]; got != float64(42) {
		t.Errorf("body max_output_tokens = %#v, want 42 from OnPayload", got)
	}
}

func TestStreamOpenAIResponseOverHTTPSendsCacheAndSessionHeaders(t *testing.T) {
	sessionID := "session-1"
	sendSessionID := false
	tests := []struct {
		name            string
		cache           ai.CacheRetention
		compat          *ai.OpenAIResponsesCompat
		wantCacheKey    bool
		wantRetention   any
		wantSessionID   bool
		wantClientReqID bool
	}{
		{
			name:            "short cache",
			cache:           ai.CacheRetentionShort,
			wantCacheKey:    true,
			wantSessionID:   true,
			wantClientReqID: true,
		},
		{
			name:            "long cache",
			cache:           ai.CacheRetentionLong,
			wantCacheKey:    true,
			wantRetention:   "24h",
			wantSessionID:   true,
			wantClientReqID: true,
		},
		{
			name:            "session header disabled",
			cache:           ai.CacheRetentionShort,
			compat:          &ai.OpenAIResponsesCompat{SendSessionIdHeader: &sendSessionID},
			wantCacheKey:    true,
			wantClientReqID: true,
		},
		{
			// Without caching, the session ID must not reach the API at all.
			name:  "no cache",
			cache: ai.CacheRetentionNone,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := providertest.NewServer(t, oair.Reply(oair.Text("msg_1", "hi"), oair.Completed("resp_1", oair.Usage{})))
			model := oair.Model()
			model.Compat = tt.compat
			cache := tt.cache

			_, _, err := streamFromServer(t, t.Context(), server, model, ai.ModelContext{}, &ai.OpenAIResponseOptions{
				StreamOptions: ai.StreamOptions{SessionId: &sessionID, Cache: &cache},
			})
			if err != nil {
				t.Fatal(err)
			}

			req := server.Requests()[0]
			body := req.JSON(t)
			if _, got := body["prompt_cache_key"]; got != tt.wantCacheKey {
				t.Errorf("body has prompt_cache_key = %v, want %v (%#v)", got, tt.wantCacheKey, body["prompt_cache_key"])
			}
			if got := body["prompt_cache_retention"]; got != tt.wantRetention {
				t.Errorf("body prompt_cache_retention = %#v, want %#v", got, tt.wantRetention)
			}
			if got := req.Header.Get("Session_id") != ""; got != tt.wantSessionID {
				t.Errorf("has session_id header = %v, want %v", got, tt.wantSessionID)
			}
			if got := req.Header.Get("X-Client-Request-Id") != ""; got != tt.wantClientReqID {
				t.Errorf("has x-client-request-id header = %v, want %v", got, tt.wantClientReqID)
			}
		})
	}
}

// retryImmediately sets Retry-After-Ms to 0, so retries of a retryable status
// don't wait out the backoff.
func retryImmediately(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Retry-After-Ms", "0")
		h.ServeHTTP(w, r)
	})
}

func TestStreamOpenAIResponseOverHTTPHonorsMaxRetries(t *testing.T) {
	server := providertest.NewServer(t, retryImmediately(
		oair.ErrorResponse(http.StatusInternalServerError, "server_error", "The server had an error processing your request."),
	))
	maxRetries := int64(0)

	_, msg, _ := streamFromServer(t, t.Context(), server, oair.Model(), ai.ModelContext{}, &ai.OpenAIResponseOptions{
		StreamOptions: ai.StreamOptions{MaxRetries: &maxRetries},
	})

	if n := len(server.Requests()); n != 1 {
		t.Errorf("got %d requests with MaxRetries 0, want 1", n)
	}
	if msg.StopReason != ai.StopReasonError {
		t.Errorf("StopReason = %q, want %q", msg.StopReason, ai.StopReasonError)
	}
}

func TestStreamOpenAIResponseOverHTTPHonorsTimeout(t *testing.T) {
	ms := func(v int64) *int64 { return &v }
	tests := []struct {
		name      string
		timeoutMs *int64
		// want is how long the request lasts. A timeout fails it; with no
		// timeout the stalled handler's late reply succeeds.
		want        time.Duration
		wantTimeout bool
	}{
		{name: "set", timeoutMs: ms(1000), want: time.Second, wantTimeout: true},
		{name: "nil uses the default", want: ai.DefaultTimeout, wantTimeout: true},
		{name: "zero disables it", timeoutMs: ms(0), want: time.Hour},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				// The handler stalls for an hour of bubble time. It replies eventually
				// so that an ignored timeout fails the test instead of deadlocking the
				// bubble, which would panic and abort the rest of the package's tests.
				reply := oair.Reply(oair.Text("msg_1", "late"), oair.Completed("resp_1", oair.Usage{}))
				server := providertest.NewServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					select {
					case <-r.Context().Done():
					case <-time.After(time.Hour):
						reply.ServeHTTP(w, r)
					}
				}))
				maxRetries := int64(0)
				start := time.Now()

				_, msg, err := streamFromServer(t, t.Context(), server, oair.Model(), ai.ModelContext{}, &ai.OpenAIResponseOptions{
					StreamOptions: ai.StreamOptions{TimeoutMs: tt.timeoutMs, MaxRetries: &maxRetries},
				})

				if elapsed := time.Since(start); elapsed != tt.want {
					t.Errorf("request ended after %v, want %v", elapsed, tt.want)
				}
				if !tt.wantTimeout {
					if err != nil {
						t.Fatalf("stream failed with no timeout set: %v", err)
					}
					return
				}
				if err == nil || msg.ErrorMessage == "" {
					t.Fatalf("stream ended with %q %q, want a timeout error", msg.StopReason, msg.ErrorMessage)
				}
				// Nobody canceled the request, so a timeout is an error, not an abort.
				if msg.StopReason != ai.StopReasonError {
					t.Errorf("StopReason = %q, want %q (error %q)", msg.StopReason, ai.StopReasonError, msg.ErrorMessage)
				}
			})
		})
	}
}

// TimeoutMs bounds the wait for a response to start, not the whole stream, so a
// long generation is never cut off.
func TestStreamOpenAIResponseOverHTTPTimeoutAllowsLongStream(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		text := oair.Text("msg_1", "slow")
		server := providertest.NewServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			oair.WriteEvents(w, text[:2])
			time.Sleep(5 * time.Second)
			oair.WriteEvents(w, text[2:], oair.Completed("resp_1", oair.Usage{}))
		}))
		timeoutMs := int64(1000)

		_, msg, err := streamFromServer(t, t.Context(), server, oair.Model(), ai.ModelContext{}, &ai.OpenAIResponseOptions{
			StreamOptions: ai.StreamOptions{TimeoutMs: &timeoutMs},
		})
		if err != nil {
			t.Fatalf("stream failed after its response started: %v", err)
		}
		if text, ok := msg.Content[0].(ai.TextContent); !ok || text.Text != "slow" {
			t.Errorf("Content[0] = %#v, want TextContent slow", msg.Content[0])
		}
	})
}

// MaxRetriesDelayMs caps how long a retry may wait. When the server asks for a
// longer wait, the request fails at once with the requested delay in the error,
// in pi-mono's format, so callers can surface it instead of stalling.
func TestStreamOpenAIResponseOverHTTPCapsRetryDelay(t *testing.T) {
	ms := func(v int64) *int64 { return &v }
	const apiMessage = "Rate limit reached. Please try again later."
	tests := []struct {
		name       string
		maxDelayMs *int64
		maxRetries int64
		// retryAfter sets the response's retry headers; now is bubble time.
		retryAfter   func(header http.Header, now time.Time)
		wantRequests int
		wantElapsed  time.Duration
		wantError    string
	}{
		{
			name:         "over the limit fails at once",
			maxDelayMs:   ms(1000),
			maxRetries:   1,
			retryAfter:   func(h http.Header, _ time.Time) { h.Set("Retry-After-Ms", "120000") },
			wantRequests: 1,
			wantError:    "Server requested 120s retry delay (max: 1s). " + apiMessage,
		},
		{
			name:         "nil limit defaults to 60s",
			maxRetries:   1,
			retryAfter:   func(h http.Header, _ time.Time) { h.Set("Retry-After", "120") },
			wantRequests: 1,
			wantError:    "Server requested 120s retry delay (max: 60s). " + apiMessage,
		},
		{
			name:       "RFC 850 date over the limit",
			maxRetries: 1,
			retryAfter: func(h http.Header, now time.Time) {
				h.Set("Retry-After", now.Add(2*time.Minute).UTC().Format("Monday, 02-Jan-06 15:04:05 GMT"))
			},
			wantRequests: 1,
			wantError:    "Server requested 120s retry delay (max: 60s). " + apiMessage,
		},
		{
			name:         "delay within the default is waited out",
			maxRetries:   1,
			retryAfter:   func(h http.Header, _ time.Time) { h.Set("Retry-After", "30") },
			wantRequests: 2,
			wantElapsed:  30 * time.Second,
			wantError:    apiMessage,
		},
		{
			name:         "zero disables the limit",
			maxDelayMs:   ms(0),
			maxRetries:   1,
			retryAfter:   func(h http.Header, _ time.Time) { h.Set("Retry-After-Ms", "120000") },
			wantRequests: 2,
			wantElapsed:  120 * time.Second,
			wantError:    apiMessage,
		},
		{
			// No retry would follow, so there is no wait to limit.
			name:         "last attempt returns the API error",
			maxRetries:   0,
			retryAfter:   func(h http.Header, _ time.Time) { h.Set("Retry-After-Ms", "120000") },
			wantRequests: 1,
			wantError:    apiMessage,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				rateLimited := oair.ErrorResponse(http.StatusTooManyRequests, "rate_limit_exceeded", apiMessage)
				server := providertest.NewServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					tt.retryAfter(w.Header(), time.Now())
					rateLimited.ServeHTTP(w, r)
				}))
				start := time.Now()

				_, msg, _ := streamFromServer(t, t.Context(), server, oair.Model(), ai.ModelContext{}, &ai.OpenAIResponseOptions{
					StreamOptions: ai.StreamOptions{MaxRetriesDelayMs: tt.maxDelayMs, MaxRetries: &tt.maxRetries},
				})

				if n := len(server.Requests()); n != tt.wantRequests {
					t.Errorf("got %d requests, want %d", n, tt.wantRequests)
				}
				if elapsed := time.Since(start); elapsed != tt.wantElapsed {
					t.Errorf("request ended after %v, want %v", elapsed, tt.wantElapsed)
				}
				if msg.StopReason != ai.StopReasonError {
					t.Errorf("StopReason = %q, want %q", msg.StopReason, ai.StopReasonError)
				}
				if strings.HasPrefix(tt.wantError, "Server requested") {
					if msg.ErrorMessage != tt.wantError {
						t.Errorf("ErrorMessage = %q, want %q", msg.ErrorMessage, tt.wantError)
					}
				} else if !strings.Contains(msg.ErrorMessage, tt.wantError) || strings.Contains(msg.ErrorMessage, "retry delay") {
					t.Errorf("ErrorMessage = %q, want the API's error %q", msg.ErrorMessage, tt.wantError)
				}
			})
		})
	}
}

func TestStreamOpenAIResponseOverHTTPRetries(t *testing.T) {
	const serverError = "The server had an error processing your request."
	failing := oair.ErrorResponse(http.StatusInternalServerError, "server_error", serverError)
	ok := oair.Reply(oair.Text("msg_1", "ok"), oair.Completed("resp_1", oair.Usage{}))
	// dropConnection fails the request without a response.
	dropConnection := http.HandlerFunc(func(http.ResponseWriter, *http.Request) { panic(http.ErrAbortHandler) })
	stall := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done():
		case <-time.After(time.Hour):
			ok.ServeHTTP(w, r)
		}
	})
	ms := func(v int64) *int64 { return &v }
	tests := []struct {
		name      string
		handlers  []http.Handler
		timeoutMs *int64
		// cancelAfter cancels the request's context after this long, if set.
		cancelAfter time.Duration
		// Without a requested delay the backoff before retry n is 0.5s * 2^n,
		// less up to a quarter of jitter.
		wantElapsedMin, wantElapsedMax time.Duration
		wantStop                       ai.StopReason
		wantError                      string
	}{
		{
			name:           "server error is retried after a backoff",
			handlers:       []http.Handler{failing, ok},
			wantElapsedMin: 375 * time.Millisecond, wantElapsedMax: 500 * time.Millisecond,
			wantStop: ai.StopReasonStop,
		},
		{
			name:           "connection error is retried",
			handlers:       []http.Handler{dropConnection, ok},
			wantElapsedMin: 375 * time.Millisecond, wantElapsedMax: 500 * time.Millisecond,
			wantStop: ai.StopReasonStop,
		},
		{
			name:           "timed-out attempt is retried",
			handlers:       []http.Handler{stall, ok},
			timeoutMs:      ms(1000),
			wantElapsedMin: 1375 * time.Millisecond, wantElapsedMax: 1500 * time.Millisecond,
			wantStop: ai.StopReasonStop,
		},
		{
			name:           "gives up after the default retries",
			handlers:       []http.Handler{failing, failing, failing},
			wantElapsedMin: 1125 * time.Millisecond, wantElapsedMax: 1500 * time.Millisecond,
			wantStop: ai.StopReasonError, wantError: serverError,
		},
		{
			name:        "cancel during the wait aborts",
			handlers:    []http.Handler{retryAfterSeconds("30", failing)},
			cancelAfter: 10 * time.Second,
			// Cancellation ends the 30s wait early.
			wantElapsedMin: 10 * time.Second, wantElapsedMax: 10 * time.Second,
			wantStop: ai.StopReasonAborted,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				server := providertest.NewServer(t, providertest.Sequence(t, tt.handlers...))
				ctx := t.Context()
				if tt.cancelAfter > 0 {
					var cancel context.CancelFunc
					ctx, cancel = context.WithTimeout(ctx, tt.cancelAfter)
					defer cancel()
				}
				start := time.Now()

				_, msg, _ := streamFromServer(t, ctx, server, oair.Model(), ai.ModelContext{}, &ai.OpenAIResponseOptions{
					StreamOptions: ai.StreamOptions{TimeoutMs: tt.timeoutMs},
				})

				if elapsed := time.Since(start); elapsed < tt.wantElapsedMin || elapsed > tt.wantElapsedMax {
					t.Errorf("request ended after %v, want between %v and %v", elapsed, tt.wantElapsedMin, tt.wantElapsedMax)
				}
				if msg.StopReason != tt.wantStop || !strings.Contains(msg.ErrorMessage, tt.wantError) {
					t.Errorf("result = %q %q, want %q with %q", msg.StopReason, msg.ErrorMessage, tt.wantStop, tt.wantError)
				}
				requests := server.Requests()
				if len(requests) != len(tt.handlers) {
					t.Fatalf("got %d requests, want %d", len(requests), len(tt.handlers))
				}
				for i, request := range requests {
					if got, want := request.Header.Get("X-Stainless-Retry-Count"), strconv.Itoa(i); got != want {
						t.Errorf("request %d X-Stainless-Retry-Count = %q, want %q", i, got, want)
					}
					if i > 0 && !bytes.Equal(request.Body, requests[0].Body) {
						t.Errorf("request %d body differs from the first", i)
					}
				}
			})
		})
	}
}

// retryAfterSeconds asks for a retry after the given number of seconds.
func retryAfterSeconds(seconds string, h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Retry-After", seconds)
		h.ServeHTTP(w, r)
	})
}

// OnResponse sees the response the request ended with, not each attempt's.
func TestStreamOpenAIResponseOverHTTPCallsOnResponseOnceAfterRetries(t *testing.T) {
	server := providertest.NewServer(t, providertest.Sequence(t,
		retryImmediately(oair.ErrorResponse(http.StatusInternalServerError, "server_error", "The server had an error processing your request.")),
		oair.Reply(oair.Text("msg_1", "hi"), oair.Completed("resp_1", oair.Usage{})),
	))
	var statuses []int

	_, _, err := streamFromServer(t, t.Context(), server, oair.Model(), ai.ModelContext{}, &ai.OpenAIResponseOptions{
		StreamOptions: ai.StreamOptions{
			OnResponse: func(response ai.ProviderResponse, _ ai.Model[ai.API]) error {
				statuses = append(statuses, response.Status)
				return nil
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(statuses) != 1 || statuses[0] != http.StatusOK {
		t.Errorf("OnResponse statuses = %v, want [200]", statuses)
	}
}

// A multi-tenant server's OpenAI environment must not reach a request made
// with a user's key.
func TestStreamOpenAIResponseOverHTTPIgnoresEnvironment(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "sk-env")
	t.Setenv("OPENAI_BASE_URL", "https://env.openai.test/env/v1")
	t.Setenv("OPENAI_ORG_ID", "org-env")
	t.Setenv("OPENAI_PROJECT_ID", "proj-env")
	t.Setenv("OPENAI_CUSTOM_HEADERS", "X-Env: leaked")
	server := providertest.NewServer(t, oair.Reply(
		oair.Text("msg_1", "ok"),
		oair.Completed("resp_1", oair.Usage{Input: 1, Output: 1}),
	))
	model := oair.Model()
	model.BaseURL = ""

	if _, _, err := streamFromServer(t, t.Context(), server, model, ai.ModelContext{
		Messages: []ai.Message{userText("hi")},
	}, nil); err != nil {
		t.Fatal(err)
	}

	requests := server.Requests()
	if len(requests) != 1 {
		t.Fatalf("got %d requests, want 1", len(requests))
	}
	req := requests[0]
	if req.Path != "/v1/responses" {
		t.Errorf("path = %q, want the default /v1/responses, not OPENAI_BASE_URL's", req.Path)
	}
	for header, want := range map[string]string{
		"Authorization":       "Bearer test-key",
		"Openai-Organization": "",
		"Openai-Project":      "",
		"X-Env":               "",
	} {
		if got := req.Header.Get(header); got != want {
			t.Errorf("header %s = %q, want %q", header, got, want)
		}
	}
}

func TestStreamSimpleOpenAIResponsesNeedsAPIKey(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "sk-env")
	for _, options := range []*ai.SimpleStreamOptions{nil, {}} {
		if _, err := ai.StreamSimpleOpenAIResponses(oair.Model(), ai.ModelContext{}, options); err == nil || !strings.Contains(err.Error(), "No API key") {
			t.Errorf("StreamSimpleOpenAIResponses(%v) error = %v, want no API key, not OPENAI_API_KEY", options, err)
		}
	}
}
