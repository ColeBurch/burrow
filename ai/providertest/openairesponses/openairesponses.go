// Package openairesponses builds OpenAI Responses API fixtures for
// providertest servers.
//
// Builders return the JSON of one or more stream events, so a reply is a list
// of them:
//
//	server := providertest.NewServer(t, openairesponses.Reply(
//		openairesponses.Text("msg_1", "hello"),
//		openairesponses.Completed("resp_1", openairesponses.Usage{Input: 3, Output: 1}),
//	))
//
// Events not covered by a builder can be written as raw JSON in a []string.
package openairesponses

import (
	"encoding/json"
	"fmt"
	"net/http"
	"slices"
	"testing"

	"github.com/ColeBurch/burrow/ai"
	"github.com/ColeBurch/burrow/ai/providertest"
)

// ContextOverflowMessage is the message OpenAI sends when the input exceeds the
// model's context window.
const ContextOverflowMessage = "Your input exceeds the context window of this model. Please adjust your input and try again."

// Model returns an OpenAI Responses model for tests. A providertest client
// routes every host to its server, so the realistic base URL still exercises
// the SDK's path handling.
func Model() ai.Model[ai.API] {
	return ai.Model[ai.API]{
		ID:            "gpt-4.1-mini",
		Name:          "GPT-4.1 mini",
		API:           ai.APIOpenAIResponses,
		Provider:      ai.ProviderOpenAI,
		BaseURL:       "https://api.openai.test/v1",
		Input:         []string{"text"},
		ContextWindow: 1000,
		MaxTokens:     200,
	}
}

// Register registers the real OpenAI Responses provider if no provider is
// registered for its API, so providertest.Server.Route can wrap it.
func Register() {
	if _, ok := ai.GetApiProvider(ai.APIOpenAIResponses); ok {
		return
	}
	ai.RegisterApiProvider(ai.ApiProvider[ai.API, ai.StreamOptions]{
		Api:          ai.APIOpenAIResponses,
		Stream:       ai.StreamOpenAIResponses,
		StreamSimple: ai.StreamSimpleOpenAIResponses,
	}, nil)
}

// Reply returns a handler that streams the given events.
func Reply(events ...[]string) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		WriteEvents(w, events...)
	})
}

// WriteEvents streams the given events as server-sent events named by each
// event's type, as the real API sends them. It panics if an event is not a
// JSON object with a type.
func WriteEvents(w http.ResponseWriter, events ...[]string) {
	all := slices.Concat(events...)
	sse := make([]providertest.Event, len(all))
	for i, event := range all {
		var header struct {
			Type string `json:"type"`
		}
		if err := json.Unmarshal([]byte(event), &header); err != nil || header.Type == "" {
			panic(fmt.Sprintf("openairesponses: event without a type: %s", event))
		}
		sse[i] = providertest.Event{Name: header.Type, Data: event}
	}
	providertest.WriteSSE(w, sse...)
}

// ErrorResponse returns a handler that fails the request with an API error.
func ErrorResponse(status int, code, message string) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		providertest.WriteJSON(w, status, mustJSON(map[string]any{
			"error": map[string]any{
				"message": message,
				"type":    "invalid_request_error",
				"param":   nil,
				"code":    code,
			},
		}))
	})
}

// ContextOverflow returns a handler that rejects the request because the input
// exceeds the context window, as OpenAI does before streaming starts.
func ContextOverflow() http.Handler {
	return ErrorResponse(http.StatusBadRequest, "context_length_exceeded", ContextOverflowMessage)
}

// InputItems returns the input items of the given type in a Responses API
// request, such as "function_call" or "reasoning". It fails t if the request
// has no input list.
func InputItems(t testing.TB, request providertest.Request, itemType string) []map[string]any {
	t.Helper()
	body := request.JSON(t)
	input, ok := body["input"].([]any)
	if !ok {
		t.Fatalf("openairesponses: request input = %T, want a list", body["input"])
	}
	var items []map[string]any
	for _, raw := range input {
		if item, ok := raw.(map[string]any); ok && item["type"] == itemType {
			items = append(items, item)
		}
	}
	return items
}

// Text returns the events of an assistant message item with one text part.
func Text(itemID, text string) []string {
	return []string{
		event("response.output_item.added", map[string]any{
			"item": map[string]any{"id": itemID, "type": "message", "role": "assistant", "content": []any{}},
		}),
		event("response.content_part.added", map[string]any{
			"item_id": itemID,
			"part":    map[string]any{"type": "output_text", "text": ""},
		}),
		event("response.output_text.delta", map[string]any{"item_id": itemID, "delta": text}),
		event("response.output_item.done", map[string]any{
			"item": map[string]any{
				"id":      itemID,
				"type":    "message",
				"role":    "assistant",
				"status":  "completed",
				"content": []any{map[string]any{"type": "output_text", "text": text}},
			},
		}),
	}
}

// ToolCall returns the events of a function call item. arguments is the
// call's JSON arguments, streamed as one delta.
func ToolCall(callID, itemID, name, arguments string) []string {
	return []string{
		event("response.output_item.added", map[string]any{
			"item": map[string]any{"id": itemID, "type": "function_call", "call_id": callID, "name": name, "arguments": ""},
		}),
		event("response.function_call_arguments.delta", map[string]any{"item_id": itemID, "delta": arguments}),
		event("response.function_call_arguments.done", map[string]any{"item_id": itemID, "arguments": arguments}),
		event("response.output_item.done", map[string]any{
			"item": map[string]any{
				"id":        itemID,
				"type":      "function_call",
				"call_id":   callID,
				"name":      name,
				"arguments": arguments,
				"status":    "completed",
			},
		}),
	}
}

// Reasoning returns the events of a reasoning item with one summary part and
// the encrypted content that must be sent back on the next turn.
func Reasoning(itemID, summary, encryptedContent string) []string {
	return []string{
		event("response.output_item.added", map[string]any{
			"item": map[string]any{"id": itemID, "type": "reasoning", "summary": []any{}},
		}),
		event("response.reasoning_summary_part.added", map[string]any{
			"item_id": itemID,
			"part":    map[string]any{"type": "summary_text", "text": ""},
		}),
		event("response.reasoning_summary_text.delta", map[string]any{"item_id": itemID, "delta": summary}),
		event("response.reasoning_summary_part.done", map[string]any{
			"item_id": itemID,
			"part":    map[string]any{"type": "summary_text", "text": summary},
		}),
		event("response.output_item.done", map[string]any{
			"item": map[string]any{
				"id":                itemID,
				"type":              "reasoning",
				"summary":           []any{map[string]any{"type": "summary_text", "text": summary}},
				"encrypted_content": encryptedContent,
			},
		}),
	}
}

// Usage is the token usage reported when a response completes. Input includes
// Cached, as in the API.
type Usage struct {
	Input  int64
	Output int64
	Cached int64
}

// Completed returns the event that ends a successful response.
func Completed(responseID string, usage Usage) []string {
	return []string{event("response.completed", map[string]any{
		"response": map[string]any{
			"id":     responseID,
			"status": "completed",
			"usage": map[string]any{
				"input_tokens":         usage.Input,
				"input_tokens_details": map[string]any{"cached_tokens": usage.Cached},
				"output_tokens":        usage.Output,
				"total_tokens":         usage.Input + usage.Output,
			},
		},
	})}
}

// Failed returns the event that ends a response the API failed mid-stream.
func Failed(responseID, code, message string) []string {
	return []string{event("response.failed", map[string]any{
		"response": map[string]any{
			"id":     responseID,
			"status": "failed",
			"error":  map[string]any{"code": code, "message": message},
		},
	})}
}

// Error returns a stream-level error event.
func Error(code, message string) []string {
	return []string{event("error", map[string]any{"code": code, "message": message})}
}

func event(eventType string, fields map[string]any) string {
	fields["type"] = eventType
	return mustJSON(fields)
}

func mustJSON(v any) string {
	data, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return string(data)
}
