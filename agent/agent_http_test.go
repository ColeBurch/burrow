package agent

import (
	"context"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/ColeBurch/burrow/ai"
	"github.com/ColeBurch/burrow/ai/providertest"
	"github.com/ColeBurch/burrow/ai/providertest/openairesponses"
	"github.com/openai/openai-go/v3/responses"
)

// httpTestModel returns an OpenAI Responses model whose requests go to server
// through DefaultStreamFn and the real provider.
func httpTestModel(t *testing.T, server *providertest.Server, model ai.Model[ai.API]) ai.Model[ai.API] {
	t.Helper()
	openairesponses.Register()
	return server.Route(t, model)
}

func httpTestAuth(context.Context, ai.Model[ai.API]) (ai.ModelAuth, error) {
	key := "test-key"
	return ai.ModelAuth{APIKey: &key}, nil
}

func TestAgentHTTPToolCallLoop(t *testing.T) {
	server := providertest.NewServer(t, providertest.Sequence(t,
		openairesponses.Reply(
			openairesponses.Text("msg_1", "Let me calculate that."),
			openairesponses.ToolCall("call_1", "fc_1", "calculate", `{"expression":"123 * 456"}`),
			openairesponses.Completed("resp_1", openairesponses.Usage{Input: 20, Output: 10}),
		),
		openairesponses.Reply(
			openairesponses.Text("msg_2", "The result is 56088."),
			openairesponses.Completed("resp_2", openairesponses.Usage{Input: 40, Output: 5}),
		),
	))
	model := httpTestModel(t, server, openairesponses.Model())
	a := NewAgent(&AgentOptions{
		InitialState: &InitialAgentState{Model: &model, Tools: []AgentTool[any, any]{calculateToolE2E()}},
		ResolveAuth:  httpTestAuth,
	})

	if err := a.PromptText(context.Background(), "What is 123 * 456?"); err != nil {
		t.Fatal(err)
	}

	s := a.State()
	if s.IsStreaming || s.ErrorMessage != "" || len(s.Messages) != 4 {
		t.Fatalf("unexpected state: streaming=%v error=%q messages=%#v", s.IsStreaming, s.ErrorMessage, s.Messages)
	}
	call := s.Messages[1].(ai.AssistantMessage)
	if call.StopReason != ai.StopReasonToolUse {
		t.Fatalf("first reply StopReason = %q, want tool use", call.StopReason)
	}
	result, ok := s.Messages[2].(ai.ToolResultMessage)
	if !ok || result.ToolCallID != "call_1|fc_1" || result.IsError || textOfMessage(result) != "123 * 456 = 56088" {
		t.Fatalf("tool result = %#v", s.Messages[2])
	}
	if got := textOfMessage(s.Messages[3]); got != "The result is 56088." {
		t.Fatalf("final text = %q", got)
	}

	requests := server.Requests()
	if len(requests) != 2 {
		t.Fatalf("got %d requests, want 2", len(requests))
	}
	tools, _ := requests[0].JSON(t)["tools"].([]any)
	if len(tools) != 1 || tools[0].(map[string]any)["name"] != "calculate" {
		t.Errorf("first request tools = %#v, want calculate", tools)
	}
	calls := openairesponses.InputItems(t, requests[1], "function_call")
	if len(calls) != 1 || calls[0]["call_id"] != "call_1" || calls[0]["id"] != "fc_1" || calls[0]["name"] != "calculate" || calls[0]["arguments"] != `{"expression":"123 * 456"}` {
		t.Errorf("second request function_call items = %#v", calls)
	}
	outputs := openairesponses.InputItems(t, requests[1], "function_call_output")
	if len(outputs) != 1 || outputs[0]["call_id"] != "call_1" || outputs[0]["output"] != "123 * 456 = 56088" {
		t.Errorf("second request function_call_output items = %#v", outputs)
	}
}

func TestAgentHTTPSettingsReachTheWire(t *testing.T) {
	server := providertest.NewServer(t, openairesponses.Reply(
		openairesponses.Text("msg_1", "ok"),
		openairesponses.Completed("resp_1", openairesponses.Usage{Input: 5, Output: 1}),
	))
	base := openairesponses.Model()
	base.Reasoning = true
	model := httpTestModel(t, server, base)
	sessionID := "session-1"
	level := ai.ThinkingLevelHigh
	var statuses []int
	a := NewAgent(&AgentOptions{
		InitialState: &InitialAgentState{Model: &model, ThinkingLevel: &level},
		ResolveAuth:  httpTestAuth,
		SessionID:    &sessionID,
		OnPayload: func(payload any, _ ai.Model[ai.API]) (any, error) {
			params, ok := payload.(responses.ResponseNewParams)
			if !ok {
				t.Errorf("OnPayload payload = %T, want responses.ResponseNewParams", payload)
				return nil, nil
			}
			params.Model = "gpt-replaced"
			return params, nil
		},
		OnResponse: func(response ai.ProviderResponse, _ ai.Model[ai.API]) error {
			statuses = append(statuses, response.Status)
			return nil
		},
	})

	if err := a.PromptText(context.Background(), "hi"); err != nil {
		t.Fatal(err)
	}
	if s := a.State(); s.ErrorMessage != "" {
		t.Fatalf("agent error = %q", s.ErrorMessage)
	}

	requests := server.Requests()
	if len(requests) != 1 {
		t.Fatalf("got %d requests, want 1", len(requests))
	}
	req := requests[0]
	for header, want := range map[string]string{
		"X-Client-Request-Id": sessionID,
		"Session_id":          sessionID,
	} {
		if got := req.Header.Get(header); got != want {
			t.Errorf("header %s = %q, want %q", header, got, want)
		}
	}
	body := req.JSON(t)
	if got := body["prompt_cache_key"]; got != sessionID {
		t.Errorf("body prompt_cache_key = %#v, want %q", got, sessionID)
	}
	if got := body["model"]; got != "gpt-replaced" {
		t.Errorf("body model = %#v, want OnPayload's gpt-replaced", got)
	}
	reasoning, _ := body["reasoning"].(map[string]any)
	if got := reasoning["effort"]; got != "high" {
		t.Errorf("body reasoning.effort = %#v, want high; reasoning = %#v", got, body["reasoning"])
	}
	if len(statuses) != 1 || statuses[0] != http.StatusOK {
		t.Errorf("OnResponse statuses = %v, want [200]", statuses)
	}
}

func TestAgentHTTPAbortCancelsRequest(t *testing.T) {
	requestCanceled := make(chan struct{})
	server := providertest.NewServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Stream part of a message, then hold the response open.
		openairesponses.WriteEvents(w, openairesponses.Text("msg_1", "partial answer")[:3])
		<-r.Context().Done()
		close(requestCanceled)
	}))
	// On the in-memory network a canceled response body usually ends in a
	// clean EOF rather than context.Canceled, which the provider reports as a
	// truncated stream. A loopback listener behaves like a real connection.
	server.Start()
	base := openairesponses.Model()
	base.BaseURL = server.URL + "/v1"
	model := httpTestModel(t, server, base)
	a := NewAgent(&AgentOptions{InitialState: &InitialAgentState{Model: &model}, ResolveAuth: httpTestAuth})

	streaming := make(chan struct{})
	var once sync.Once
	a.Subscribe(func(e AgentEvent, _ context.Context) error {
		if update, ok := e.(MessageUpdateEvent); ok && update.AssistantMessageEvent.EventType() == "text_delta" {
			once.Do(func() { close(streaming) })
		}
		return nil
	})

	done := make(chan error, 1)
	go func() { done <- a.PromptText(context.Background(), "Write a long answer.") }()
	select {
	case <-streaming:
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for streamed text")
	}
	a.Abort()
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for the prompt to return after Abort")
	}
	select {
	case <-requestCanceled:
	case <-time.After(5 * time.Second):
		t.Fatal("the HTTP request was not canceled")
	}

	s := a.State()
	last := s.Messages[len(s.Messages)-1].(ai.AssistantMessage)
	if s.IsStreaming || last.StopReason != ai.StopReasonAborted || last.ErrorMessage == "" || s.ErrorMessage != last.ErrorMessage {
		t.Fatalf("unexpected abort state: streaming=%v error=%q last=%#v", s.IsStreaming, s.ErrorMessage, last)
	}
	if got := textOfMessage(last); got != "partial answer" {
		t.Fatalf("partial text = %q, want %q", got, "partial answer")
	}
}

func TestAgentHTTPProviderErrorInTranscript(t *testing.T) {
	server := providertest.NewServer(t, providertest.Sequence(t,
		openairesponses.ErrorResponse(http.StatusUnauthorized, "invalid_api_key", "Incorrect API key provided: test-key."),
		openairesponses.Reply(
			openairesponses.Text("msg_1", "recovered"),
			openairesponses.Completed("resp_1", openairesponses.Usage{Input: 5, Output: 1}),
		),
	))
	model := httpTestModel(t, server, openairesponses.Model())
	a := NewAgent(&AgentOptions{InitialState: &InitialAgentState{Model: &model}, ResolveAuth: httpTestAuth})

	if err := a.PromptText(context.Background(), "hi"); err != nil {
		t.Fatal(err)
	}
	s := a.State()
	if len(s.Messages) != 2 {
		t.Fatalf("got %d messages, want user and assistant error: %#v", len(s.Messages), s.Messages)
	}
	failed := s.Messages[1].(ai.AssistantMessage)
	if s.IsStreaming || failed.StopReason != ai.StopReasonError || !strings.Contains(failed.ErrorMessage, "Incorrect API key") || s.ErrorMessage != failed.ErrorMessage {
		t.Fatalf("unexpected error state: streaming=%v error=%q last=%#v", s.IsStreaming, s.ErrorMessage, failed)
	}
	if n := len(server.Requests()); n != 1 {
		t.Fatalf("got %d requests after a 401, want 1", n)
	}

	// The agent is idle again and the next prompt goes through.
	if err := a.PromptText(context.Background(), "try again"); err != nil {
		t.Fatal(err)
	}
	s = a.State()
	if s.ErrorMessage != "" || textOfMessage(s.Messages[len(s.Messages)-1]) != "recovered" {
		t.Fatalf("unexpected state after retry: error=%q messages=%#v", s.ErrorMessage, s.Messages)
	}
}

func TestAgentHTTPOmitsEmptySystemPrompt(t *testing.T) {
	server := providertest.NewServer(t, openairesponses.Reply(
		openairesponses.Text("msg_1", "hello"),
		openairesponses.Completed("resp_1", openairesponses.Usage{Input: 5, Output: 1}),
	))
	model := httpTestModel(t, server, openairesponses.Model())
	a := NewAgent(&AgentOptions{InitialState: &InitialAgentState{Model: &model}, ResolveAuth: httpTestAuth})

	if err := a.PromptText(context.Background(), "hi"); err != nil {
		t.Fatal(err)
	}

	input, _ := server.Requests()[0].JSON(t)["input"].([]any)
	for _, raw := range input {
		if item, ok := raw.(map[string]any); ok && (item["role"] == "system" || item["role"] == "developer") {
			t.Errorf("input has a %s message %#v, want none for an agent without a system prompt", item["role"], item)
		}
	}
}

func TestAgentHTTPAuthReachesTheWire(t *testing.T) {
	server := providertest.NewServer(t, openairesponses.Reply(
		openairesponses.Text("msg_1", "ok"),
		openairesponses.Completed("resp_1", openairesponses.Usage{Input: 5, Output: 1}),
	))
	model := httpTestModel(t, server, openairesponses.Model())
	a := NewAgent(&AgentOptions{
		InitialState: &InitialAgentState{Model: &model},
		ResolveAuth: func(context.Context, ai.Model[ai.API]) (ai.ModelAuth, error) {
			key, baseURL := "sk-oauth", "https://auth.openai.test/proxy/v1"
			return ai.ModelAuth{APIKey: &key, Headers: map[string]string{"X-Account": "acct_1"}, BaseURL: &baseURL}, nil
		},
	})

	if err := a.PromptText(context.Background(), "hi"); err != nil {
		t.Fatal(err)
	}
	if s := a.State(); s.ErrorMessage != "" {
		t.Fatalf("agent error = %q", s.ErrorMessage)
	}
	requests := server.Requests()
	if len(requests) != 1 {
		t.Fatalf("got %d requests, want 1", len(requests))
	}
	req := requests[0]
	if req.Path != "/proxy/v1/responses" {
		t.Errorf("path = %q, want the auth base URL's /proxy/v1/responses", req.Path)
	}
	for header, want := range map[string]string{"Authorization": "Bearer sk-oauth", "X-Account": "acct_1"} {
		if got := req.Header.Get(header); got != want {
			t.Errorf("header %s = %q, want %q", header, got, want)
		}
	}
}
