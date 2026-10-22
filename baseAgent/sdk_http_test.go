package baseAgent_test

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"reflect"
	"strings"
	"sync"
	"testing"
	"testing/synctest"
	"time"

	"github.com/ColeBurch/burrow/agent"
	"github.com/ColeBurch/burrow/ai"
	"github.com/ColeBurch/burrow/ai/providertest"
	"github.com/ColeBurch/burrow/ai/providertest/openairesponses"
	"github.com/ColeBurch/burrow/baseAgent"
	"github.com/ColeBurch/burrow/baseAgent/sessionstoretest"
)

// ============================================================================
// Sessions Over HTTP
// ============================================================================

// These tests build sessions the way an app does, with CreateAgentSession, a
// model registry, and a store that encodes entries as a database would. Every
// request goes through the real OpenAI Responses provider to a providertest
// server, and a session can be reopened from the store as a restarted app
// would.

const httpSessionAPIKey = "test-key"

// staticAuth is an AuthStore with one key for OpenAI.
type staticAuth string

func (a staticAuth) APIKey(provider ai.Provider) (string, bool) {
	return string(a), provider == ai.ProviderOpenAI
}

type httpSessionHarness struct {
	ctx    context.Context
	server *providertest.Server
	store  *sessionstoretest.MemoryStore
	// model is routed to server and is the only model in registry, so sessions
	// that do not name a model use it.
	model    ai.Model[ai.API]
	registry *baseAgent.ModelRegistry
}

func newHTTPSessionHarness(t *testing.T, base ai.Model[ai.API], handler http.Handler) *httpSessionHarness {
	t.Helper()
	// Bounds a hang; inside a synctest bubble it also turns a regression that
	// waits out a long retry delay into a prompt failure.
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	t.Cleanup(cancel)

	openairesponses.Register()
	h := &httpSessionHarness{ctx: ctx, store: sessionstoretest.NewMemoryStore()}
	h.server = providertest.NewServer(t, handler)
	h.model = h.server.Route(t, base)
	h.registry = baseAgent.NewModelRegistryWithModels(staticAuth(httpSessionAPIKey), map[ai.Provider]map[string]ai.Model[ai.API]{
		h.model.Provider: {h.model.ID: h.model},
	})
	return h
}

// seed stores a new session holding messages and returns its ID.
func (h *httpSessionHarness) seed(t *testing.T, messages ...agent.AgentMessage) string {
	t.Helper()
	sm, err := baseAgent.NewSessionManager(h.ctx, nil, nil, true, h.store, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, message := range messages {
		if _, err := sm.AppendMessage(h.ctx, message); err != nil {
			t.Fatal(err)
		}
	}
	return sm.GetSessionID()
}

// open loads the stored session id, or creates one when id is empty, and
// returns the session CreateAgentSession builds for it with the harness's
// store and registry.
func (h *httpSessionHarness) open(t *testing.T, id string, options baseAgent.CreateAgentSessionOptions) *baseAgent.AgentSession {
	t.Helper()
	var sessionID *string
	if id != "" {
		sessionID = &id
	}
	sm, err := baseAgent.NewSessionManager(h.ctx, nil, sessionID, true, h.store, nil)
	if err != nil {
		t.Fatal(err)
	}
	options.SessionManager = sm
	options.ModelRegistry = h.registry
	result, err := baseAgent.CreateAgentSession(h.ctx, options)
	if err != nil {
		t.Fatalf("CreateAgentSession: %v", err)
	}
	if result.ModelFallbackMessage != nil {
		t.Fatalf("CreateAgentSession fell back: %s", *result.ModelFallbackMessage)
	}
	return result.Session
}

// compactionEnds records the session's compaction end events.
type compactionEnds struct {
	mu   sync.Mutex
	ends []baseAgent.CompactionEndEvent
}

func recordCompactionEnds(t *testing.T, session *baseAgent.AgentSession) *compactionEnds {
	c := &compactionEnds{}
	t.Cleanup(session.Subscribe(func(event baseAgent.AgentSessionEvent) {
		if end, ok := event.(baseAgent.CompactionEndEvent); ok {
			c.mu.Lock()
			c.ends = append(c.ends, end)
			c.mu.Unlock()
		}
	}))
	return c
}

func (c *compactionEnds) get() []baseAgent.CompactionEndEvent {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]baseAgent.CompactionEndEvent(nil), c.ends...)
}

func persistedCompactions(session *baseAgent.AgentSession) int {
	n := 0
	for _, entry := range session.SessionManager.GetEntries() {
		if _, ok := entry.(*baseAgent.CompactionEntry); ok {
			n++
		}
	}
	return n
}

func reply(itemID, text string, usage openairesponses.Usage) http.Handler {
	return openairesponses.Reply(
		openairesponses.Text(itemID, text),
		openairesponses.Completed("resp_"+itemID, usage),
	)
}

// seedTurns returns user and assistant messages alternating through texts, as
// the provider would have stored them.
func seedTurns(model ai.Model[ai.API], texts ...string) []agent.AgentMessage {
	messages := make([]agent.AgentMessage, len(texts))
	for i, text := range texts {
		content := ai.TextContent{Type: ai.ContentTypeText, Text: text}
		if i%2 == 0 {
			messages[i] = ai.UserMessage{Role: ai.RoleUser, Content: []ai.UserContent{content}, Timestamp: 1}
			continue
		}
		messages[i] = ai.AssistantMessage{
			Role:       ai.RoleAssistant,
			Content:    []ai.AssistantContent{content},
			Api:        model.API,
			Provider:   model.Provider,
			Model:      model.ID,
			Usage:      ai.Usage{Input: 10, Output: 1, Total: 11},
			StopReason: ai.StopReasonStop,
			Timestamp:  1,
		}
	}
	return messages
}

// lookupTool answers {"key": k} with "value-k".
func lookupTool() agent.AgentTool[any, any] {
	return agent.AgentTool[any, any]{
		Tool: ai.Tool{Name: "lookup", Description: "Look up a key.", Parameters: map[string]any{
			"type":       "object",
			"properties": map[string]any{"key": map[string]any{"type": "string"}},
			"required":   []any{"key"},
		}},
		Execute: func(_ context.Context, _ string, params any, _ agent.AgentToolUpdateCallback[any]) (agent.AgentToolResult[any], error) {
			key, _ := params.(map[string]any)["key"].(string)
			return agent.AgentToolResult[any]{Content: []ai.ToolResultContent{
				ai.TextContent{Type: ai.ContentTypeText, Text: "value-" + key},
			}}, nil
		},
	}
}

func requestInput(t *testing.T, request providertest.Request) []any {
	t.Helper()
	input, ok := request.JSON(t)["input"].([]any)
	if !ok {
		t.Fatalf("request input is not a list: %s", request.Body)
	}
	return input
}

// wireMessage is the role and concatenated text of a message item in a
// Responses request's input.
type wireMessage struct{ Role, Text string }

// wireMessages returns the message items of a request's input, skipping items
// such as function calls that have no role.
func wireMessages(t *testing.T, request providertest.Request) []wireMessage {
	t.Helper()
	var messages []wireMessage
	for _, raw := range requestInput(t, request) {
		item, _ := raw.(map[string]any)
		role, _ := item["role"].(string)
		if role == "" {
			continue
		}
		var text strings.Builder
		switch content := item["content"].(type) {
		case string:
			text.WriteString(content)
		case []any:
			for _, part := range content {
				partText, _ := part.(map[string]any)["text"].(string)
				text.WriteString(partText)
			}
		}
		messages = append(messages, wireMessage{Role: role, Text: text.String()})
	}
	return messages
}

func assertWireMessages(t *testing.T, request providertest.Request, want []wireMessage) {
	t.Helper()
	if got := wireMessages(t, request); !reflect.DeepEqual(got, want) {
		t.Errorf("request input messages = %+v\nwant %+v", got, want)
	}
}

func mustJSON(t *testing.T, v any) string {
	t.Helper()
	data, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}

// ============================================================================
// Tests
// ============================================================================

func TestCreateAgentSessionHTTPSendsSessionSettings(t *testing.T) {
	base := openairesponses.Model()
	base.Reasoning = true
	base.Headers = &map[string]string{"X-Model-Header": "from-model"}
	h := newHTTPSessionHarness(t, base, providertest.Sequence(t,
		reply("msg_1", "first reply", openairesponses.Usage{Input: 20, Output: 2}),
		reply("msg_2", "second reply", openairesponses.Usage{Input: 30, Output: 2}),
		reply("msg_summary", "summary", openairesponses.Usage{Input: 30, Output: 2}),
	))
	systemPrompt := "You are a test agent."
	options := baseAgent.CreateAgentSessionOptions{
		SystemPrompt: &systemPrompt,
		Tools:        []agent.AgentTool[any, any]{lookupTool()},
		// Keeps "second" (2 tokens) and "second reply" (3), the whole last turn,
		// so compaction sends one summary request rather than a split turn's two.
		CompactionSettings: &baseAgent.CompactionSettings{
			Enabled: true, ReservedTokens: 100, KeepRecentTokens: 5,
		},
	}

	created := options
	created.ThinkingLevel = ai.ThinkingLevelHigh
	session := h.open(t, "", created)
	id := session.SessionManager.GetSessionID()
	if err := session.Prompt(h.ctx, "first", nil); err != nil {
		t.Fatal(err)
	}
	// Reopened without a thinking level, the session restores the stored one.
	session = h.open(t, id, options)
	if err := session.Prompt(h.ctx, "second", nil); err != nil {
		t.Fatal(err)
	}
	if _, err := session.Compact(h.ctx, nil); err != nil {
		t.Fatalf("Compact: %v", err)
	}

	requests := h.server.Requests()
	if len(requests) != 3 {
		t.Fatalf("got %d requests, want two prompts and a summary", len(requests))
	}
	// Every request, including compaction's, carries the registry's key and
	// the model's headers.
	for i, request := range requests {
		for header, want := range map[string]string{
			"Authorization":  "Bearer " + httpSessionAPIKey,
			"X-Model-Header": "from-model",
		} {
			if got := request.Header.Get(header); got != want {
				t.Errorf("request %d header %s = %q, want %q", i, header, got, want)
			}
		}
		if request.Path != "/v1/responses" {
			t.Errorf("request %d path = %q, want /v1/responses", i, request.Path)
		}
	}

	for i, request := range requests[:2] {
		if got := request.Header.Get("Session_id"); got != id {
			t.Errorf("prompt %d session_id header = %q, want the session ID %q", i, got, id)
		}
		body := request.JSON(t)
		if got := body["prompt_cache_key"]; got != id {
			t.Errorf("prompt %d prompt_cache_key = %#v, want the session ID %q", i, got, id)
		}
		if got := body["model"]; got != base.ID {
			t.Errorf("prompt %d model = %#v, want %q", i, got, base.ID)
		}
		reasoning, _ := body["reasoning"].(map[string]any)
		if got := reasoning["effort"]; got != "high" {
			t.Errorf("prompt %d reasoning.effort = %#v, want high", i, got)
		}
		tools, _ := body["tools"].([]any)
		if len(tools) != 1 || tools[0].(map[string]any)["name"] != "lookup" {
			t.Errorf("prompt %d tools = %s, want lookup", i, mustJSON(t, tools))
		}
	}

	assertWireMessages(t, requests[1], []wireMessage{
		{"developer", systemPrompt},
		{"user", "first"},
		{"assistant", "first reply"},
		{"user", "second"},
	})
}

func TestAgentSessionHTTPResumeReplaysToolCallAndReasoning(t *testing.T) {
	base := openairesponses.Model()
	base.Reasoning = true
	h := newHTTPSessionHarness(t, base, providertest.Sequence(t,
		openairesponses.Reply(
			openairesponses.Reasoning("rs_1", "Need to look it up.", "encrypted-reasoning-1"),
			openairesponses.ToolCall("call_1", "fc_1", "lookup", `{"key":"a"}`),
			openairesponses.Completed("resp_1", openairesponses.Usage{Input: 20, Output: 10}),
		),
		reply("msg_2", "a is value-a", openairesponses.Usage{Input: 40, Output: 5}),
		reply("msg_3", "still value-a", openairesponses.Usage{Input: 50, Output: 5}),
	))
	options := baseAgent.CreateAgentSessionOptions{Tools: []agent.AgentTool[any, any]{lookupTool()}}

	session := h.open(t, "", options)
	if err := session.Prompt(h.ctx, "look up a", nil); err != nil {
		t.Fatal(err)
	}
	session = h.open(t, session.SessionManager.GetSessionID(), options)
	if err := session.Prompt(h.ctx, "again", nil); err != nil {
		t.Fatal(err)
	}

	requests := h.server.Requests()
	if len(requests) != 3 {
		t.Fatalf("got %d requests, want the tool call, its result, and the resumed prompt", len(requests))
	}
	live := requests[1]
	reasoning := openairesponses.InputItems(t, live, "reasoning")
	if len(reasoning) != 1 || reasoning[0]["id"] != "rs_1" || reasoning[0]["encrypted_content"] != "encrypted-reasoning-1" {
		t.Errorf("reasoning items = %s, want rs_1 with its encrypted content", mustJSON(t, reasoning))
	}
	calls := openairesponses.InputItems(t, live, "function_call")
	if len(calls) != 1 || calls[0]["call_id"] != "call_1" || calls[0]["id"] != "fc_1" || calls[0]["arguments"] != `{"key":"a"}` {
		t.Errorf("function_call items = %s, want call_1/fc_1", mustJSON(t, calls))
	}
	outputs := openairesponses.InputItems(t, live, "function_call_output")
	if len(outputs) != 1 || outputs[0]["call_id"] != "call_1" || outputs[0]["output"] != "value-a" {
		t.Errorf("function_call_output items = %s, want call_1's value-a", mustJSON(t, outputs))
	}

	// After the store and codec round trip, the resumed request replays the
	// live history item for item, then adds the reply and the new prompt.
	liveInput := requestInput(t, live)
	resumedInput := requestInput(t, requests[2])
	if len(resumedInput) != len(liveInput)+2 {
		t.Fatalf("resumed input has %d items, want %d: %s", len(resumedInput), len(liveInput)+2, mustJSON(t, resumedInput))
	}
	for i := range liveInput {
		if !reflect.DeepEqual(resumedInput[i], liveInput[i]) {
			t.Errorf("resumed input item %d = %s\nwant the live item %s", i, mustJSON(t, resumedInput[i]), mustJSON(t, liveInput[i]))
		}
	}
	// The reply goes back under the ID the API gave it.
	replayed, _ := resumedInput[len(liveInput)].(map[string]any)
	if replayed["type"] != "message" || replayed["id"] != "msg_2" {
		t.Errorf("replayed reply = %s, want message msg_2", mustJSON(t, replayed))
	}
}

func TestAgentSessionHTTPCompactionRetriesServerError(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		h := newHTTPSessionHarness(t, openairesponses.Model(), providertest.Sequence(t,
			openairesponses.ErrorResponse(http.StatusInternalServerError, "server_error", "The server had an error processing your request."),
			reply("msg_summary", "summary", openairesponses.Usage{Input: 30, Output: 2}),
		))
		id := h.seed(t, seedTurns(openairesponses.Model(), "aaaa", "bbbb", "cccc", "dddd")...)
		session := h.open(t, id, baseAgent.CreateAgentSessionOptions{
			CompactionSettings: &baseAgent.CompactionSettings{Enabled: true, ReservedTokens: 100, KeepRecentTokens: 2},
		})

		result, err := session.Compact(h.ctx, nil)
		if err != nil {
			t.Fatalf("Compact: %v", err)
		}
		if result.Summary != "summary" {
			t.Errorf("summary = %q, want the retried request's", result.Summary)
		}
		requests := h.server.Requests()
		if len(requests) != 2 {
			t.Fatalf("got %d requests, want the failed summary request and its retry", len(requests))
		}
		for i, request := range requests {
			if got, want := request.Header.Get("X-Stainless-Retry-Count"), fmt.Sprint(i); got != want {
				t.Errorf("request %d retry count = %q, want %q", i, got, want)
			}
		}
		if n := persistedCompactions(session); n != 1 {
			t.Errorf("persisted compactions = %d, want 1", n)
		}
	})
}

func TestAgentSessionHTTPCompactionRetryDelayOverLimit(t *testing.T) {
	const apiMessage = "Rate limit reached. Please try again later."
	const wantError = "Server requested 120s retry delay (max: 60s). " + apiMessage
	rateLimited := func() http.Handler {
		limited := openairesponses.ErrorResponse(http.StatusTooManyRequests, "rate_limit_exceeded", apiMessage)
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Retry-After", "120")
			limited.ServeHTTP(w, r)
		})
	}
	settings := &baseAgent.CompactionSettings{Enabled: true, ReservedTokens: 100, KeepRecentTokens: 2}
	history := seedTurns(openairesponses.Model(), "aaaa", "bbbb", "cccc", "dddd")

	// Compaction requests use the default 60s limit, so a longer delay fails
	// at once instead of stalling the session.
	t.Run("manual", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			h := newHTTPSessionHarness(t, openairesponses.Model(), providertest.Sequence(t,
				rateLimited(),
				reply("msg_after", "after reply", openairesponses.Usage{Input: 30, Output: 2}),
			))
			session := h.open(t, h.seed(t, history...), baseAgent.CreateAgentSessionOptions{CompactionSettings: settings})
			start := time.Now()

			_, err := session.Compact(h.ctx, nil)
			if err == nil || !strings.Contains(err.Error(), wantError) {
				t.Fatalf("Compact error = %v, want %q", err, wantError)
			}
			if elapsed := time.Since(start); elapsed != 0 {
				t.Errorf("Compact failed after %v, want at once", elapsed)
			}
			if n := persistedCompactions(session); n != 0 {
				t.Errorf("persisted compactions = %d, want 0", n)
			}

			// The session is unchanged and the next prompt sends the full history.
			if err := session.Prompt(h.ctx, "after", nil); err != nil {
				t.Fatal(err)
			}
			requests := h.server.Requests()
			if len(requests) != 2 {
				t.Fatalf("got %d requests, want the summary request and the prompt", len(requests))
			}
			if !strings.Contains(string(requests[1].Body), "aaaa") {
				t.Errorf("prompt after the failed compaction lost the history: %s", requests[1].Body)
			}
		})
	})

	t.Run("automatic", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			h := newHTTPSessionHarness(t, openairesponses.Model(), providertest.Sequence(t,
				// Above the threshold of ContextWindow 1000 - ReservedTokens 100.
				// "done" and "next" are a token each, so KeepRecentTokens 2 keeps
				// the whole turn and compaction sends one summary request, not a
				// split turn's two.
				reply("msg_next", "done", openairesponses.Usage{Input: 950, Output: 5}),
				rateLimited(),
			))
			session := h.open(t, h.seed(t, history...), baseAgent.CreateAgentSessionOptions{CompactionSettings: settings})
			ends := recordCompactionEnds(t, session)
			start := time.Now()

			if err := session.Prompt(h.ctx, "next", nil); err != nil {
				t.Fatalf("Prompt error = %v, want nil when only compaction fails", err)
			}
			if elapsed := time.Since(start); elapsed != 0 {
				t.Errorf("Prompt returned after %v, want at once", elapsed)
			}
			if n := len(h.server.Requests()); n != 2 {
				t.Errorf("got %d requests, want the prompt and one summary request", n)
			}
			got := ends.get()
			if len(got) != 1 || got[0].Reason != baseAgent.CompactionReasonThreshold || !strings.Contains(got[0].ErrorMessage, wantError) {
				t.Errorf("compaction ends = %+v, want one threshold end with %q", got, wantError)
			}
			if n := persistedCompactions(session); n != 0 {
				t.Errorf("persisted compactions = %d, want 0", n)
			}
		})
	})
}

func TestAgentSessionHTTPSplitTurnFailureCancelsOtherSummaryRequest(t *testing.T) {
	const apiMessage = "Invalid summary request."
	// Prompt text that tells the two summary requests apart, free of characters
	// JSON escapes.
	const historyMarker = "The messages above are a conversation to summarize"
	const turnPrefixMarker = "This is the PREFIX of a turn"
	tests := []struct {
		name          string
		failingMarker string
		heldMarker    string
		wantError     string
	}{
		{name: "history fails", failingMarker: historyMarker, heldMarker: turnPrefixMarker, wantError: "summarization failed"},
		{name: "turn prefix fails", failingMarker: turnPrefixMarker, heldMarker: historyMarker, wantError: "turn prefix summarization failed"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			heldArrived := make(chan struct{})
			heldCanceled := make(chan struct{})
			held := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				close(heldArrived)
				select {
				case <-r.Context().Done():
					close(heldCanceled)
				case <-time.After(5 * time.Second):
					// Answer, so a regression fails the test instead of hanging it.
					reply("msg_held", "held summary", openairesponses.Usage{Input: 30, Output: 2}).ServeHTTP(w, r)
				}
			})
			failing := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				// Fail only once both requests are in flight.
				select {
				case <-heldArrived:
				case <-time.After(5 * time.Second):
					t.Error("the other summary request never arrived; the two did not run in parallel")
				}
				openairesponses.ErrorResponse(http.StatusBadRequest, "invalid_request_error", apiMessage).ServeHTTP(w, r)
			})
			h := newHTTPSessionHarness(t, openairesponses.Model(), providertest.Match(t,
				providertest.Case{When: providertest.BodyContains(tt.failingMarker), Handler: failing},
				providertest.Case{When: providertest.BodyContains(tt.heldMarker), Handler: held},
			))
			// The last reply is 5 tokens, so KeepRecentTokens 2 cuts inside the
			// last turn: aaaa and bbbb are history and cccc is the turn prefix.
			id := h.seed(t, seedTurns(openairesponses.Model(), "aaaa", "bbbb", "cccc", "dddd dddd dddd dddd")...)
			session := h.open(t, id, baseAgent.CreateAgentSessionOptions{
				CompactionSettings: &baseAgent.CompactionSettings{Enabled: true, ReservedTokens: 100, KeepRecentTokens: 2},
			})

			_, err := session.Compact(h.ctx, nil)
			if err == nil || !strings.HasPrefix(err.Error(), tt.wantError) || !strings.Contains(err.Error(), apiMessage) {
				t.Errorf("Compact error = %v, want %q with the API's message", err, tt.wantError)
			}
			select {
			case <-heldCanceled:
			case <-time.After(5 * time.Second):
				t.Fatal("the held summary request was not canceled on the server")
			}
			if n := len(h.server.Requests()); n != 2 {
				t.Errorf("got %d requests, want the two summary requests", n)
			}
			if n := persistedCompactions(session); n != 0 {
				t.Errorf("persisted compactions = %d, want 0", n)
			}
		})
	}
}

func TestAgentSessionHTTPBranchSummaryAndCustomMessage(t *testing.T) {
	h := newHTTPSessionHarness(t, openairesponses.Model(),
		reply("msg_eeee", "ffff", openairesponses.Usage{Input: 30, Output: 2}))
	id := h.seed(t, seedTurns(openairesponses.Model(), "aaaa", "bbbb", "cccc", "dddd")...)
	session := h.open(t, id, baseAgent.CreateAgentSessionOptions{})
	sm := session.SessionManager

	var bbbb string
	for _, entry := range sm.GetEntries() {
		if message, ok := entry.(*baseAgent.SessionMessageEntry); ok {
			if reply, ok := message.Message.(ai.AssistantMessage); ok && reply.Content[0].(ai.TextContent).Text == "bbbb" {
				bbbb = message.ID
			}
		}
	}
	if bbbb == "" {
		t.Fatal("seeded reply bbbb not found")
	}
	// Leave the cccc turn for a branch from bbbb, then add a note to it.
	if _, err := sm.BranchWithSummary(h.ctx, &bbbb, "Tried cccc; it did not work.", nil, nil); err != nil {
		t.Fatal(err)
	}
	note := []ai.UserContent{ai.TextContent{Type: ai.ContentTypeText, Text: "Remember the note."}}
	if _, err := sm.AppendCustomMessageEntry(h.ctx, "note", note, true, nil); err != nil {
		t.Fatal(err)
	}

	// The session loads the branch when it is opened.
	session = h.open(t, id, baseAgent.CreateAgentSessionOptions{})
	if err := session.Prompt(h.ctx, "eeee", nil); err != nil {
		t.Fatal(err)
	}

	requests := h.server.Requests()
	if len(requests) != 1 {
		t.Fatalf("got %d requests, want 1", len(requests))
	}
	assertWireMessages(t, requests[0], []wireMessage{
		{"user", "aaaa"},
		{"assistant", "bbbb"},
		{"user", baseAgent.BranchSummaryPrefix + "Tried cccc; it did not work." + baseAgent.BranchSummarySuffix},
		{"user", "Remember the note."},
		{"user", "eeee"},
	})
}

func TestAgentSessionHTTPCompactsTwice(t *testing.T) {
	h := newHTTPSessionHarness(t, openairesponses.Model(), providertest.Sequence(t,
		reply("msg_bbbb", "bbbb", openairesponses.Usage{Input: 20, Output: 1}),
		reply("msg_dddd", "dddd", openairesponses.Usage{Input: 30, Output: 1}),
		reply("msg_summary_1", "summary one", openairesponses.Usage{Input: 30, Output: 2}),
		reply("msg_ffff", "ffff", openairesponses.Usage{Input: 30, Output: 1}),
		reply("msg_summary_2", "summary two", openairesponses.Usage{Input: 30, Output: 2}),
		reply("msg_hhhh", "hhhh", openairesponses.Usage{Input: 30, Output: 1}),
	))
	// Every message is one token, so each compaction keeps the last turn.
	session := h.open(t, "", baseAgent.CreateAgentSessionOptions{
		CompactionSettings: &baseAgent.CompactionSettings{Enabled: true, ReservedTokens: 100, KeepRecentTokens: 2},
	})
	prompt := func(text string) {
		t.Helper()
		if err := session.Prompt(h.ctx, text, nil); err != nil {
			t.Fatalf("Prompt %q: %v", text, err)
		}
	}
	compact := func(want string) {
		t.Helper()
		result, err := session.Compact(h.ctx, nil)
		if err != nil {
			t.Fatalf("Compact: %v", err)
		}
		if result.Summary != want {
			t.Fatalf("summary = %q, want %q", result.Summary, want)
		}
	}
	summary := func(text string) wireMessage {
		return wireMessage{"user", baseAgent.CompactionSummaryPrefix + text + baseAgent.CompactionSummarySuffix}
	}

	prompt("aaaa")
	prompt("cccc")
	compact("summary one")
	prompt("eeee")
	compact("summary two")
	prompt("gggg")

	requests := h.server.Requests()
	if len(requests) != 6 {
		t.Fatalf("got %d requests, want prompt, prompt, summary, prompt, summary, prompt", len(requests))
	}
	assertWireMessages(t, requests[2], []wireMessage{
		{"system", baseAgent.SUMMARIZATION_SYSTEM_PROMPT},
		{"user", "<conversation>\n[User]: aaaa\n\n[Assistant]: bbbb\n</conversation>\n\n" + baseAgent.SUMMARIZATION_PROMPT},
	})
	assertWireMessages(t, requests[3], []wireMessage{
		summary("summary one"),
		{"user", "cccc"},
		{"assistant", "dddd"},
		{"user", "eeee"},
	})
	// The second compaction summarizes what the first kept, updating its summary.
	assertWireMessages(t, requests[4], []wireMessage{
		{"system", baseAgent.SUMMARIZATION_SYSTEM_PROMPT},
		{"user", "<conversation>\n[User]: cccc\n\n[Assistant]: dddd\n</conversation>\n\n" +
			"<previous-summary>\nsummary one\n</previous-summary>\n\n" + baseAgent.UPDATE_SUMMARIZATION_PROMPT},
	})
	// Only the newest summary is in context.
	assertWireMessages(t, requests[5], []wireMessage{
		summary("summary two"),
		{"user", "eeee"},
		{"assistant", "ffff"},
		{"user", "gggg"},
	})
}
