package baseAgent

import (
	"context"
	"net/http"
	"strings"
	"sync"
	"testing"

	"github.com/ColeBurch/burrow/agent"
	"github.com/ColeBurch/burrow/ai"
	"github.com/ColeBurch/burrow/ai/providertest"
	"github.com/ColeBurch/burrow/ai/providertest/openairesponses"
)

// ============================================================================
// Compaction Over HTTP
// ============================================================================

// These tests send both the agent's requests and compaction's summary requests
// through the real OpenAI Responses provider to a providertest server, so they
// check what the session actually puts on the wire.

const (
	httpCompactionSummary      = "wire summary"
	httpCompactionSystemPrompt = "You are a test agent."
)

type httpCompactionHarness struct {
	ctx     context.Context
	server  *providertest.Server
	model   ai.Model[ai.API]
	session *AgentSession

	mu     sync.Mutex
	starts []CompactionStartEvent
	ends   []CompactionEndEvent
}

// newHTTPCompactionHarness seeds the aaaa–dddd history and serves the session's
// requests with handlers, in order.
func newHTTPCompactionHarness(t *testing.T, handlers ...http.Handler) *httpCompactionHarness {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), autoCompactionTestTimeout)
	t.Cleanup(cancel)

	openairesponses.Register()
	h := &httpCompactionHarness{ctx: ctx}
	h.server = providertest.NewServer(t, providertest.Sequence(t, handlers...))
	h.model = h.server.Route(t, openairesponses.Model())

	sm, err := InMemorySessionManager(ctx)
	if err != nil {
		t.Fatal(err)
	}
	// Seed with the unrouted model: replies from the provider carry its API.
	messages := seedManualCompactionHistory(t, ctx, sm, openairesponses.Model())
	systemPrompt := httpCompactionSystemPrompt
	a := agent.NewAgent(&agent.AgentOptions{
		InitialState: &agent.InitialAgentState{SystemPrompt: &systemPrompt, Model: &h.model, Messages: messages},
		ConvertToLLM: ConvertToLLM,
		ResolveAuth:  testAuth,
	})
	h.session = NewAgentSession(a, sm)
	h.session.SetCompactionSettings(CompactionSettings{Enabled: true, ReservedTokens: 100, KeepRecentTokens: 2})
	t.Cleanup(h.session.Subscribe(func(event AgentSessionEvent) {
		h.mu.Lock()
		defer h.mu.Unlock()
		switch e := event.(type) {
		case CompactionStartEvent:
			h.starts = append(h.starts, e)
		case CompactionEndEvent:
			h.ends = append(h.ends, e)
		}
	}))
	return h
}

func (h *httpCompactionHarness) compactionEvents() ([]CompactionStartEvent, []CompactionEndEvent) {
	h.mu.Lock()
	defer h.mu.Unlock()
	return append([]CompactionStartEvent(nil), h.starts...), append([]CompactionEndEvent(nil), h.ends...)
}

func (h *httpCompactionHarness) persistedCompactions() []*CompactionEntry {
	var compactions []*CompactionEntry
	for _, entry := range h.session.SessionManager.GetEntries() {
		if compaction, ok := entry.(*CompactionEntry); ok {
			compactions = append(compactions, compaction)
		}
	}
	return compactions
}

// summaryReply answers a summary request with httpCompactionSummary.
func summaryReply() http.Handler {
	return openairesponses.Reply(
		openairesponses.Text("msg_summary", httpCompactionSummary),
		openairesponses.Completed("resp_summary", openairesponses.Usage{Input: 50, Output: 10}),
	)
}

func textReply(text string, usage openairesponses.Usage) http.Handler {
	return openairesponses.Reply(
		openairesponses.Text("msg_"+text, text),
		openairesponses.Completed("resp_"+text, usage),
	)
}

// inputMessage is a role and the concatenated text of one message item in a
// Responses request's input.
type inputMessage struct {
	Role string
	Text string
}

func requestInputMessages(t *testing.T, request providertest.Request) []inputMessage {
	t.Helper()
	input, ok := request.JSON(t)["input"].([]any)
	if !ok {
		t.Fatalf("request input is not a list: %s", request.Body)
	}
	var messages []inputMessage
	for _, raw := range input {
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
				if part, ok := part.(map[string]any); ok {
					partText, _ := part["text"].(string)
					text.WriteString(partText)
				}
			}
		}
		messages = append(messages, inputMessage{Role: role, Text: text.String()})
	}
	return messages
}

func isSummaryRequest(t *testing.T, request providertest.Request) bool {
	t.Helper()
	messages := requestInputMessages(t, request)
	return len(messages) > 0 && messages[0].Role == "system" && messages[0].Text == SUMMARIZATION_SYSTEM_PROMPT
}

func countSummaryRequests(t *testing.T, requests []providertest.Request) int {
	t.Helper()
	count := 0
	for _, request := range requests {
		if isSummaryRequest(t, request) {
			count++
		}
	}
	return count
}

// assertCompactedWireRequest checks that an agent request after compaction
// starts with the system prompt and the summary, drops the summarized history,
// and ends with prompt.
func assertCompactedWireRequest(t *testing.T, request providertest.Request, prompt string) {
	t.Helper()
	messages := requestInputMessages(t, request)
	if len(messages) < 3 {
		t.Fatalf("compacted request has %d messages, want the system prompt, summary, and prompt: %+v", len(messages), messages)
	}
	if want := (inputMessage{Role: "system", Text: httpCompactionSystemPrompt}); messages[0] != want {
		t.Errorf("first input message = %+v, want %+v", messages[0], want)
	}
	wantSummary := CompactionSummaryPrefix + httpCompactionSummary + CompactionSummarySuffix
	if messages[1].Role != "user" || messages[1].Text != wantSummary {
		t.Errorf("second input message = %+v, want user summary %q", messages[1], wantSummary)
	}
	for _, message := range messages {
		if message.Text == "aaaa" {
			t.Error("request carries history that should have been summarized")
		}
	}
	if last := messages[len(messages)-1]; last.Role != "user" || last.Text != prompt {
		t.Errorf("last input message = %+v, want user %q", last, prompt)
	}
}

func TestAgentSessionHTTPContextOverflowRecoveryCompactsAndRetries(t *testing.T) {
	h := newHTTPCompactionHarness(t,
		openairesponses.ContextOverflow(),
		summaryReply(),
		textReply("recovered", openairesponses.Usage{Input: 60, Output: 5}),
	)
	// The overflow reply has no content, so this keeps exactly its user turn.
	settings := h.session.GetCompactionSettings()
	settings.KeepRecentTokens = 1
	h.session.SetCompactionSettings(settings)

	if err := h.session.Prompt(h.ctx, "next", nil); err != nil {
		t.Fatalf("Prompt: %v", err)
	}

	requests := h.server.Requests()
	if len(requests) != 3 {
		t.Fatalf("got %d requests, want overflow, summary, retry", len(requests))
	}
	first := requestInputMessages(t, requests[0])
	if len(first) != 6 || first[1].Text != "aaaa" || first[5].Text != "next" {
		t.Errorf("first request input = %+v, want the system prompt, full history, then next", first)
	}
	if !isSummaryRequest(t, requests[1]) {
		t.Errorf("second request is not a summary request: %+v", requestInputMessages(t, requests[1]))
	}
	if isSummaryRequest(t, requests[2]) {
		t.Fatal("third request is a summary request, want the retried prompt")
	}
	assertCompactedWireRequest(t, requests[2], "next")

	starts, ends := h.compactionEvents()
	if len(starts) != 1 || starts[0].Reason != CompactionReasonOverflow {
		t.Errorf("compaction starts = %+v, want one overflow start", starts)
	}
	if len(ends) != 1 || !ends[0].WillRetry || ends[0].ErrorMessage != "" {
		t.Errorf("compaction ends = %+v, want one successful end that retries", ends)
	}
	compactions := h.persistedCompactions()
	if len(compactions) != 1 || compactions[0].Summary != httpCompactionSummary {
		t.Fatalf("persisted compactions = %+v, want one with the wire summary", compactions)
	}

	// The real overflow error stays in the transcript; the live context ends
	// with the successful retry.
	var overflows int
	for _, message := range h.session.SessionManager.BuildSessionContext().Messages {
		if assistant, ok := message.(ai.AssistantMessage); ok && ai.IsContextOverflow(assistant, h.model.ContextWindow) {
			overflows++
			if !strings.Contains(assistant.ErrorMessage, "context_length_exceeded") {
				t.Errorf("persisted overflow error = %q, want the provider's context_length_exceeded error", assistant.ErrorMessage)
			}
		}
	}
	if overflows != 1 {
		t.Errorf("persisted overflow responses = %d, want 1", overflows)
	}
	messages := h.session.Agent.State().Messages
	last, ok := messages[len(messages)-1].(ai.AssistantMessage)
	if !ok || last.StopReason != ai.StopReasonStop || assistantText(t, last) != "recovered" {
		t.Errorf("last live message = %#v, want the recovered reply", messages[len(messages)-1])
	}
}

func TestAgentSessionHTTPContextOverflowRecoveryStopsAfterSecondOverflow(t *testing.T) {
	h := newHTTPCompactionHarness(t,
		openairesponses.ContextOverflow(),
		summaryReply(),
		openairesponses.ContextOverflow(),
	)
	settings := h.session.GetCompactionSettings()
	settings.KeepRecentTokens = 1
	h.session.SetCompactionSettings(settings)

	// compaction_end reports exhausted recovery; the overflow stays in the transcript.
	if err := h.session.Prompt(h.ctx, "next", nil); err != nil {
		t.Errorf("Prompt error = %v, want nil after exhausted recovery", err)
	}

	requests := h.server.Requests()
	if len(requests) != 3 {
		t.Fatalf("got %d requests, want overflow, summary, overflowing retry", len(requests))
	}
	if got := countSummaryRequests(t, requests); got != 1 {
		t.Errorf("summary requests = %d, want 1", got)
	}
	assertCompactedWireRequest(t, requests[2], "next")
	if got := len(h.persistedCompactions()); got != 1 {
		t.Errorf("persisted compactions = %d, want 1", got)
	}
	_, ends := h.compactionEvents()
	if len(ends) != 2 || ends[1].ErrorMessage != "context overflow recovery failed after one compact-and-retry attempt" {
		t.Errorf("compaction ends = %+v, want a successful end then exhausted recovery", ends)
	}
	messages := h.session.Agent.State().Messages
	last, ok := messages[len(messages)-1].(ai.AssistantMessage)
	if !ok || !ai.IsContextOverflow(last, h.model.ContextWindow) {
		t.Errorf("last live message = %#v, want the second overflow", messages[len(messages)-1])
	}
}

func TestAgentSessionHTTPAutoCompactionFromReportedUsage(t *testing.T) {
	// The threshold is ContextWindow 1000 - ReservedTokens 100. The API reports
	// cached tokens inside input_tokens; total_tokens decides compaction.
	t.Run("above", func(t *testing.T) {
		h := newHTTPCompactionHarness(t,
			textReply("done", openairesponses.Usage{Input: 880, Cached: 400, Output: 21}),
			summaryReply(),
			textReply("again", openairesponses.Usage{Input: 60, Output: 5}),
		)
		if err := h.session.Prompt(h.ctx, "next", nil); err != nil {
			t.Fatalf("Prompt: %v", err)
		}

		messages := h.session.SessionManager.BuildSessionContext().Messages
		var usage *ai.Usage
		for _, message := range messages {
			if assistant, ok := message.(ai.AssistantMessage); ok && len(assistant.Content) > 0 && assistantText(t, assistant) == "done" {
				usage = &assistant.Usage
			}
		}
		if usage == nil {
			t.Fatal("reply was not persisted")
		}
		if usage.Input != 480 || usage.CacheRead != 400 || usage.Output != 21 || usage.Total != 901 {
			t.Errorf("persisted usage = %+v, want input 480, cache read 400, output 21, total 901", *usage)
		}

		requests := h.server.Requests()
		if len(requests) != 2 || !isSummaryRequest(t, requests[1]) {
			t.Fatalf("got %d requests, want the prompt then a summary request", len(requests))
		}
		starts, ends := h.compactionEvents()
		if len(starts) != 1 || starts[0].Reason != CompactionReasonThreshold {
			t.Errorf("compaction starts = %+v, want one threshold start", starts)
		}
		if len(ends) != 1 || ends[0].WillRetry || ends[0].ErrorMessage != "" {
			t.Errorf("compaction ends = %+v, want one successful end without retry", ends)
		}
		if got := len(h.persistedCompactions()); got != 1 {
			t.Errorf("persisted compactions = %d, want 1", got)
		}

		if err := h.session.Prompt(h.ctx, "after", nil); err != nil {
			t.Fatalf("second Prompt: %v", err)
		}
		requests = h.server.Requests()
		if len(requests) != 3 {
			t.Fatalf("got %d requests, want a third for the next prompt", len(requests))
		}
		assertCompactedWireRequest(t, requests[2], "after")
	})

	t.Run("below", func(t *testing.T) {
		h := newHTTPCompactionHarness(t,
			textReply("done", openairesponses.Usage{Input: 879, Cached: 400, Output: 20}),
		)
		if err := h.session.Prompt(h.ctx, "next", nil); err != nil {
			t.Fatalf("Prompt: %v", err)
		}
		if got := len(h.server.Requests()); got != 1 {
			t.Errorf("got %d requests, want only the prompt", got)
		}
		if starts, _ := h.compactionEvents(); len(starts) != 0 {
			t.Errorf("compaction starts = %+v, want none at 899 tokens", starts)
		}
	})
}

func TestAgentSessionHTTPSummaryRequestBody(t *testing.T) {
	h := newHTTPCompactionHarness(t, summaryReply())

	result, err := h.session.Compact(h.ctx, &ManualCompactionOptions{CustomInstructions: "keep file names"})
	if err != nil {
		t.Fatal(err)
	}
	if result.Summary != httpCompactionSummary {
		t.Errorf("summary = %q, want %q", result.Summary, httpCompactionSummary)
	}

	requests := h.server.Requests()
	if len(requests) != 1 {
		t.Fatalf("got %d requests, want 1 summary request", len(requests))
	}
	request := requests[0]
	if request.Path != "/v1/responses" || request.Header.Get("Authorization") != "Bearer test-key" {
		t.Errorf("request = %s Authorization=%q, want /v1/responses with the agent's key", request.Path, request.Header.Get("Authorization"))
	}
	body := request.JSON(t)
	for field, want := range map[string]any{
		"model": h.model.ID,
		// GenerateSummary allows 4/5 of ReservedTokens for the summary.
		"max_output_tokens": float64(80),
		"store":             false,
		"stream":            true,
	} {
		if got := body[field]; got != want {
			t.Errorf("body %s = %#v, want %#v", field, got, want)
		}
	}

	// KeepRecentTokens 2 keeps cccc and dddd, so only the first turn is summarized.
	wantPrompt := "<conversation>\n[User]: aaaa\n\n[Assistant]: bbbb\n</conversation>\n\n" +
		SUMMARIZATION_PROMPT + "\n\nAdditional focus: keep file names"
	messages := requestInputMessages(t, request)
	want := []inputMessage{
		{Role: "system", Text: SUMMARIZATION_SYSTEM_PROMPT},
		{Role: "user", Text: wantPrompt},
	}
	if len(messages) != len(want) {
		t.Fatalf("input messages = %+v, want system prompt and summarization prompt", messages)
	}
	for i := range want {
		if messages[i] != want[i] {
			t.Errorf("input message %d = %+v\nwant %+v", i, messages[i], want[i])
		}
	}
}
