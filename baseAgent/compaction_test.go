package baseAgent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ColeBurch/burrow/agent"
	"github.com/ColeBurch/burrow/ai"
)

// ============================================================================
// Helper Functions
// ============================================================================

func testEntryBase(entryType, id string) SessionEntryBase {
	return SessionEntryBase{Type: entryType, ID: id, Timestamp: "2025-01-01T00:00:00Z"}
}

func testMessageEntry(id string, message agent.AgentMessage) *SessionMessageEntry {
	return &SessionMessageEntry{
		SessionEntryBase: testEntryBase("message", id),
		Message:          message,
	}
}

func testUserMessage(text string) ai.UserMessage {
	return ai.UserMessage{
		Role: ai.RoleUser,
		Content: []ai.UserContent{
			ai.TextContent{Type: ai.ContentTypeText, Text: text},
		},
	}
}

func testAssistantMessage(text string, usage ai.Usage, stopReason ai.StopReason) ai.AssistantMessage {
	return ai.AssistantMessage{
		Role: ai.RoleAssistant,
		Content: []ai.AssistantContent{
			ai.TextContent{Type: ai.ContentTypeText, Text: text},
		},
		Usage:      usage,
		StopReason: stopReason,
	}
}

func testToolResultMessage(text string) ai.ToolResultMessage {
	return ai.ToolResultMessage{
		Role: ai.RoleToolResult,
		Content: []ai.ToolResultContent{
			ai.TextContent{Type: ai.ContentTypeText, Text: text},
		},
	}
}

func TestGetMessageFromEntry(t *testing.T) {
	const timestamp = "2025-01-02T03:04:05.678Z"
	timestampMillis := time.Date(2025, time.January, 2, 3, 4, 5, 678_000_000, time.UTC).UnixMilli()

	base := func(entryType string) SessionEntryBase {
		return SessionEntryBase{
			Type:      entryType,
			ID:        "entry-id",
			Timestamp: timestamp,
		}
	}

	userMessage := ai.UserMessage{
		Role: ai.RoleUser,
		Content: []ai.UserContent{
			ai.TextContent{Type: ai.ContentTypeText, Text: "user message"},
		},
		Timestamp: 123,
	}
	customContent := []ai.UserContent{
		ai.TextContent{Type: ai.ContentTypeText, Text: "custom message"},
	}
	customDetails := map[string]any{"source": "test"}

	tests := []struct {
		name  string
		entry SessionEntry
		want  agent.AgentMessage
	}{
		{
			name: "session message",
			entry: &SessionMessageEntry{
				SessionEntryBase: base("message"),
				Message:          userMessage,
			},
			want: userMessage,
		},
		{
			name: "custom message",
			entry: &CustomMessageEntry{
				SessionEntryBase: base("custom_message"),
				CustomType:       "notice",
				Content:          customContent,
				Details:          customDetails,
				Display:          true,
			},
			want: CustomMessage{
				Role:       MessageRoleCustom,
				CustomType: "notice",
				Content:    customContent,
				Details:    customDetails,
				Display:    true,
				Timestamp:  timestampMillis,
			},
		},
		{
			name: "branch summary",
			entry: &BranchSummaryEntry{
				SessionEntryBase: base("branch_summary"),
				FromID:           "branch-end-id",
				Summary:          "branch summary",
			},
			want: BranchSummaryMessage{
				Role:      MessageRoleBranchSummary,
				Summary:   "branch summary",
				FromID:    "branch-end-id",
				Timestamp: timestampMillis,
			},
		},
		{
			name: "compaction summary",
			entry: &CompactionEntry{
				SessionEntryBase: base("compaction"),
				Summary:          "compaction summary",
				TokensBefore:     456,
			},
			want: CompactionSummaryMessage{
				Role:         MessageRoleCompactionSummary,
				Summary:      "compaction summary",
				TokensBefore: 456,
				Timestamp:    timestampMillis,
			},
		},
		{
			name:  "thinking level change",
			entry: &ThinkingLevelChangeEntry{SessionEntryBase: base("thinking_level_change")},
		},
		{
			name:  "model change",
			entry: &ModelChangeEntry{SessionEntryBase: base("model_change")},
		},
		{
			name:  "custom entry",
			entry: &CustomEntry{SessionEntryBase: base("custom")},
		},
		{
			name:  "label",
			entry: &LabelEntry{SessionEntryBase: base("label")},
		},
		{
			name:  "session info",
			entry: &SessionInfoEntry{SessionEntryBase: base("session_info")},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := getMessageFromEntry(test.entry)
			if !reflect.DeepEqual(got, test.want) {
				t.Fatalf("getMessageFromEntry() = %#v, want %#v", got, test.want)
			}
		})
	}
}

func TestGetMessageFromEntryForCompaction(t *testing.T) {
	userMessage := ai.UserMessage{
		Role: ai.RoleUser,
		Content: []ai.UserContent{
			ai.TextContent{Type: ai.ContentTypeText, Text: "keep this message"},
		},
		Timestamp: 123,
	}

	tests := []struct {
		name  string
		entry SessionEntry
		want  agent.AgentMessage
	}{
		{
			name: "keeps a session message",
			entry: &SessionMessageEntry{
				SessionEntryBase: SessionEntryBase{Type: "message", ID: "message-id"},
				Message:          userMessage,
			},
			want: userMessage,
		},
		{
			name: "excludes a compaction summary",
			entry: &CompactionEntry{
				SessionEntryBase: SessionEntryBase{Type: "compaction", ID: "compaction-id"},
				Summary:          "old summary",
				TokensBefore:     456,
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := getMessageFromEntryForCompaction(test.entry)
			if !reflect.DeepEqual(got, test.want) {
				t.Fatalf("getMessageFromEntryForCompaction() = %#v, want %#v", got, test.want)
			}
		})
	}
}

func TestCompactGeneratesSplitTurnSummariesInParallel(t *testing.T) {
	type requestRecord struct {
		prompt    string
		maxTokens int64
		reasoning *ai.ModelThinkingLevel
	}

	var mu sync.Mutex
	var requests []requestRecord
	started := 0
	bothStarted := make(chan struct{})

	model := registerSummaryTestProvider(t, true, func(
		_ ai.Model[ai.API],
		modelContext ai.ModelContext,
		options *ai.SimpleStreamOptions,
	) (*ai.AssistantMessageEventStream, error) {
		if len(modelContext.Messages) != 1 {
			return nil, fmt.Errorf("got %d messages, want 1", len(modelContext.Messages))
		}
		message, ok := modelContext.Messages[0].(ai.UserMessage)
		if !ok || len(message.Content) != 1 {
			return nil, fmt.Errorf("unexpected summarization message")
		}
		text, ok := message.Content[0].(ai.TextContent)
		if !ok {
			return nil, fmt.Errorf("unexpected summarization content")
		}

		mu.Lock()
		requests = append(requests, requestRecord{
			prompt:    text.Text,
			maxTokens: *options.MaxTokens,
			reasoning: options.Reasoning,
		})
		started++
		if started == 2 {
			close(bothStarted)
		}
		mu.Unlock()

		select {
		case <-bothStarted:
		case <-time.After(time.Second):
			return nil, fmt.Errorf("summary requests did not run in parallel")
		}

		responseText := "history summary"
		if strings.Contains(text.Text, TURN_PREFIX_SUMMARIZATION_PROMPT) {
			responseText = "turn prefix summary"
		}
		response := ai.AssistantMessage{
			Role:       ai.RoleAssistant,
			Content:    []ai.AssistantContent{ai.TextContent{Type: ai.ContentTypeText, Text: responseText}},
			StopReason: ai.StopReasonStop,
		}
		return completedSummaryTestStream(response), nil
	})

	previousSummary := "previous summary"
	preparation := &CompactionPreparation{
		FirstKeptEntryID: "kept-entry",
		MessagesToSummarize: []agent.AgentMessage{
			ai.UserMessage{Role: ai.RoleUser, Content: []ai.UserContent{ai.TextContent{Type: ai.ContentTypeText, Text: "history"}}},
		},
		TurnPrefixMessages: []agent.AgentMessage{
			ai.UserMessage{Role: ai.RoleUser, Content: []ai.UserContent{ai.TextContent{Type: ai.ContentTypeText, Text: "prefix"}}},
		},
		IsSplitTurn:     true,
		TokensBefore:    321,
		PreviousSummary: &previousSummary,
		Settings:        CompactionSettings{ReservedTokens: 100},
	}
	thinkingLevel := ai.ThinkingLevelHigh

	result, err := Compact(context.Background(), preparation, model, "test-key", &CompactOptions{
		Headers:            map[string]string{"X-Test": "value"},
		CustomInstructions: "focus on tests",
		ThinkingLevel:      thinkingLevel,
	})
	if err != nil {
		t.Fatalf("Compact() error = %v", err)
	}

	wantSummary := "history summary\n\n---\n\n**Turn Context (split turn):**\n\nturn prefix summary"
	if result.Summary != wantSummary {
		t.Fatalf("Compact() summary = %q, want %q", result.Summary, wantSummary)
	}
	if result.FirstKeptEntryID != "kept-entry" || result.TokensBefore != 321 {
		t.Fatalf("Compact() result = %#v", result)
	}

	mu.Lock()
	defer mu.Unlock()
	if len(requests) != 2 {
		t.Fatalf("request count = %d, want 2", len(requests))
	}
	for _, request := range requests {
		if request.reasoning == nil || *request.reasoning != thinkingLevel {
			t.Errorf("reasoning = %v, want %q", request.reasoning, thinkingLevel)
		}
		if strings.Contains(request.prompt, TURN_PREFIX_SUMMARIZATION_PROMPT) {
			if request.maxTokens != 50 {
				t.Errorf("turn prefix max tokens = %d, want 50", request.maxTokens)
			}
			continue
		}
		if request.maxTokens != 80 {
			t.Errorf("history max tokens = %d, want 80", request.maxTokens)
		}
		if !strings.Contains(request.prompt, "<previous-summary>\nprevious summary\n</previous-summary>") {
			t.Errorf("history prompt does not contain the previous summary")
		}
		if !strings.Contains(request.prompt, "Additional focus: focus on tests") {
			t.Errorf("history prompt does not contain custom instructions")
		}
	}
}

func TestCompactSplitTurnWithoutNewHistory(t *testing.T) {
	tests := []struct {
		name            string
		previousSummary *string
		wantHistory     string
	}{
		{
			// Only the latest compaction is kept in context, so the new summary
			// must carry the previous one forward.
			name:            "keeps the previous summary",
			previousSummary: func() *string { s := "previous summary"; return &s }(),
			wantHistory:     "previous summary",
		},
		{
			name:        "notes missing history",
			wantHistory: "No prior history.",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var requests atomic.Int32
			model := registerSummaryTestProvider(t, false, func(
				ai.Model[ai.API],
				ai.ModelContext,
				*ai.SimpleStreamOptions,
			) (*ai.AssistantMessageEventStream, error) {
				requests.Add(1)
				return completedSummaryTestStream(summaryTestTextMessage("turn prefix summary")), nil
			})
			preparation := &CompactionPreparation{
				FirstKeptEntryID: "kept-entry",
				TurnPrefixMessages: []agent.AgentMessage{
					testUserMessage("prefix"),
				},
				IsSplitTurn:     true,
				PreviousSummary: test.previousSummary,
				Settings:        CompactionSettings{ReservedTokens: 100},
			}

			result, err := Compact(context.Background(), preparation, model, "test-key", nil)
			if err != nil {
				t.Fatalf("Compact() error = %v", err)
			}
			want := test.wantHistory + "\n\n---\n\n**Turn Context (split turn):**\n\nturn prefix summary"
			if result.Summary != want {
				t.Fatalf("Compact() summary = %q, want %q", result.Summary, want)
			}
			if got := requests.Load(); got != 1 {
				t.Fatalf("summary requests = %d, want only the turn prefix request", got)
			}
		})
	}
}

func TestCompactSplitTurnRejectsTruncatedTurnPrefixSummary(t *testing.T) {
	model := registerSummaryTestProvider(t, false, func(
		_ ai.Model[ai.API],
		modelContext ai.ModelContext,
		_ *ai.SimpleStreamOptions,
	) (*ai.AssistantMessageEventStream, error) {
		response := summaryTestTextMessage("history summary")
		// Both requests run concurrently, so inspect the prompt without t.Fatal.
		prompt := ""
		if message, ok := modelContext.Messages[0].(ai.UserMessage); ok && len(message.Content) == 1 {
			if text, ok := message.Content[0].(ai.TextContent); ok {
				prompt = text.Text
			}
		}
		if strings.Contains(prompt, TURN_PREFIX_SUMMARIZATION_PROMPT) {
			response = summaryTestTextMessage("partial turn prefix")
			response.StopReason = ai.StopReasonLength
		}
		return completedSummaryTestStream(response), nil
	})
	preparation := &CompactionPreparation{
		FirstKeptEntryID:    "kept-entry",
		MessagesToSummarize: []agent.AgentMessage{testUserMessage("history")},
		TurnPrefixMessages:  []agent.AgentMessage{testUserMessage("prefix")},
		IsSplitTurn:         true,
		Settings:            CompactionSettings{ReservedTokens: 100},
	}

	_, err := Compact(context.Background(), preparation, model, "test-key", nil)
	want := "turn prefix summarization failed: summary was truncated at the output token limit"
	if err == nil || !strings.Contains(err.Error(), want) {
		t.Fatalf("Compact() error = %v, want %q", err, want)
	}
}

func TestCompactSplitTurnFailureCancelsOtherSummary(t *testing.T) {
	historyCancelled := make(chan struct{})
	model := registerSummaryTestProvider(t, false, func(
		_ ai.Model[ai.API],
		modelContext ai.ModelContext,
		options *ai.SimpleStreamOptions,
	) (*ai.AssistantMessageEventStream, error) {
		prompt := ""
		if message, ok := modelContext.Messages[0].(ai.UserMessage); ok && len(message.Content) == 1 {
			if text, ok := message.Content[0].(ai.TextContent); ok {
				prompt = text.Text
			}
		}
		if strings.Contains(prompt, TURN_PREFIX_SUMMARIZATION_PROMPT) {
			return nil, errors.New("turn prefix failed")
		}
		select {
		case <-options.Signal.Done():
			close(historyCancelled)
			return nil, options.Signal.Err()
		case <-time.After(5 * time.Second):
			return completedSummaryTestStream(summaryTestTextMessage("history summary")), nil
		}
	})
	preparation := &CompactionPreparation{
		FirstKeptEntryID:    "kept-entry",
		MessagesToSummarize: []agent.AgentMessage{testUserMessage("history")},
		TurnPrefixMessages:  []agent.AgentMessage{testUserMessage("prefix")},
		IsSplitTurn:         true,
		Settings:            CompactionSettings{ReservedTokens: 100},
	}

	_, err := Compact(context.Background(), preparation, model, "test-key", nil)
	if err == nil || err.Error() != "turn prefix summarization failed: turn prefix failed" {
		t.Fatalf("Compact() error = %v, want the turn prefix failure", err)
	}
	select {
	case <-historyCancelled:
	default:
		t.Fatal("history summary request was not cancelled")
	}
}

// ============================================================================
// Token Calculation
// ============================================================================

func TestCalculateContextTokens(t *testing.T) {
	tests := []struct {
		name  string
		usage ai.Usage
		want  int64
	}{
		{
			name: "uses native total when present",
			usage: ai.Usage{
				Input:      10,
				Output:     20,
				CacheRead:  30,
				CacheWrite: 40,
				Total:      999,
			},
			want: 999,
		},
		{
			name: "calculates total from components when native total is zero",
			usage: ai.Usage{
				Input:      1000,
				Output:     500,
				CacheRead:  200,
				CacheWrite: 100,
			},
			want: 1800,
		},
		{
			name:  "returns zero when all values are zero",
			usage: ai.Usage{},
			want:  0,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := CalculateContextTokens(test.usage); got != test.want {
				t.Fatalf("CalculateContextTokens() = %d, want %d", got, test.want)
			}
		})
	}
}

func TestGetLastAssistantUsage(t *testing.T) {
	firstUsage := ai.Usage{Input: 70, Output: 30, Total: 100}
	secondUsage := ai.Usage{Input: 150, Output: 50, Total: 200}
	abortedUsage := ai.Usage{Input: 800, Output: 88, Total: 888}
	errorUsage := ai.Usage{Input: 900, Output: 99, Total: 999}
	lengthUsage := ai.Usage{Input: 250, Output: 50, Total: 300}
	toolUseUsage := ai.Usage{Input: 325, Output: 75, Total: 400}
	var nilMessageEntry *SessionMessageEntry

	tests := []struct {
		name    string
		entries []SessionEntry
		want    *ai.Usage
	}{
		{
			name: "returns the latest valid assistant usage",
			entries: []SessionEntry{
				testMessageEntry("user-1", testUserMessage("first prompt")),
				testMessageEntry("assistant-1", testAssistantMessage("first response", firstUsage, ai.StopReasonStop)),
				testMessageEntry("user-2", testUserMessage("second prompt")),
				testMessageEntry("assistant-2", testAssistantMessage("second response", secondUsage, ai.StopReasonStop)),
			},
			want: &secondUsage,
		},
		{
			name: "skips an aborted assistant message",
			entries: []SessionEntry{
				testMessageEntry("assistant-1", testAssistantMessage("valid response", firstUsage, ai.StopReasonStop)),
				testMessageEntry("assistant-2", testAssistantMessage("aborted response", abortedUsage, ai.StopReasonAborted)),
			},
			want: &firstUsage,
		},
		{
			name: "skips an error assistant message",
			entries: []SessionEntry{
				testMessageEntry("assistant-1", testAssistantMessage("valid response", firstUsage, ai.StopReasonStop)),
				testMessageEntry("assistant-2", testAssistantMessage("error response", errorUsage, ai.StopReasonError)),
			},
			want: &firstUsage,
		},
		{
			name: "skips entries without assistant messages",
			entries: []SessionEntry{
				testMessageEntry("assistant-1", testAssistantMessage("valid response", firstUsage, ai.StopReasonStop)),
				&ModelChangeEntry{SessionEntryBase: SessionEntryBase{Type: "model_change", ID: "model"}},
				&ThinkingLevelChangeEntry{SessionEntryBase: SessionEntryBase{Type: "thinking_level_change", ID: "thinking"}},
				&CompactionEntry{SessionEntryBase: SessionEntryBase{Type: "compaction", ID: "compaction"}},
				&LabelEntry{SessionEntryBase: SessionEntryBase{Type: "label", ID: "label"}},
				nilMessageEntry,
				&SessionMessageEntry{SessionEntryBase: SessionEntryBase{Type: "message", ID: "empty-message"}},
				testMessageEntry("user-1", testUserMessage("trailing prompt")),
			},
			want: &firstUsage,
		},
		{
			name: "returns nil without assistant messages",
			entries: []SessionEntry{
				testMessageEntry("user-1", testUserMessage("prompt")),
				&ModelChangeEntry{SessionEntryBase: SessionEntryBase{Type: "model_change", ID: "model"}},
			},
		},
		{
			name: "returns nil when all assistant messages are invalid",
			entries: []SessionEntry{
				testMessageEntry("assistant-1", testAssistantMessage("aborted response", abortedUsage, ai.StopReasonAborted)),
				testMessageEntry("assistant-2", testAssistantMessage("error response", errorUsage, ai.StopReasonError)),
			},
		},
		{
			name: "accepts usage from a length-limited assistant message",
			entries: []SessionEntry{
				testMessageEntry("assistant-1", testAssistantMessage("partial response", lengthUsage, ai.StopReasonLength)),
			},
			want: &lengthUsage,
		},
		{
			name: "accepts usage from an assistant tool call",
			entries: []SessionEntry{
				testMessageEntry("assistant-1", testAssistantMessage("tool call", toolUseUsage, ai.StopReasonToolUse)),
			},
			want: &toolUseUsage,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := GetLastAssistantUsage(test.entries)
			if !reflect.DeepEqual(got, test.want) {
				t.Fatalf("GetLastAssistantUsage() = %#v, want %#v", got, test.want)
			}
		})
	}
}

func TestShouldTriggerCompaction(t *testing.T) {
	tests := []struct {
		name          string
		contextTokens int64
		contextWindow int64
		settings      CompactionSettings
		want          bool
	}{
		{
			name:          "does not trigger below threshold",
			contextTokens: 89_999,
			contextWindow: 100_000,
			settings:      CompactionSettings{Enabled: true, ReservedTokens: 10_000},
		},
		{
			name:          "does not trigger at threshold",
			contextTokens: 90_000,
			contextWindow: 100_000,
			settings:      CompactionSettings{Enabled: true, ReservedTokens: 10_000},
		},
		{
			name:          "triggers above threshold",
			contextTokens: 90_001,
			contextWindow: 100_000,
			settings:      CompactionSettings{Enabled: true, ReservedTokens: 10_000},
			want:          true,
		},
		{
			name:          "does not trigger when disabled",
			contextTokens: 99_999,
			contextWindow: 100_000,
			settings:      CompactionSettings{Enabled: false, ReservedTokens: 10_000},
		},
		{
			name:          "does not trigger for an unknown context window",
			contextTokens: 1,
			contextWindow: 0,
			settings:      CompactionSettings{Enabled: true, ReservedTokens: 10_000},
		},
		{
			name:          "does not trigger for a negative context window",
			contextTokens: 1,
			contextWindow: -1,
			settings:      CompactionSettings{Enabled: true, ReservedTokens: 10_000},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := ShouldTriggerCompaction(test.contextTokens, test.contextWindow, test.settings)
			if got != test.want {
				t.Fatalf("ShouldTriggerCompaction() = %t, want %t", got, test.want)
			}
		})
	}
}

func TestEstimateTokens(t *testing.T) {
	tests := []struct {
		name    string
		message agent.AgentMessage
		want    int64
	}{
		{
			name: "user text blocks",
			message: ai.UserMessage{
				Role: ai.RoleUser,
				Content: []ai.UserContent{
					ai.TextContent{Type: ai.ContentTypeText, Text: "abc"},
					ai.TextContent{Type: ai.ContentTypeText, Text: "defgh"},
				},
			},
			want: 2,
		},
		{
			name: "user image counts as 4800 characters",
			message: ai.UserMessage{
				Role: ai.RoleUser,
				Content: []ai.UserContent{
					ai.ImageContent{Type: ai.ContentTypeImage, Image: "image-data", MimeType: "image/png"},
				},
			},
			want: 1200,
		},
		{
			name: "assistant text blocks",
			message: ai.AssistantMessage{
				Role: ai.RoleAssistant,
				Content: []ai.AssistantContent{
					ai.TextContent{Type: ai.ContentTypeText, Text: "abc"},
					ai.TextContent{Type: ai.ContentTypeText, Text: "defgh"},
				},
			},
			want: 2,
		},
		{
			name: "assistant thinking",
			message: ai.AssistantMessage{
				Role: ai.RoleAssistant,
				Content: []ai.AssistantContent{
					ai.ThinkingContent{Type: ai.ContentTypeThinking, Thinking: "abcdefghi"},
				},
			},
			want: 3,
		},
		{
			name: "assistant tool call name and arguments",
			message: ai.AssistantMessage{
				Role: ai.RoleAssistant,
				Content: []ai.AssistantContent{
					ai.ToolCall{
						Type: ai.ContentTypeToolCall,
						Id:   "call-id",
						Name: "run",
						Args: map[string]any{"x": 1},
					},
				},
			},
			want: 3,
		},
		{
			// "run" + {"cmd":"a<b&c"}: 18 characters, not the 28 with HTML escaping.
			name: "assistant tool call arguments without HTML escaping",
			message: ai.AssistantMessage{
				Role: ai.RoleAssistant,
				Content: []ai.AssistantContent{
					ai.ToolCall{
						Type: ai.ContentTypeToolCall,
						Id:   "call-id",
						Name: "run",
						Args: map[string]any{"cmd": "a<b&c"},
					},
				},
			},
			want: 5,
		},
		{
			name: "custom text",
			message: CustomMessage{
				Role: MessageRoleCustom,
				Content: []ai.UserContent{
					ai.TextContent{Type: ai.ContentTypeText, Text: "abcdefghi"},
				},
			},
			want: 3,
		},
		{
			name: "custom image",
			message: CustomMessage{
				Role: MessageRoleCustom,
				Content: []ai.UserContent{
					ai.ImageContent{Type: ai.ContentTypeImage, Image: "image-data", MimeType: "image/png"},
				},
			},
			want: 1200,
		},
		{
			name: "tool result text",
			message: ai.ToolResultMessage{
				Role: ai.RoleToolResult,
				Content: []ai.ToolResultContent{
					ai.TextContent{Type: ai.ContentTypeText, Text: "abcdefghi"},
				},
			},
			want: 3,
		},
		{
			name: "tool result image",
			message: ai.ToolResultMessage{
				Role: ai.RoleToolResult,
				Content: []ai.ToolResultContent{
					ai.ImageContent{Type: ai.ContentTypeImage, Image: "image-data", MimeType: "image/png"},
				},
			},
			want: 1200,
		},
		{
			name: "branch summary",
			message: BranchSummaryMessage{
				Role:    MessageRoleBranchSummary,
				Summary: "abcdefghi",
			},
			want: 3,
		},
		{
			name: "compaction summary",
			message: CompactionSummaryMessage{
				Role:    MessageRoleCompactionSummary,
				Summary: "abcde",
			},
			want: 2,
		},
		{
			name:    "unicode uses UTF-16 length",
			message: testUserMessage("😀😀😀"),
			want:    2,
		},
		{
			name:    "unknown message type",
			message: unknownCompactionAgentMessage{},
		},
		{
			name: "nil message",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := EstimateTokens(test.message); got != test.want {
				t.Fatalf("EstimateTokens() = %d, want %d", got, test.want)
			}
		})
	}
}

func TestUTF16Length(t *testing.T) {
	tests := []struct {
		name  string
		value string
		want  int64
	}{
		{name: "empty", value: "", want: 0},
		{name: "ASCII", value: "abcd", want: 4},
		{name: "one emoji", value: "😀", want: 2},
		{name: "mixed text", value: "a😀b", want: 4},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := utf16Length(test.value); got != test.want {
				t.Fatalf("utf16Length() = %d, want %d", got, test.want)
			}
		})
	}
}

func TestEstimateContextTokens(t *testing.T) {
	tests := []struct {
		name               string
		messages           []agent.AgentMessage
		wantTokens         int64
		wantUsageTokens    int64
		wantTrailingTokens int64
		wantLastUsageIndex int
		hasLastUsage       bool
	}{
		{
			name: "empty messages",
		},
		{
			name: "estimates all messages when no assistant usage exists",
			messages: []agent.AgentMessage{
				testUserMessage("abcde"),
				testToolResultMessage("abcdefghi"),
			},
			wantTokens:         5,
			wantTrailingTokens: 5,
		},
		{
			name: "uses the last assistant usage",
			messages: []agent.AgentMessage{
				testUserMessage("this earlier message is included in provider usage"),
				testAssistantMessage("response", ai.Usage{Total: 125}, ai.StopReasonStop),
			},
			wantTokens:         125,
			wantUsageTokens:    125,
			wantLastUsageIndex: 1,
			hasLastUsage:       true,
		},
		{
			name: "adds a trailing user estimate",
			messages: []agent.AgentMessage{
				testAssistantMessage("response", ai.Usage{Total: 100}, ai.StopReasonStop),
				testUserMessage("abcde"),
			},
			wantTokens:         102,
			wantUsageTokens:    100,
			wantTrailingTokens: 2,
			wantLastUsageIndex: 0,
			hasLastUsage:       true,
		},
		{
			name: "adds a trailing tool result estimate",
			messages: []agent.AgentMessage{
				testAssistantMessage("response", ai.Usage{Total: 100}, ai.StopReasonStop),
				testToolResultMessage("abcdefghi"),
			},
			wantTokens:         103,
			wantUsageTokens:    100,
			wantTrailingTokens: 3,
			wantLastUsageIndex: 0,
			hasLastUsage:       true,
		},
		{
			name: "ignores aborted assistant usage and estimates its content",
			messages: []agent.AgentMessage{
				testAssistantMessage("response", ai.Usage{Total: 100}, ai.StopReasonStop),
				testAssistantMessage("abcdefgh", ai.Usage{Total: 999}, ai.StopReasonAborted),
			},
			wantTokens:         102,
			wantUsageTokens:    100,
			wantTrailingTokens: 2,
			wantLastUsageIndex: 0,
			hasLastUsage:       true,
		},
		{
			name: "ignores error assistant usage and estimates its content",
			messages: []agent.AgentMessage{
				testAssistantMessage("response", ai.Usage{Total: 100}, ai.StopReasonStop),
				testAssistantMessage("abcdefghijkl", ai.Usage{Total: 999}, ai.StopReasonError),
			},
			wantTokens:         103,
			wantUsageTokens:    100,
			wantTrailingTokens: 3,
			wantLastUsageIndex: 0,
			hasLastUsage:       true,
		},
		{
			name: "uses estimates when all assistant messages are invalid",
			messages: []agent.AgentMessage{
				testAssistantMessage("abcdefgh", ai.Usage{Total: 888}, ai.StopReasonAborted),
				testAssistantMessage("abcdefghijkl", ai.Usage{Total: 999}, ai.StopReasonError),
			},
			wantTokens:         5,
			wantTrailingTokens: 5,
		},
		{
			name: "uses only messages after the latest valid usage as trailing messages",
			messages: []agent.AgentMessage{
				testAssistantMessage("first response", ai.Usage{Total: 100}, ai.StopReasonStop),
				testUserMessage("abcd"),
				testAssistantMessage("second response", ai.Usage{Total: 250}, ai.StopReasonStop),
				testUserMessage("abcdefgh"),
			},
			wantTokens:         252,
			wantUsageTokens:    250,
			wantTrailingTokens: 2,
			wantLastUsageIndex: 2,
			hasLastUsage:       true,
		},
		{
			name: "calculates assistant usage from components",
			messages: []agent.AgentMessage{
				testAssistantMessage("response", ai.Usage{
					Input:      10,
					Output:     5,
					CacheRead:  3,
					CacheWrite: 2,
				}, ai.StopReasonStop),
			},
			wantTokens:         20,
			wantUsageTokens:    20,
			wantLastUsageIndex: 0,
			hasLastUsage:       true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := EstimateContextTokens(test.messages)
			if got.tokens != test.wantTokens {
				t.Errorf("tokens = %d, want %d", got.tokens, test.wantTokens)
			}
			if got.usageTokens != test.wantUsageTokens {
				t.Errorf("usageTokens = %d, want %d", got.usageTokens, test.wantUsageTokens)
			}
			if got.trailingTokens != test.wantTrailingTokens {
				t.Errorf("trailingTokens = %d, want %d", got.trailingTokens, test.wantTrailingTokens)
			}
			if test.hasLastUsage {
				if got.lastUsageIndex == nil {
					t.Fatalf("lastUsageIndex = nil, want %d", test.wantLastUsageIndex)
				}
				if *got.lastUsageIndex != test.wantLastUsageIndex {
					t.Errorf("lastUsageIndex = %d, want %d", *got.lastUsageIndex, test.wantLastUsageIndex)
				}
			} else if got.lastUsageIndex != nil {
				t.Errorf("lastUsageIndex = %d, want nil", *got.lastUsageIndex)
			}
		})
	}
}

type unknownCompactionAgentMessage struct{}

func (unknownCompactionAgentMessage) AgentMessageType() string {
	return "unknown"
}

// ============================================================================
// Cutpoint Tests
// ============================================================================

func TestFindValidCutPoints(t *testing.T) {
	var nilMessageEntry *SessionMessageEntry
	entries := []SessionEntry{
		testMessageEntry("user", testUserMessage("user")),
		testMessageEntry("assistant", cutPointAssistantMessage("assistant")),
		testMessageEntry("tool-result", testToolResultMessage("tool result")),
		testMessageEntry("custom-message", CustomMessage{Role: MessageRoleCustom}),
		testMessageEntry("branch-message", BranchSummaryMessage{Role: MessageRoleBranchSummary}),
		testMessageEntry("compaction-message", CompactionSummaryMessage{Role: MessageRoleCompactionSummary}),
		&BranchSummaryEntry{SessionEntryBase: testEntryBase("branch_summary", "branch-entry")},
		&CustomMessageEntry{SessionEntryBase: testEntryBase("custom_message", "custom-entry")},
		&ModelChangeEntry{SessionEntryBase: testEntryBase("model_change", "model")},
		nilMessageEntry,
	}

	tests := []struct {
		name       string
		startIndex int
		endIndex   int
		want       []int
	}{
		{
			name:       "returns every supported cut point",
			startIndex: 0,
			endIndex:   len(entries),
			want:       []int{0, 1, 3, 4, 5, 6, 7},
		},
		{
			name:       "respects range boundaries",
			startIndex: 1,
			endIndex:   7,
			want:       []int{1, 3, 4, 5, 6},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := findValidCutPoints(entries, test.startIndex, test.endIndex)
			if !reflect.DeepEqual(got, test.want) {
				t.Fatalf("findValidCutPoints() = %v, want %v", got, test.want)
			}
		})
	}
}

func TestFindTurnStartIndex(t *testing.T) {
	tests := []struct {
		name       string
		entries    []SessionEntry
		entryIndex int
		startIndex int
		want       int
	}{
		{
			name: "finds the preceding user message",
			entries: []SessionEntry{
				testMessageEntry("user", testUserMessage("request")),
				testMessageEntry("assistant", cutPointAssistantMessage("response")),
				testMessageEntry("tool-result", testToolResultMessage("result")),
			},
			entryIndex: 2,
			want:       0,
		},
		{
			name: "uses a custom message entry as the turn start",
			entries: []SessionEntry{
				testMessageEntry("user", testUserMessage("request")),
				&CustomMessageEntry{SessionEntryBase: testEntryBase("custom_message", "custom")},
				testMessageEntry("assistant", cutPointAssistantMessage("response")),
			},
			entryIndex: 2,
			want:       1,
		},
		{
			name: "uses a branch summary entry as the turn start",
			entries: []SessionEntry{
				testMessageEntry("user", testUserMessage("request")),
				&BranchSummaryEntry{SessionEntryBase: testEntryBase("branch_summary", "branch")},
				testMessageEntry("assistant", cutPointAssistantMessage("response")),
			},
			entryIndex: 2,
			want:       1,
		},
		{
			name: "returns minus one without a preceding turn start",
			entries: []SessionEntry{
				testMessageEntry("assistant", cutPointAssistantMessage("response")),
				testMessageEntry("tool-result", testToolResultMessage("result")),
			},
			entryIndex: 1,
			want:       -1,
		},
		{
			name: "does not search before start index",
			entries: []SessionEntry{
				testMessageEntry("user", testUserMessage("request")),
				testMessageEntry("assistant-1", cutPointAssistantMessage("first response")),
				testMessageEntry("assistant-2", cutPointAssistantMessage("second response")),
			},
			entryIndex: 2,
			startIndex: 1,
			want:       -1,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := FindTurnStartIndex(test.entries, test.entryIndex, test.startIndex)
			if got != test.want {
				t.Fatalf("FindTurnStartIndex() = %d, want %d", got, test.want)
			}
		})
	}
}

func TestFindCutPointAtUserMessageDoesNotSplitTurn(t *testing.T) {
	entries := []SessionEntry{
		testMessageEntry("user-1", testUserMessage("abcd")),
		testMessageEntry("assistant-1", cutPointAssistantMessage("abcd")),
		testMessageEntry("user-2", testUserMessage("abcd")),
		testMessageEntry("assistant-2", cutPointAssistantMessage("abcd")),
	}

	got := FindCutPoint(entries, 0, len(entries), 2)
	want := CutPointResult{FirstKeptEntryIndex: 2, TurnStartIndex: -1, IsSplitTurn: false}
	if got != want {
		t.Fatalf("FindCutPoint() = %#v, want %#v", got, want)
	}
}

func TestFindCutPointAtAssistantMessageSplitsTurn(t *testing.T) {
	entries := []SessionEntry{
		testMessageEntry("user-1", testUserMessage("abcd")),
		testMessageEntry("assistant-1", cutPointAssistantMessage("abcd")),
		testMessageEntry("user-2", testUserMessage("abcd")),
		testMessageEntry("assistant-2", cutPointAssistantMessage("abcd")),
	}

	got := FindCutPoint(entries, 0, len(entries), 1)
	want := CutPointResult{FirstKeptEntryIndex: 3, TurnStartIndex: 2, IsSplitTurn: true}
	if got != want {
		t.Fatalf("FindCutPoint() = %#v, want %#v", got, want)
	}
}

func TestFindCutPointKeepsToolResultWithAssistantToolCall(t *testing.T) {
	entries := []SessionEntry{
		testMessageEntry("user", testUserMessage("abcd")),
		testMessageEntry("assistant", ai.AssistantMessage{
			Role: ai.RoleAssistant,
			Content: []ai.AssistantContent{
				ai.ToolCall{Type: ai.ContentTypeToolCall, Id: "call-id", Name: "run", Args: nil},
			},
		}),

		testMessageEntry("tool-result", testToolResultMessage("abcd")),
	}

	got := FindCutPoint(entries, 0, len(entries), 2)
	if got.FirstKeptEntryIndex != 1 {
		t.Fatalf("FirstKeptEntryIndex = %d, want 1", got.FirstKeptEntryIndex)
	}
	if entries[got.FirstKeptEntryIndex].(*SessionMessageEntry).Message.AgentMessageType() != string(ai.RoleAssistant) {
		t.Fatalf("first kept entry is not the assistant tool call")
	}
	if entries[got.FirstKeptEntryIndex+1].(*SessionMessageEntry).Message.AgentMessageType() != string(ai.RoleToolResult) {
		t.Fatalf("the tool result does not follow the first kept entry")
	}
}

func TestFindCutPointIncludesPrecedingNonMessageEntries(t *testing.T) {
	entries := []SessionEntry{
		testMessageEntry("user-1", testUserMessage("abcd")),
		testMessageEntry("assistant-1", cutPointAssistantMessage("abcd")),
		&ModelChangeEntry{SessionEntryBase: testEntryBase("model_change", "model")},
		&ThinkingLevelChangeEntry{SessionEntryBase: testEntryBase("thinking_level_change", "thinking")},
		testMessageEntry("user-2", testUserMessage("abcd")),
		testMessageEntry("assistant-2", cutPointAssistantMessage("abcd")),
	}

	got := FindCutPoint(entries, 0, len(entries), 2)
	if got.FirstKeptEntryIndex != 2 {
		t.Fatalf("FirstKeptEntryIndex = %d, want 2", got.FirstKeptEntryIndex)
	}
}

func TestFindCutPointStopsAtPreviousCompactionBoundary(t *testing.T) {
	entries := []SessionEntry{
		testMessageEntry("user-1", testUserMessage("abcd")),
		testMessageEntry("assistant-1", cutPointAssistantMessage("abcd")),
		&CompactionEntry{SessionEntryBase: testEntryBase("compaction", "compaction")},
		&ModelChangeEntry{SessionEntryBase: testEntryBase("model_change", "model")},
		testMessageEntry("user-2", testUserMessage("abcd")),
		testMessageEntry("assistant-2", cutPointAssistantMessage("abcd")),
	}

	got := FindCutPoint(entries, 0, len(entries), 2)
	if got.FirstKeptEntryIndex != 3 {
		t.Fatalf("FirstKeptEntryIndex = %d, want 3", got.FirstKeptEntryIndex)
	}
}

func TestFindCutPointWithoutValidCutPointsReturnsStartIndex(t *testing.T) {
	entries := []SessionEntry{
		testMessageEntry("user", testUserMessage("abcd")),
		testMessageEntry("tool-result", testToolResultMessage("abcd")),
		&ModelChangeEntry{SessionEntryBase: testEntryBase("model_change", "model")},
	}

	got := FindCutPoint(entries, 1, len(entries), 1)
	want := CutPointResult{FirstKeptEntryIndex: 1, TurnStartIndex: -1, IsSplitTurn: false}
	if got != want {
		t.Fatalf("FindCutPoint() = %#v, want %#v", got, want)
	}
}

func TestFindCutPointKeepsEverythingWithinBudget(t *testing.T) {
	entries := []SessionEntry{
		testMessageEntry("user", testUserMessage("abcd")),
		testMessageEntry("assistant", cutPointAssistantMessage("abcd")),
	}

	got := FindCutPoint(entries, 0, len(entries), 100)
	want := CutPointResult{FirstKeptEntryIndex: 0, TurnStartIndex: -1, IsSplitTurn: false}
	if got != want {
		t.Fatalf("FindCutPoint() = %#v, want %#v", got, want)
	}
}

func TestFindCutPointIgnoresRepliesTheModelNeverSees(t *testing.T) {
	for _, stopReason := range []ai.StopReason{ai.StopReasonAborted, ai.StopReasonError} {
		t.Run(string(stopReason), func(t *testing.T) {
			// A long partial reply that TransformMessages drops before the provider.
			unseen := testAssistantMessage(strings.Repeat("x", 400), ai.Usage{}, stopReason)
			entries := []SessionEntry{
				testMessageEntry("user-1", testUserMessage("abcd")),
				testMessageEntry("assistant-1", cutPointAssistantMessage("abcd")),
				testMessageEntry("user-2", testUserMessage("abcd")),
				testMessageEntry("unseen", unseen),
			}

			// user-2 alone fills the budget; the unseen reply must not.
			got := FindCutPoint(entries, 0, len(entries), 1)
			want := CutPointResult{FirstKeptEntryIndex: 2, TurnStartIndex: -1, IsSplitTurn: false}
			if got != want {
				t.Fatalf("FindCutPoint() = %#v, want %#v", got, want)
			}
		})
	}
}

func TestFindCutPointCanCutAtUnseenReplyWhenNothingElseFits(t *testing.T) {
	// Overflow recovery after a huge tool result: the only cut point after the
	// result is the overflow error, so the whole turn must become the prefix.
	overflow := testAssistantMessage("", ai.Usage{}, ai.StopReasonError)
	overflow.ErrorMessage = "prompt is too long"
	entries := []SessionEntry{
		testMessageEntry("user", testUserMessage("abcd")),
		testMessageEntry("assistant", ai.AssistantMessage{
			Role: ai.RoleAssistant,
			Content: []ai.AssistantContent{
				ai.ToolCall{Type: ai.ContentTypeToolCall, Id: "call-id", Name: "run", Args: nil},
			},
		}),
		testMessageEntry("tool-result", testToolResultMessage(strings.Repeat("x", 4000))),
		testMessageEntry("overflow", overflow),
	}

	got := FindCutPoint(entries, 0, len(entries), 2)
	want := CutPointResult{FirstKeptEntryIndex: 3, TurnStartIndex: 0, IsSplitTurn: true}
	if got != want {
		t.Fatalf("FindCutPoint() = %#v, want %#v", got, want)
	}
}

func cutPointAssistantMessage(text string) ai.AssistantMessage {
	return testAssistantMessage(text, ai.Usage{}, "")
}

// ============================================================================
// Serialization Tests
// ============================================================================

func TestSerializeConversationTruncatesLongToolResults(t *testing.T) {
	longContent := strings.Repeat("x", 5000)
	messages := []ai.Message{
		serializationToolResultMessage(
			ai.TextContent{Type: ai.ContentTypeText, Text: longContent},
		),
	}

	got := serializeConversation(messages)
	want := "[Tool result]: " + strings.Repeat("x", 2000) + "\n\n[... 3000 more characters truncated]"
	if got != want {
		t.Fatalf("serializeConversation() returned an unexpected truncated tool result")
	}
}

func TestSerializeConversationDoesNotTruncateShortToolResults(t *testing.T) {
	shortContent := strings.Repeat("x", 1500)
	messages := []ai.Message{
		serializationToolResultMessage(
			ai.TextContent{Type: ai.ContentTypeText, Text: shortContent},
		),
	}

	got := serializeConversation(messages)
	want := "[Tool result]: " + shortContent
	if got != want {
		t.Fatalf("serializeConversation() = %q, want %q", got, want)
	}
}

func TestSerializeConversationDoesNotTruncateUserOrAssistantMessages(t *testing.T) {
	longContent := strings.Repeat("y", 5000)
	messages := []ai.Message{
		ai.UserMessage{
			Role: ai.RoleUser,
			Content: []ai.UserContent{
				ai.TextContent{Type: ai.ContentTypeText, Text: longContent},
			},
		},
		ai.AssistantMessage{
			Role: ai.RoleAssistant,
			Content: []ai.AssistantContent{
				ai.TextContent{Type: ai.ContentTypeText, Text: longContent},
			},
		},
	}

	got := serializeConversation(messages)
	if strings.Contains(got, "truncated") {
		t.Fatalf("serializeConversation() truncated a user or assistant message")
	}
	want := "[User]: " + longContent + "\n\n[Assistant]: " + longContent
	if got != want {
		t.Fatalf("serializeConversation() did not preserve the complete messages")
	}
}

func TestSerializeConversationFormatsEveryMessageSection(t *testing.T) {
	messages := []ai.Message{
		ai.UserMessage{
			Role: ai.RoleUser,
			Content: []ai.UserContent{
				ai.TextContent{Type: ai.ContentTypeText, Text: "first"},
				ai.ImageContent{Type: ai.ContentTypeImage, Image: "image-data", MimeType: "image/png"},
				ai.TextContent{Type: ai.ContentTypeText, Text: " second"},
			},
		},
		ai.AssistantMessage{
			Role: ai.RoleAssistant,
			Content: []ai.AssistantContent{
				ai.TextContent{Type: ai.ContentTypeText, Text: "answer one"},
				ai.ThinkingContent{Type: ai.ContentTypeThinking, Thinking: "plan one"},
				ai.ToolCall{
					Type: ai.ContentTypeToolCall,
					Id:   "edit-call",
					Name: "edit",
					Args: map[string]any{"z": "last", "a": 1},
				},
				ai.ThinkingContent{Type: ai.ContentTypeThinking, Thinking: "plan two"},
				ai.TextContent{Type: ai.ContentTypeText, Text: "answer two"},
				ai.ToolCall{
					Type: ai.ContentTypeToolCall,
					Id:   "read-call",
					Name: "read",
					Args: map[string]any{"path": "file.go"},
				},
			},
		},
		serializationToolResultMessage(
			ai.TextContent{Type: ai.ContentTypeText, Text: "part one"},
			ai.ImageContent{Type: ai.ContentTypeImage, Image: "image-data", MimeType: "image/png"},
			ai.TextContent{Type: ai.ContentTypeText, Text: "part two"},
		),
	}

	got := serializeConversation(messages)
	want := `[User]: first second

[Assistant thinking]: plan one
plan two

[Assistant]: answer one
answer two

[Assistant tool calls]: edit(a=1, z="last"); read(path="file.go")

[Tool result]: part onepart two`
	if got != want {
		t.Fatalf("serializeConversation() = %q, want %q", got, want)
	}
}

func TestSerializeConversationOmitsEmptySections(t *testing.T) {
	messages := []ai.Message{
		ai.UserMessage{
			Role: ai.RoleUser,
			Content: []ai.UserContent{
				ai.ImageContent{Type: ai.ContentTypeImage, Image: "image-data", MimeType: "image/png"},
			},
		},
		ai.AssistantMessage{Role: ai.RoleAssistant},
		serializationToolResultMessage(
			ai.ImageContent{Type: ai.ContentTypeImage, Image: "image-data", MimeType: "image/png"},
		),
	}

	if got := serializeConversation(messages); got != "" {
		t.Fatalf("serializeConversation() = %q, want empty text", got)
	}
}

func TestFormatToolArguments(t *testing.T) {
	type argumentObject struct {
		Z string `json:"z"`
		A int    `json:"a"`
	}

	tests := []struct {
		name string
		args any
		want string
	}{
		{
			name: "sorts map keys",
			args: map[string]any{"z": "last", "a": 1, "m": true},
			want: `a=1, m=true, z="last"`,
		},
		{
			name: "formats nested JSON",
			args: map[string]any{
				"options": map[string]any{"retries": 3, "force": true},
			},
			want: `options={"force":true,"retries":3}`,
		},
		{
			name: "does not HTML-escape values",
			args: map[string]any{"cmd": "a < b && c > d", "nested": map[string]any{"html": "<p>"}},
			want: `cmd="a < b && c > d", nested={"html":"<p>"}`,
		},
		{
			name: "normalizes an object to sorted arguments",
			args: argumentObject{Z: "last", A: 1},
			want: `a=1, z="last"`,
		},
		{
			name: "marks an unserializable argument value",
			args: map[string]any{"callback": func() {}},
			want: "callback=[unserializable]",
		},
		{
			name: "marks unserializable top-level arguments",
			args: func() {},
			want: "[unserializable]",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := formatToolArguments(test.args); got != test.want {
				t.Fatalf("formatToolArguments() = %q, want %q", got, test.want)
			}
		})
	}
}

func TestTruncateForSummaryBoundaries(t *testing.T) {
	tests := []struct {
		name string
		text string
		want string
	}{
		{
			name: "exact ASCII limit",
			text: strings.Repeat("a", 2000),
			want: strings.Repeat("a", 2000),
		},
		{
			name: "one ASCII unit over limit",
			text: strings.Repeat("a", 2001),
			want: strings.Repeat("a", 2000) + "\n\n[... 1 more characters truncated]",
		},
		{
			name: "exact UTF-16 limit",
			text: strings.Repeat("a", 1998) + "😀",
			want: strings.Repeat("a", 1998) + "😀",
		},
		{
			name: "one UTF-16 unit over limit",
			text: strings.Repeat("a", 1998) + "😀b",
			want: strings.Repeat("a", 1998) + "😀\n\n[... 1 more characters truncated]",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := truncateForSummary(test.text, 2000); got != test.want {
				t.Fatalf("truncateForSummary() = %q, want %q", got, test.want)
			}
		})
	}
}

func serializationToolResultMessage(content ...ai.ToolResultContent) ai.ToolResultMessage {
	return ai.ToolResultMessage{
		Role:       ai.RoleToolResult,
		ToolCallID: "call-id",
		ToolName:   "tool",
		Content:    content,
	}
}

// ============================================================================
// Preparation Tests
// ============================================================================

func TestPrepareCompactionTrivialCases(t *testing.T) {
	settings := CompactionSettings{Enabled: true, ReservedTokens: 100, KeepRecentTokens: 100}

	t.Run("empty entries", func(t *testing.T) {
		if got := PrepareCompaction(nil, settings); got != nil {
			t.Fatalf("PrepareCompaction() = %#v, want nil", got)
		}
	})

	t.Run("final compaction entry", func(t *testing.T) {
		entries := linkPreparationEntries(
			testMessageEntry("user", testUserMessage("request")),
			&CompactionEntry{
				SessionEntryBase: testEntryBase("compaction", "compaction"),
				Summary:          "summary",
				FirstKeptEntryID: "user",
				TokensBefore:     100,
			},
		)
		if got := PrepareCompaction(entries, settings); got != nil {
			t.Fatalf("PrepareCompaction() = %#v, want nil", got)
		}
	})

	t.Run("empty first kept entry ID", func(t *testing.T) {
		entries := linkPreparationEntries(
			testMessageEntry("", testUserMessage("request")),
		)
		if got := PrepareCompaction(entries, settings); got != nil {
			t.Fatalf("PrepareCompaction() = %#v, want nil", got)
		}
	})

	t.Run("all messages fit", func(t *testing.T) {
		entries := linkPreparationEntries(testMessageEntry("user", testUserMessage("request")))
		if got := PrepareCompaction(entries, settings); got != nil {
			t.Fatalf("PrepareCompaction() = %#v, want nil", got)
		}
	})
}

func TestPrepareCompactionWithoutPreviousCompaction(t *testing.T) {
	userOne := testUserMessage("aaaa")
	assistantOne := preparationAssistantMessage("bbbb", 100)
	userTwo := testUserMessage("cccc")
	assistantTwo := preparationAssistantMessage("dddd", 4321)
	entries := linkPreparationEntries(
		testMessageEntry("user-1", userOne),
		testMessageEntry("assistant-1", assistantOne),
		testMessageEntry("user-2", userTwo),
		testMessageEntry("assistant-2", assistantTwo),
	)
	settings := CompactionSettings{Enabled: true, ReservedTokens: 500, KeepRecentTokens: 2}

	got := PrepareCompaction(entries, settings)
	if got == nil {
		t.Fatalf("PrepareCompaction() = nil, want preparation")
	}
	if got.FirstKeptEntryID != "user-2" {
		t.Errorf("FirstKeptEntryID = %q, want %q", got.FirstKeptEntryID, "user-2")
	}
	wantSummaryMessages := []agent.AgentMessage{userOne, assistantOne}
	if !reflect.DeepEqual(got.MessagesToSummarize, wantSummaryMessages) {
		t.Errorf("MessagesToSummarize = %#v, want %#v", got.MessagesToSummarize, wantSummaryMessages)
	}
	if len(got.TurnPrefixMessages) != 0 || got.IsSplitTurn {
		t.Errorf("unexpected split-turn preparation: %#v", got)
	}
	if got.PreviousSummary != nil {
		t.Errorf("PreviousSummary = %q, want nil", *got.PreviousSummary)
	}
	if got.TokensBefore != 4321 {
		t.Errorf("TokensBefore = %d, want 4321", got.TokensBefore)
	}
	contextBefore := BuildSessionContext(entries, nil, nil)
	if want := EstimateContextTokens(contextBefore.Messages).tokens; got.TokensBefore != want {
		t.Errorf("TokensBefore = %d, want context estimate %d", got.TokensBefore, want)
	}
	if got.Settings != settings {
		t.Errorf("Settings = %#v, want %#v", got.Settings, settings)
	}
}

func TestPrepareCompactionForSplitTurn(t *testing.T) {
	userOne := testUserMessage("aaaa")
	assistantOne := preparationAssistantMessage("bbbb", 100)
	userTwo := testUserMessage("cccc")
	assistantToolCall := ai.AssistantMessage{
		Role: ai.RoleAssistant,
		Content: []ai.AssistantContent{
			ai.ToolCall{Type: ai.ContentTypeToolCall, Id: "call-id", Name: "read", Args: nil},
		},
		StopReason: ai.StopReasonToolUse,
		Usage:      ai.Usage{Total: 300},
	}
	toolResult := ai.ToolResultMessage{
		Role:       ai.RoleToolResult,
		ToolCallID: "call-id",
		ToolName:   "read",
		Content: []ai.ToolResultContent{
			ai.TextContent{Type: ai.ContentTypeText, Text: "dddd"},
		},
	}
	assistantFinal := preparationAssistantMessage("eeee", 500)
	entries := linkPreparationEntries(
		testMessageEntry("user-1", userOne),
		testMessageEntry("assistant-1", assistantOne),
		testMessageEntry("user-2", userTwo),
		testMessageEntry("assistant-tool", assistantToolCall),
		testMessageEntry("tool-result", toolResult),
		testMessageEntry("assistant-final", assistantFinal),
	)
	settings := CompactionSettings{Enabled: true, ReservedTokens: 500, KeepRecentTokens: 1}

	got := PrepareCompaction(entries, settings)
	if got == nil {
		t.Fatalf("PrepareCompaction() = nil, want preparation")
	}
	if got.FirstKeptEntryID != "assistant-final" {
		t.Errorf("FirstKeptEntryID = %q, want %q", got.FirstKeptEntryID, "assistant-final")
	}
	if !got.IsSplitTurn {
		t.Errorf("IsSplitTurn = false, want true")
	}
	wantSummaryMessages := []agent.AgentMessage{userOne, assistantOne}
	if !reflect.DeepEqual(got.MessagesToSummarize, wantSummaryMessages) {
		t.Errorf("MessagesToSummarize = %#v, want %#v", got.MessagesToSummarize, wantSummaryMessages)
	}
	wantPrefixMessages := []agent.AgentMessage{userTwo, assistantToolCall, toolResult}
	if !reflect.DeepEqual(got.TurnPrefixMessages, wantPrefixMessages) {
		t.Errorf("TurnPrefixMessages = %#v, want %#v", got.TurnPrefixMessages, wantPrefixMessages)
	}
	if got.TokensBefore != 500 {
		t.Errorf("TokensBefore = %d, want 500", got.TokensBefore)
	}
}

func TestPrepareCompactionSkipsWhenPreviouslyKeptMessagesFit(t *testing.T) {
	userOne := testUserMessage("user msg 1")
	assistantOne := preparationAssistantMessage("assistant msg 1", 100)
	userTwo := testUserMessage("user msg 2 - kept by compaction1")
	assistantTwo := preparationAssistantMessage("assistant msg 2", 200)
	userThree := testUserMessage("user msg 3 - kept by compaction1")
	assistantThree := preparationAssistantMessage("assistant msg 3", 300)
	userFour := testUserMessage("user msg 4 - new after compaction1")
	assistantFour := preparationAssistantMessage("assistant msg 4", 10_000)
	entries := linkPreparationEntries(
		testMessageEntry("user-1", userOne),
		testMessageEntry("assistant-1", assistantOne),
		testMessageEntry("user-2", userTwo),
		testMessageEntry("assistant-2", assistantTwo),
		testMessageEntry("user-3", userThree),
		testMessageEntry("assistant-3", assistantThree),
		&CompactionEntry{
			SessionEntryBase: testEntryBase("compaction", "compaction-1"),
			Summary:          "First summary",
			FirstKeptEntryID: "user-2",
			TokensBefore:     1000,
		},
		testMessageEntry("user-4", userFour),
		testMessageEntry("assistant-4", assistantFour),
	)
	settings := CompactionSettings{Enabled: true, ReservedTokens: 500, KeepRecentTokens: 20_000}

	// Messages kept by the previous compaction and everything after it fit in
	// KeepRecentTokens. A second compaction would summarize nothing and replace
	// the first summary with a copy of itself.
	if got := PrepareCompaction(entries, settings); got != nil {
		t.Fatalf("PrepareCompaction() = %#v, want nil", got)
	}
}

func TestPrepareCompactionResummarizesPreviouslyKeptMessages(t *testing.T) {
	userOne := testUserMessage(strings.Repeat("user msg 1 ", 4))
	assistantOne := preparationAssistantMessage(strings.Repeat("assistant msg 1 ", 4), 100)
	userTwoText := strings.Repeat("user msg 2 - kept by compaction1 ", 12)
	userTwo := testUserMessage(userTwoText)
	assistantTwo := preparationAssistantMessage(strings.Repeat("assistant msg 2 ", 12), 200)
	userThreeText := strings.Repeat("user msg 3 - kept by compaction1 ", 12)
	userThree := testUserMessage(userThreeText)
	assistantThree := preparationAssistantMessage(strings.Repeat("assistant msg 3 ", 12), 300)
	userFour := testUserMessage(strings.Repeat("user msg 4 - new after compaction1 ", 12))
	assistantFour := preparationAssistantMessage(strings.Repeat("assistant msg 4 ", 12), 10_000)
	entries := linkPreparationEntries(
		testMessageEntry("user-1", userOne),
		testMessageEntry("assistant-1", assistantOne),
		testMessageEntry("user-2", userTwo),
		testMessageEntry("assistant-2", assistantTwo),
		testMessageEntry("user-3", userThree),
		testMessageEntry("assistant-3", assistantThree),
		&CompactionEntry{
			SessionEntryBase: testEntryBase("compaction", "compaction-1"),
			Summary:          "First summary",
			FirstKeptEntryID: "user-2",
			TokensBefore:     1000,
		},
		testMessageEntry("user-4", userFour),
		testMessageEntry("assistant-4", assistantFour),
	)
	settings := CompactionSettings{Enabled: true, ReservedTokens: 500, KeepRecentTokens: 100}

	got := PrepareCompaction(entries, settings)
	if got == nil {
		t.Fatalf("PrepareCompaction() = nil, want preparation")
	}
	if got.PreviousSummary == nil || *got.PreviousSummary != "First summary" {
		t.Errorf("PreviousSummary = %v, want %q", got.PreviousSummary, "First summary")
	}
	if !preparationMessagesContainText(got.MessagesToSummarize, userTwoText) {
		t.Errorf("MessagesToSummarize does not contain the second user message")
	}
	if !preparationMessagesContainText(got.MessagesToSummarize, userThreeText) {
		t.Errorf("MessagesToSummarize does not contain the third user message")
	}
	if preparationMessagesContainText(got.MessagesToSummarize, "First summary") {
		t.Errorf("MessagesToSummarize contains the previous compaction summary")
	}
}

func TestPrepareCompactionUsesEntryAfterMissingPreviousBoundary(t *testing.T) {
	userOne := testUserMessage("aaaa")
	assistantOne := preparationAssistantMessage("bbbb", 100)
	userTwo := testUserMessage("cccc")
	assistantTwo := preparationAssistantMessage("dddd", 200)
	userThree := testUserMessage("eeee")
	assistantThree := preparationAssistantMessage("ffff", 400)
	entries := linkPreparationEntries(
		testMessageEntry("user-1", userOne),
		testMessageEntry("assistant-1", assistantOne),
		&CompactionEntry{
			SessionEntryBase: testEntryBase("compaction", "compaction-1"),
			Summary:          "First summary",
			FirstKeptEntryID: "missing-entry",
			TokensBefore:     100,
		},
		testMessageEntry("user-2", userTwo),
		testMessageEntry("assistant-2", assistantTwo),
		testMessageEntry("user-3", userThree),
		testMessageEntry("assistant-3", assistantThree),
	)
	settings := CompactionSettings{Enabled: true, ReservedTokens: 500, KeepRecentTokens: 2}

	got := PrepareCompaction(entries, settings)
	if got == nil {
		t.Fatalf("PrepareCompaction() = nil, want preparation")
	}
	if got.FirstKeptEntryID != "user-3" {
		t.Errorf("FirstKeptEntryID = %q, want %q", got.FirstKeptEntryID, "user-3")
	}
	wantSummaryMessages := []agent.AgentMessage{userTwo, assistantTwo}
	if !reflect.DeepEqual(got.MessagesToSummarize, wantSummaryMessages) {
		t.Errorf("MessagesToSummarize = %#v, want %#v", got.MessagesToSummarize, wantSummaryMessages)
	}
	if got.PreviousSummary == nil || *got.PreviousSummary != "First summary" {
		t.Errorf("PreviousSummary = %v, want %q", got.PreviousSummary, "First summary")
	}
}

func preparationAssistantMessage(text string, totalTokens int64) ai.AssistantMessage {
	return testAssistantMessage(text, ai.Usage{Total: totalTokens}, ai.StopReasonStop)
}

func linkPreparationEntries(entries ...SessionEntry) []SessionEntry {
	var parentID *string
	for _, entry := range entries {
		entry.GetBase().ParentID = parentID
		id := entry.GetBase().ID
		parentID = &id
	}
	return entries
}

func preparationMessagesContainText(messages []agent.AgentMessage, text string) bool {
	for _, message := range messages {
		switch message := message.(type) {
		case ai.UserMessage:
			for _, content := range message.Content {
				if textContent, ok := content.(ai.TextContent); ok && strings.Contains(textContent.Text, text) {
					return true
				}
			}
		case ai.AssistantMessage:
			for _, content := range message.Content {
				if textContent, ok := content.(ai.TextContent); ok && strings.Contains(textContent.Text, text) {
					return true
				}
			}
		case ai.ToolResultMessage:
			for _, content := range message.Content {
				if textContent, ok := content.(ai.TextContent); ok && strings.Contains(textContent.Text, text) {
					return true
				}
			}
		case CompactionSummaryMessage:
			if strings.Contains(message.Summary, text) {
				return true
			}
		}
	}
	return false
}

// ============================================================================
// Summarization Tests
// ============================================================================

type summaryTestRequest struct {
	modelContext ai.ModelContext
	options      ai.SimpleStreamOptions
}

func TestGenerateSummaryBuildsInitialRequest(t *testing.T) {
	var request summaryTestRequest
	model := registerSummaryTestProvider(t, true, func(
		_ ai.Model[ai.API],
		modelContext ai.ModelContext,
		options *ai.SimpleStreamOptions,
	) (*ai.AssistantMessageEventStream, error) {
		request = summaryTestRequest{modelContext: modelContext, options: *options}
		return completedSummaryTestStream(ai.AssistantMessage{
			Role:       ai.RoleAssistant,
			Content:    []ai.AssistantContent{ai.TextContent{Type: ai.ContentTypeText, Text: "generated summary"}},
			StopReason: ai.StopReasonStop,
		}), nil
	})

	type contextKey string
	ctx := context.WithValue(context.Background(), contextKey("request"), "summary")
	headers := map[string]string{"X-Test": "value"}
	result, err := GenerateSummary(
		ctx,
		[]agent.AgentMessage{testUserMessage("summarize this")},
		model,
		101,
		"test-key",
		&GenerateSummaryOptions{
			Headers:       headers,
			ThinkingLevel: ai.ThinkingLevelHigh,
		},
	)
	if err != nil {
		t.Fatalf("GenerateSummary() error = %v", err)
	}
	if result != "generated summary" {
		t.Fatalf("GenerateSummary() = %q, want %q", result, "generated summary")
	}

	if request.modelContext.SystemPrompt == nil || *request.modelContext.SystemPrompt != SUMMARIZATION_SYSTEM_PROMPT {
		t.Errorf("system prompt = %v, want SUMMARIZATION_SYSTEM_PROMPT", request.modelContext.SystemPrompt)
	}
	wantPrompt := "<conversation>\n[User]: summarize this\n</conversation>\n\n" + SUMMARIZATION_PROMPT
	if got := summaryTestPrompt(t, request.modelContext); got != wantPrompt {
		t.Errorf("prompt = %q, want %q", got, wantPrompt)
	}
	if request.options.MaxTokens == nil || *request.options.MaxTokens != 80 {
		t.Errorf("MaxTokens = %v, want 80", request.options.MaxTokens)
	}
	if request.options.ApiKey == nil || *request.options.ApiKey != "test-key" {
		t.Errorf("ApiKey = %v, want test-key", request.options.ApiKey)
	}
	if !reflect.DeepEqual(request.options.Headers, headers) {
		t.Errorf("Headers = %#v, want %#v", request.options.Headers, headers)
	}
	if request.options.Signal != ctx {
		t.Errorf("Signal does not contain the supplied context")
	}
	if request.options.Reasoning == nil || *request.options.Reasoning != ai.ThinkingLevelHigh {
		t.Errorf("Reasoning = %v, want %q", request.options.Reasoning, ai.ThinkingLevelHigh)
	}
}

func TestGenerateSummaryBuildsUpdateRequest(t *testing.T) {
	var request summaryTestRequest
	model := registerSummaryTestProvider(t, false, func(
		_ ai.Model[ai.API],
		modelContext ai.ModelContext,
		options *ai.SimpleStreamOptions,
	) (*ai.AssistantMessageEventStream, error) {
		request = summaryTestRequest{modelContext: modelContext, options: *options}
		return completedSummaryTestStream(summaryTestTextMessage("summary")), nil
	})

	_, err := GenerateSummary(
		context.Background(),
		[]agent.AgentMessage{testUserMessage("new work")},
		model,
		100,
		"test-key",
		&GenerateSummaryOptions{
			PreviousSummary:    "old summary",
			CustomInstructions: "focus on errors",
		},
	)
	if err != nil {
		t.Fatalf("GenerateSummary() error = %v", err)
	}

	wantPrompt := "<conversation>\n[User]: new work\n</conversation>\n\n" +
		"<previous-summary>\nold summary\n</previous-summary>\n\n" +
		UPDATE_SUMMARIZATION_PROMPT + "\n\nAdditional focus: focus on errors"
	if got := summaryTestPrompt(t, request.modelContext); got != wantPrompt {
		t.Fatalf("prompt = %q, want %q", got, wantPrompt)
	}
}

func TestGenerateSummaryReasoningOptions(t *testing.T) {
	tests := []struct {
		name           string
		modelReasoning bool
		thinkingLevel  ai.ModelThinkingLevel
		wantReasoning  ai.ModelThinkingLevel
		hasReasoning   bool
	}{
		{
			name:           "uses the provided level for a reasoning model",
			modelReasoning: true,
			thinkingLevel:  ai.ThinkingLevelMedium,
			wantReasoning:  ai.ThinkingLevelMedium,
			hasReasoning:   true,
		},
		{
			name:           "omits reasoning when thinking is off",
			modelReasoning: true,
			thinkingLevel:  ai.ThinkingLevelOff,
		},
		{
			name:           "omits reasoning for a non-reasoning model",
			modelReasoning: false,
			thinkingLevel:  ai.ThinkingLevelMedium,
		},
		{
			name:           "omits reasoning without a thinking level",
			modelReasoning: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var gotReasoning ai.ModelThinkingLevel
			var hasReasoning bool
			model := registerSummaryTestProvider(t, test.modelReasoning, func(
				_ ai.Model[ai.API],
				_ ai.ModelContext,
				options *ai.SimpleStreamOptions,
			) (*ai.AssistantMessageEventStream, error) {
				if options.Reasoning != nil {
					hasReasoning = true
					gotReasoning = *options.Reasoning
				}
				return completedSummaryTestStream(summaryTestTextMessage("summary")), nil
			})

			_, err := GenerateSummary(
				context.Background(),
				[]agent.AgentMessage{testUserMessage("message")},
				model,
				100,
				"test-key",
				&GenerateSummaryOptions{ThinkingLevel: test.thinkingLevel},
			)
			if err != nil {
				t.Fatalf("GenerateSummary() error = %v", err)
			}
			if hasReasoning != test.hasReasoning {
				t.Fatalf("Reasoning presence = %t, want %t", hasReasoning, test.hasReasoning)
			}
			if hasReasoning && gotReasoning != test.wantReasoning {
				t.Fatalf("Reasoning = %q, want %q", gotReasoning, test.wantReasoning)
			}
		})
	}
}

func TestGenerateSummaryErrors(t *testing.T) {
	t.Run("provider error", func(t *testing.T) {
		providerError := errors.New("provider failed")
		model := registerSummaryTestProvider(t, false, func(
			ai.Model[ai.API],
			ai.ModelContext,
			*ai.SimpleStreamOptions,
		) (*ai.AssistantMessageEventStream, error) {
			return nil, providerError
		})

		_, err := GenerateSummary(context.Background(), nil, model, 100, "test-key", nil)
		if !errors.Is(err, providerError) {
			t.Fatalf("GenerateSummary() error = %v, want wrapped provider error", err)
		}
		if err.Error() != "summarization failed: provider failed" {
			t.Fatalf("GenerateSummary() error = %q", err)
		}
	})

	tests := []struct {
		name         string
		errorMessage string
		want         string
	}{
		{
			name:         "response error message",
			errorMessage: "model failed",
			want:         "summarization failed: model failed",
		},
		{
			name: "response error without message",
			want: "summarization failed: unknown error",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			model := registerSummaryTestProvider(t, false, func(
				ai.Model[ai.API],
				ai.ModelContext,
				*ai.SimpleStreamOptions,
			) (*ai.AssistantMessageEventStream, error) {
				return completedSummaryTestStream(ai.AssistantMessage{
					StopReason:   ai.StopReasonError,
					ErrorMessage: test.errorMessage,
				}), nil
			})

			_, err := GenerateSummary(context.Background(), nil, model, 100, "test-key", nil)
			if err == nil || err.Error() != test.want {
				t.Fatalf("GenerateSummary() error = %v, want %q", err, test.want)
			}
		})
	}
}

// A summary replaces the history it covers, so a partial or empty response
// must fail compaction rather than be persisted.
func TestGenerateSummaryRejectsUnusableResponses(t *testing.T) {
	text := []ai.AssistantContent{ai.TextContent{Type: ai.ContentTypeText, Text: "partial summary"}}
	tests := []struct {
		name     string
		response ai.AssistantMessage
		want     string
	}{
		{
			name:     "truncated at the output limit",
			response: ai.AssistantMessage{Content: text, StopReason: ai.StopReasonLength},
			want:     "summarization failed: summary was truncated at the output token limit",
		},
		{
			name:     "aborted",
			response: ai.AssistantMessage{Content: text, StopReason: ai.StopReasonAborted},
			want:     "summarization failed: response was aborted",
		},
		{
			name:     "no content",
			response: ai.AssistantMessage{StopReason: ai.StopReasonStop},
			want:     "summarization failed: summary is empty",
		},
		{
			name: "whitespace text",
			response: ai.AssistantMessage{
				Content:    []ai.AssistantContent{ai.TextContent{Type: ai.ContentTypeText, Text: " \n\t"}},
				StopReason: ai.StopReasonStop,
			},
			want: "summarization failed: summary is empty",
		},
		{
			name: "no text content",
			response: ai.AssistantMessage{
				Content: []ai.AssistantContent{
					ai.ThinkingContent{Type: ai.ContentTypeThinking, Thinking: "private reasoning"},
					ai.ToolCall{Type: ai.ContentTypeToolCall, Id: "call-id", Name: "tool", Args: map[string]any{}},
				},
				StopReason: ai.StopReasonStop,
			},
			want: "summarization failed: summary is empty",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			model := registerSummaryTestProvider(t, false, func(
				ai.Model[ai.API],
				ai.ModelContext,
				*ai.SimpleStreamOptions,
			) (*ai.AssistantMessageEventStream, error) {
				return completedSummaryTestStream(test.response), nil
			})

			_, err := GenerateSummary(context.Background(), nil, model, 100, "test-key", nil)
			if err == nil || err.Error() != test.want {
				t.Fatalf("GenerateSummary() error = %v, want %q", err, test.want)
			}
		})
	}
}

func TestGenerateSummaryReturnsOnlyTextContent(t *testing.T) {
	tests := []struct {
		name    string
		content []ai.AssistantContent
		want    string
	}{
		{
			name: "joins text and ignores non-text content",
			content: []ai.AssistantContent{
				ai.ThinkingContent{Type: ai.ContentTypeThinking, Thinking: "private reasoning"},
				ai.TextContent{Type: ai.ContentTypeText, Text: "first"},
				ai.ToolCall{Type: ai.ContentTypeToolCall, Id: "call-id", Name: "tool", Args: map[string]any{}},
				ai.TextContent{Type: ai.ContentTypeText, Text: "second"},
			},
			want: "first\nsecond",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			model := registerSummaryTestProvider(t, false, func(
				ai.Model[ai.API],
				ai.ModelContext,
				*ai.SimpleStreamOptions,
			) (*ai.AssistantMessageEventStream, error) {
				return completedSummaryTestStream(ai.AssistantMessage{
					Content:    test.content,
					StopReason: ai.StopReasonStop,
				}), nil
			})

			got, err := GenerateSummary(context.Background(), nil, model, 100, "test-key", nil)
			if err != nil {
				t.Fatalf("GenerateSummary() error = %v", err)
			}
			if got != test.want {
				t.Fatalf("GenerateSummary() = %q, want %q", got, test.want)
			}
		})
	}
}

func TestCompactUsesCustomSummaryPrompts(t *testing.T) {
	var mu sync.Mutex
	var systemPrompts []string
	var prompts []string
	model := registerSummaryTestProvider(t, false, func(
		_ ai.Model[ai.API],
		modelContext ai.ModelContext,
		_ *ai.SimpleStreamOptions,
	) (*ai.AssistantMessageEventStream, error) {
		mu.Lock()
		systemPrompts = append(systemPrompts, *modelContext.SystemPrompt)
		prompts = append(prompts, summaryTestPrompt(t, modelContext))
		mu.Unlock()
		return completedSummaryTestStream(ai.AssistantMessage{
			Content:    []ai.AssistantContent{ai.TextContent{Type: ai.ContentTypeText, Text: "summary"}},
			StopReason: ai.StopReasonStop,
		}), nil
	})

	previousSummary := "previous summary"
	preparation := &CompactionPreparation{
		FirstKeptEntryID: "kept-entry",
		MessagesToSummarize: []agent.AgentMessage{
			ai.UserMessage{Role: ai.RoleUser, Content: []ai.UserContent{ai.TextContent{Type: ai.ContentTypeText, Text: "history"}}},
		},
		TurnPrefixMessages: []agent.AgentMessage{
			ai.UserMessage{Role: ai.RoleUser, Content: []ai.UserContent{ai.TextContent{Type: ai.ContentTypeText, Text: "prefix"}}},
		},
		IsSplitTurn:     true,
		PreviousSummary: &previousSummary,
		Settings:        CompactionSettings{ReservedTokens: 100},
	}
	_, err := Compact(context.Background(), preparation, model, "test-key", &CompactOptions{
		Prompts: SummaryPrompts{
			System:     "custom system",
			Initial:    "custom initial",
			Update:     "custom update",
			TurnPrefix: "custom turn prefix",
		},
	})
	if err != nil {
		t.Fatalf("Compact() error = %v", err)
	}

	sort.Strings(prompts)
	if len(prompts) != 2 ||
		!strings.Contains(prompts[0], "history") || !strings.HasSuffix(prompts[0], "</previous-summary>\n\ncustom update") ||
		!strings.Contains(prompts[1], "prefix") || !strings.HasSuffix(prompts[1], "</conversation>\n\ncustom turn prefix") {
		t.Errorf("summary prompts = %q, want the custom update and turn prefix prompts", prompts)
	}
	if want := []string{"custom system", "custom system"}; !reflect.DeepEqual(systemPrompts, want) {
		t.Errorf("system prompts = %q, want %q", systemPrompts, want)
	}
}

func registerSummaryTestProvider(
	t *testing.T,
	reasoning bool,
	streamSimple ai.StreamFunction[ai.API, ai.SimpleStreamOptions],
) ai.Model[ai.API] {
	t.Helper()

	name := strings.NewReplacer("/", "-", " ", "-").Replace(t.Name())
	testAPI := ai.API("summary-test-" + name)
	sourceID := string(testAPI)
	ai.RegisterApiProvider(ai.ApiProvider[ai.API, ai.StreamOptions]{
		Api:          testAPI,
		StreamSimple: streamSimple,
	}, &sourceID)
	t.Cleanup(func() {
		ai.UnregisterApiProvider(sourceID)
	})

	return ai.Model[ai.API]{
		ID:        "summary-test-model",
		API:       testAPI,
		Reasoning: reasoning,
	}
}

func completedSummaryTestStream(message ai.AssistantMessage) *ai.AssistantMessageEventStream {
	stream := ai.NewAssistantMessageEventStream(1)
	stream.Push(ai.DoneEvent{Type: "done", Message: message})
	return stream
}

func summaryTestTextMessage(text string) ai.AssistantMessage {
	return ai.AssistantMessage{
		Role:       ai.RoleAssistant,
		Content:    []ai.AssistantContent{ai.TextContent{Type: ai.ContentTypeText, Text: text}},
		StopReason: ai.StopReasonStop,
	}
}

func summaryTestPrompt(t *testing.T, modelContext ai.ModelContext) string {
	t.Helper()
	if len(modelContext.Messages) != 1 {
		t.Fatalf("message count = %d, want 1", len(modelContext.Messages))
	}
	message, ok := modelContext.Messages[0].(ai.UserMessage)
	if !ok {
		t.Fatalf("message type = %T, want ai.UserMessage", modelContext.Messages[0])
	}
	if len(message.Content) != 1 {
		t.Fatalf("content count = %d, want 1", len(message.Content))
	}
	text, ok := message.Content[0].(ai.TextContent)
	if !ok {
		t.Fatalf("content type = %T, want ai.TextContent", message.Content[0])
	}
	return text.Text
}

// ============================================================================
// Full Integration Tests
// ============================================================================

func TestCompactionPrepareCompactAppendAndReloadFlow(t *testing.T) {
	ctx := context.Background()
	sessionManager, err := InMemorySessionManager(ctx)
	if err != nil {
		t.Fatalf("InMemorySessionManager() error = %v", err)
	}

	appendCompactionFlowMessage(t, ctx, sessionManager, "old-user", testUserMessage("old1"))
	appendCompactionFlowMessage(t, ctx, sessionManager, "old-assistant", preparationAssistantMessage("old2", 50))
	appendCompactionFlowMessage(t, ctx, sessionManager, "kept-user", testUserMessage("keep"))
	appendCompactionFlowMessage(t, ctx, sessionManager, "kept-assistant", preparationAssistantMessage("kept", 100))

	var prompts []string
	model := registerSummaryTestProvider(t, false, func(
		_ ai.Model[ai.API],
		modelContext ai.ModelContext,
		_ *ai.SimpleStreamOptions,
	) (*ai.AssistantMessageEventStream, error) {
		prompt := summaryTestPrompt(t, modelContext)
		prompts = append(prompts, prompt)

		summary := "first summary"
		if strings.Contains(prompt, "<previous-summary>\nfirst summary\n</previous-summary>") {
			summary = "second summary"
		}
		return completedSummaryTestStream(ai.AssistantMessage{
			Role: ai.RoleAssistant,
			Content: []ai.AssistantContent{
				ai.TextContent{Type: ai.ContentTypeText, Text: summary},
			},
			StopReason: ai.StopReasonStop,
		}), nil
	})
	settings := CompactionSettings{Enabled: true, ReservedTokens: 100, KeepRecentTokens: 2}

	firstPreparation := PrepareCompaction(sessionManager.GetEntries(), settings)
	if firstPreparation == nil {
		t.Fatalf("first PrepareCompaction() = nil, want preparation")
	}
	if firstPreparation.FirstKeptEntryID != "kept-user" {
		t.Fatalf("first FirstKeptEntryID = %q, want %q", firstPreparation.FirstKeptEntryID, "kept-user")
	}

	firstResult, err := Compact(ctx, firstPreparation, model, "test-key", nil)
	if err != nil {
		t.Fatalf("first Compact() error = %v", err)
	}
	if firstResult.Summary != "first summary" {
		t.Fatalf("first summary = %q, want %q", firstResult.Summary, "first summary")
	}
	if _, err := sessionManager.AppendCompaction(
		ctx,
		firstResult.Summary,
		firstResult.FirstKeptEntryID,
		firstResult.TokensBefore,
		nil,
		nil,
	); err != nil {
		t.Fatalf("first AppendCompaction() error = %v", err)
	}

	appendCompactionFlowMessage(t, ctx, sessionManager, "new-user", testUserMessage("new1"))
	appendCompactionFlowMessage(t, ctx, sessionManager, "new-assistant", preparationAssistantMessage("new2", 200))

	firstContext := sessionManager.BuildSessionContext()
	wantFirstContext := []string{
		"summary:first summary",
		"user:keep",
		"assistant:kept",
		"user:new1",
		"assistant:new2",
	}
	if got := compactionFlowContextValues(t, firstContext.Messages); !reflect.DeepEqual(got, wantFirstContext) {
		t.Fatalf("first compacted context = %v, want %v", got, wantFirstContext)
	}

	secondPreparation := PrepareCompaction(sessionManager.GetEntries(), settings)
	if secondPreparation == nil {
		t.Fatalf("second PrepareCompaction() = nil, want preparation")
	}
	if secondPreparation.FirstKeptEntryID != "new-user" {
		t.Fatalf("second FirstKeptEntryID = %q, want %q", secondPreparation.FirstKeptEntryID, "new-user")
	}
	if secondPreparation.PreviousSummary == nil || *secondPreparation.PreviousSummary != "first summary" {
		t.Fatalf("second PreviousSummary = %v, want %q", secondPreparation.PreviousSummary, "first summary")
	}

	secondResult, err := Compact(ctx, secondPreparation, model, "test-key", nil)
	if err != nil {
		t.Fatalf("second Compact() error = %v", err)
	}
	if secondResult.Summary != "second summary" {
		t.Fatalf("second summary = %q, want %q", secondResult.Summary, "second summary")
	}
	if _, err := sessionManager.AppendCompaction(
		ctx,
		secondResult.Summary,
		secondResult.FirstKeptEntryID,
		secondResult.TokensBefore,
		nil,
		nil,
	); err != nil {
		t.Fatalf("second AppendCompaction() error = %v", err)
	}

	secondContext := sessionManager.BuildSessionContext()
	wantSecondContext := []string{
		"summary:second summary",
		"user:new1",
		"assistant:new2",
	}
	if got := compactionFlowContextValues(t, secondContext.Messages); !reflect.DeepEqual(got, wantSecondContext) {
		t.Fatalf("second compacted context = %v, want %v", got, wantSecondContext)
	}

	if len(prompts) != 2 {
		t.Fatalf("summary request count = %d, want 2", len(prompts))
	}
	if !strings.Contains(prompts[0], "[User]: old1") || !strings.Contains(prompts[0], "[Assistant]: old2") {
		t.Errorf("first prompt does not contain the discarded history")
	}
	if strings.Contains(prompts[0], "[User]: keep") {
		t.Errorf("first prompt contains a kept message")
	}
	if !strings.Contains(prompts[1], "<previous-summary>\nfirst summary\n</previous-summary>") {
		t.Errorf("second prompt does not contain the first summary")
	}
	if !strings.Contains(prompts[1], "[User]: keep") || !strings.Contains(prompts[1], "[Assistant]: kept") {
		t.Errorf("second prompt does not contain the newly discarded history")
	}
	if strings.Contains(prompts[1], "[User]: new1") {
		t.Errorf("second prompt contains a kept message")
	}
}

func appendCompactionFlowMessage(
	t *testing.T,
	ctx context.Context,
	sessionManager *SessionManager,
	id string,
	message agent.AgentMessage,
) {
	t.Helper()
	if _, err := sessionManager.AppendMessageWithID(ctx, id, message); err != nil {
		t.Fatalf("AppendMessageWithID(%q) error = %v", id, err)
	}
}

func compactionFlowContextValues(t *testing.T, messages []agent.AgentMessage) []string {
	t.Helper()
	values := make([]string, 0, len(messages))
	for _, message := range messages {
		switch message := message.(type) {
		case CompactionSummaryMessage:
			values = append(values, "summary:"+message.Summary)
		case ai.UserMessage:
			values = append(values, "user:"+compactionFlowUserText(t, message))
		case ai.AssistantMessage:
			values = append(values, "assistant:"+compactionFlowAssistantText(t, message))
		default:
			t.Fatalf("unexpected context message type %T", message)
		}
	}
	return values
}

func compactionFlowUserText(t *testing.T, message ai.UserMessage) string {
	t.Helper()
	if len(message.Content) != 1 {
		t.Fatalf("user content count = %d, want 1", len(message.Content))
	}
	content, ok := message.Content[0].(ai.TextContent)
	if !ok {
		t.Fatalf("user content type = %T, want ai.TextContent", message.Content[0])
	}
	return content.Text
}

func compactionFlowAssistantText(t *testing.T, message ai.AssistantMessage) string {
	t.Helper()
	if len(message.Content) != 1 {
		t.Fatalf("assistant content count = %d, want 1", len(message.Content))
	}
	content, ok := message.Content[0].(ai.TextContent)
	if !ok {
		t.Fatalf("assistant content type = %T, want ai.TextContent", message.Content[0])
	}
	return content.Text
}

// ============================================================================
// Compaction Hook Tests
// ============================================================================

// jsonCompactionSessionStore keeps compaction entries as JSON and decodes them
// on load, as a database store would. Other entries are returned unchanged.
type jsonCompactionSessionStore struct {
	appendErrorSessionStore

	mu      sync.Mutex
	header  *SessionHeader
	entries []SessionEntry
	encoded map[string][]byte

	// onAppendCompaction runs before a compaction entry is stored.
	onAppendCompaction func(ctx context.Context)
}

func (s *jsonCompactionSessionStore) CreateSession(_ context.Context, header *SessionHeader) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.header = header
	return nil
}

func (s *jsonCompactionSessionStore) AppendEntry(ctx context.Context, _ *SessionHeader, entry SessionEntry) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if compaction, ok := entry.(*CompactionEntry); ok {
		if s.onAppendCompaction != nil {
			s.onAppendCompaction(ctx)
		}
		encoded, err := json.Marshal(compaction)
		if err != nil {
			return err
		}
		if s.encoded == nil {
			s.encoded = map[string][]byte{}
		}
		s.encoded[compaction.ID] = encoded
	}
	s.entries = append(s.entries, entry)
	return nil
}

func (s *jsonCompactionSessionStore) LoadSession(context.Context, string) (*SessionHeader, []SessionEntry, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	entries := make([]SessionEntry, len(s.entries))
	for i, entry := range s.entries {
		entries[i] = entry
		if encoded, ok := s.encoded[entry.GetBase().ID]; ok {
			var decoded CompactionEntry
			if err := json.Unmarshal(encoded, &decoded); err != nil {
				return nil, nil, err
			}
			entries[i] = &decoded
		}
	}
	return s.header, entries, nil
}

type hookTestSummaryRequest struct {
	systemPrompt string
	prompt       string
}

// hookTestSession is a persistent session whose summary requests are recorded.
type hookTestSession struct {
	ctx     context.Context
	model   ai.Model[ai.API]
	store   *jsonCompactionSessionStore
	session *AgentSession

	mu       sync.Mutex
	requests []hookTestSummaryRequest
}

func newHookTestSession(t *testing.T) *hookTestSession {
	t.Helper()
	h := &hookTestSession{ctx: context.Background(), store: &jsonCompactionSessionStore{}}
	h.model = registerManualCompactionProviderWithStream(t, func(
		_ ai.Model[ai.API],
		modelContext ai.ModelContext,
		_ *ai.SimpleStreamOptions,
	) (*ai.AssistantMessageEventStream, error) {
		request := hookTestSummaryRequest{}
		if modelContext.SystemPrompt != nil {
			request.systemPrompt = *modelContext.SystemPrompt
		}
		request.prompt = userText(t, modelContext.Messages[0])
		h.mu.Lock()
		h.requests = append(h.requests, request)
		h.mu.Unlock()
		return autoCompactionStream(manualCompactionAssistantMessage("default summary", h.model)), nil
	})

	sm, err := NewSessionManager(h.ctx, nil, nil, true, h.store, nil)
	if err != nil {
		t.Fatal(err)
	}
	seedManualCompactionHistory(t, h.ctx, sm, h.model)
	h.session = h.newSession(t, sm)
	return h
}

func (h *hookTestSession) newSession(t *testing.T, sm *SessionManager) *AgentSession {
	t.Helper()
	apiKey := "test-key"
	a := agent.NewAgent(&agent.AgentOptions{
		InitialState: &agent.InitialAgentState{
			Model:    &h.model,
			Messages: sm.BuildSessionContext().Messages,
		},
		GetAPIKey: func(string) *string { return &apiKey },
	})
	session := NewAgentSession(a, sm)
	session.SetCompactionSettings(CompactionSettings{Enabled: true, ReservedTokens: 100, KeepRecentTokens: 2})
	return session
}

func (h *hookTestSession) summaryRequests() []hookTestSummaryRequest {
	h.mu.Lock()
	defer h.mu.Unlock()
	return append([]hookTestSummaryRequest(nil), h.requests...)
}

// assertNothingCompacted checks that no compaction was saved and that the live
// context still matches the session.
func (h *hookTestSession) assertNothingCompacted(t *testing.T) {
	t.Helper()
	if latest := GetLatestCompactionEntry(h.session.SessionManager.GetEntries()); latest != nil {
		t.Errorf("compaction entry was saved: %#v", latest)
	}
	for _, entry := range h.store.entries {
		if _, ok := entry.(*CompactionEntry); ok {
			t.Errorf("compaction entry was persisted: %#v", entry)
		}
	}
	if got, want := h.session.Agent.State().Messages, h.session.SessionManager.BuildSessionContext().Messages; !reflect.DeepEqual(got, want) {
		t.Errorf("live context differs from persisted context:\ngot  %#v\nwant %#v", got, want)
	}
	if got := len(h.session.Agent.State().Messages); got != 4 {
		t.Errorf("live message count = %d, want the 4 uncompacted messages", got)
	}
}

type hookTestFileOps struct {
	Files []string `json:"files"`
}

func TestAgentSessionCompactionHookAugmentsDefaultAndDetailsSurviveReload(t *testing.T) {
	h := newHookTestSession(t)
	var mu sync.Mutex
	var previous []hookTestFileOps
	var reasons []CompactionReason
	fileOpsHook := func(file string) CompactionHook {
		return func(ctx context.Context, req CompactionRequest, next CompactFunc) (*CompactionResult, error) {
			ops, ok, err := DecodeCompactionDetails[hookTestFileOps](req.Preparation.PreviousDetails)
			if err != nil {
				return nil, err
			}
			mu.Lock()
			reasons = append(reasons, req.Reason)
			if ok {
				previous = append(previous, ops)
			}
			mu.Unlock()
			ops.Files = append(ops.Files, file)

			req.Prompts.Initial = "custom initial prompt"
			result, err := next(ctx, req)
			if err != nil {
				return nil, err
			}
			result.Summary += "\nfiles: " + strings.Join(ops.Files, ", ")
			result.Details = ops
			return result, nil
		}
	}
	h.session.SetCompactionHook(fileOpsHook("a.go"))

	result, err := h.session.Compact(h.ctx, &ManualCompactionOptions{CustomInstructions: "focus on files"})
	if err != nil {
		t.Fatalf("Compact() error = %v", err)
	}
	if want := "default summary\nfiles: a.go"; result.Summary != want {
		t.Errorf("summary = %q, want %q", result.Summary, want)
	}
	requests := h.summaryRequests()
	if len(requests) != 1 {
		t.Fatalf("summary requests = %d, want 1", len(requests))
	}
	// Unchanged prompts keep their defaults; changed ones reach the model.
	if requests[0].systemPrompt != SUMMARIZATION_SYSTEM_PROMPT {
		t.Errorf("summary system prompt = %q, want the default", requests[0].systemPrompt)
	}
	if want := "custom initial prompt\n\nAdditional focus: focus on files"; !strings.HasSuffix(requests[0].prompt, want) {
		t.Errorf("summary prompt = %q, want suffix %q", requests[0].prompt, want)
	}

	latest := GetLatestCompactionEntry(h.session.SessionManager.GetEntries())
	if latest == nil || latest.Summary != result.Summary {
		t.Fatalf("saved compaction = %#v, want summary %q", latest, result.Summary)
	}
	if latest.FromHook != nil {
		t.Errorf("fromHook = %v, want unset when the hook used the default summary", *latest.FromHook)
	}
	if got := h.session.Agent.State().Messages[0].(CompactionSummaryMessage); !strings.Contains(got.Summary, "files: a.go") {
		t.Errorf("live summary = %q, want the hook's summary", got.Summary)
	}

	// The next compaction sees the same details whether or not the session was
	// reloaded, even though a reload decodes them as map[string]any.
	settings := h.session.GetCompactionSettings()
	sessionID := h.session.SessionManager.GetSessionID()
	sm, err := NewSessionManager(h.ctx, nil, &sessionID, true, h.store, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, sm := range []*SessionManager{h.session.SessionManager, sm} {
		for _, message := range []agent.AgentMessage{
			manualCompactionUserMessage("eeee"),
			manualCompactionAssistantMessage("ffff", h.model),
		} {
			if _, err := sm.AppendMessage(h.ctx, message); err != nil {
				t.Fatal(err)
			}
		}
	}
	live := PrepareCompaction(h.session.SessionManager.GetBranch(), settings)
	reloaded := PrepareCompaction(sm.GetBranch(), settings)
	if live == nil || reloaded == nil {
		t.Fatal("second compaction has nothing to compact")
	}
	if _, ok := GetLatestCompactionEntry(sm.GetEntries()).Details.(map[string]any); !ok {
		t.Fatalf("reloaded details = %T, want map[string]any from JSON", GetLatestCompactionEntry(sm.GetEntries()).Details)
	}
	if string(live.PreviousDetails) != string(reloaded.PreviousDetails) {
		t.Errorf("previous details before reload = %s, after reload = %s", live.PreviousDetails, reloaded.PreviousDetails)
	}

	reloadedSession := h.newSession(t, sm)
	reloadedSession.SetCompactionHook(fileOpsHook("b.go"))
	result, err = reloadedSession.Compact(h.ctx, nil)
	if err != nil {
		t.Fatalf("Compact() after reload error = %v", err)
	}
	if want := "default summary\nfiles: a.go, b.go"; result.Summary != want {
		t.Errorf("summary after reload = %q, want %q", result.Summary, want)
	}
	if want := []hookTestFileOps{{Files: []string{"a.go"}}}; !reflect.DeepEqual(previous, want) {
		t.Errorf("previous details seen by hook = %#v, want %#v", previous, want)
	}
	if want := []CompactionReason{CompactionReasonManual, CompactionReasonManual}; !reflect.DeepEqual(reasons, want) {
		t.Errorf("hook reasons = %v, want %v", reasons, want)
	}
	requests = h.summaryRequests()
	if len(requests) != 2 || !strings.Contains(requests[1].prompt, "<previous-summary>\ndefault summary\nfiles: a.go\n</previous-summary>") {
		t.Errorf("second summary request = %#v, want the hook's previous summary", requests)
	}
}

func TestAgentSessionCompactionHookReplacesSummary(t *testing.T) {
	h := newHookTestSession(t)
	var branchEntries int
	h.session.SetCompactionHook(func(_ context.Context, req CompactionRequest, _ CompactFunc) (*CompactionResult, error) {
		branchEntries = len(req.BranchEntries)
		return &CompactionResult{
			Summary:          "custom summary",
			FirstKeptEntryID: req.Preparation.FirstKeptEntryID,
			TokensBefore:     req.Preparation.TokensBefore,
			Details:          map[string]any{"source": "hook"},
		}, nil
	})

	result, err := h.session.Compact(h.ctx, nil)
	if err != nil {
		t.Fatalf("Compact() error = %v", err)
	}
	if result.Summary != "custom summary" {
		t.Errorf("summary = %q, want the hook's summary", result.Summary)
	}
	if got := len(h.summaryRequests()); got != 0 {
		t.Errorf("summary requests = %d, want 0 when the hook replaces the summary", got)
	}
	if branchEntries != 4 {
		t.Errorf("hook saw %d branch entries, want 4", branchEntries)
	}

	latest := GetLatestCompactionEntry(h.session.SessionManager.GetEntries())
	if latest == nil || latest.Summary != "custom summary" {
		t.Fatalf("saved compaction = %#v, want the hook's summary", latest)
	}
	if latest.FromHook == nil || !*latest.FromHook {
		t.Errorf("fromHook = %v, want true", latest.FromHook)
	}
	if got := string(latest.Details.(json.RawMessage)); got != `{"source":"hook"}` {
		t.Errorf("saved details = %s, want the hook's details", got)
	}
	if got := h.session.Agent.State().Messages[0].(CompactionSummaryMessage); got.Summary != "custom summary" {
		t.Errorf("live summary = %q, want the hook's summary", got.Summary)
	}
}

func TestAgentSessionCompactionHookCancelsManualCompaction(t *testing.T) {
	h := newHookTestSession(t)
	h.session.SetCompactionHook(func(context.Context, CompactionRequest, CompactFunc) (*CompactionResult, error) {
		return nil, ErrCompactionCancelled
	})
	var ends []CompactionEndEvent
	unsubscribe := h.session.Subscribe(func(event AgentSessionEvent) {
		if end, ok := event.(CompactionEndEvent); ok {
			ends = append(ends, end)
		}
	})
	defer unsubscribe()

	if _, err := h.session.Compact(h.ctx, nil); !errors.Is(err, ErrCompactionCancelled) {
		t.Fatalf("Compact() error = %v, want ErrCompactionCancelled", err)
	}
	if len(ends) != 1 || !ends[0].Aborted || ends[0].ErrorMessage != "" || ends[0].Result != nil {
		t.Errorf("compaction_end events = %#v, want one aborted event without an error message", ends)
	}
	if got := len(h.summaryRequests()); got != 0 {
		t.Errorf("summary requests = %d, want 0", got)
	}
	h.assertNothingCompacted(t)
}

// Once the summary is complete, cancelling compaction must not interrupt the
// write, but a stuck store must still fail.
func TestAgentSessionCompactionPersistsDespiteCancellationDuringWrite(t *testing.T) {
	h := newHookTestSession(t)
	var writeErr error
	var hasDeadline bool
	h.store.onAppendCompaction = func(ctx context.Context) {
		h.session.AbortCompaction()
		writeErr = ctx.Err()
		_, hasDeadline = ctx.Deadline()
	}

	result, err := h.session.Compact(h.ctx, nil)
	if err != nil {
		t.Fatalf("Compact() error = %v", err)
	}
	if writeErr != nil {
		t.Errorf("write context error = %v, want a write unaffected by AbortCompaction", writeErr)
	}
	if !hasDeadline {
		t.Error("write context has no deadline")
	}
	latest := GetLatestCompactionEntry(h.session.SessionManager.GetEntries())
	if latest == nil || latest.Summary != result.Summary {
		t.Fatalf("latest compaction = %#v, want the returned result", latest)
	}
	if got, want := h.session.Agent.State().Messages, h.session.SessionManager.BuildSessionContext().Messages; !reflect.DeepEqual(got, want) {
		t.Errorf("live context differs from persisted context:\ngot  %#v\nwant %#v", got, want)
	}
}

func TestAgentSessionCompactionTimeoutIsFailure(t *testing.T) {
	h := newHookTestSession(t)
	h.session.SetCompactionHook(func(ctx context.Context, _ CompactionRequest, _ CompactFunc) (*CompactionResult, error) {
		<-ctx.Done()
		return nil, ctx.Err()
	})
	var ends []CompactionEndEvent
	unsubscribe := h.session.Subscribe(func(event AgentSessionEvent) {
		if end, ok := event.(CompactionEndEvent); ok {
			ends = append(ends, end)
		}
	})
	defer unsubscribe()

	ctx, cancel := context.WithTimeout(h.ctx, 10*time.Millisecond)
	defer cancel()
	_, err := h.session.Compact(ctx, nil)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("Compact() error = %v, want context.DeadlineExceeded", err)
	}
	want := "Compaction failed: compaction timed out: context deadline exceeded"
	if len(ends) != 1 || ends[0].Aborted || ends[0].ErrorMessage != want {
		t.Errorf("compaction_end events = %#v, want one failure %q", ends, want)
	}
	h.assertNothingCompacted(t)
}

func TestIsCompactionAbort(t *testing.T) {
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	expired, cancelExpired := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
	defer cancelExpired()
	live := context.Background()
	failure := errors.New("provider failed")

	tests := []struct {
		name string
		ctx  context.Context
		err  error
		want bool
	}{
		{name: "cancelled compaction", ctx: cancelled, err: failure, want: true},
		{name: "expired compaction", ctx: expired, err: context.DeadlineExceeded},
		{name: "hook cancelled", ctx: live, err: fmt.Errorf("hook: %w", ErrCompactionCancelled), want: true},
		{name: "cancelled request", ctx: live, err: fmt.Errorf("request: %w", context.Canceled), want: true},
		{name: "request timeout", ctx: live, err: fmt.Errorf("request: %w", context.DeadlineExceeded)},
		{name: "provider failure", ctx: live, err: failure},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := isCompactionAbort(test.ctx, test.err); got != test.want {
				t.Fatalf("isCompactionAbort() = %t, want %t", got, test.want)
			}
		})
	}
}

func TestAgentSessionCompactionHookCancelsAutomaticCompactionBeforePrompt(t *testing.T) {
	h := newAutoCompactionHarness(t, false)
	var reason atomic.Value
	h.session.SetCompactionHook(func(_ context.Context, req CompactionRequest, _ CompactFunc) (*CompactionResult, error) {
		reason.Store(req.Reason)
		return nil, ErrCompactionCancelled
	})
	h.appendMessages(t, manualCompactionUserMessage("big"), h.response(950, ai.StopReasonStop))
	h.responses = []ai.AssistantMessage{h.response(40, ai.StopReasonStop)}
	log, unsubscribe := recordSessionEvents(h)
	defer unsubscribe()

	if err := h.session.Prompt(h.ctx, "next", nil); err != nil {
		t.Fatalf("Prompt error = %v, want nil after the hook cancelled compaction", err)
	}
	end := assertPromptSentAfterFailedCompaction(t, h, log, "next", true)
	if end.ErrorMessage != "" {
		t.Errorf("cancelled compaction_end errorMessage = %q, want empty", end.ErrorMessage)
	}
	if got := reason.Load(); got != CompactionReasonThreshold {
		t.Errorf("hook reason = %v, want %v", got, CompactionReasonThreshold)
	}
	if got := h.summaryCalls.Load(); got != 0 {
		t.Errorf("summary requests = %d, want 0", got)
	}
}

func TestAgentSessionCompactionHookRejectsInvalidResults(t *testing.T) {
	for _, test := range []struct {
		name   string
		result func(req CompactionRequest) *CompactionResult
		want   string
	}{
		{
			name: "unknown first kept entry",
			result: func(req CompactionRequest) *CompactionResult {
				return &CompactionResult{Summary: "custom", FirstKeptEntryID: "missing"}
			},
			want: `first kept entry "missing" is not on the current branch`,
		},
		{
			name: "empty first kept entry",
			result: func(req CompactionRequest) *CompactionResult {
				return &CompactionResult{Summary: "custom"}
			},
			want: `first kept entry "" is not on the current branch`,
		},
		{
			name: "details that do not marshal",
			result: func(req CompactionRequest) *CompactionResult {
				return &CompactionResult{Summary: "custom", FirstKeptEntryID: req.Preparation.FirstKeptEntryID, Details: make(chan int)}
			},
			want: "compaction details:",
		},
		{
			name:   "no result",
			result: func(CompactionRequest) *CompactionResult { return nil },
			want:   "compaction hook returned no result",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			h := newHookTestSession(t)
			h.session.SetCompactionHook(func(_ context.Context, req CompactionRequest, _ CompactFunc) (*CompactionResult, error) {
				return test.result(req), nil
			})

			if _, err := h.session.Compact(h.ctx, nil); err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("Compact() error = %v, want %q", err, test.want)
			}
			h.assertNothingCompacted(t)
		})
	}
}
