package baseAgent

import (
	"reflect"
	"strings"
	"testing"

	"github.com/ColeBurch/burrow/ai"
)

func ptr(s string) *string { return &s }

func entryBase(typ, id string, parentID *string) SessionEntryBase {
	return SessionEntryBase{Type: typ, ID: id, ParentID: parentID, Timestamp: "2025-01-01T00:00:00Z"}
}

func msg(id string, parentID *string, role ai.Role, text string) SessionEntry {
	base := entryBase("message", id, parentID)
	if role == ai.RoleUser {
		return &SessionMessageEntry{SessionEntryBase: base, Message: ai.UserMessage{
			Role:      role,
			Content:   []ai.UserContent{ai.TextContent{Type: ai.ContentTypeText, Text: text}},
			Timestamp: 1,
		}}
	}
	return &SessionMessageEntry{SessionEntryBase: base, Message: ai.AssistantMessage{
		Role:       role,
		Content:    []ai.AssistantContent{ai.TextContent{Type: ai.ContentTypeText, Text: text}},
		Api:        ai.API("anthropic-messages"),
		Provider:   ai.Provider("anthropic"),
		Model:      "claude-test",
		Usage:      ai.Usage{Input: 1, Output: 1, CacheRead: 0, CacheWrite: 0, Total: 2},
		StopReason: ai.StopReasonStop,
		Timestamp:  1,
	}}
}

func compaction(id string, parentID *string, summary, firstKeptEntryID string) SessionEntry {
	return &CompactionEntry{SessionEntryBase: entryBase("compaction", id, parentID), Summary: summary, FirstKeptEntryID: firstKeptEntryID, TokensBefore: 1000}
}

func branchSummary(id string, parentID *string, summary, fromID string) SessionEntry {
	return &BranchSummaryEntry{SessionEntryBase: entryBase("branch_summary", id, parentID), Summary: summary, FromID: fromID}
}

func thinkingLevel(id string, parentID *string, level string) SessionEntry {
	return &ThinkingLevelChangeEntry{SessionEntryBase: entryBase("thinking_level_change", id, parentID), ThinkingLevel: level}
}

func modelChange(id string, parentID *string, provider, modelID string) SessionEntry {
	return &ModelChangeEntry{SessionEntryBase: entryBase("model_change", id, parentID), Provider: provider, ModelID: modelID}
}

func buildCtx(entries []SessionEntry, leafIDs ...string) SessionContext {
	var leafID *string
	if len(leafIDs) > 0 {
		leafID = &leafIDs[0]
	}
	return BuildSessionContext(entries, leafID, nil)
}

func userText(t *testing.T, m any) string {
	t.Helper()
	u, ok := m.(ai.UserMessage)
	if !ok {
		t.Fatalf("message is %T, want ai.UserMessage", m)
	}
	c, ok := u.Content[0].(ai.TextContent)
	if !ok {
		t.Fatalf("content is %T, want ai.TextContent", u.Content[0])
	}
	return c.Text
}

func assistantText(t *testing.T, m any) string {
	t.Helper()
	a, ok := m.(ai.AssistantMessage)
	if !ok {
		t.Fatalf("message is %T, want ai.AssistantMessage", m)
	}
	c, ok := a.Content[0].(ai.TextContent)
	if !ok {
		t.Fatalf("content is %T, want ai.TextContent", a.Content[0])
	}
	return c.Text
}

func TestBuildSessionContextTrivialCases(t *testing.T) {
	t.Run("empty entries returns empty context", func(t *testing.T) {
		ctx := buildCtx(nil)
		if len(ctx.Messages) != 0 || ctx.ThinkingLevel != "off" || ctx.Model != nil {
			t.Fatalf("unexpected context: %#v", ctx)
		}
	})

	t.Run("single user message", func(t *testing.T) {
		ctx := buildCtx([]SessionEntry{msg("1", nil, ai.RoleUser, "hello")})
		if len(ctx.Messages) != 1 || ctx.Messages[0].AgentMessageType() != "user" {
			t.Fatalf("unexpected messages: %#v", ctx.Messages)
		}
	})

	t.Run("simple conversation", func(t *testing.T) {
		entries := []SessionEntry{msg("1", nil, ai.RoleUser, "hello"), msg("2", ptr("1"), ai.RoleAssistant, "hi there"), msg("3", ptr("2"), ai.RoleUser, "how are you"), msg("4", ptr("3"), ai.RoleAssistant, "great")}
		ctx := buildCtx(entries)
		got := []string{}
		for _, m := range ctx.Messages {
			got = append(got, m.AgentMessageType())
		}
		want := []string{"user", "assistant", "user", "assistant"}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("roles = %#v, want %#v", got, want)
		}
	})

	t.Run("tracks thinking level changes", func(t *testing.T) {
		ctx := buildCtx([]SessionEntry{msg("1", nil, ai.RoleUser, "hello"), thinkingLevel("2", ptr("1"), "high"), msg("3", ptr("2"), ai.RoleAssistant, "thinking hard")})
		if ctx.ThinkingLevel != "high" || len(ctx.Messages) != 2 {
			t.Fatalf("unexpected context: %#v", ctx)
		}
	})

	t.Run("tracks model from assistant message", func(t *testing.T) {
		ctx := buildCtx([]SessionEntry{msg("1", nil, ai.RoleUser, "hello"), msg("2", ptr("1"), ai.RoleAssistant, "hi")})
		want := &SessionModel{Provider: "anthropic", ModelID: "claude-test"}
		if !reflect.DeepEqual(ctx.Model, want) {
			t.Fatalf("model = %#v, want %#v", ctx.Model, want)
		}
	})

	t.Run("tracks model from model change entry", func(t *testing.T) {
		ctx := buildCtx([]SessionEntry{msg("1", nil, ai.RoleUser, "hello"), modelChange("2", ptr("1"), "openai", "gpt-4"), msg("3", ptr("2"), ai.RoleAssistant, "hi")})
		want := &SessionModel{Provider: "anthropic", ModelID: "claude-test"}
		if !reflect.DeepEqual(ctx.Model, want) {
			t.Fatalf("model = %#v, want %#v", ctx.Model, want)
		}
	})
}

func TestBuildSessionContextWithCompaction(t *testing.T) {
	t.Run("includes summary before kept messages", func(t *testing.T) {
		entries := []SessionEntry{msg("1", nil, ai.RoleUser, "first"), msg("2", ptr("1"), ai.RoleAssistant, "response1"), msg("3", ptr("2"), ai.RoleUser, "second"), msg("4", ptr("3"), ai.RoleAssistant, "response2"), compaction("5", ptr("4"), "Summary of first two turns", "3"), msg("6", ptr("5"), ai.RoleUser, "third"), msg("7", ptr("6"), ai.RoleAssistant, "response3")}
		ctx := buildCtx(entries)
		if len(ctx.Messages) != 5 {
			t.Fatalf("len = %d", len(ctx.Messages))
		}
		if !strings.Contains(ctx.Messages[0].(CompactionSummaryMessage).Summary, "Summary of first two turns") {
			t.Fatal("missing summary")
		}
		if userText(t, ctx.Messages[1]) != "second" || assistantText(t, ctx.Messages[2]) != "response2" || userText(t, ctx.Messages[3]) != "third" || assistantText(t, ctx.Messages[4]) != "response3" {
			t.Fatalf("unexpected messages: %#v", ctx.Messages)
		}
	})

	t.Run("handles compaction keeping from first message", func(t *testing.T) {
		entries := []SessionEntry{msg("1", nil, ai.RoleUser, "first"), msg("2", ptr("1"), ai.RoleAssistant, "response"), compaction("3", ptr("2"), "Empty summary", "1"), msg("4", ptr("3"), ai.RoleUser, "second")}
		ctx := buildCtx(entries)
		if len(ctx.Messages) != 4 || !strings.Contains(ctx.Messages[0].(CompactionSummaryMessage).Summary, "Empty summary") {
			t.Fatalf("unexpected messages: %#v", ctx.Messages)
		}
	})

	t.Run("multiple compactions uses latest", func(t *testing.T) {
		entries := []SessionEntry{msg("1", nil, ai.RoleUser, "a"), msg("2", ptr("1"), ai.RoleAssistant, "b"), compaction("3", ptr("2"), "First summary", "1"), msg("4", ptr("3"), ai.RoleUser, "c"), msg("5", ptr("4"), ai.RoleAssistant, "d"), compaction("6", ptr("5"), "Second summary", "4"), msg("7", ptr("6"), ai.RoleUser, "e")}
		ctx := buildCtx(entries)
		if len(ctx.Messages) != 4 || !strings.Contains(ctx.Messages[0].(CompactionSummaryMessage).Summary, "Second summary") {
			t.Fatalf("unexpected messages: %#v", ctx.Messages)
		}
	})
}

func TestBuildSessionContextWithBranches(t *testing.T) {
	t.Run("follows path to specified leaf", func(t *testing.T) {
		entries := []SessionEntry{msg("1", nil, ai.RoleUser, "start"), msg("2", ptr("1"), ai.RoleAssistant, "response"), msg("3", ptr("2"), ai.RoleUser, "branch A"), msg("4", ptr("2"), ai.RoleUser, "branch B")}
		ctxA := buildCtx(entries, "3")
		if len(ctxA.Messages) != 3 || userText(t, ctxA.Messages[2]) != "branch A" {
			t.Fatalf("ctxA = %#v", ctxA.Messages)
		}
		ctxB := buildCtx(entries, "4")
		if len(ctxB.Messages) != 3 || userText(t, ctxB.Messages[2]) != "branch B" {
			t.Fatalf("ctxB = %#v", ctxB.Messages)
		}
	})

	t.Run("includes branch summary in path", func(t *testing.T) {
		entries := []SessionEntry{msg("1", nil, ai.RoleUser, "start"), msg("2", ptr("1"), ai.RoleAssistant, "response"), msg("3", ptr("2"), ai.RoleUser, "abandoned path"), branchSummary("4", ptr("2"), "Summary of abandoned work", "3"), msg("5", ptr("4"), ai.RoleUser, "new direction")}
		ctx := buildCtx(entries, "5")
		if len(ctx.Messages) != 4 || !strings.Contains(ctx.Messages[2].(BranchSummaryMessage).Summary, "Summary of abandoned work") || userText(t, ctx.Messages[3]) != "new direction" {
			t.Fatalf("unexpected messages: %#v", ctx.Messages)
		}
	})

	t.Run("complex tree with multiple branches and compaction", func(t *testing.T) {
		entries := []SessionEntry{msg("1", nil, ai.RoleUser, "start"), msg("2", ptr("1"), ai.RoleAssistant, "r1"), msg("3", ptr("2"), ai.RoleUser, "q2"), msg("4", ptr("3"), ai.RoleAssistant, "r2"), compaction("5", ptr("4"), "Compacted history", "3"), msg("6", ptr("5"), ai.RoleUser, "q3"), msg("7", ptr("6"), ai.RoleAssistant, "r3"), msg("8", ptr("3"), ai.RoleUser, "wrong path"), msg("9", ptr("8"), ai.RoleAssistant, "wrong response"), branchSummary("10", ptr("3"), "Tried wrong approach", "9"), msg("11", ptr("10"), ai.RoleUser, "better approach")}
		ctxMain := buildCtx(entries, "7")
		if len(ctxMain.Messages) != 5 || !strings.Contains(ctxMain.Messages[0].(CompactionSummaryMessage).Summary, "Compacted history") || userText(t, ctxMain.Messages[1]) != "q2" || assistantText(t, ctxMain.Messages[2]) != "r2" || userText(t, ctxMain.Messages[3]) != "q3" || assistantText(t, ctxMain.Messages[4]) != "r3" {
			t.Fatalf("main = %#v", ctxMain.Messages)
		}
		ctxBranch := buildCtx(entries, "11")
		if len(ctxBranch.Messages) != 5 || userText(t, ctxBranch.Messages[0]) != "start" || assistantText(t, ctxBranch.Messages[1]) != "r1" || userText(t, ctxBranch.Messages[2]) != "q2" || !strings.Contains(ctxBranch.Messages[3].(BranchSummaryMessage).Summary, "Tried wrong approach") || userText(t, ctxBranch.Messages[4]) != "better approach" {
			t.Fatalf("branch = %#v", ctxBranch.Messages)
		}
	})
}

func TestBuildSessionContextEdgeCases(t *testing.T) {
	t.Run("uses last entry when leafId not found", func(t *testing.T) {
		ctx := buildCtx([]SessionEntry{msg("1", nil, ai.RoleUser, "hello"), msg("2", ptr("1"), ai.RoleAssistant, "hi")}, "nonexistent")
		if len(ctx.Messages) != 2 {
			t.Fatalf("len = %d", len(ctx.Messages))
		}
	})

	t.Run("handles orphaned entries gracefully", func(t *testing.T) {
		ctx := buildCtx([]SessionEntry{msg("1", nil, ai.RoleUser, "hello"), msg("2", ptr("missing"), ai.RoleAssistant, "orphan")}, "2")
		if len(ctx.Messages) != 1 {
			t.Fatalf("len = %d", len(ctx.Messages))
		}
	})
}
