package baseAgent

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/ColeBurch/burrow/ai"
)

// ============================================================================
// Helper Functions
// ============================================================================

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

// ============================================================================
// BuildSessionContext
// ============================================================================

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

// ============================================================================
// ResetLeaf
// ============================================================================

func resetLeafUserMessage(text string) ai.UserMessage {
	return ai.UserMessage{
		Role: ai.RoleUser,
		Content: []ai.UserContent{
			ai.TextContent{Type: ai.ContentTypeText, Text: text},
		},
	}
}

func TestSessionManagerResetLeafBuildsEmptyContext(t *testing.T) {
	ctx := context.Background()
	sm, err := InMemorySessionManager(ctx)
	if err != nil {
		t.Fatal(err)
	}

	if _, err := sm.AppendMessage(ctx, resetLeafUserMessage("Use SQLite")); err != nil {
		t.Fatal(err)
	}
	if _, err := sm.AppendThinkingLevelChange(ctx, "high"); err != nil {
		t.Fatal(err)
	}
	if _, err := sm.AppendModelChange(ctx, "openai", "gpt-test"); err != nil {
		t.Fatal(err)
	}

	if err := sm.ResetLeaf(ctx); err != nil {
		t.Fatal(err)
	}
	got := sm.BuildSessionContext()

	if len(got.Messages) != 0 {
		t.Errorf("message count after ResetLeaf = %d, want 0", len(got.Messages))
	}
	if got.ThinkingLevel != "off" {
		t.Errorf("thinking level after ResetLeaf = %q, want %q", got.ThinkingLevel, "off")
	}
	if got.Model != nil {
		t.Errorf("model after ResetLeaf = %#v, want nil", got.Model)
	}
}

func TestSessionManagerAppendAfterResetCreatesIsolatedRoot(t *testing.T) {
	ctx := context.Background()
	sm, err := InMemorySessionManager(ctx)
	if err != nil {
		t.Fatal(err)
	}

	if _, err := sm.AppendMessage(ctx, resetLeafUserMessage("Use SQLite")); err != nil {
		t.Fatal(err)
	}

	if err := sm.ResetLeaf(ctx); err != nil {
		t.Fatal(err)
	}
	newID, err := sm.AppendMessage(ctx, resetLeafUserMessage("Use PostgreSQL"))
	if err != nil {
		t.Fatal(err)
	}

	newEntry := sm.GetEntry(newID)
	if newEntry == nil {
		t.Fatal("new root entry was not found")
	}
	if newEntry.GetBase().ParentID != nil {
		t.Errorf("new root parent = %q, want nil", *newEntry.GetBase().ParentID)
	}

	tree := sm.GetTree()
	if len(tree) != 2 {
		t.Errorf("root count = %d, want 2", len(tree))
	}

	got := sm.BuildSessionContext()
	if len(got.Messages) != 1 {
		t.Fatalf("message count for new root = %d, want 1", len(got.Messages))
	}
	if text := userText(t, got.Messages[0]); text != "Use PostgreSQL" {
		t.Errorf("new root message = %q, want %q", text, "Use PostgreSQL")
	}
}

func TestSessionManagerResetLeafAfterCompactionBuildsEmptyContext(t *testing.T) {
	ctx := context.Background()
	sm, err := InMemorySessionManager(ctx)
	if err != nil {
		t.Fatal(err)
	}

	firstID, err := sm.AppendMessage(ctx, resetLeafUserMessage("old root"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := sm.AppendCompaction(ctx, "old summary", firstID, 1000, nil, nil); err != nil {
		t.Fatal(err)
	}

	if err := sm.ResetLeaf(ctx); err != nil {
		t.Fatal(err)
	}
	got := sm.BuildSessionContext()

	if len(got.Messages) != 0 {
		t.Fatalf("message count after resetting a compacted branch = %d, want 0", len(got.Messages))
	}
}

func TestSessionManagerResetLeafDoesNotSelectLatestRoot(t *testing.T) {
	ctx := context.Background()
	sm, err := InMemorySessionManager(ctx)
	if err != nil {
		t.Fatal(err)
	}

	firstRootID, err := sm.AppendMessage(ctx, resetLeafUserMessage("first root"))
	if err != nil {
		t.Fatal(err)
	}
	if err := sm.ResetLeaf(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err := sm.AppendMessage(ctx, resetLeafUserMessage("latest root")); err != nil {
		t.Fatal(err)
	}
	if err := sm.SetLeaf(ctx, firstRootID); err != nil {
		t.Fatal(err)
	}

	if err := sm.ResetLeaf(ctx); err != nil {
		t.Fatal(err)
	}
	got := sm.BuildSessionContext()

	if len(got.Messages) != 0 {
		t.Fatalf("message count after resetting from an older root = %d, want 0", len(got.Messages))
	}
}

// ============================================================================
// Session Store
// ============================================================================

// positionSessionStore keeps sessions in memory and checks the current position
// the way the SessionStore contract requires. The full contract is checked by
// sessionstoretest, which this package cannot import.
type positionSessionStore struct {
	appendErrorSessionStore
	sessions map[string]*LoadedSession
}

func newPositionSessionStore() *positionSessionStore {
	return &positionSessionStore{sessions: map[string]*LoadedSession{}}
}

func (s *positionSessionStore) CreateSession(_ context.Context, header SessionHeader, entries []SessionEntry) error {
	session := &LoadedSession{Header: header, Entries: entries}
	if len(entries) > 0 {
		session.CurrentEntryID = &entries[len(entries)-1].GetBase().ID
	}
	s.sessions[header.ID] = session
	return nil
}

func (s *positionSessionStore) AppendEntry(_ context.Context, sessionID string, entry SessionEntry) error {
	session := s.sessions[sessionID]
	if !reflect.DeepEqual(entry.GetBase().ParentID, session.CurrentEntryID) {
		return ErrSessionConflict
	}
	session.Entries = append(session.Entries, entry)
	session.CurrentEntryID = &entry.GetBase().ID
	return nil
}

func (s *positionSessionStore) SetCurrentEntry(_ context.Context, sessionID string, expected, entryID *string) error {
	session := s.sessions[sessionID]
	if !reflect.DeepEqual(expected, session.CurrentEntryID) {
		return ErrSessionConflict
	}
	session.CurrentEntryID = entryID
	return nil
}

func (s *positionSessionStore) LoadSession(_ context.Context, sessionID string) (LoadedSession, error) {
	session, ok := s.sessions[sessionID]
	if !ok {
		return LoadedSession{}, ErrSessionNotFound
	}
	return *session, nil
}

func newStoredSession(t *testing.T, ctx context.Context, store SessionStore, texts ...string) (*SessionManager, []string) {
	t.Helper()
	sm, err := NewSessionManager(ctx, nil, nil, true, store, nil)
	if err != nil {
		t.Fatal(err)
	}
	ids := make([]string, len(texts))
	for i, text := range texts {
		if ids[i], err = sm.AppendMessage(ctx, resetLeafUserMessage(text)); err != nil {
			t.Fatal(err)
		}
	}
	return sm, ids
}

func reloadSession(t *testing.T, ctx context.Context, store SessionStore, id string) *SessionManager {
	t.Helper()
	sm, err := NewSessionManager(ctx, nil, &id, true, store, nil)
	if err != nil {
		t.Fatal(err)
	}
	return sm
}

// After a branch, the newest entry is on the abandoned branch. Reloading must
// resume at the saved position, not at the last entry.
func TestSessionManagerReloadResumesSavedPosition(t *testing.T) {
	ctx := context.Background()
	store := newPositionSessionStore()
	sm, ids := newStoredSession(t, ctx, store, "a", "b", "c")

	if err := sm.SetLeaf(ctx, ids[0]); err != nil {
		t.Fatal(err)
	}
	if got := reloadSession(t, ctx, store, sm.GetSessionID()).GetLeafID(); got == nil || *got != ids[0] {
		t.Fatalf("reloaded leaf = %v, want %s", got, ids[0])
	}

	if err := sm.ResetLeaf(ctx); err != nil {
		t.Fatal(err)
	}
	if got := reloadSession(t, ctx, store, sm.GetSessionID()).GetLeafID(); got != nil {
		t.Fatalf("reloaded leaf after ResetLeaf = %s, want nil", *got)
	}
}

// Two live managers for one session: the second writer's stale view is
// rejected instead of silently forking the history, and Reload recovers.
func TestSessionManagerDetectsAnotherWriter(t *testing.T) {
	ctx := context.Background()
	store := newPositionSessionStore()
	first, ids := newStoredSession(t, ctx, store, "a")
	second := reloadSession(t, ctx, store, first.GetSessionID())

	latest, err := first.AppendMessage(ctx, resetLeafUserMessage("from first"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := second.AppendMessage(ctx, resetLeafUserMessage("from second")); !errors.Is(err, ErrSessionConflict) {
		t.Fatalf("stale AppendMessage() error = %v, want ErrSessionConflict", err)
	}
	if err := second.ResetLeaf(ctx); !errors.Is(err, ErrSessionConflict) {
		t.Fatalf("stale ResetLeaf() error = %v, want ErrSessionConflict", err)
	}
	if got := second.GetLeafID(); got == nil || *got != ids[0] {
		t.Errorf("leaf after a rejected ResetLeaf = %v, want unchanged %s", got, ids[0])
	}

	if err := second.Reload(ctx); err != nil {
		t.Fatal(err)
	}
	id, err := second.AppendMessage(ctx, resetLeafUserMessage("from second"))
	if err != nil {
		t.Fatalf("AppendMessage() after Reload error = %v", err)
	}
	if parent := second.GetEntry(id).GetBase().ParentID; parent == nil || *parent != latest {
		t.Errorf("parent after Reload = %v, want the other writer's entry %s", parent, latest)
	}
}

// Concurrent writers on one manager must each build on the previous write.
func TestSessionManagerSerializesConcurrentAppends(t *testing.T) {
	ctx := context.Background()
	sm, _ := newStoredSession(t, ctx, newPositionSessionStore())
	var wg sync.WaitGroup
	errs := make(chan error, 20)
	for range 20 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, err := sm.AppendMessage(ctx, resetLeafUserMessage("concurrent"))
			errs <- err
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatalf("concurrent AppendMessage() error = %v", err)
		}
	}
	if got := len(sm.GetBranch()); got != 20 {
		t.Errorf("branch length = %d, want every append on one path (20)", got)
	}
}

// BranchWithSummary moves the stored position before appending, so the
// summary entry passes the store's parent check.
func TestSessionManagerBranchWithSummary(t *testing.T) {
	ctx := context.Background()
	store := newPositionSessionStore()
	sm, ids := newStoredSession(t, ctx, store, "a", "b")

	id, err := sm.BranchWithSummary(ctx, &ids[0], "tried b", nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	reloaded := reloadSession(t, ctx, store, sm.GetSessionID())
	if got := reloaded.GetLeafID(); got == nil || *got != id {
		t.Fatalf("reloaded leaf = %v, want the summary %s", got, id)
	}
	messages := reloaded.BuildSessionContext().Messages
	if len(messages) != 2 {
		t.Fatalf("context = %d messages, want a and the summary", len(messages))
	}
	if summary, ok := messages[1].(BranchSummaryMessage); !ok || summary.Summary != "tried b" {
		t.Errorf("context[1] = %#v, want the branch summary", messages[1])
	}
}

// ============================================================================
// Forking
// ============================================================================

func TestSessionManagerForkKeepsEntryIDsAndLabels(t *testing.T) {
	ctx := context.Background()
	store := newPositionSessionStore()
	header := &SessionHeader{ID: "custom-id"}
	sm, err := NewSessionManager(ctx, header, nil, true, store, nil)
	if err != nil {
		t.Fatal(err)
	}
	a, _ := sm.AppendMessage(ctx, resetLeafUserMessage("a"))
	label := "start"
	labelID, err := sm.AppendLabelChange(ctx, a, &label)
	if err != nil {
		t.Fatal(err)
	}
	b, _ := sm.AppendMessage(ctx, resetLeafUserMessage("b"))

	forked, err := sm.Fork(ctx, &b)
	if err != nil {
		t.Fatal(err)
	}
	reloaded := reloadSession(t, ctx, store, forked.GetSessionID())
	if got := reloaded.GetHeader().ParentSession; got == nil || *got != "custom-id" {
		t.Errorf("ParentSession = %v, want custom-id", got)
	}
	entries := reloaded.GetEntries()
	var got []string
	for _, entry := range entries {
		got = append(got, entry.GetBase().ID)
	}
	if len(got) != 3 || got[0] != a || got[1] != b || got[2] == labelID {
		t.Fatalf("forked entries = %v, want %s, %s and a new label entry", got, a, b)
	}
	if parent := entries[1].GetBase().ParentID; parent == nil || *parent != a {
		t.Errorf("b's parent = %v, want %s once the old label is dropped", parent, a)
	}
	if got := reloaded.GetLabel(a); got == nil || *got != label {
		t.Errorf("forked label = %v, want %q", got, label)
	}
	if got, want := entries[2].GetBase().Timestamp, sm.GetEntry(labelID).GetBase().Timestamp; got != want {
		t.Errorf("label timestamp = %s, want the original %s", got, want)
	}
	if leaf := reloaded.GetLeafID(); leaf == nil || *leaf != got[2] {
		t.Errorf("forked leaf = %v, want the label entry %s", leaf, got[2])
	}

	root, err := sm.Fork(ctx, nil)
	if err != nil {
		t.Fatal(err)
	}
	if entries := reloadSession(t, ctx, store, root.GetSessionID()).GetEntries(); len(entries) != 0 {
		t.Errorf("root fork entries = %d, want 0", len(entries))
	}
}

// ============================================================================
// GetBranches
// ============================================================================

func TestSessionManagerGetBranches(t *testing.T) {
	ctx := context.Background()
	sm, err := InMemorySessionManager(ctx)
	if err != nil {
		t.Fatal(err)
	}
	user := func(text string) string {
		id, err := sm.AppendMessage(ctx, resetLeafUserMessage(text))
		if err != nil {
			t.Fatal(err)
		}
		return id
	}
	reply := func() string {
		id, err := sm.AppendMessage(ctx, ai.AssistantMessage{Role: ai.RoleAssistant})
		if err != nil {
			t.Fatal(err)
		}
		return id
	}
	setLeaf := func(id string) {
		if err := sm.SetLeaf(ctx, id); err != nil {
			t.Fatal(err)
		}
	}

	// q1 ─┬─ reply1                    (regenerated)
	//     └─ reply2 ─┬─ q2 ─ reply3
	//                └─ q3 ─ label "alt"   (current)
	q1 := user("q1")
	reply1 := reply()
	setLeaf(q1)
	reply2 := reply()
	q2 := user("q2")
	reply3 := reply()
	setLeaf(reply2)
	q3 := user("q3")
	alt := "alt"
	labelEntry, err := sm.AppendLabelChange(ctx, q3, &alt)
	if err != nil {
		t.Fatal(err)
	}

	want := []SessionBranch{
		{LeafID: reply1, BranchPointID: &q1, FirstEntryID: reply1, FirstMessage: "q1"},
		{LeafID: reply3, BranchPointID: &reply2, FirstEntryID: q2, FirstMessage: "q2"},
		{LeafID: labelEntry, BranchPointID: &reply2, FirstEntryID: q3, Label: &alt, FirstMessage: "q3", Active: true},
	}
	if got := sm.GetBranches(); !reflect.DeepEqual(got, want) {
		t.Errorf("GetBranches() = %+v, want %+v", got, want)
	}

	// At a branch point, before the next append, no branch is active.
	setLeaf(reply2)
	for _, branch := range sm.GetBranches() {
		if branch.Active {
			t.Errorf("branch %s is active at a branch point", branch.LeafID)
		}
	}
}
