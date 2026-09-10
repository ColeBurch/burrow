package baseAgent

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/ColeBurch/burrow/agent"
	"github.com/ColeBurch/burrow/ai"
)

type appendErrorSessionStore struct {
	appendErr error
}

func (*appendErrorSessionStore) CreateSession(context.Context, *SessionHeader) error {
	return nil
}

func (*appendErrorSessionStore) LoadSession(context.Context, string) (*SessionHeader, []SessionEntry, error) {
	panic("unexpected LoadSession call")
}

func (s *appendErrorSessionStore) AppendEntry(context.Context, *SessionHeader, SessionEntry) error {
	return s.appendErr
}

func (*appendErrorSessionStore) ListSessions(context.Context, map[string]any) ([]SessionInfo, error) {
	panic("unexpected ListSessions call")
}

func (*appendErrorSessionStore) FindMostRecentSession(context.Context, map[string]any) (*SessionInfo, error) {
	panic("unexpected FindMostRecentSession call")
}

func (*appendErrorSessionStore) CreateBranch(context.Context, string, SessionHeader) error {
	panic("unexpected CreateBranch call")
}

func TestAgentSessionPromptReturnsSessionStoreError(t *testing.T) {
	ctx := context.Background()
	appendErr := errors.New("session store append failed")
	store := &appendErrorSessionStore{appendErr: appendErr}

	sessionManager, err := NewSessionManager(ctx, nil, nil, true, store, nil)
	if err != nil {
		t.Fatalf("NewSessionManager() error = %v", err)
	}

	model := ai.Model[ai.API]{ID: "test-model", Name: "test-model"}
	a := agent.NewAgent(&agent.AgentOptions{
		InitialState: &agent.InitialAgentState{Model: &model},
	})
	session := NewAgentSession(a, sessionManager)

	err = session.Prompt(ctx, "hello", nil)
	if !errors.Is(err, appendErr) {
		t.Fatalf("Prompt() error = %v, want %v", err, appendErr)
	}
}

// ============================================================================
// Compaction Tests
// ============================================================================

func manualCompactionUserMessage(text string) ai.UserMessage {
	return ai.UserMessage{
		Role: ai.RoleUser,
		Content: []ai.UserContent{
			ai.TextContent{Type: ai.ContentTypeText, Text: text},
		},
	}
}

func manualCompactionAssistantMessage(text string, model ai.Model[ai.API]) ai.AssistantMessage {
	return ai.AssistantMessage{
		Role: ai.RoleAssistant,
		Content: []ai.AssistantContent{
			ai.TextContent{Type: ai.ContentTypeText, Text: text},
		},
		Api:        model.API,
		Provider:   model.Provider,
		Model:      model.ID,
		Usage:      ai.Usage{Input: 30, Output: 10, Total: 40},
		StopReason: ai.StopReasonStop,
	}
}

func registerManualCompactionProviderWithStream(
	t *testing.T,
	streamSimple func(
		ai.Model[ai.API],
		ai.ModelContext,
		*ai.SimpleStreamOptions,
	) (*ai.AssistantMessageEventStream, error),
) ai.Model[ai.API] {
	t.Helper()

	apiName := ai.API("manual-compaction-" + strings.ReplaceAll(t.Name(), "/", "-"))
	sourceID := string(apiName)
	ai.RegisterApiProvider(ai.ApiProvider[ai.API, ai.StreamOptions]{
		Api:          apiName,
		StreamSimple: streamSimple,
	}, &sourceID)
	t.Cleanup(func() { ai.UnregisterApiProvider(sourceID) })

	return ai.Model[ai.API]{
		ID:            "manual-compaction-model",
		Name:          "Manual Compaction Model",
		API:           apiName,
		Provider:      ai.Provider("manual-compaction-provider"),
		ContextWindow: 1000,
		MaxTokens:     200,
	}
}

func registerManualCompactionProvider(t *testing.T, summary string) ai.Model[ai.API] {
	t.Helper()

	return registerManualCompactionProviderWithStream(t, func(
		_ ai.Model[ai.API],
		_ ai.ModelContext,
		_ *ai.SimpleStreamOptions,
	) (*ai.AssistantMessageEventStream, error) {
		response := ai.AssistantMessage{
			Role: ai.RoleAssistant,
			Content: []ai.AssistantContent{
				ai.TextContent{Type: ai.ContentTypeText, Text: summary},
			},
			StopReason: ai.StopReasonStop,
		}
		stream := ai.NewAssistantMessageEventStream(1)
		stream.Push(ai.DoneEvent{Type: "done", Reason: ai.StopReasonStop, Message: response})
		return stream, nil
	})
}

func seedManualCompactionHistory(
	t *testing.T,
	ctx context.Context,
	sm *SessionManager,
	model ai.Model[ai.API],
) []agent.AgentMessage {
	t.Helper()

	messages := []agent.AgentMessage{
		manualCompactionUserMessage("aaaa"),
		manualCompactionAssistantMessage("bbbb", model),
		manualCompactionUserMessage("cccc"),
		manualCompactionAssistantMessage("dddd", model),
	}
	for _, message := range messages {
		if _, err := sm.AppendMessage(ctx, message); err != nil {
			t.Fatal(err)
		}
	}
	return messages
}

func TestAgentSessionManualCompactionPersistsAndReplacesContext(t *testing.T) {
	ctx := context.Background()
	model := registerManualCompactionProvider(t, "compacted history")

	sm, err := InMemorySessionManager(ctx)
	if err != nil {
		t.Fatal(err)
	}
	messages := seedManualCompactionHistory(t, ctx, sm, model)

	apiKey := "test-key"
	a := agent.NewAgent(&agent.AgentOptions{
		InitialState: &agent.InitialAgentState{
			Model:    &model,
			Messages: messages,
		},
		GetAPIKey: func(string) *string { return &apiKey },
	})
	session := NewAgentSession(a, sm)
	session.SetCompactionSettings(CompactionSettings{
		Enabled:          true,
		ReservedTokens:   100,
		KeepRecentTokens: 2,
	})

	result, err := session.Compact(ctx, nil)
	if err != nil {
		t.Fatal(err)
	}
	if result.Summary != "compacted history" {
		t.Fatalf("summary = %q, want %q", result.Summary, "compacted history")
	}

	latest := GetLatestCompactionEntry(sm.GetEntries())
	if latest == nil {
		t.Fatal("compaction entry was not persisted")
	}
	if latest.Summary != result.Summary || latest.FirstKeptEntryID != result.FirstKeptEntryID {
		t.Fatalf("persisted compaction = %#v, result = %#v", latest, result)
	}

	got := a.State().Messages
	if len(got) != 3 {
		t.Fatalf("agent message count after compaction = %d, want 3", len(got))
	}
	if _, ok := got[0].(CompactionSummaryMessage); !ok {
		t.Fatalf("first agent message = %T, want CompactionSummaryMessage", got[0])
	}
	if text := userText(t, got[1]); text != "cccc" {
		t.Fatalf("first kept user message = %q, want %q", text, "cccc")
	}
	if text := assistantText(t, got[2]); text != "dddd" {
		t.Fatalf("first kept assistant message = %q, want %q", text, "dddd")
	}

	if _, err := session.Compact(ctx, nil); err == nil || !strings.Contains(err.Error(), "already compacted") {
		t.Fatalf("second compaction error = %v, want already compacted", err)
	}

	entryCount := len(sm.GetEntries())
	if err := session.Prompt(ctx, "eeee", nil); err != nil {
		t.Fatalf("prompt after compaction failed: %v", err)
	}
	if got := len(sm.GetEntries()); got != entryCount+2 {
		t.Fatalf("entry count after prompt = %d, want %d", got, entryCount+2)
	}
}

func TestAgentSessionManualCompactionRejectsEmptySession(t *testing.T) {
	ctx := context.Background()
	model := registerManualCompactionProvider(t, "unused")
	sm, err := InMemorySessionManager(ctx)
	if err != nil {
		t.Fatal(err)
	}
	apiKey := "test-key"
	a := agent.NewAgent(&agent.AgentOptions{
		InitialState: &agent.InitialAgentState{Model: &model},
		GetAPIKey:    func(string) *string { return &apiKey },
	})
	session := NewAgentSession(a, sm)

	if _, err := session.Compact(ctx, nil); err == nil || !strings.Contains(err.Error(), "nothing to compact") {
		t.Fatalf("compaction error = %v, want nothing to compact", err)
	}
}

func TestAgentSessionManualCompactionAbortsActiveRun(t *testing.T) {
	ctx := context.Background()
	model := registerManualCompactionProvider(t, "compacted active run")
	sm, err := InMemorySessionManager(ctx)
	if err != nil {
		t.Fatal(err)
	}
	messages := seedManualCompactionHistory(t, ctx, sm, model)

	streamStarted := make(chan struct{})
	apiKey := "test-key"
	a := agent.NewAgent(&agent.AgentOptions{
		InitialState: &agent.InitialAgentState{Model: &model, Messages: messages},
		GetAPIKey:    func(string) *string { return &apiKey },
		StreamFn: func(runCtx context.Context, _ ai.Model[ai.API], _ ai.ModelContext, _ *ai.SimpleStreamOptions) (*ai.AssistantMessageEventStream, error) {
			close(streamStarted)
			stream := ai.NewAssistantMessageEventStream(2)
			go func() {
				stream.Push(ai.StartEvent{Type: "start", Partial: manualCompactionAssistantMessage("", model)})
				<-runCtx.Done()
				aborted := manualCompactionAssistantMessage("aborted response", model)
				aborted.StopReason = ai.StopReasonAborted
				aborted.ErrorMessage = "aborted"
				stream.Push(ai.ErrorEvent{Type: "error", Reason: ai.StopReasonAborted, Message: aborted})
			}()
			return stream, nil
		},
	})
	session := NewAgentSession(a, sm)
	session.SetCompactionSettings(CompactionSettings{Enabled: true, ReservedTokens: 100, KeepRecentTokens: 2})

	promptDone := make(chan error, 1)
	go func() {
		promptDone <- session.Prompt(ctx, "active prompt", nil)
	}()

	select {
	case <-streamStarted:
	case <-time.After(time.Second):
		t.Fatal("active agent run did not start")
	}

	result, err := session.Compact(ctx, nil)
	if err != nil {
		t.Fatal(err)
	}
	if result.Summary != "compacted active run" {
		t.Fatalf("summary = %q, want %q", result.Summary, "compacted active run")
	}

	select {
	case <-promptDone:
	case <-time.After(time.Second):
		t.Fatal("aborted prompt did not finish")
	}

	for _, entry := range sm.GetEntries() {
		messageEntry, ok := entry.(*SessionMessageEntry)
		if !ok {
			continue
		}
		assistantMessage, ok := messageEntry.Message.(ai.AssistantMessage)
		if ok && assistantMessage.StopReason == ai.StopReasonAborted {
			t.Fatal("aborted assistant message was persisted")
		}
	}
	for _, message := range a.State().Messages {
		assistantMessage, ok := message.(ai.AssistantMessage)
		if ok && assistantMessage.StopReason == ai.StopReasonAborted {
			t.Fatal("aborted assistant message remained in agent context")
		}
	}
}

func TestAgentSessionAbortCompactionCancelsProviderRequest(t *testing.T) {
	ctx := context.Background()
	providerStarted := make(chan struct{})
	model := registerManualCompactionProviderWithStream(t, func(
		_ ai.Model[ai.API],
		_ ai.ModelContext,
		options *ai.SimpleStreamOptions,
	) (*ai.AssistantMessageEventStream, error) {
		close(providerStarted)
		<-options.Signal.Done()
		return nil, options.Signal.Err()
	})

	sm, err := InMemorySessionManager(ctx)
	if err != nil {
		t.Fatal(err)
	}
	messages := seedManualCompactionHistory(t, ctx, sm, model)
	apiKey := "test-key"
	a := agent.NewAgent(&agent.AgentOptions{
		InitialState: &agent.InitialAgentState{Model: &model, Messages: messages},
		GetAPIKey:    func(string) *string { return &apiKey },
	})
	session := NewAgentSession(a, sm)
	session.SetCompactionSettings(CompactionSettings{Enabled: true, ReservedTokens: 100, KeepRecentTokens: 2})

	type compactResult struct {
		result *CompactionResult
		err    error
	}
	compactDone := make(chan compactResult, 1)
	go func() {
		result, err := session.Compact(ctx, nil)
		compactDone <- compactResult{result: result, err: err}
	}()

	select {
	case <-providerStarted:
	case <-time.After(time.Second):
		t.Fatal("compaction provider request did not start")
	}
	if !session.IsCompacting() {
		t.Fatal("session did not report active compaction")
	}

	session.AbortCompaction()

	var outcome compactResult
	select {
	case outcome = <-compactDone:
	case <-time.After(time.Second):
		t.Fatal("cancelled compaction did not finish")
	}
	if outcome.result != nil {
		t.Fatalf("cancelled compaction result = %#v, want nil", outcome.result)
	}
	if outcome.err == nil || !strings.Contains(outcome.err.Error(), "cancel") {
		t.Fatalf("cancelled compaction error = %v, want cancellation error", outcome.err)
	}
	if session.IsCompacting() {
		t.Fatal("session still reports active compaction")
	}
	if latest := GetLatestCompactionEntry(sm.GetEntries()); latest != nil {
		t.Fatalf("cancelled compaction persisted entry %#v", latest)
	}
	if got := len(a.State().Messages); got != len(messages) {
		t.Fatalf("agent message count after cancellation = %d, want %d", got, len(messages))
	}
}
