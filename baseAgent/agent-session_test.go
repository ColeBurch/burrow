package baseAgent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ColeBurch/burrow/agent"
	"github.com/ColeBurch/burrow/ai"
)

type appendErrorSessionStore struct {
	appendErr error
}

func (*appendErrorSessionStore) CreateSession(context.Context, SessionHeader, []SessionEntry) error {
	return nil
}

func (*appendErrorSessionStore) LoadSession(context.Context, string) (LoadedSession, error) {
	panic("unexpected LoadSession call")
}

func (s *appendErrorSessionStore) AppendEntry(context.Context, string, SessionEntry) error {
	return s.appendErr
}

func (*appendErrorSessionStore) SetCurrentEntry(context.Context, string, *string, *string) error {
	panic("unexpected SetCurrentEntry call")
}

func (*appendErrorSessionStore) ListSessions(context.Context, SessionQuery) ([]SessionInfo, error) {
	panic("unexpected ListSessions call")
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

// contextSessionStore honors ctx like database/sql: a done context fails the
// write. When block is set, writes wait until the context is done.
type contextSessionStore struct {
	appendErrorSessionStore
	block bool

	mu          sync.Mutex
	entries     []SessionEntry
	hadDeadline bool
}

func (s *contextSessionStore) AppendEntry(ctx context.Context, _ string, entry SessionEntry) error {
	_, hasDeadline := ctx.Deadline()
	s.mu.Lock()
	s.hadDeadline = s.hadDeadline || hasDeadline
	s.mu.Unlock()
	if s.block {
		<-ctx.Done()
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.entries = append(s.entries, entry)
	return nil
}

func (s *contextSessionStore) storedMessages() []agent.AgentMessage {
	s.mu.Lock()
	defer s.mu.Unlock()
	var messages []agent.AgentMessage
	for _, entry := range s.entries {
		if messageEntry, ok := entry.(*SessionMessageEntry); ok {
			messages = append(messages, messageEntry.Message)
		}
	}
	return messages
}

func countAbortedAssistants(messages []agent.AgentMessage) int {
	count := 0
	for _, message := range messages {
		if assistant, ok := message.(ai.AssistantMessage); ok && assistant.StopReason == ai.StopReasonAborted {
			count++
		}
	}
	return count
}

func TestAgentSessionPersistsAbortedMessageAfterRunCancellation(t *testing.T) {
	ctx := context.Background()
	store := &contextSessionStore{}
	sm, err := NewSessionManager(ctx, nil, nil, true, store, nil)
	if err != nil {
		t.Fatal(err)
	}
	model := registerManualCompactionProvider(t, "unused")
	streamStarted := make(chan struct{})
	apiKey := "test-key"
	a := agent.NewAgent(&agent.AgentOptions{
		InitialState: &agent.InitialAgentState{Model: &model},
		GetAPIKey:    func(string) *string { return &apiKey },
		StreamFn: func(runCtx context.Context, _ ai.Model[ai.API], _ ai.ModelContext, _ *ai.SimpleStreamOptions) (*ai.AssistantMessageEventStream, error) {
			close(streamStarted)
			stream := ai.NewAssistantMessageEventStream(2)
			go func() {
				stream.Push(ai.StartEvent{Type: "start", Partial: manualCompactionAssistantMessage("", model)})
				<-runCtx.Done()
				aborted := manualCompactionAssistantMessage("partial answer", model)
				aborted.StopReason = ai.StopReasonAborted
				aborted.ErrorMessage = "aborted"
				stream.Push(ai.ErrorEvent{Type: "error", Reason: ai.StopReasonAborted, Message: aborted})
			}()
			return stream, nil
		},
	})
	session := NewAgentSession(a, sm)
	var agentEnds atomic.Int32
	defer session.Subscribe(func(event AgentSessionEvent) {
		if _, ok := event.(agent.EndEvent); ok {
			agentEnds.Add(1)
		}
	})()

	promptDone := make(chan error, 1)
	go func() { promptDone <- session.Prompt(ctx, "hello", nil) }()
	select {
	case <-streamStarted:
	case <-time.After(time.Second):
		t.Fatal("agent run did not start")
	}
	a.Abort()

	select {
	case err := <-promptDone:
		if err != nil {
			t.Fatalf("Prompt() error = %v, want nil for a user abort", err)
		}
	case <-time.After(time.Second):
		t.Fatal("aborted prompt did not finish")
	}

	if got := agentEnds.Load(); got != 1 {
		t.Errorf("agent_end events = %d, want 1", got)
	}
	if got := countAbortedAssistants(a.State().Messages); got != 1 {
		t.Errorf("live aborted assistant messages = %d, want 1", got)
	}
	stored := store.storedMessages()
	if len(stored) != 2 || stored[0].AgentMessageType() != string(ai.RoleUser) || countAbortedAssistants(stored) != 1 {
		t.Fatalf("stored messages = %#v, want user then aborted assistant", stored)
	}
	// The pre-prompt compaction check reads the branch, so it must see the abort.
	if got := countAbortedAssistants(sm.BuildSessionContext().Messages); got != 1 {
		t.Errorf("session context aborted assistant messages = %d, want 1", got)
	}
}

func TestAgentSessionMessagePersistenceTimesOut(t *testing.T) {
	previousTimeout := messagePersistTimeout
	messagePersistTimeout = 20 * time.Millisecond
	t.Cleanup(func() { messagePersistTimeout = previousTimeout })

	ctx := context.Background()
	store := &contextSessionStore{block: true}
	sm, err := NewSessionManager(ctx, nil, nil, true, store, nil)
	if err != nil {
		t.Fatal(err)
	}
	model := registerManualCompactionProvider(t, "unused")
	apiKey := "test-key"
	a := agent.NewAgent(&agent.AgentOptions{
		InitialState: &agent.InitialAgentState{Model: &model},
		GetAPIKey:    func(string) *string { return &apiKey },
	})
	session := NewAgentSession(a, sm)

	promptDone := make(chan error, 1)
	// The caller context has no deadline; only the persistence timeout can end the write.
	go func() { promptDone <- session.Prompt(ctx, "hello", nil) }()
	select {
	case err := <-promptDone:
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("Prompt() error = %v, want %v", err, context.DeadlineExceeded)
		}
	case <-time.After(time.Second):
		t.Fatal("blocked store write was not bounded by the persistence timeout")
	}

	store.mu.Lock()
	hadDeadline := store.hadDeadline
	store.mu.Unlock()
	if !hadDeadline {
		t.Error("store write context had no deadline")
	}
	if len(store.storedMessages()) != 0 || len(sm.GetEntries()) != 0 {
		t.Error("timed-out write was recorded")
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

	if _, err := session.Compact(ctx, nil); !errors.Is(err, ErrNothingToCompact) {
		t.Fatalf("compaction error = %v, want ErrNothingToCompact", err)
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
	// The persisted aborted reply uses no budget, so the cut keeps the whole
	// aborted turn instead of splitting it.
	if result.Summary != "compacted active run" {
		t.Fatalf("summary = %q, want %q", result.Summary, "compacted active run")
	}

	select {
	case <-promptDone:
	case <-time.After(time.Second):
		t.Fatal("aborted prompt did not finish")
	}

	// Persistence stays subscribed while Compact aborts the run.
	var persisted []agent.AgentMessage
	for _, entry := range sm.GetEntries() {
		if messageEntry, ok := entry.(*SessionMessageEntry); ok {
			persisted = append(persisted, messageEntry.Message)
		}
	}
	if got := countAbortedAssistants(persisted); got != 1 {
		t.Errorf("persisted aborted assistant messages = %d, want 1", got)
	}
	if live, want := a.State().Messages, sm.BuildSessionContext().Messages; !reflect.DeepEqual(live, want) {
		t.Errorf("live context differs from persisted context:\ngot  %#v\nwant %#v", live, want)
	}
}

// Messages queued for a run that Compact aborts are handed back instead of
// being stranded in the agent's queue until some later prompt.
func TestAgentSessionManualCompactionRestoresQueuedMessages(t *testing.T) {
	ctx := context.Background()
	model := registerManualCompactionProvider(t, "compacted active run")
	sm, err := InMemorySessionManager(ctx)
	if err != nil {
		t.Fatal(err)
	}
	messages := seedManualCompactionHistory(t, ctx, sm, model)

	streamStarted := make(chan struct{})
	var requests atomic.Int32
	apiKey := "test-key"
	a := agent.NewAgent(&agent.AgentOptions{
		InitialState: &agent.InitialAgentState{Model: &model, Messages: messages},
		GetAPIKey:    func(string) *string { return &apiKey },
		StreamFn: func(runCtx context.Context, _ ai.Model[ai.API], _ ai.ModelContext, _ *ai.SimpleStreamOptions) (*ai.AssistantMessageEventStream, error) {
			if requests.Add(1) == 1 {
				close(streamStarted)
			}
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
	var ends []CompactionEndEvent
	unsubscribe := session.Subscribe(func(event AgentSessionEvent) {
		if end, ok := event.(CompactionEndEvent); ok {
			ends = append(ends, end)
		}
	})
	defer unsubscribe()

	promptDone := make(chan error, 1)
	go func() {
		promptDone <- session.Prompt(ctx, "active prompt", nil)
	}()
	select {
	case <-streamStarted:
	case <-time.After(time.Second):
		t.Fatal("active agent run did not start")
	}
	// Queue modes do not apply: every queued message comes back, steering first.
	for _, text := range []string{"follow-up 1", "follow-up 2"} {
		if err := session.Prompt(ctx, text, &PromptOptions{StreamingBehavior: StreamingBehaviorFollowUp}); err != nil {
			t.Fatal(err)
		}
	}
	if err := session.Prompt(ctx, "steer", &PromptOptions{StreamingBehavior: StreamingBehaviorSteer}); err != nil {
		t.Fatal(err)
	}

	if _, err := session.Compact(ctx, nil); err != nil {
		t.Fatal(err)
	}
	select {
	case <-promptDone:
	case <-time.After(time.Second):
		t.Fatal("aborted prompt did not finish")
	}

	if len(ends) != 1 {
		t.Fatalf("compaction_end events = %d, want 1", len(ends))
	}
	var restored []string
	for _, message := range ends[0].RestoredMessages {
		restored = append(restored, userText(t, message))
	}
	if want := []string{"steer", "follow-up 1", "follow-up 2"}; !reflect.DeepEqual(restored, want) {
		t.Errorf("restored messages = %q, want %q", restored, want)
	}
	if a.HasQueuedMessages() {
		t.Error("restored messages are still queued")
	}
	if got := requests.Load(); got != 1 {
		t.Errorf("agent requests = %d, want only the aborted run", got)
	}
}

func TestAgentSessionManualCompactionFailureKeepsSessionInSync(t *testing.T) {
	ctx := context.Background()
	failure := errors.New("529 overloaded")
	model := registerManualCompactionProviderWithStream(t, func(ai.Model[ai.API], ai.ModelContext, *ai.SimpleStreamOptions) (*ai.AssistantMessageEventStream, error) {
		return nil, failure
	})
	sm, err := InMemorySessionManager(ctx)
	if err != nil {
		t.Fatal(err)
	}
	seeded := seedManualCompactionHistory(t, ctx, sm, model)

	const toolOutput = "41 passed, 1 failed: TestParseConfig"
	toolStarted := make(chan struct{})
	var mu sync.Mutex
	var sent [][]ai.Message
	apiKey := "test-key"
	a := agent.NewAgent(&agent.AgentOptions{
		InitialState: &agent.InitialAgentState{
			Model:    &model,
			Messages: seeded,
			Tools: []agent.AgentTool[any, any]{{
				Tool: ai.Tool{Name: "run_tests", Parameters: map[string]any{"type": "object"}},
				Execute: func(toolCtx context.Context, _ string, _ any, _ agent.AgentToolUpdateCallback[any]) (agent.AgentToolResult[any], error) {
					close(toolStarted)
					// Compact's abort interrupts the tool, which returns what it collected.
					<-toolCtx.Done()
					return agent.AgentToolResult[any]{Content: []ai.ToolResultContent{
						ai.TextContent{Type: ai.ContentTypeText, Text: toolOutput},
					}}, nil
				},
			}},
		},
		ConvertToLLM: ConvertToLLM,
		GetAPIKey:    func(string) *string { return &apiKey },
		StreamFn: func(runCtx context.Context, _ ai.Model[ai.API], modelContext ai.ModelContext, _ *ai.SimpleStreamOptions) (*ai.AssistantMessageEventStream, error) {
			reply := manualCompactionAssistantMessage("ok", model)
			if runCtx.Err() != nil {
				reply.StopReason = ai.StopReasonAborted
				reply.ErrorMessage = "aborted"
				return autoCompactionStream(reply), nil
			}
			mu.Lock()
			defer mu.Unlock()
			sent = append(sent, modelContext.Messages)
			if len(sent) == 1 {
				reply.StopReason = ai.StopReasonToolUse
				reply.Content = []ai.AssistantContent{ai.ToolCall{Type: ai.ContentTypeToolCall, Id: "call-1", Name: "run_tests", Args: map[string]any{}}}
			}
			return autoCompactionStream(reply), nil
		},
	})
	session := NewAgentSession(a, sm)
	session.SetCompactionSettings(CompactionSettings{Enabled: false, ReservedTokens: 100, KeepRecentTokens: 2})

	promptDone := make(chan error, 1)
	go func() { promptDone <- session.Prompt(ctx, "run the tests", nil) }()
	select {
	case <-toolStarted:
	case <-time.After(time.Second):
		t.Fatal("tool did not start")
	}

	if _, err := session.Compact(ctx, nil); !errors.Is(err, failure) {
		t.Fatalf("Compact error = %v, want %v", err, failure)
	}
	select {
	case <-promptDone:
	case <-time.After(time.Second):
		t.Fatal("aborted prompt did not finish")
	}

	persisted := sm.BuildSessionContext().Messages
	if live := a.State().Messages; !reflect.DeepEqual(live, persisted) {
		t.Fatalf("live context differs from persisted context after failed Compact:\ngot  %#v\nwant %#v", live, persisted)
	}
	foundToolResult := false
	for _, message := range persisted {
		if result, ok := message.(ai.ToolResultMessage); ok && result.ToolCallID == "call-1" {
			foundToolResult = result.Content[0].(ai.TextContent).Text == toolOutput
		}
	}
	if !foundToolResult {
		t.Fatal("tool result that ended during Compact's abort was not persisted")
	}

	if err := session.Prompt(ctx, "which test failed?", nil); err != nil {
		t.Fatal(err)
	}
	// Reloading must rebuild exactly the history the model was sent.
	persisted = sm.BuildSessionContext().Messages
	mu.Lock()
	lastSent := sent[len(sent)-1]
	mu.Unlock()
	reloaded := ConvertToLLM(persisted[:len(persisted)-1])
	if got, want := ai.TransformMessages(reloaded, model, nil), ai.TransformMessages(lastSent, model, nil); !reflect.DeepEqual(got, want) {
		t.Errorf("reloaded history differs from what the model saw:\ngot  %#v\nwant %#v", got, want)
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

// ============================================================================
// Automatic Compaction Tests
// ============================================================================

const autoCompactionTestTimeout = 5 * time.Second
const autoCompactionTestSummary = "threshold summary"

type autoCompactionRequest struct {
	context   ai.ModelContext
	summaries int32
}

type autoCompactionHarness struct {
	ctx            context.Context
	cancel         context.CancelFunc
	session        *AgentSession
	model          ai.Model[ai.API]
	summaryCalls   atomic.Int32
	summaryStarted chan struct{}
	summaryRelease chan struct{}
	mu             sync.Mutex
	requests       []autoCompactionRequest
	responses      []ai.AssistantMessage
}

func autoCompactionStream(message ai.AssistantMessage) *ai.AssistantMessageEventStream {
	stream := ai.NewAssistantMessageEventStream(1)
	if message.StopReason == ai.StopReasonError || message.StopReason == ai.StopReasonAborted {
		stream.Push(ai.ErrorEvent{Type: "error", Reason: message.StopReason, Message: message})
	} else {
		stream.Push(ai.DoneEvent{Type: "done", Reason: message.StopReason, Message: message})
	}
	return stream
}

func newAutoCompactionHarness(t *testing.T, blockSummary bool) *autoCompactionHarness {
	t.Helper()
	h := &autoCompactionHarness{
		summaryStarted: make(chan struct{}),
		summaryRelease: make(chan struct{}),
	}
	h.ctx, h.cancel = context.WithTimeout(context.Background(), autoCompactionTestTimeout)
	t.Cleanup(h.cancel)
	h.model = registerManualCompactionProviderWithStream(t, func(model ai.Model[ai.API], _ ai.ModelContext, options *ai.SimpleStreamOptions) (*ai.AssistantMessageEventStream, error) {
		if h.summaryCalls.Add(1) == 1 {
			close(h.summaryStarted)
		}
		if blockSummary {
			// Block in the provider itself, so no stream producer goroutine can leak.
			select {
			case <-h.summaryRelease:
			case <-options.Signal.Done():
				return nil, options.Signal.Err()
			}
		}
		return autoCompactionStream(manualCompactionAssistantMessage(autoCompactionTestSummary, model)), nil
	})
	sm, err := InMemorySessionManager(h.ctx)
	if err != nil {
		t.Fatal(err)
	}
	messages := seedManualCompactionHistory(t, h.ctx, sm, h.model)
	apiKey := "test-key"
	a := agent.NewAgent(&agent.AgentOptions{
		InitialState: &agent.InitialAgentState{Model: &h.model, Messages: messages},
		ConvertToLLM: ConvertToLLM,
		GetAPIKey:    func(string) *string { return &apiKey },
		StreamFn: func(_ context.Context, _ ai.Model[ai.API], modelContext ai.ModelContext, _ *ai.SimpleStreamOptions) (*ai.AssistantMessageEventStream, error) {
			h.mu.Lock()
			defer h.mu.Unlock()
			index := len(h.requests)
			h.requests = append(h.requests, autoCompactionRequest{context: modelContext, summaries: h.summaryCalls.Load()})
			if index >= len(h.responses) {
				return nil, fmt.Errorf("unexpected agent request %d", index+1)
			}
			return autoCompactionStream(h.responses[index]), nil
		},
	})
	h.session = NewAgentSession(a, sm)
	h.session.SetCompactionSettings(CompactionSettings{Enabled: true, ReservedTokens: 100, KeepRecentTokens: 2})
	return h
}

func (h *autoCompactionHarness) response(tokens int64, reason ai.StopReason) ai.AssistantMessage {
	// Keep the reply below KeepRecentTokens so compaction retains the whole turn.
	message := manualCompactionAssistantMessage("done", h.model)
	message.Usage = ai.Usage{Input: tokens, Total: tokens}
	message.StopReason = reason
	if reason == ai.StopReasonError || reason == ai.StopReasonAborted {
		message.ErrorMessage = "test response failed"
	}
	if reason == ai.StopReasonError {
		// Like agent.AssistantErrorMessage: provider errors carry no content.
		// Aborted replies keep theirs, as real partial responses do.
		message.Content = []ai.AssistantContent{}
	}
	return message
}

// retainOverflowedTurn keeps exactly the overflowed turn on compaction when its
// reply uses no budget: an error reply never reaches the model, and a zero-output
// reply has no content. The turn's only budgeted content is its one-token prompt.
func (h *autoCompactionHarness) retainOverflowedTurn() {
	settings := h.session.GetCompactionSettings()
	settings.KeepRecentTokens = 1
	h.session.SetCompactionSettings(settings)
}

func (h *autoCompactionHarness) appendMessages(t *testing.T, messages ...agent.AgentMessage) {
	t.Helper()
	for _, message := range messages {
		if _, err := h.session.SessionManager.AppendMessage(h.ctx, message); err != nil {
			t.Fatal(err)
		}
	}
	if err := h.session.Agent.ReplaceMessages(h.session.SessionManager.BuildSessionContext().Messages); err != nil {
		t.Fatal(err)
	}
}

func (h *autoCompactionHarness) recordedRequests() []autoCompactionRequest {
	h.mu.Lock()
	defer h.mu.Unlock()
	return append([]autoCompactionRequest(nil), h.requests...)
}

func (h *autoCompactionHarness) assertCounts(t *testing.T, summaries, requests int) {
	t.Helper()
	if got := h.summaryCalls.Load(); got != int32(summaries) {
		t.Errorf("summary requests = %d, want %d", got, summaries)
	}
	if got := len(h.recordedRequests()); got != requests {
		t.Errorf("agent requests = %d, want %d (threshold compaction must not retry without queued messages)", got, requests)
	}
	compactions := 0
	for _, entry := range h.session.SessionManager.GetEntries() {
		if _, ok := entry.(*CompactionEntry); ok {
			compactions++
		}
	}
	if compactions != summaries {
		t.Errorf("persisted compactions = %d, want %d", compactions, summaries)
	}
	if h.session.IsCompacting() || h.session.IsStreaming() {
		t.Error("operation returned before compaction/agent run finished")
	}
	if got, want := h.session.Agent.State().Messages, h.session.SessionManager.BuildSessionContext().Messages; !reflect.DeepEqual(got, want) {
		t.Errorf("live context differs from persisted context:\ngot  %#v\nwant %#v", got, want)
	}
}

func assertAutoCompactedRequest(t *testing.T, request autoCompactionRequest, prompt string) {
	t.Helper()
	messages := request.context.Messages
	if len(messages) < 2 {
		t.Fatalf("compacted request has only %d messages", len(messages))
	}
	wantSummary := CompactionSummaryPrefix + autoCompactionTestSummary + CompactionSummarySuffix
	if got := userText(t, messages[0]); got != wantSummary {
		t.Errorf("first provider message = %q, want summary %q", got, wantSummary)
	}
	for _, message := range messages {
		if user, ok := message.(ai.UserMessage); ok && userText(t, user) == "aaaa" {
			t.Error("provider received old history that should have been summarized")
		}
	}
	if got := userText(t, messages[len(messages)-1]); got != prompt {
		t.Errorf("last provider message = %q, want %q", got, prompt)
	}
}

type autoCompactionOperation struct {
	done chan struct{}
	err  error
}

func (h *autoCompactionHarness) start(t *testing.T, operation func(context.Context) error) *autoCompactionOperation {
	t.Helper()
	result := &autoCompactionOperation{done: make(chan struct{})}
	go func() {
		defer close(result.done)
		result.err = operation(h.ctx)
	}()
	t.Cleanup(func() {
		// Cancel the shared context before joining: a second Prompt may be waiting
		// for operationMu while the first is blocked in the summary provider.
		h.cancel()
		h.session.Agent.Abort()
		h.session.AbortCompaction()
		select {
		case <-result.done:
		case <-time.After(autoCompactionTestTimeout):
			t.Error("operation goroutine did not exit during cleanup")
		}
	})
	return result
}

func (operation *autoCompactionOperation) wait(t *testing.T) error {
	t.Helper()
	select {
	case <-operation.done:
		return operation.err
	case <-time.After(autoCompactionTestTimeout):
		t.Fatal("operation did not finish")
		return nil
	}
}

func waitAutoCompactionSignal(t *testing.T, signal <-chan struct{}) {
	t.Helper()
	select {
	case <-signal:
	case <-time.After(autoCompactionTestTimeout):
		t.Fatal("provider did not reach the expected blocking point")
	}
}

func TestAgentSessionAutoCompactionThreshold(t *testing.T) {
	for _, phase := range []string{"pre_prompt", "agent_end"} {
		t.Run(phase, func(t *testing.T) {
			for _, test := range []struct {
				name    string
				tokens  int64
				enabled bool
				want    int
			}{
				{name: "above", tokens: 901, enabled: true, want: 1},
				{name: "below", tokens: 899, enabled: true},
				{name: "at_threshold", tokens: 900, enabled: true},
				{name: "disabled", tokens: 950},
			} {
				t.Run(test.name, func(t *testing.T) {
					h := newAutoCompactionHarness(t, false)
					settings := h.session.GetCompactionSettings()
					settings.Enabled = test.enabled
					h.session.SetCompactionSettings(settings)
					response := h.response(test.tokens, ai.StopReasonStop)
					if phase == "pre_prompt" {
						h.appendMessages(t, manualCompactionUserMessage("seed"), response)
						response = h.response(40, ai.StopReasonStop)
					}
					h.responses = []ai.AssistantMessage{response}
					if err := h.session.Prompt(h.ctx, "next", nil); err != nil {
						t.Fatal(err)
					}
					h.assertCounts(t, test.want, 1)
					request := h.recordedRequests()[0]
					wantBefore := int32(0)
					if phase == "pre_prompt" {
						wantBefore = int32(test.want)
					}
					if request.summaries != wantBefore {
						t.Errorf("summaries before agent request = %d, want %d", request.summaries, wantBefore)
					}
					if wantBefore == 1 {
						assertAutoCompactedRequest(t, request, "next")
					} else if got := userText(t, request.context.Messages[0]); got != "aaaa" {
						t.Errorf("uncompacted request starts with %q, want original history", got)
					}
					if test.want == 1 {
						entry := GetLatestCompactionEntry(h.session.SessionManager.GetEntries())
						if entry == nil || entry.Summary != autoCompactionTestSummary || entry.TokensBefore != test.tokens {
							t.Errorf("persisted compaction = %#v, want summary with %d tokens before", entry, test.tokens)
						}
					}
				})
			}
		})
	}
}

// A threshold crossing with everything inside KeepRecentTokens has nothing to
// summarize. Like pi, it ends quietly instead of reporting a failure.
func TestAgentSessionAutoCompactionThresholdWithNothingToCompact(t *testing.T) {
	h := newAutoCompactionHarness(t, false)
	settings := h.session.GetCompactionSettings()
	settings.KeepRecentTokens = 1_000_000
	h.session.SetCompactionSettings(settings)
	log, unsubscribe := recordSessionEvents(h)
	defer unsubscribe()
	h.responses = []ai.AssistantMessage{h.response(901, ai.StopReasonStop)}

	if err := h.session.Prompt(h.ctx, "next", nil); err != nil {
		t.Fatal(err)
	}
	h.assertCounts(t, 0, 1)

	observations := log.snapshot(true)
	if got := sessionEventTypes(observations); !reflect.DeepEqual(got, []string{"compaction_start", "compaction_end"}) {
		t.Fatalf("compaction events = %v, want start and end", got)
	}
	assertSessionEventStart(t, observations[0], CompactionReasonThreshold)
	end := observations[1].event.(CompactionEndEvent)
	if end.ErrorMessage != "" || end.Aborted || end.WillRetry || end.Result != nil {
		t.Errorf("end event = %#v, want a silent no-op", end)
	}
}

func TestAgentSessionAutoCompactionAbortedSkipsAgentEndButCompactsPrePrompt(t *testing.T) {
	h := newAutoCompactionHarness(t, false)
	h.responses = []ai.AssistantMessage{h.response(950, ai.StopReasonAborted), h.response(40, ai.StopReasonStop)}
	if err := h.session.Prompt(h.ctx, "abort", nil); err != nil {
		t.Fatal(err)
	}
	h.assertCounts(t, 0, 1)
	messages := h.session.Agent.State().Messages
	if last := messages[len(messages)-1].(ai.AssistantMessage); last.StopReason != ai.StopReasonAborted {
		t.Fatalf("last stop reason = %q, want aborted", last.StopReason)
	}
	if err := h.session.Prompt(h.ctx, "next", nil); err != nil {
		t.Fatal(err)
	}
	h.assertCounts(t, 1, 2)
	request := h.recordedRequests()[1]
	if request.summaries != 1 {
		t.Error("aborted high-usage response did not trigger pre-prompt compaction")
	}
	assertAutoCompactedRequest(t, request, "next")
}

func TestAgentSessionAutoCompactionErrorUsesLastSuccessfulUsage(t *testing.T) {
	for _, phase := range []string{"pre_prompt", "agent_end"} {
		t.Run(phase, func(t *testing.T) {
			for _, test := range []struct {
				name          string
				successTokens int64
				errorTokens   int64
				want          int
			}{
				{name: "successful_usage_plus_trailing_messages", successTokens: 890, want: 1},
				{name: "ignore_error_usage", successTokens: 40, errorTokens: 950},
			} {
				t.Run(test.name, func(t *testing.T) {
					h := newAutoCompactionHarness(t, false)
					// The successful usage alone is below threshold. The trailing user
					// message pushes the estimate above it, even with zero error usage.
					h.appendMessages(t, manualCompactionUserMessage("seed"), h.response(test.successTokens, ai.StopReasonStop))
					failed := h.response(test.errorTokens, ai.StopReasonError)
					prompt := strings.Repeat("x", 80)
					if phase == "pre_prompt" {
						h.appendMessages(t, manualCompactionUserMessage(prompt), failed)
						h.responses = []ai.AssistantMessage{h.response(40, ai.StopReasonStop)}
					} else {
						h.responses = []ai.AssistantMessage{failed}
					}
					if err := h.session.Prompt(h.ctx, prompt, nil); err != nil {
						t.Fatal(err)
					}
					h.assertCounts(t, test.want, 1)
					wantBefore := int32(0)
					if phase == "pre_prompt" {
						wantBefore = int32(test.want)
					}
					if got := h.recordedRequests()[0].summaries; got != wantBefore {
						t.Errorf("summaries before agent request = %d, want %d", got, wantBefore)
					}
					if test.want == 1 {
						entry := GetLatestCompactionEntry(h.session.SessionManager.GetEntries())
						wantTokens := test.successTokens + EstimateTokens(manualCompactionUserMessage(prompt)) + EstimateTokens(failed)
						if entry == nil || entry.TokensBefore != wantTokens {
							t.Errorf("compaction = %#v, want last successful usage plus trailing estimate %d", entry, wantTokens)
						}
					}
				})
			}
		})
	}
}

func TestAgentSessionAutoCompactionIgnoresRetainedUsage(t *testing.T) {
	for _, reason := range []ai.StopReason{ai.StopReasonStop, ai.StopReasonError} {
		t.Run(string(reason), func(t *testing.T) {
			h := newAutoCompactionHarness(t, false)
			high := h.response(950, ai.StopReasonStop)
			// A retained message can look newer by timestamp; branch order, not
			// timestamps or the live context's last successful usage, is the boundary.
			high.Timestamp = time.Now().Add(time.Hour).UnixMilli()
			h.responses = []ai.AssistantMessage{high, h.response(0, reason), h.response(40, ai.StopReasonStop)}
			if err := h.session.Prompt(h.ctx, "first", nil); err != nil {
				t.Fatal(err)
			}
			h.assertCounts(t, 1, 1)
			messages := h.session.Agent.State().Messages
			if !reflect.DeepEqual(messages[len(messages)-1], high) {
				t.Fatal("test requires high-usage response to remain in compacted context")
			}
			for index, prompt := range []string{"second", "third"} {
				if err := h.session.Prompt(h.ctx, prompt, nil); err != nil {
					t.Fatal(err)
				}
				h.assertCounts(t, 1, index+2)
				assertAutoCompactedRequest(t, h.recordedRequests()[index+1], prompt)
			}
		})
	}
}

func TestAgentSessionAutoCompactionQueuedPromptsContinue(t *testing.T) {
	for _, behavior := range []StreamingBehavior{StreamingBehaviorSteer, StreamingBehaviorFollowUp} {
		t.Run(string(behavior), func(t *testing.T) {
			h := newAutoCompactionHarness(t, true)
			h.responses = []ai.AssistantMessage{h.response(950, ai.StopReasonStop), h.response(40, ai.StopReasonStop)}
			first := h.start(t, func(ctx context.Context) error { return h.session.Prompt(ctx, "first", nil) })
			waitAutoCompactionSignal(t, h.summaryStarted)
			if !h.session.IsCompacting() || h.session.IsStreaming() {
				t.Fatal("expected blocked automatic summary with an idle agent")
			}
			queued := h.start(t, func(ctx context.Context) error {
				return h.session.Prompt(ctx, "queued", &PromptOptions{StreamingBehavior: behavior})
			})
			if err := queued.wait(t); err != nil {
				t.Fatalf("queue during summary: %v", err)
			}
			if !h.session.Agent.HasQueuedMessages() {
				t.Fatal("Prompt did not queue the message during automatic summary")
			}
			if got := len(h.recordedRequests()); got != 1 {
				t.Fatalf("agent requests during blocked summary = %d, want 1", got)
			}
			select {
			case <-first.done:
				t.Fatal("original Prompt returned before summary and continuation completed")
			default:
			}
			close(h.summaryRelease)
			if err := first.wait(t); err != nil {
				t.Fatal(err)
			}
			h.assertCounts(t, 1, 2)
			if h.session.Agent.HasQueuedMessages() {
				t.Error("continuation did not drain the queued message")
			}
			assertAutoCompactedRequest(t, h.recordedRequests()[1], "queued")
			queuedCount := 0
			for _, entry := range h.session.SessionManager.GetEntries() {
				if message, ok := getMessageFromEntry(entry).(ai.UserMessage); ok && userText(t, message) == "queued" {
					queuedCount++
				}
			}
			if queuedCount != 1 {
				t.Errorf("persisted queued messages = %d, want 1", queuedCount)
			}
		})
	}
}

func TestAgentSessionAutoCompactionNormalPromptWaitsForSummary(t *testing.T) {
	h := newAutoCompactionHarness(t, true)
	h.responses = []ai.AssistantMessage{h.response(950, ai.StopReasonStop), h.response(40, ai.StopReasonStop)}
	first := h.start(t, func(ctx context.Context) error { return h.session.Prompt(ctx, "first", nil) })
	waitAutoCompactionSignal(t, h.summaryStarted)
	secondStarted := make(chan struct{})
	second := h.start(t, func(ctx context.Context) error {
		close(secondStarted)
		return h.session.Prompt(ctx, "second", nil)
	})
	waitAutoCompactionSignal(t, secondStarted)
	select {
	case <-second.done:
		t.Fatalf("normal Prompt returned during blocked summary: %v", second.err)
	case <-time.After(50 * time.Millisecond):
	}
	if h.session.Agent.HasQueuedMessages() {
		t.Error("normal Prompt was queued rather than waiting for compaction")
	}
	if got := len(h.recordedRequests()); got != 1 {
		t.Fatalf("agent requests during blocked summary = %d, want 1", got)
	}
	close(h.summaryRelease)
	if err := first.wait(t); err != nil {
		t.Fatal(err)
	}
	if err := second.wait(t); err != nil {
		t.Fatal(err)
	}
	h.assertCounts(t, 1, 2)
	assertAutoCompactedRequest(t, h.recordedRequests()[1], "second")
}

func TestAgentSessionAutoCompactionManualCompactSuppressesAutomatic(t *testing.T) {
	h := newAutoCompactionHarness(t, true)
	streamStarted := make(chan struct{})
	h.session.Agent.StreamFn = func(ctx context.Context, _ ai.Model[ai.API], _ ai.ModelContext, _ *ai.SimpleStreamOptions) (*ai.AssistantMessageEventStream, error) {
		close(streamStarted)
		<-ctx.Done()
		return autoCompactionStream(h.response(950, ai.StopReasonAborted)), nil
	}
	prompt := h.start(t, func(ctx context.Context) error { return h.session.Prompt(ctx, "active", nil) })
	waitAutoCompactionSignal(t, streamStarted)
	manual := h.start(t, func(ctx context.Context) error {
		_, err := h.session.Compact(ctx, nil)
		return err
	})
	waitAutoCompactionSignal(t, h.summaryStarted)
	if err := prompt.wait(t); err != nil && !errors.Is(err, context.Canceled) {
		t.Fatalf("aborted Prompt: %v", err)
	}
	if !h.session.IsCompacting() || h.summaryCalls.Load() != 1 {
		t.Fatal("expected only the blocked manual summary")
	}
	close(h.summaryRelease)
	if err := manual.wait(t); err != nil {
		t.Fatal(err)
	}
	h.assertCounts(t, 1, 0)
	// The aborted response is recorded like any other abort. Its high usage must
	// not trigger automatic compaction on top of the manual one.
	var persisted []agent.AgentMessage
	for _, entry := range h.session.SessionManager.GetEntries() {
		if message := getMessageFromEntry(entry); message != nil {
			persisted = append(persisted, message)
		}
	}
	if got := countAbortedAssistants(persisted); got != 1 {
		t.Errorf("persisted aborted responses = %d, want 1", got)
	}
}

// A prompt sent during manual compaction neither fails nor joins the aborted
// run's queue. It waits and then runs as a normal turn on the compacted context.
func TestAgentSessionPromptWaitsForManualCompaction(t *testing.T) {
	h := newAutoCompactionHarness(t, true)
	h.responses = []ai.AssistantMessage{h.response(40, ai.StopReasonStop)}
	manual := h.start(t, func(ctx context.Context) error {
		_, err := h.session.Compact(ctx, nil)
		return err
	})
	waitAutoCompactionSignal(t, h.summaryStarted)

	waiting := h.start(t, func(ctx context.Context) error {
		return h.session.Prompt(ctx, "during manual", &PromptOptions{StreamingBehavior: StreamingBehaviorFollowUp})
	})
	cancelled, cancel := context.WithCancel(h.ctx)
	abandoned := h.start(t, func(context.Context) error { return h.session.Prompt(cancelled, "abandoned", nil) })
	cancel()
	if err := abandoned.wait(t); !errors.Is(err, context.Canceled) {
		t.Fatalf("Prompt cancelled while waiting = %v, want context.Canceled", err)
	}
	select {
	case <-waiting.done:
		t.Fatalf("Prompt finished during compaction: %v", waiting.err)
	default:
	}
	if h.session.Agent.HasQueuedMessages() {
		t.Fatal("prompt during manual compaction was queued on the agent")
	}

	close(h.summaryRelease)
	if err := manual.wait(t); err != nil {
		t.Fatal(err)
	}
	if err := waiting.wait(t); err != nil {
		t.Fatal(err)
	}
	h.assertCounts(t, 1, 1)
	assertAutoCompactedRequest(t, h.recordedRequests()[0], "during manual")
}

// failAutoCompactionSummaries makes every summary request fail, like a provider outage.
func failAutoCompactionSummaries(t *testing.T, h *autoCompactionHarness, failure error) {
	t.Helper()
	registerManualCompactionProviderWithStream(t, func(ai.Model[ai.API], ai.ModelContext, *ai.SimpleStreamOptions) (*ai.AssistantMessageEventStream, error) {
		h.summaryCalls.Add(1)
		return nil, failure
	})
}

// assertPromptSentAfterFailedCompaction checks that the prompt reached the
// provider and was persisted with its reply, and that no compaction was saved.
func assertPromptSentAfterFailedCompaction(t *testing.T, h *autoCompactionHarness, log *sessionEventLog, prompt string, aborted bool) CompactionEndEvent {
	t.Helper()
	observations := log.snapshot(true)
	if got, want := sessionEventTypes(observations), []string{"compaction_start", "compaction_end"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("compaction events = %v, want %v", got, want)
	}
	end := observations[1].event.(CompactionEndEvent)
	if end.Reason != CompactionReasonThreshold || end.Result != nil || end.WillRetry || end.Aborted != aborted {
		t.Errorf("compaction_end = %#v, want failed threshold compaction with aborted=%v", end, aborted)
	}

	requests := h.recordedRequests()
	if len(requests) != 1 {
		t.Fatalf("agent requests = %d, want 1", len(requests))
	}
	sent := requests[0].context.Messages
	if got := userText(t, sent[len(sent)-1]); got != prompt {
		t.Errorf("last provider message = %q, want %q", got, prompt)
	}

	if GetLatestCompactionEntry(h.session.SessionManager.GetEntries()) != nil {
		t.Error("failed compaction was persisted")
	}
	persisted := h.session.SessionManager.BuildSessionContext().Messages
	if got := userText(t, persisted[len(persisted)-2]); got != prompt {
		t.Errorf("persisted prompt = %q, want %q", got, prompt)
	}
	if reply, ok := persisted[len(persisted)-1].(ai.AssistantMessage); !ok || reply.StopReason != ai.StopReasonStop {
		t.Errorf("persisted reply = %#v, want successful assistant message", persisted[len(persisted)-1])
	}
	if live := h.session.Agent.State().Messages; !reflect.DeepEqual(live, persisted) {
		t.Errorf("live context differs from persisted context:\ngot  %#v\nwant %#v", live, persisted)
	}
	return end
}

func TestAgentSessionAutoCompactionFailureAfterTurnDoesNotFailPrompt(t *testing.T) {
	h := newAutoCompactionHarness(t, false)
	failAutoCompactionSummaries(t, h, errors.New("529 overloaded"))
	h.responses = []ai.AssistantMessage{h.response(950, ai.StopReasonStop)}
	log, unsubscribe := recordSessionEvents(h)
	defer unsubscribe()

	// The turn succeeds and crosses the threshold; its compaction then fails.
	if err := h.session.Prompt(h.ctx, "first", nil); err != nil {
		t.Fatalf("Prompt error = %v, want nil for a successful turn", err)
	}
	end := assertPromptSentAfterFailedCompaction(t, h, log, "first", false)
	if !strings.HasPrefix(end.ErrorMessage, "Auto-compaction failed:") || !strings.Contains(end.ErrorMessage, "529 overloaded") {
		t.Errorf("compaction_end errorMessage = %q, want auto-compaction failure", end.ErrorMessage)
	}
	if got := h.summaryCalls.Load(); got != 1 {
		t.Errorf("summary requests = %d, want 1", got)
	}
}

func TestAgentSessionAutoCompactionFailureBeforePromptStillSends(t *testing.T) {
	h := newAutoCompactionHarness(t, false)
	failAutoCompactionSummaries(t, h, errors.New("529 overloaded"))
	h.appendMessages(t, manualCompactionUserMessage("big"), h.response(950, ai.StopReasonStop))
	h.responses = []ai.AssistantMessage{h.response(40, ai.StopReasonStop)}
	log, unsubscribe := recordSessionEvents(h)
	defer unsubscribe()

	if err := h.session.Prompt(h.ctx, "next", nil); err != nil {
		t.Fatalf("Prompt error = %v, want nil after failed pre-prompt compaction", err)
	}
	end := assertPromptSentAfterFailedCompaction(t, h, log, "next", false)
	if !strings.HasPrefix(end.ErrorMessage, "Auto-compaction failed:") || !strings.Contains(end.ErrorMessage, "529 overloaded") {
		t.Errorf("compaction_end errorMessage = %q, want auto-compaction failure", end.ErrorMessage)
	}
	if got := h.summaryCalls.Load(); got != 1 {
		t.Errorf("summary requests = %d, want 1", got)
	}
}

func TestAgentSessionAbortCompactionBeforePromptStillSends(t *testing.T) {
	h := newAutoCompactionHarness(t, true)
	h.appendMessages(t, manualCompactionUserMessage("big"), h.response(950, ai.StopReasonStop))
	h.responses = []ai.AssistantMessage{h.response(40, ai.StopReasonStop)}
	log, unsubscribe := recordSessionEvents(h)
	defer unsubscribe()

	prompt := h.start(t, func(ctx context.Context) error { return h.session.Prompt(ctx, "next", nil) })
	waitAutoCompactionSignal(t, h.summaryStarted)
	if got := len(h.recordedRequests()); got != 0 {
		t.Fatalf("agent requests before compaction finished = %d, want 0", got)
	}
	h.session.AbortCompaction()

	if err := prompt.wait(t); err != nil {
		t.Fatalf("Prompt error = %v, want nil after AbortCompaction", err)
	}
	end := assertPromptSentAfterFailedCompaction(t, h, log, "next", true)
	if end.ErrorMessage != "" {
		t.Errorf("aborted compaction_end errorMessage = %q, want empty", end.ErrorMessage)
	}
	if err := h.ctx.Err(); err != nil {
		t.Errorf("AbortCompaction canceled caller context: %v", err)
	}
}

func TestAgentSessionAutoCompactionContinuesPromptsQueuedAtAgentEnd(t *testing.T) {
	for _, tokens := range []int64{40, 950} {
		t.Run(fmt.Sprint(tokens), func(t *testing.T) {
			h := newAutoCompactionHarness(t, false)
			h.responses = []ai.AssistantMessage{h.response(tokens, ai.StopReasonStop), h.response(40, ai.StopReasonStop)}
			queued := false
			unsubscribe := h.session.Agent.Subscribe(func(event agent.AgentEvent, ctx context.Context) error {
				if _, ok := event.(agent.EndEvent); ok && !queued {
					queued = true
					return h.session.Prompt(ctx, "queued", &PromptOptions{StreamingBehavior: StreamingBehaviorFollowUp})
				}
				return nil
			})
			defer unsubscribe()
			if err := h.session.Prompt(h.ctx, "first", nil); err != nil {
				t.Fatal(err)
			}
			wantSummaries := 0
			if tokens > 900 {
				wantSummaries = 1
			}
			h.assertCounts(t, wantSummaries, 2)
			if h.session.Agent.HasQueuedMessages() {
				t.Fatal("message accepted at agent_end was stranded")
			}
			request := h.recordedRequests()[1]
			if got := userText(t, request.context.Messages[len(request.context.Messages)-1]); got != "queued" {
				t.Errorf("continued prompt = %q, want queued", got)
			}
		})
	}
}

// ============================================================================
// Context Overflow Recovery Tests
// ============================================================================

func assertOverflowRecoveryCounts(t *testing.T, h *autoCompactionHarness, summaries, compactions int, requestSummaries ...int32) {
	t.Helper()
	if got := h.summaryCalls.Load(); got != int32(summaries) {
		t.Errorf("summary requests = %d, want %d", got, summaries)
	}
	persisted := 0
	for _, entry := range h.session.SessionManager.GetEntries() {
		if _, ok := entry.(*CompactionEntry); ok {
			persisted++
		}
	}
	if persisted != compactions {
		t.Errorf("persisted compactions = %d, want %d", persisted, compactions)
	}
	requests := h.recordedRequests()
	if len(requests) != len(requestSummaries) {
		t.Fatalf("agent requests = %d, want %d", len(requests), len(requestSummaries))
	}
	for i, want := range requestSummaries {
		if requests[i].summaries != want {
			t.Errorf("summaries before agent request %d = %d, want %d", i+1, requests[i].summaries, want)
		}
	}
	if h.session.IsCompacting() || h.session.IsStreaming() {
		t.Error("Prompt returned before compaction/retry finished")
	}
}

func assertOverflowMessageCount(t *testing.T, messages []agent.AgentMessage, wantMessage ai.AssistantMessage, wantCount int) {
	t.Helper()
	count := 0
	for _, message := range messages {
		if reflect.DeepEqual(message, wantMessage) {
			count++
		}
	}
	if count != wantCount {
		t.Errorf("occurrences of response (stop=%q, error=%q, usage=%+v) = %d, want %d", wantMessage.StopReason, wantMessage.ErrorMessage, wantMessage.Usage, count, wantCount)
	}
}

func TestAgentSessionContextOverflowRecoveryCompactsAndRetries(t *testing.T) {
	for _, kind := range []string{"provider_error", "silent_success", "zero_output_length"} {
		t.Run(kind, func(t *testing.T) {
			h := newAutoCompactionHarness(t, true)
			overflow := h.response(0, ai.StopReasonError)
			overflow.ErrorMessage = "prompt is too long"
			switch kind {
			case "provider_error":
				h.retainOverflowedTurn()
			case "silent_success":
				overflow = h.response(600, ai.StopReasonStop)
				overflow.Usage.CacheRead = 401
				overflow.Usage.Total = 1001
			case "zero_output_length":
				overflow = h.response(600, ai.StopReasonLength)
				overflow.Content = nil
				overflow.Usage.CacheRead = 390
				overflow.Usage.Total = 990
				h.retainOverflowedTurn()
			}
			success := h.response(40, ai.StopReasonStop)
			h.responses = []ai.AssistantMessage{overflow, success}
			prompt := h.start(t, func(ctx context.Context) error { return h.session.Prompt(ctx, "next", nil) })
			waitAutoCompactionSignal(t, h.summaryStarted)
			if !h.session.IsCompacting() || h.session.IsStreaming() {
				t.Fatal("expected blocked recovery summary with an idle agent")
			}
			select {
			case <-prompt.done:
				t.Fatalf("Prompt returned before recovery completed: %v", prompt.err)
			default:
			}
			if got := len(h.recordedRequests()); got != 1 {
				t.Fatalf("agent requests during summary = %d, want 1", got)
			}
			assertOverflowMessageCount(t, h.session.Agent.State().Messages, overflow, 0)
			assertOverflowMessageCount(t, h.session.SessionManager.BuildSessionContext().Messages, overflow, 1)
			if entry := GetLatestCompactionEntry(h.session.SessionManager.GetEntries()); entry != nil {
				t.Fatal("compaction persisted before summary completed")
			}

			close(h.summaryRelease)
			if err := prompt.wait(t); err != nil {
				t.Fatalf("Prompt recovery: %v", err)
			}
			assertOverflowRecoveryCounts(t, h, 1, 1, 0, 1)
			retry := h.recordedRequests()[1]
			assertAutoCompactedRequest(t, retry, "next")
			for _, message := range retry.context.Messages {
				if assistant, ok := message.(ai.AssistantMessage); ok && ai.IsContextOverflow(assistant, h.model.ContextWindow) {
					t.Errorf("retry provider received overflow response: %#v", assistant)
				}
			}

			entries := h.session.SessionManager.GetEntries()
			compaction := GetLatestCompactionEntry(entries)
			if compaction == nil || compaction.Summary != autoCompactionTestSummary {
				t.Fatalf("persisted compaction = %#v, want recovery summary", compaction)
			}
			var persisted []agent.AgentMessage
			users := 0
			for _, entry := range entries {
				message := getMessageFromEntry(entry)
				if message != nil {
					persisted = append(persisted, message)
				}
				if user, ok := message.(ai.UserMessage); ok && userText(t, user) == "next" {
					users++
					if compaction.FirstKeptEntryID != entry.GetBase().ID {
						t.Error("fixture did not retain the overflow's whole user turn")
					}
				}
			}
			if users != 1 {
				t.Errorf("persisted user prompts = %d, want 1 (retry must not resend the user message)", users)
			}
			assertOverflowMessageCount(t, persisted, overflow, 1)
			assertOverflowMessageCount(t, persisted, success, 1)
			reloaded := h.session.SessionManager.BuildSessionContext().Messages
			assertOverflowMessageCount(t, reloaded, overflow, 1)
			var wantLive []agent.AgentMessage
			for _, message := range reloaded {
				if !reflect.DeepEqual(message, overflow) {
					wantLive = append(wantLive, message)
				}
			}
			if got := h.session.Agent.State().Messages; !reflect.DeepEqual(got, wantLive) {
				t.Errorf("live context must match retained history except for stripped overflow:\ngot  %#v\nwant %#v", got, wantLive)
			}
		})
	}
}

func TestAgentSessionContextOverflowRecoveryStopsAfterSecondOverflow(t *testing.T) {
	h := newAutoCompactionHarness(t, false)
	h.retainOverflowedTurn()
	first := h.response(0, ai.StopReasonError)
	first.ErrorMessage = "prompt is too long"
	second := first
	second.ErrorMessage = "context_length_exceeded"
	h.responses = []ai.AssistantMessage{first, second}
	// compaction_end reports exhausted recovery; the overflow stays in the transcript.
	if err := h.session.Prompt(h.ctx, "next", nil); err != nil {
		t.Errorf("Prompt error = %v, want nil after exhausted recovery", err)
	}
	assertOverflowRecoveryCounts(t, h, 1, 1, 0, 1)
	assertAutoCompactedRequest(t, h.recordedRequests()[1], "next")
	assertOverflowMessageCount(t, h.session.SessionManager.BuildSessionContext().Messages, first, 1)
	assertOverflowMessageCount(t, h.session.SessionManager.BuildSessionContext().Messages, second, 1)
	assertOverflowMessageCount(t, h.session.Agent.State().Messages, first, 0)
	messages := h.session.Agent.State().Messages
	if len(messages) == 0 || !reflect.DeepEqual(messages[len(messages)-1], second) {
		t.Errorf("live context must end with the second overflow, got %#v", messages)
	}
}

func TestAgentSessionContextOverflowRecoverySkipped(t *testing.T) {
	for _, kind := range []string{"other_model", "other_provider", "disabled_error", "disabled_silent_success", "disabled_zero_output_length"} {
		t.Run(kind, func(t *testing.T) {
			h := newAutoCompactionHarness(t, false)
			overflow := h.response(0, ai.StopReasonError)
			overflow.ErrorMessage = "prompt is too long"
			switch kind {
			case "other_model":
				overflow.Model = "other-model"
			case "other_provider":
				overflow.Provider = "other-provider"
			case "disabled_silent_success":
				overflow = h.response(1001, ai.StopReasonStop)
			case "disabled_zero_output_length":
				overflow = h.response(990, ai.StopReasonLength)
				overflow.Content = nil
			}
			if strings.HasPrefix(kind, "disabled_") {
				settings := h.session.GetCompactionSettings()
				settings.Enabled = false
				h.session.SetCompactionSettings(settings)
			}
			h.responses = []ai.AssistantMessage{overflow}
			if err := h.session.Prompt(h.ctx, "next", nil); err != nil {
				t.Fatalf("Prompt: %v", err)
			}
			h.assertCounts(t, 0, 1)
			assertOverflowMessageCount(t, h.session.Agent.State().Messages, overflow, 1)
			if got := h.recordedRequests()[0].summaries; got != 0 {
				t.Errorf("summaries before request = %d, want 0", got)
			}
		})
	}
}

func TestAgentSessionContextOverflowRecoveryCancellation(t *testing.T) {
	for _, method := range []string{"abort_compaction", "caller_context"} {
		t.Run(method, func(t *testing.T) {
			h := newAutoCompactionHarness(t, true)
			h.retainOverflowedTurn()
			overflow := h.response(0, ai.StopReasonError)
			overflow.ErrorMessage = "prompt is too long"
			h.responses = []ai.AssistantMessage{overflow, h.response(40, ai.StopReasonStop)}
			callerCtx, cancel := context.WithCancel(h.ctx)
			defer cancel()
			prompt := h.start(t, func(context.Context) error { return h.session.Prompt(callerCtx, "next", nil) })
			waitAutoCompactionSignal(t, h.summaryStarted)
			if !h.session.IsCompacting() || h.session.IsStreaming() {
				t.Fatal("expected blocked recovery summary with an idle agent")
			}
			select {
			case <-prompt.done:
				t.Fatalf("Prompt returned before cancellation: %v", prompt.err)
			default:
			}
			if method == "abort_compaction" {
				h.session.AbortCompaction()
			} else {
				cancel()
			}
			err := prompt.wait(t)
			if method == "abort_compaction" {
				// AbortCompaction skips recovery without failing the prompt.
				if err != nil {
					t.Errorf("Prompt error = %v, want nil", err)
				}
			} else if !errors.Is(err, context.Canceled) {
				t.Errorf("Prompt error = %v, want context.Canceled", err)
			}
			if method == "abort_compaction" && callerCtx.Err() != nil {
				t.Errorf("AbortCompaction canceled caller context: %v", callerCtx.Err())
			}
			assertOverflowRecoveryCounts(t, h, 1, 0, 0)
			assertOverflowMessageCount(t, h.session.SessionManager.BuildSessionContext().Messages, overflow, 1)
			assertOverflowMessageCount(t, h.session.Agent.State().Messages, overflow, 0)
		})
	}
}

func TestAgentSessionContextOverflowRecoveryResets(t *testing.T) {
	for _, reset := range []string{"new_user_after_failed_retry", "successful_response"} {
		t.Run(reset, func(t *testing.T) {
			h := newAutoCompactionHarness(t, false)
			h.retainOverflowedTurn()
			overflow := h.response(0, ai.StopReasonError)
			overflow.ErrorMessage = "prompt is too long"
			success := h.response(40, ai.StopReasonStop)
			firstRetry := success
			wantAttempted := reset == "new_user_after_failed_retry"
			if wantAttempted {
				firstRetry = overflow
			}
			h.responses = []ai.AssistantMessage{overflow, firstRetry, overflow, success}
			// Exhausted recovery is reported by compaction_end, not Prompt.
			if err := h.session.Prompt(h.ctx, "first", nil); err != nil {
				t.Fatalf("first Prompt: %v", err)
			}
			assertOverflowRecoveryCounts(t, h, 1, 1, 0, 1)
			// Inspect before sending another user message so its reset cannot mask
			// a missing reset on the successful assistant response.
			h.session.mu.Lock()
			attempted := h.session.overflowAttempted
			h.session.mu.Unlock()
			if attempted != wantAttempted {
				t.Errorf("recovery attempted after first Prompt = %v, want %v", attempted, wantAttempted)
			}
			if err := h.session.Prompt(h.ctx, "next", nil); err != nil {
				t.Fatalf("new turn recovery: %v", err)
			}
			assertOverflowRecoveryCounts(t, h, 2, 2, 0, 1, 1, 2)
			assertAutoCompactedRequest(t, h.recordedRequests()[3], "next")
			messages := h.session.Agent.State().Messages
			if len(messages) == 0 || !reflect.DeepEqual(messages[len(messages)-1], success) {
				t.Errorf("new turn did not finish with successful retry: %#v", messages)
			}
			h.session.mu.Lock()
			attempted = h.session.overflowAttempted
			h.session.mu.Unlock()
			if attempted {
				t.Error("successful retry did not reset recovery for the next overflow")
			}
		})
	}
}

// ============================================================================
// Session Event Tests
// ============================================================================

// These snapshots are taken inside the callback, not after Prompt/Compact returns:
// completion events must describe already-persisted/reloaded state before retry.
type sessionEventObservation struct {
	event      AgentSessionEvent
	compacting bool
	streaming  bool
	requests   int
	summaries  int32
	latest     *CompactionEntry
	live       []agent.AgentMessage
	persisted  []agent.AgentMessage
}

type sessionEventLog struct {
	mu           sync.Mutex
	observations []sessionEventObservation
}

func recordSessionEvents(h *autoCompactionHarness) (*sessionEventLog, func()) {
	log := &sessionEventLog{}
	unsubscribe := h.session.Subscribe(func(event AgentSessionEvent) {
		observation := sessionEventObservation{event: event}
		switch event.(type) {
		case CompactionStartEvent, CompactionEndEvent:
			observation.compacting = h.session.IsCompacting()
			observation.streaming = h.session.IsStreaming()
			observation.requests = len(h.recordedRequests())
			observation.summaries = h.summaryCalls.Load()
			observation.latest = GetLatestCompactionEntry(h.session.SessionManager.GetEntries())
			observation.live = h.session.Agent.State().Messages
			observation.persisted = h.session.SessionManager.BuildSessionContext().Messages
		}
		log.mu.Lock()
		log.observations = append(log.observations, observation)
		log.mu.Unlock()
	})
	return log, unsubscribe
}

func (log *sessionEventLog) snapshot(compactionsOnly bool) []sessionEventObservation {
	log.mu.Lock()
	defer log.mu.Unlock()
	var observations []sessionEventObservation
	for _, observation := range log.observations {
		if compactionsOnly {
			switch observation.event.(type) {
			case CompactionStartEvent, CompactionEndEvent:
			default:
				continue
			}
		}
		observations = append(observations, observation)
	}
	return observations
}

func sessionEventTypes(observations []sessionEventObservation) []string {
	types := make([]string, 0, len(observations))
	for _, observation := range observations {
		types = append(types, observation.event.EventType())
	}
	return types
}

func assertSessionEventStart(t *testing.T, observation sessionEventObservation, reason CompactionReason) {
	t.Helper()
	start, ok := observation.event.(CompactionStartEvent)
	if !ok || start.Type != "compaction_start" || start.EventType() != start.Type || start.Reason != reason {
		t.Errorf("start event = %#v, want compaction_start/%s", observation.event, reason)
	}
	if !observation.compacting || observation.streaming {
		t.Errorf("at start: compacting=%v streaming=%v, want active compaction and idle agent", observation.compacting, observation.streaming)
	}
	if observation.summaries != 0 || observation.latest != nil {
		t.Errorf("start arrived after summary/persistence: calls=%d latest=%#v", observation.summaries, observation.latest)
	}
}

func assertSessionEventSuccess(t *testing.T, observation sessionEventObservation, reason CompactionReason, willRetry bool) *CompactionResult {
	t.Helper()
	end, ok := observation.event.(CompactionEndEvent)
	if !ok {
		t.Fatalf("event = %T, want CompactionEndEvent", observation.event)
	}
	if end.Type != "compaction_end" || end.EventType() != end.Type || end.Reason != reason || end.Aborted || end.WillRetry != willRetry || end.ErrorMessage != "" {
		t.Errorf("successful end = %#v, want reason=%s willRetry=%v", end, reason, willRetry)
	}
	if end.Result == nil {
		t.Fatal("successful end has no result")
	}
	if end.Result.Summary != autoCompactionTestSummary || end.Result.FirstKeptEntryID == "" || end.Result.TokensBefore <= 0 {
		t.Errorf("incomplete compaction result: %#v", end.Result)
	}
	latest := observation.latest
	if latest == nil || latest.Summary != end.Result.Summary || latest.FirstKeptEntryID != end.Result.FirstKeptEntryID || latest.TokensBefore != end.Result.TokensBefore {
		t.Errorf("end preceded persistence: entry=%#v result=%#v", latest, end.Result)
	}
	if !reflect.DeepEqual(observation.live, observation.persisted) {
		t.Errorf("end preceded context reload:\nlive %#v\npersisted %#v", observation.live, observation.persisted)
	}
	if len(observation.live) == 0 {
		t.Error("end has empty live context")
	} else if _, ok := observation.live[0].(CompactionSummaryMessage); !ok {
		t.Errorf("first live message at end = %T, want CompactionSummaryMessage", observation.live[0])
	}
	if observation.streaming {
		t.Error("end arrived after retry/continuation started")
	}
	return end.Result
}

// Use a separate, bounded runner for subscription tests so a callback-lock
// regression fails instead of hanging the test goroutine itself.
func runSessionEventTest(t *testing.T, operation func()) {
	t.Helper()
	done := make(chan struct{})
	go func() {
		defer close(done)
		operation()
	}()
	select {
	case <-done:
	case <-time.After(autoCompactionTestTimeout):
		t.Fatal("session event operation timed out (possible callback deadlock)")
	}
}

func TestAgentSessionEventsTypesAndJSON(t *testing.T) {
	for _, tc := range []struct {
		reason CompactionReason
		wire   string
	}{
		{CompactionReasonManual, "manual"},
		{CompactionReasonThreshold, "threshold"},
		{CompactionReasonOverflow, "overflow"},
	} {
		t.Run(tc.wire, func(t *testing.T) {
			if string(tc.reason) != tc.wire {
				t.Errorf("reason = %q, want %q", tc.reason, tc.wire)
			}
			var start AgentSessionEvent = CompactionStartEvent{Type: "compaction_start", Reason: tc.reason}
			if start.EventType() != "compaction_start" || (CompactionStartEvent{}).EventType() != "compaction_start" || (CompactionEndEvent{}).EventType() != "compaction_end" {
				t.Fatal("event type must be independent of the Type field")
			}
			data, err := json.Marshal(start)
			if err != nil {
				t.Fatal(err)
			}
			var fields map[string]any
			if err := json.Unmarshal(data, &fields); err != nil {
				t.Fatal(err)
			}
			if want := map[string]any{"type": "compaction_start", "reason": tc.wire}; !reflect.DeepEqual(fields, want) {
				t.Errorf("start JSON = %s, want %#v", data, want)
			}
			for _, end := range []CompactionEndEvent{
				{Type: "compaction_end", Reason: tc.reason},
				{Type: "compaction_end", Reason: tc.reason, Aborted: true},
				{Type: "compaction_end", Reason: tc.reason, ErrorMessage: "summary failed"},
				{Type: "compaction_end", Reason: tc.reason, Result: &CompactionResult{Summary: "summary", FirstKeptEntryID: "kept", TokensBefore: 950}, WillRetry: true},
			} {
				var event AgentSessionEvent = end
				data, err := json.Marshal(event)
				if err != nil {
					t.Fatal(err)
				}
				fields = nil
				if err := json.Unmarshal(data, &fields); err != nil {
					t.Fatal(err)
				}
				_, hasResult := fields["result"]
				_, hasError := fields["errorMessage"]
				if fields["type"] != "compaction_end" || fields["reason"] != tc.wire || fields["aborted"] != end.Aborted || fields["willRetry"] != end.WillRetry || hasResult != (end.Result != nil) || hasError != (end.ErrorMessage != "") {
					t.Errorf("end JSON fields/omissions = %s for %#v", data, end)
				}
				var decoded CompactionEndEvent
				if err := json.Unmarshal(data, &decoded); err != nil || !reflect.DeepEqual(decoded, end) {
					t.Errorf("end JSON round trip = %#v, %v; want %#v", decoded, err, end)
				}
			}
		})
	}
}

func TestAgentSessionEventsForwardBeforePersistence(t *testing.T) {
	h := newAutoCompactionHarness(t, false)
	response := h.response(40, ai.StopReasonStop)
	h.session.Agent.StreamFn = func(context.Context, ai.Model[ai.API], ai.ModelContext, *ai.SimpleStreamOptions) (*ai.AssistantMessageEventStream, error) {
		stream := ai.NewAssistantMessageEventStream(3)
		stream.Push(ai.StartEvent{Type: "start", Partial: manualCompactionAssistantMessage("", h.model)})
		stream.Push(ai.TextDeltaEvent{Type: "text_delta", Delta: "done", Partial: response})
		stream.Push(ai.DoneEvent{Type: "done", Reason: ai.StopReasonStop, Message: response})
		return stream, nil
	}
	var forwarded, original []AgentSessionEvent
	var order []string
	initialEntries := len(h.session.SessionManager.GetEntries())
	messageEnds := 0
	unsubscribe := h.session.Subscribe(func(event AgentSessionEvent) {
		forwarded = append(forwarded, event)
		order = append(order, "session:"+event.EventType())
		if end, ok := event.(agent.MessageEndEvent); ok {
			entries := h.session.SessionManager.GetEntries()
			if len(entries) != initialEntries+messageEnds {
				t.Errorf("entry count during message_end = %d, want %d", len(entries), initialEntries+messageEnds)
			}
			for _, entry := range entries {
				if entry.GetBase().ID == end.MessageID {
					t.Errorf("message %s persisted before session notification", end.MessageID)
				}
			}
			messageEnds++
		}
	})
	defer unsubscribe()
	unsubscribeAgent := h.session.Agent.Subscribe(func(event agent.AgentEvent, _ context.Context) error {
		original = append(original, event)
		order = append(order, "agent:"+event.EventType())
		if end, ok := event.(agent.MessageEndEvent); ok {
			found := false
			for _, entry := range h.session.SessionManager.GetEntries() {
				if entry.GetBase().ID == end.MessageID && reflect.DeepEqual(getMessageFromEntry(entry), end.Message) {
					found = true
				}
			}
			if !found {
				t.Errorf("message %s not persisted after session listener returned", end.MessageID)
			}
		}
		return nil
	})
	defer unsubscribeAgent()
	operation := h.start(t, func(ctx context.Context) error { return h.session.Prompt(ctx, "events", nil) })
	if err := operation.wait(t); err != nil {
		t.Fatal(err)
	}
	wantTypes := []string{"agent_start", "turn_start", "message_start", "message_end", "message_start", "message_update", "message_end", "turn_end", "agent_end"}
	var gotTypes, wantOrder []string
	for _, event := range forwarded {
		gotTypes = append(gotTypes, event.EventType())
	}
	for _, eventType := range wantTypes {
		wantOrder = append(wantOrder, "session:"+eventType, "agent:"+eventType)
	}
	if !reflect.DeepEqual(gotTypes, wantTypes) || !reflect.DeepEqual(order, wantOrder) {
		t.Errorf("forwarding order = %v, want %v", order, wantOrder)
	}
	if !reflect.DeepEqual(forwarded, original) {
		t.Errorf("forwarded payloads differ:\nsession %#v\nagent %#v", forwarded, original)
	}
	if messageEnds != 2 || len(h.session.SessionManager.GetEntries()) != initialEntries+2 {
		t.Error("expected both user and assistant messages to be forwarded and persisted once")
	}
}

func TestAgentSessionEventsForwardToolEvents(t *testing.T) {
	h := newAutoCompactionHarness(t, false)
	toolReply := h.response(40, ai.StopReasonToolUse)
	toolReply.Content = []ai.AssistantContent{ai.ToolCall{Type: ai.ContentTypeToolCall, Id: "call-1", Name: "offline", Args: map[string]any{}}}
	h.responses = []ai.AssistantMessage{toolReply, h.response(40, ai.StopReasonStop)}
	partial := agent.AgentToolResult[any]{Content: []ai.ToolResultContent{ai.TextContent{Type: ai.ContentTypeText, Text: "partial"}}}
	complete := agent.AgentToolResult[any]{Content: []ai.ToolResultContent{ai.TextContent{Type: ai.ContentTypeText, Text: "complete"}}}
	oldAgent := h.session.Agent
	settings := h.session.GetCompactionSettings()
	h.session = NewAgentSession(agent.NewAgent(&agent.AgentOptions{
		InitialState: &agent.InitialAgentState{
			Model:    &h.model,
			Messages: oldAgent.State().Messages,
			Tools: []agent.AgentTool[any, any]{{
				Tool: ai.Tool{Name: "offline", Parameters: map[string]any{"type": "object"}},
				Execute: func(_ context.Context, _ string, _ any, update agent.AgentToolUpdateCallback[any]) (agent.AgentToolResult[any], error) {
					update(partial)
					return complete, nil
				},
			}},
		},
		GetAPIKey: oldAgent.GetAPIKey,
		StreamFn:  oldAgent.StreamFn,
	}), h.session.SessionManager)
	h.session.SetCompactionSettings(settings)
	var forwarded, original []AgentSessionEvent
	var timeline []string
	unsubscribe := h.session.Subscribe(func(event AgentSessionEvent) {
		forwarded = append(forwarded, event)
		timeline = append(timeline, "session:"+event.EventType())
		if end, ok := event.(agent.MessageEndEvent); ok {
			for _, entry := range h.session.SessionManager.GetEntries() {
				if entry.GetBase().ID == end.MessageID {
					t.Errorf("message %s persisted before notification", end.MessageID)
				}
			}
		}
	})
	defer unsubscribe()
	unsubscribeAgent := h.session.Agent.Subscribe(func(event agent.AgentEvent, _ context.Context) error {
		original = append(original, event)
		timeline = append(timeline, "agent:"+event.EventType())
		return nil
	})
	defer unsubscribeAgent()
	prompt := h.start(t, func(ctx context.Context) error { return h.session.Prompt(ctx, "use offline tool", nil) })
	if err := prompt.wait(t); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(forwarded, original) {
		t.Error("forwarded tool-run events differ from original agent payloads")
	}
	var tools []AgentSessionEvent
	var wantTimeline []string
	for _, event := range original {
		wantTimeline = append(wantTimeline, "session:"+event.EventType(), "agent:"+event.EventType())
		switch event.(type) {
		case agent.ToolExecutionStartEvent, agent.ToolExecutionUpdateEvent, agent.ToolExecutionEndEvent:
			tools = append(tools, event)
		}
	}
	if !reflect.DeepEqual(timeline, wantTimeline) {
		t.Errorf("tool-run forwarding order = %v, want %v", timeline, wantTimeline)
	}
	wantTools := []AgentSessionEvent{
		agent.ToolExecutionStartEvent{Type: "tool_execution_start", ToolCallID: "call-1", ToolName: "offline", ToolArgs: map[string]any{}},
		agent.ToolExecutionUpdateEvent{Type: "tool_execution_update", ToolCallID: "call-1", ToolName: "offline", ToolArgs: map[string]any{}, PartialResult: partial},
		agent.ToolExecutionEndEvent{Type: "tool_execution_end", ToolCallID: "call-1", ToolName: "offline", Result: complete},
	}
	if !reflect.DeepEqual(tools, wantTools) {
		t.Errorf("tool events = %#v, want %#v", tools, wantTools)
	}
	h.assertCounts(t, 0, 2)
}

func TestAgentSessionEventsDoNotSwallowPromptPersistenceError(t *testing.T) {
	h := newAutoCompactionHarness(t, false)
	failure := errors.New("ordinary message persistence failed")
	h.session.SessionManager.state.persist = true
	h.session.SessionManager.state.store = &appendErrorSessionStore{appendErr: failure}
	log, unsubscribe := recordSessionEvents(h)
	defer unsubscribe()
	initialEntries := len(h.session.SessionManager.GetEntries())
	prompt := h.start(t, func(ctx context.Context) error { return h.session.Prompt(ctx, "not persisted", nil) })
	if err := prompt.wait(t); !errors.Is(err, failure) {
		t.Errorf("Prompt error = %v, want original persistence error", err)
	}
	messageEnds := 0
	for _, observation := range log.snapshot(false) {
		if end, ok := observation.event.(agent.MessageEndEvent); ok {
			messageEnds++
			if _, ok := end.Message.(ai.UserMessage); !ok {
				t.Errorf("failed persistence notification contains %T, want user message", end.Message)
			}
		}
	}
	if messageEnds != 1 || len(log.snapshot(true)) != 0 || len(h.recordedRequests()) != 0 {
		t.Error("expected one forwarded user message_end and no provider call/compaction")
	}
	if len(h.session.SessionManager.GetEntries()) != initialEntries {
		t.Error("failed message append changed persisted history")
	}
}

func TestAgentSessionEventsSubscriptions(t *testing.T) {
	runSessionEventTest(t, func() {
		session := NewAgentSession(nil, nil)
		var calls []string
		first := session.Subscribe(func(AgentSessionEvent) { calls = append(calls, "first") })
		duplicate := AgentSessionEventListener(func(AgentSessionEvent) { calls = append(calls, "duplicate") })
		removeA := session.Subscribe(duplicate)
		removeB := session.Subscribe(duplicate)
		last := session.Subscribe(func(AgentSessionEvent) { calls = append(calls, "last") })
		nilUnsubscribe := session.Subscribe(nil)
		nilUnsubscribe()
		nilUnsubscribe()
		if len(calls) != 0 {
			t.Error("Subscribe invoked a listener")
		}
		check := func(want ...string) {
			calls = nil
			session.emit(agent.StartEvent{Type: "agent_start"})
			if !reflect.DeepEqual(calls, want) {
				t.Errorf("listener order = %v, want %v", calls, want)
			}
		}
		check("first", "duplicate", "duplicate", "last")
		removeA()
		removeA()
		check("first", "duplicate", "last")
		removeB()
		removeB()
		check("first", "last")
		first()
		last()
		check()
	})
}

func TestAgentSessionEventsReentrantSubscriptionSnapshot(t *testing.T) {
	h := newAutoCompactionHarness(t, false)
	runSessionEventTest(t, func() {
		var calls []string
		var removeSelf, removeOther, removeNew func()
		removeSelf = h.session.Subscribe(func(AgentSessionEvent) {
			calls = append(calls, "self")
			removeSelf()
			removeSelf()
			removeOther()
			removeNew = h.session.Subscribe(func(AgentSessionEvent) { calls = append(calls, "new") })
			settings := h.session.GetCompactionSettings()
			h.session.SetCompactionSettings(settings)
			if h.session.IsCompacting() || h.session.IsStreaming() {
				t.Error("idle session reports busy")
			}
			if _, err := h.session.GetState(); err != nil {
				t.Errorf("GetState in listener: %v", err)
			}
			if _, err := h.session.GetModel(); err != nil {
				t.Errorf("GetModel in listener: %v", err)
			}
			h.session.AbortCompaction()
		})
		removeOther = h.session.Subscribe(func(AgentSessionEvent) { calls = append(calls, "other") })
		h.session.emit(agent.StartEvent{Type: "agent_start"})
		if want := []string{"self", "other"}; !reflect.DeepEqual(calls, want) {
			t.Errorf("in-flight snapshot = %v, want %v", calls, want)
		}
		calls = nil
		h.session.emit(agent.EndEvent{Type: "agent_end"})
		if want := []string{"new"}; !reflect.DeepEqual(calls, want) {
			t.Errorf("next snapshot = %v, want %v", calls, want)
		}
		removeNew()
	})
}

func TestAgentSessionEventsConcurrentSubscribeEmit(t *testing.T) {
	runSessionEventTest(t, func() {
		session := NewAgentSession(nil, nil)
		const workers, iterations = 8, 100
		var permanentCalls, transientCalls atomic.Int64
		removePermanent := session.Subscribe(func(AgentSessionEvent) { permanentCalls.Add(1) })
		listener := AgentSessionEventListener(func(AgentSessionEvent) { transientCalls.Add(1) })
		removeShared := session.Subscribe(listener)
		start := make(chan struct{})
		var wg sync.WaitGroup
		for worker := 0; worker < workers; worker++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				<-start
				for i := 0; i < iterations; i++ {
					remove := session.Subscribe(listener)
					session.emit(agent.StartEvent{Type: "agent_start"})
					removeShared() // Concurrent calls to the very same unsubscribe.
					remove()
					remove()
				}
			}()
		}
		close(start)
		wg.Wait()
		if got := permanentCalls.Load(); got != workers*iterations {
			t.Errorf("permanent listener calls = %d, want %d", got, workers*iterations)
		}
		if transientCalls.Load() < workers*iterations {
			t.Error("an emission missed its worker's active registration")
		}
		before := transientCalls.Load()
		session.emit(agent.EndEvent{Type: "agent_end"})
		if transientCalls.Load() != before || permanentCalls.Load() != workers*iterations+1 {
			t.Error("removed registrations leaked or permanent registration was removed")
		}
		removePermanent()
		session.emit(agent.EndEvent{Type: "agent_end"})
		if permanentCalls.Load() != workers*iterations+1 {
			t.Error("permanent listener survived unsubscribe")
		}
	})
}

func TestAgentSessionEventsManualSuccessAndReconnect(t *testing.T) {
	h := newAutoCompactionHarness(t, false)
	h.responses = []ai.AssistantMessage{h.response(40, ai.StopReasonStop), h.response(40, ai.StopReasonStop)}
	log, unsubscribe := recordSessionEvents(h)
	defer unsubscribe()
	var result *CompactionResult
	compact := h.start(t, func(ctx context.Context) error {
		var err error
		result, err = h.session.Compact(ctx, nil)
		return err
	})
	if err := compact.wait(t); err != nil {
		t.Fatal(err)
	}
	observations := log.snapshot(false)
	if got, want := sessionEventTypes(observations), []string{"compaction_start", "compaction_end"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("manual events = %v, want %v", got, want)
	}
	assertSessionEventStart(t, observations[0], CompactionReasonManual)
	if emitted := assertSessionEventSuccess(t, observations[1], CompactionReasonManual, false); !reflect.DeepEqual(emitted, result) {
		t.Errorf("end result = %#v, Compact result = %#v", emitted, result)
	}
	if h.session.IsCompacting() {
		t.Error("manual compaction did not clear cancellation state")
	}
	prompt := h.start(t, func(ctx context.Context) error { return h.session.Prompt(ctx, "after reconnect", nil) })
	if err := prompt.wait(t); err != nil {
		t.Fatal(err)
	}
	want := []string{"compaction_start", "compaction_end", "agent_start", "turn_start", "message_start", "message_end", "message_start", "message_end", "turn_end", "agent_end"}
	if got := sessionEventTypes(log.snapshot(false)); !reflect.DeepEqual(got, want) {
		t.Errorf("listener lost/duplicated after reconnect: %v, want %v", got, want)
	}
	h.assertCounts(t, 1, 1)
	unsubscribe()
	unsubscribe()
	entries := len(h.session.SessionManager.GetEntries())
	prompt = h.start(t, func(ctx context.Context) error { return h.session.Prompt(ctx, "after unsubscribe", nil) })
	if err := prompt.wait(t); err != nil {
		t.Fatal(err)
	}
	if got := sessionEventTypes(log.snapshot(false)); !reflect.DeepEqual(got, want) {
		t.Errorf("unsubscribe from before reconnect no longer works: %v", got)
	}
	if got := len(h.session.SessionManager.GetEntries()); got != entries+2 {
		t.Errorf("removing session listener disrupted agent persistence: entries=%d, want %d", got, entries+2)
	}
}

func TestAgentSessionEventsManualStartAfterAbortAndIdle(t *testing.T) {
	h := newAutoCompactionHarness(t, false)
	started := make(chan struct{})
	var providerCanceled atomic.Bool
	h.session.Agent.StreamFn = func(ctx context.Context, _ ai.Model[ai.API], _ ai.ModelContext, _ *ai.SimpleStreamOptions) (*ai.AssistantMessageEventStream, error) {
		close(started)
		<-ctx.Done()
		providerCanceled.Store(true)
		return autoCompactionStream(h.response(950, ai.StopReasonAborted)), nil
	}
	log, unsubscribe := recordSessionEvents(h)
	defer unsubscribe()
	removeCheck := h.session.Subscribe(func(event AgentSessionEvent) {
		if _, ok := event.(CompactionStartEvent); ok && !providerCanceled.Load() {
			t.Error("manual start preceded active provider cancellation")
		}
	})
	defer removeCheck()
	prompt := h.start(t, func(ctx context.Context) error { return h.session.Prompt(ctx, "active", nil) })
	waitAutoCompactionSignal(t, started)
	compact := h.start(t, func(ctx context.Context) error {
		_, err := h.session.Compact(ctx, nil)
		return err
	})
	if err := compact.wait(t); err != nil {
		t.Fatal(err)
	}
	if err := prompt.wait(t); err != nil && !errors.Is(err, context.Canceled) {
		t.Fatalf("aborted Prompt: %v", err)
	}
	observations := log.snapshot(true)
	if got, want := sessionEventTypes(observations), []string{"compaction_start", "compaction_end"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("manual abort events = %v, want %v", got, want)
	}
	assertSessionEventStart(t, observations[0], CompactionReasonManual)
	assertSessionEventSuccess(t, observations[1], CompactionReasonManual, false)
}

func TestAgentSessionEventsThresholdQueuedContinuation(t *testing.T) {
	for _, behavior := range []StreamingBehavior{StreamingBehaviorSteer, StreamingBehaviorFollowUp} {
		t.Run(string(behavior), func(t *testing.T) {
			h := newAutoCompactionHarness(t, false)
			h.responses = []ai.AssistantMessage{h.response(950, ai.StopReasonStop), h.response(40, ai.StopReasonStop)}
			log, unsubscribe := recordSessionEvents(h)
			defer unsubscribe()
			removeQueue := h.session.Subscribe(func(event AgentSessionEvent) {
				if _, ok := event.(CompactionStartEvent); ok {
					// Queueing is supported; starting/waiting for an operation here is not.
					if err := h.session.Prompt(h.ctx, "queued", &PromptOptions{StreamingBehavior: behavior}); err != nil {
						t.Errorf("queue from start listener: %v", err)
					}
				}
				if _, ok := event.(CompactionEndEvent); ok && !h.session.Agent.HasQueuedMessages() {
					t.Error("queued continuation started before compaction_end")
				}
			})
			defer removeQueue()
			prompt := h.start(t, func(ctx context.Context) error { return h.session.Prompt(ctx, "first", nil) })
			if err := prompt.wait(t); err != nil {
				t.Fatal(err)
			}
			observations := log.snapshot(true)
			if got, want := sessionEventTypes(observations), []string{"compaction_start", "compaction_end"}; !reflect.DeepEqual(got, want) {
				t.Fatalf("threshold events = %v, want %v", got, want)
			}
			assertSessionEventStart(t, observations[0], CompactionReasonThreshold)
			assertSessionEventSuccess(t, observations[1], CompactionReasonThreshold, false)
			if observations[0].requests != 1 || observations[1].requests != 1 {
				t.Error("threshold notifications must follow first request and precede continuation")
			}
			h.assertCounts(t, 1, 2)
			assertAutoCompactedRequest(t, h.recordedRequests()[1], "queued")
		})
	}
}

func TestAgentSessionEventsOverflowRetry(t *testing.T) {
	for _, secondOverflow := range []bool{false, true} {
		t.Run(fmt.Sprintf("second_overflow_%v", secondOverflow), func(t *testing.T) {
			h := newAutoCompactionHarness(t, false)
			h.retainOverflowedTurn()
			overflow := h.response(0, ai.StopReasonError)
			overflow.ErrorMessage = "prompt is too long"
			retry := h.response(40, ai.StopReasonStop)
			if secondOverflow {
				retry = overflow
				retry.ErrorMessage = "context_length_exceeded"
			}
			h.responses = []ai.AssistantMessage{overflow, retry}
			log, unsubscribe := recordSessionEvents(h)
			defer unsubscribe()
			prompt := h.start(t, func(ctx context.Context) error { return h.session.Prompt(ctx, "next", nil) })
			// A second overflow is reported by compaction_end, not Prompt.
			if err := prompt.wait(t); err != nil {
				t.Fatal(err)
			}
			observations := log.snapshot(true)
			want := []string{"compaction_start", "compaction_end"}
			if secondOverflow {
				want = append(want, "compaction_end")
			}
			if got := sessionEventTypes(observations); !reflect.DeepEqual(got, want) {
				t.Fatalf("overflow events = %v, want %v", got, want)
			}
			assertSessionEventStart(t, observations[0], CompactionReasonOverflow)
			assertSessionEventSuccess(t, observations[1], CompactionReasonOverflow, true)
			if observations[0].requests != 1 || observations[1].requests != 1 {
				t.Error("overflow compaction events did not precede retry")
			}
			var lifecycle []string
			for _, observation := range log.snapshot(false) {
				switch eventType := observation.event.EventType(); eventType {
				case "agent_start", "agent_end", "compaction_start", "compaction_end":
					lifecycle = append(lifecycle, eventType)
				}
			}
			wantLifecycle := []string{"agent_start", "agent_end", "compaction_start", "compaction_end", "agent_start", "agent_end"}
			if secondOverflow {
				wantLifecycle = append(wantLifecycle, "compaction_end")
				end, ok := observations[2].event.(CompactionEndEvent)
				if !ok || end.Type != "compaction_end" || end.Reason != CompactionReasonOverflow || end.Result != nil || end.Aborted || end.WillRetry || !strings.Contains(end.ErrorMessage, "one compact-and-retry") {
					t.Errorf("second overflow end = %#v", observations[2].event)
				}
				if observations[2].requests != 2 || observations[2].summaries != 1 || observations[2].compacting || observations[2].streaming {
					t.Errorf("extra end must follow failed retry without starting compaction: %#v", observations[2])
				}
			}
			if !reflect.DeepEqual(lifecycle, wantLifecycle) {
				t.Errorf("overflow lifecycle = %v, want %v", lifecycle, wantLifecycle)
			}
			assertOverflowRecoveryCounts(t, h, 1, 1, 0, 1)
			assertAutoCompactedRequest(t, h.recordedRequests()[1], "next")
		})
	}
}

// Fail only compaction persistence, allowing the triggering prompt to complete.
type sessionEventCompactionErrorStore struct{ appendErrorSessionStore }

func (s *sessionEventCompactionErrorStore) AppendEntry(_ context.Context, _ string, entry SessionEntry) error {
	if _, ok := entry.(*CompactionEntry); ok {
		return s.appendErr
	}
	return nil
}

func TestAgentSessionEventsCompactionFailureAndCancellation(t *testing.T) {
	for _, reason := range []CompactionReason{CompactionReasonManual, CompactionReasonThreshold, CompactionReasonOverflow} {
		for _, method := range []string{"provider_error", "persistence_error", "abort_provider", "caller_cancel", "abort_start"} {
			t.Run(string(reason)+"/"+method, func(t *testing.T) {
				cancelled := method == "abort_provider" || method == "caller_cancel" || method == "abort_start"
				h := newAutoCompactionHarness(t, cancelled)
				failure := errors.New("session event fixture failure")
				if method == "provider_error" {
					// Replace this test's registered provider, retaining the harness model/API.
					registerManualCompactionProviderWithStream(t, func(ai.Model[ai.API], ai.ModelContext, *ai.SimpleStreamOptions) (*ai.AssistantMessageEventStream, error) {
						h.summaryCalls.Add(1)
						return nil, failure
					})
				}
				if method == "persistence_error" {
					h.session.SessionManager.state.persist = true
					h.session.SessionManager.state.store = &sessionEventCompactionErrorStore{appendErrorSessionStore{appendErr: failure}}
				}
				response := h.response(950, ai.StopReasonStop)
				if reason == CompactionReasonOverflow {
					response = h.response(0, ai.StopReasonError)
					response.ErrorMessage = "prompt is too long"
				}
				h.responses = []ai.AssistantMessage{response, h.response(40, ai.StopReasonStop)}
				log, unsubscribe := recordSessionEvents(h)
				defer unsubscribe()
				if method == "abort_start" {
					removeAbort := h.session.Subscribe(func(event AgentSessionEvent) {
						if _, ok := event.(CompactionStartEvent); ok {
							if !h.session.IsCompacting() {
								t.Error("start listener cannot see cancellation state")
							}
							h.session.AbortCompaction()
						}
					})
					defer removeAbort()
				}
				callerCtx, cancel := context.WithCancel(h.ctx)
				defer cancel()
				var result *CompactionResult
				operation := h.start(t, func(context.Context) error {
					if reason == CompactionReasonManual {
						var err error
						result, err = h.session.Compact(callerCtx, nil)
						return err
					}
					return h.session.Prompt(callerCtx, "trigger", nil)
				})
				if method == "abort_provider" || method == "caller_cancel" {
					waitAutoCompactionSignal(t, h.summaryStarted)
					if method == "caller_cancel" {
						cancel()
					} else {
						h.session.AbortCompaction()
					}
				}
				err := operation.wait(t)
				switch {
				case reason != CompactionReasonManual && method != "caller_cancel":
					// Automatic compaction failures and AbortCompaction are reported by
					// compaction_end; only the caller's cancellation fails Prompt.
					if err != nil {
						t.Errorf("operation error = %v, want nil", err)
					}
				case cancelled:
					if !errors.Is(err, context.Canceled) {
						t.Errorf("operation error = %v, want existing cancellation error", err)
					}
				default:
					if err == nil || !strings.Contains(err.Error(), failure.Error()) {
						t.Errorf("operation error = %v, want fixture failure", err)
					}
				}
				if cancelled && method != "caller_cancel" && callerCtx.Err() != nil {
					t.Errorf("AbortCompaction canceled caller context: %v", callerCtx.Err())
				}
				if result != nil {
					t.Errorf("failed Compact returned result %#v", result)
				}
				observations := log.snapshot(true)
				if got, want := sessionEventTypes(observations), []string{"compaction_start", "compaction_end"}; !reflect.DeepEqual(got, want) {
					t.Fatalf("failure/cancellation events = %v, want %v", got, want)
				}
				assertSessionEventStart(t, observations[0], reason)
				end, ok := observations[1].event.(CompactionEndEvent)
				if !ok || end.Type != "compaction_end" || end.Reason != reason || end.Result != nil || end.WillRetry || end.Aborted != cancelled {
					t.Errorf("failed/canceled end = %#v, reason=%s canceled=%v", observations[1].event, reason, cancelled)
				}
				if cancelled {
					if end.ErrorMessage != "" {
						t.Errorf("canceled end errorMessage = %q, want empty", end.ErrorMessage)
					}
				} else {
					prefix := map[CompactionReason]string{CompactionReasonManual: "Compaction failed:", CompactionReasonThreshold: "Auto-compaction failed:", CompactionReasonOverflow: "Context overflow recovery failed:"}[reason]
					if !strings.HasPrefix(end.ErrorMessage, prefix) || !strings.Contains(end.ErrorMessage, failure.Error()) {
						t.Errorf("failure errorMessage = %q, want %q and fixture error", end.ErrorMessage, prefix)
					}
				}
				if observations[1].latest != nil || GetLatestCompactionEntry(h.session.SessionManager.GetEntries()) != nil {
					t.Error("failed/canceled compaction was persisted")
				}
				if !reflect.DeepEqual(observations[0].live, observations[1].live) {
					t.Error("failed/canceled compaction replaced live context")
				}
				wantRequests := 1
				if reason == CompactionReasonManual {
					wantRequests = 0
				}
				if observations[0].requests != wantRequests || observations[1].requests != wantRequests || len(h.recordedRequests()) != wantRequests {
					t.Error("failed/canceled compaction retried the agent")
				}
				if h.session.IsCompacting() || h.session.IsStreaming() {
					t.Error("operation left compaction/agent active")
				}
				if reason == CompactionReasonManual {
					// Failed and canceled Compact must reconnect forwarding too.
					h.responses = []ai.AssistantMessage{h.response(40, ai.StopReasonStop)}
					before := len(log.snapshot(false))
					prompt := h.start(t, func(ctx context.Context) error { return h.session.Prompt(ctx, "after failed compact", nil) })
					if err := prompt.wait(t); err != nil {
						t.Fatal(err)
					}
					got := sessionEventTypes(log.snapshot(false)[before:])
					want := []string{"agent_start", "turn_start", "message_start", "message_end", "message_start", "message_end", "turn_end", "agent_end"}
					if !reflect.DeepEqual(got, want) {
						t.Errorf("forwarding after failed/canceled Compact = %v, want %v", got, want)
					}
				}
			})
		}
	}
}
