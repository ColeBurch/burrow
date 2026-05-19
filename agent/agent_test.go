package agent

import (
	"context"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ColeBurch/burrow/ai"
)

func agentDoneStream(text string) *ai.AssistantMessageEventStream {
	s := ai.NewAssistantMessageEventStream(8)
	go s.Push(ai.DoneEvent{Type: "done", Reason: ai.StopReasonStop, Message: textAssistant(text)})
	return s
}

func neverEndingUntilAbortStream(ctx context.Context) *ai.AssistantMessageEventStream {
	s := ai.NewAssistantMessageEventStream(8)
	go func() {
		s.Push(ai.StartEvent{Type: "start", Partial: textAssistant("")})
		<-ctx.Done()
		msg := textAssistant("Aborted")
		msg.StopReason = ai.StopReasonAborted
		msg.ErrorMessage = "aborted"
		s.Push(ai.ErrorEvent{Type: "error", Reason: ai.StopReasonAborted, Message: msg})
	}()
	return s
}

func TestAgentCreatesDefaultState(t *testing.T) {
	a := NewAgent(nil)
	s := a.State()
	if s.SystemPrompt != "" || s.Model.ID == "" || s.ThinkingLevel != ai.ThinkingLevelOff || len(s.Tools) != 0 || len(s.Messages) != 0 || s.IsStreaming || s.StreamingMessage != nil || len(s.PendingToolCalls) != 0 || s.ErrorMessage != "" {
		t.Fatalf("unexpected default state: %#v", s)
	}
}

func TestAgentCreatesCustomInitialState(t *testing.T) {
	prompt := "You are a helpful assistant."
	model := testModel()
	level := ai.ThinkingLevelLow
	a := NewAgent(&AgentOptions{InitialState: &InitialAgentState{SystemPrompt: &prompt, Model: &model, ThinkingLevel: &level}})
	s := a.State()
	if s.SystemPrompt != prompt || !reflect.DeepEqual(s.Model, model) || s.ThinkingLevel != level {
		t.Fatalf("unexpected custom state: %#v", s)
	}
}

func TestAgentSubscribeUnsubscribeAndNoStateEvents(t *testing.T) {
	a := NewAgent(nil)
	count := 0
	unsubscribe := a.Subscribe(func(AgentEvent, context.Context) error { count++; return nil })
	if count != 0 {
		t.Fatalf("subscribe emitted event")
	}
	a.state.SystemPrompt = "Test prompt"
	if count != 0 || a.state.SystemPrompt != "Test prompt" {
		t.Fatalf("unexpected count/state")
	}
	unsubscribe()
	a.state.SystemPrompt = "Another prompt"
	if count != 0 {
		t.Fatalf("unsubscribed listener called")
	}
}

func TestAgentAwaitsSubscribersBeforePromptResolves(t *testing.T) {
	barrier := make(chan struct{})
	a := NewAgent(&AgentOptions{StreamFn: func(context.Context, ai.Model[ai.API], ai.ModelContext, *ai.SimpleStreamOptions) (*ai.AssistantMessageEventStream, error) {
		return agentDoneStream("ok"), nil
	}})
	var listenerFinished atomic.Bool
	a.Subscribe(func(e AgentEvent, ctx context.Context) error {
		if e.EventType() == "agent_end" {
			<-barrier
			listenerFinished.Store(true)
		}
		return nil
	})
	done := make(chan error, 1)
	go func() { done <- a.PromptText(context.Background(), "hello") }()
	time.Sleep(10 * time.Millisecond)
	select {
	case <-done:
		t.Fatal("prompt resolved before subscriber")
	default:
	}
	if listenerFinished.Load() || !a.State().IsStreaming {
		t.Fatalf("unexpected state while blocked")
	}
	close(barrier)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if !listenerFinished.Load() || a.State().IsStreaming {
		t.Fatalf("unexpected final state")
	}
}

func TestAgentWaitForIdleWaitsForSubscribers(t *testing.T) {
	barrier := make(chan struct{})
	started := make(chan struct{})
	a := NewAgent(&AgentOptions{StreamFn: func(context.Context, ai.Model[ai.API], ai.ModelContext, *ai.SimpleStreamOptions) (*ai.AssistantMessageEventStream, error) {
		close(started)
		return agentDoneStream("ok"), nil
	}})
	a.Subscribe(func(e AgentEvent, ctx context.Context) error {
		if ev, ok := e.(MessageEndEvent); ok && ev.Message.AgentMessageType() == string(ai.RoleAssistant) {
			<-barrier
		}
		return nil
	})
	promptDone := make(chan error, 1)
	go func() { promptDone <- a.PromptText(context.Background(), "hello") }()
	// WaitForIdle waits for the currently active run; it does not wait for work
	// that has merely been scheduled on another goroutine. Synchronize on the
	// stream function so activeRun is installed before asserting idle behavior.
	select {
	case <-started:
	case err := <-promptDone:
		t.Fatalf("prompt finished before stream started: %v", err)
	case <-time.After(time.Second):
		t.Fatal("prompt did not start")
	}
	idle := make(chan struct{})
	go func() { a.WaitForIdle(); close(idle) }()
	time.Sleep(10 * time.Millisecond)
	select {
	case <-idle:
		t.Fatal("idle resolved before subscriber")
	default:
	}
	if !a.State().IsStreaming {
		t.Fatalf("expected streaming")
	}
	close(barrier)
	<-idle
	if err := <-promptDone; err != nil {
		t.Fatal(err)
	}
	if a.State().IsStreaming {
		t.Fatalf("expected idle")
	}
}

func TestAgentPassesActiveAbortSignalToSubscribers(t *testing.T) {
	var received context.Context
	a := NewAgent(&AgentOptions{StreamFn: func(ctx context.Context, _ ai.Model[ai.API], _ ai.ModelContext, _ *ai.SimpleStreamOptions) (*ai.AssistantMessageEventStream, error) {
		return neverEndingUntilAbortStream(ctx), nil
	}})
	a.Subscribe(func(e AgentEvent, ctx context.Context) error {
		if e.EventType() == "agent_start" {
			received = ctx
		}
		return nil
	})
	done := make(chan error, 1)
	go func() { done <- a.PromptText(context.Background(), "hello") }()
	time.Sleep(10 * time.Millisecond)
	if received == nil || received.Err() != nil {
		t.Fatalf("signal not active")
	}
	a.Abort()
	<-done
	if received.Err() == nil {
		t.Fatalf("signal was not aborted")
	}
}

func TestAgentQueuesAndAbort(t *testing.T) {
	a := NewAgent(nil)
	m := userMsg("queued")
	a.Steer(m)
	a.FollowUp(m)
	a.Abort()
	if len(a.State().Messages) != 0 || !a.HasQueuedMessages() {
		t.Fatalf("queues affected state unexpectedly")
	}
}

func TestAgentRejectsPromptAndContinueWhileStreaming(t *testing.T) {
	a := NewAgent(&AgentOptions{StreamFn: func(ctx context.Context, _ ai.Model[ai.API], _ ai.ModelContext, _ *ai.SimpleStreamOptions) (*ai.AssistantMessageEventStream, error) {
		return neverEndingUntilAbortStream(ctx), nil
	}})
	done := make(chan error, 1)
	go func() { done <- a.PromptText(context.Background(), "first") }()
	time.Sleep(10 * time.Millisecond)
	if err := a.PromptText(context.Background(), "second"); err == nil || !strings.Contains(err.Error(), "already processing") {
		t.Fatalf("expected prompt rejection, got %v", err)
	}
	if err := a.Continue(context.Background()); err == nil || !strings.Contains(err.Error(), "already processing") {
		t.Fatalf("expected continue rejection, got %v", err)
	}
	a.Abort()
	<-done
}

func TestAgentContinueProcessesQueuedMessagesFromAssistantTail(t *testing.T) {
	a := NewAgent(&AgentOptions{StreamFn: func(context.Context, ai.Model[ai.API], ai.ModelContext, *ai.SimpleStreamOptions) (*ai.AssistantMessageEventStream, error) {
		return agentDoneStream("Processed"), nil
	}})
	a.state.Messages = []AgentMessage{userMsg("Initial"), textAssistant("Initial response")}
	a.FollowUp(userMsg("Queued follow-up"))
	if err := a.Continue(context.Background()); err != nil {
		t.Fatal(err)
	}
	msgs := a.State().Messages
	if msgs[len(msgs)-1].AgentMessageType() != string(ai.RoleAssistant) {
		t.Fatalf("expected assistant tail")
	}
	found := false
	for _, m := range msgs {
		if u, ok := m.(ai.UserMessage); ok && userText(u) == "Queued follow-up" {
			found = true
		}
	}
	if !found {
		t.Fatalf("queued follow-up not appended: %#v", msgs)
	}
}

func TestAgentContinueOneAtATimeSteeringFromAssistantTail(t *testing.T) {
	var n int
	a := NewAgent(&AgentOptions{StreamFn: func(context.Context, ai.Model[ai.API], ai.ModelContext, *ai.SimpleStreamOptions) (*ai.AssistantMessageEventStream, error) {
		n++
		return agentDoneStream("Processed"), nil
	}})
	a.state.Messages = []AgentMessage{userMsg("Initial"), textAssistant("Initial response")}
	a.Steer(userMsg("Steering 1"))
	a.Steer(userMsg("Steering 2"))
	if err := a.Continue(context.Background()); err != nil {
		t.Fatal(err)
	}
	msgs := a.State().Messages
	recent := msgs[len(msgs)-4:]
	roles := []string{recent[0].AgentMessageType(), recent[1].AgentMessageType(), recent[2].AgentMessageType(), recent[3].AgentMessageType()}
	if !reflect.DeepEqual(roles, []string{"user", "assistant", "user", "assistant"}) || n != 2 {
		t.Fatalf("roles=%v n=%d", roles, n)
	}
}

func TestAgentForwardsSessionIDToStreamFnOptions(t *testing.T) {
	sid := "session-abc"
	var received *string
	a := NewAgent(&AgentOptions{SessionID: &sid, StreamFn: func(_ context.Context, _ ai.Model[ai.API], _ ai.ModelContext, opts *ai.SimpleStreamOptions) (*ai.AssistantMessageEventStream, error) {
		received = opts.SessionId
		return agentDoneStream("ok"), nil
	}})
	if err := a.PromptText(context.Background(), "hello"); err != nil {
		t.Fatal(err)
	}
	if received == nil || *received != "session-abc" {
		t.Fatalf("session id not forwarded")
	}
	sid2 := "session-def"
	a.SessionID = &sid2
	if err := a.PromptText(context.Background(), "hello again"); err != nil {
		t.Fatal(err)
	}
	if received == nil || *received != "session-def" {
		t.Fatalf("updated session id not forwarded")
	}
}
