package agent

import (
	"context"
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/ColeBurch/burrow/internal/ai"
)

func textOfMessage(m AgentMessage) string {
	parts := []string{}
	switch msg := m.(type) {
	case ai.AssistantMessage:
		for _, b := range msg.Content {
			if t, ok := b.(ai.TextContent); ok {
				parts = append(parts, t.Text)
			}
		}
	case ai.ToolResultMessage:
		for _, b := range msg.Content {
			if t, ok := b.(ai.TextContent); ok {
				parts = append(parts, t.Text)
			}
		}
	}
	return strings.Join(parts, "\n")
}

func streamFromMessages(messages ...ai.AssistantMessage) StreamFn {
	i := 0
	return func(context.Context, ai.Model[ai.API], ai.ModelContext, *ai.SimpleStreamOptions) (*ai.AssistantMessageEventStream, error) {
		if i >= len(messages) {
			return doneStream(textAssistant("")), nil
		}
		msg := messages[i]
		i++
		return doneStream(msg), nil
	}
}

func streamingTextStream(text string, delay time.Duration) StreamFn {
	return func(ctx context.Context, model ai.Model[ai.API], _ ai.ModelContext, _ *ai.SimpleStreamOptions) (*ai.AssistantMessageEventStream, error) {
		s := ai.NewAssistantMessageEventStream(32)
		go func() {
			partial := ai.AssistantMessage{Role: ai.RoleAssistant, Api: model.API, Provider: model.Provider, Model: model.ID, Content: []ai.AssistantContent{ai.TextContent{Type: ai.ContentTypeText, Text: ""}}, Timestamp: nowMs()}
			s.Push(ai.StartEvent{Type: "start", Partial: partial})
			acc := ""
			for _, tok := range strings.Split(text, " ") {
				select {
				case <-ctx.Done():
					partial.StopReason = ai.StopReasonAborted
					partial.ErrorMessage = ctx.Err().Error()
					s.Push(ai.ErrorEvent{Type: "error", Reason: ai.StopReasonAborted, Message: partial})
					return
				case <-time.After(delay):
				}
				if acc != "" {
					acc += " "
				}
				acc += tok
				partial.Content = []ai.AssistantContent{ai.TextContent{Type: ai.ContentTypeText, Text: acc}}
				s.Push(ai.TextDeltaEvent{Type: "text_delta", ContentIndex: 0, Delta: tok, Partial: partial})
			}
			partial.StopReason = ai.StopReasonStop
			s.Push(ai.DoneEvent{Type: "done", Reason: ai.StopReasonStop, Message: partial})
		}()
		return s, nil
	}
}

func calculateToolE2E() AgentTool[any, any] {
	return AgentTool[any, any]{Tool: ai.Tool{Name: "calculate", Description: "Calculate", Parameters: map[string]any{"type": "object"}}, Execute: func(_ context.Context, _ string, params any, _ AgentToolUpdateCallback[any]) (AgentToolResult[any], error) {
		expr := params.(map[string]any)["expression"].(string)
		if expr != "123 * 456" && expr != "5 + 3" {
			return AgentToolResult[any]{}, fmt.Errorf("unexpected expression %s", expr)
		}
		res := map[string]string{"123 * 456": "56088", "5 + 3": "8"}[expr]
		return AgentToolResult[any]{Content: []ai.ToolResultContent{ai.TextContent{Type: ai.ContentTypeText, Text: expr + " = " + res}}}, nil
	}}
}

func TestAgentE2EBasicPrompt(t *testing.T) {
	prompt := "You are a helpful assistant. Keep your responses concise."
	model := testModel()
	a := NewAgent(&AgentOptions{InitialState: &InitialAgentState{SystemPrompt: &prompt, Model: &model}, StreamFn: streamFromMessages(textAssistant("4"))})
	if err := a.PromptText(context.Background(), "What is 2+2? Answer with just the number."); err != nil {
		t.Fatal(err)
	}
	s := a.State()
	if s.IsStreaming || len(s.Messages) != 2 || s.Messages[0].AgentMessageType() != "user" || s.Messages[1].AgentMessageType() != "assistant" {
		t.Fatalf("unexpected state %#v", s)
	}
	if !strings.Contains(textOfMessage(s.Messages[1]), "4") {
		t.Fatalf("assistant text = %q", textOfMessage(s.Messages[1]))
	}
}

func TestAgentE2EToolExecutionAndPendingCalls(t *testing.T) {
	model := testModel()
	prompt := "You are a helpful assistant. Always use the calculator tool for math."
	first := assistantMsg([]ai.AssistantContent{ai.TextContent{Type: ai.ContentTypeText, Text: "Let me calculate that."}, ai.ToolCall{Type: ai.ContentTypeToolCall, Id: "calc-1", Name: "calculate", Args: map[string]any{"expression": "123 * 456"}}}, ai.StopReasonToolUse)
	a := NewAgent(&AgentOptions{InitialState: &InitialAgentState{SystemPrompt: &prompt, Model: &model, Tools: []AgentTool[any, any]{calculateToolE2E()}}, StreamFn: streamFromMessages(first, textAssistant("The result is 56088."))})
	seen := []struct {
		typ string
		ids []string
	}{}
	a.Subscribe(func(e AgentEvent, _ context.Context) error {
		if e.EventType() == "tool_execution_start" || e.EventType() == "tool_execution_end" {
			ids := []string{}
			for id := range a.State().PendingToolCalls {
				ids = append(ids, id)
			}
			seen = append(seen, struct {
				typ string
				ids []string
			}{e.EventType(), ids})
		}
		return nil
	})
	if err := a.PromptText(context.Background(), "Calculate 123 * 456 using the calculator tool."); err != nil {
		t.Fatal(err)
	}
	s := a.State()
	var toolResult AgentMessage
	for _, m := range s.Messages {
		if m.AgentMessageType() == "toolResult" {
			toolResult = m
		}
	}
	if toolResult == nil || !strings.Contains(textOfMessage(toolResult), "123 * 456 = 56088") {
		t.Fatalf("missing tool result: %#v", s.Messages)
	}
	if !strings.Contains(textOfMessage(s.Messages[len(s.Messages)-1]), "56088") || len(s.PendingToolCalls) != 0 {
		t.Fatalf("unexpected final state %#v", s)
	}
	if !reflect.DeepEqual(seen, []struct {
		typ string
		ids []string
	}{{"tool_execution_start", []string{"calc-1"}}, {"tool_execution_end", []string{}}}) {
		t.Fatalf("pending snapshots %#v", seen)
	}
}

func TestAgentE2EAbortDuringStreaming(t *testing.T) {
	model := testModel()
	a := NewAgent(&AgentOptions{InitialState: &InitialAgentState{Model: &model}, StreamFn: streamingTextStream("one two three four five six seven eight nine ten", 20*time.Millisecond)})
	done := make(chan error, 1)
	go func() { done <- a.PromptText(context.Background(), "Count slowly from 1 to 20.") }()
	time.Sleep(30 * time.Millisecond)
	a.Abort()
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	s := a.State()
	last := s.Messages[len(s.Messages)-1].(ai.AssistantMessage)
	if s.IsStreaming || last.StopReason != ai.StopReasonAborted || last.ErrorMessage == "" || s.ErrorMessage != last.ErrorMessage {
		t.Fatalf("unexpected abort state %#v last %#v", s, last)
	}
}

func TestAgentE2EStateUpdates(t *testing.T) {
	model := testModel()
	a := NewAgent(&AgentOptions{InitialState: &InitialAgentState{Model: &model}, StreamFn: streamingTextStream("1 2 3 4 5", 0)})
	events := []string{}
	a.Subscribe(func(e AgentEvent, _ context.Context) error { events = append(events, e.EventType()); return nil })
	if err := a.PromptText(context.Background(), "Count from 1 to 5."); err != nil {
		t.Fatal(err)
	}
	for _, typ := range []string{"agent_start", "turn_start", "message_start", "message_update", "message_end", "turn_end", "agent_end"} {
		if !containsString(events, typ) {
			t.Fatalf("missing %s in %v", typ, events)
		}
	}
	if indexOf(events, "agent_start") >= indexOf(events, "message_start") || indexOf(events, "message_start") >= indexOf(events, "message_end") || indexOf(events, "message_end") >= lastIndexOf(events, "agent_end") {
		t.Fatalf("bad event order %v", events)
	}
	if s := a.State(); s.IsStreaming || len(s.Messages) != 2 {
		t.Fatalf("unexpected state %#v", s)
	}
}

func TestAgentE2EMultiTurnAndThinkingContent(t *testing.T) {
	model := testModel()
	model.Reasoning = true
	prompt := "You are a helpful assistant."
	streamFn := func(ctx context.Context, m ai.Model[ai.API], mc ai.ModelContext, o *ai.SimpleStreamOptions) (*ai.AssistantMessageEventStream, error) {
		if len(mc.Messages) == 1 {
			return doneStream(textAssistant("Nice to meet you, Alice.")), nil
		}
		hasAlice := false
		for _, msg := range mc.Messages {
			if u, ok := msg.(ai.UserMessage); ok && strings.Contains(userText(u), "Alice") {
				hasAlice = true
			}
		}
		if hasAlice {
			return doneStream(textAssistant("Your name is Alice.")), nil
		}
		return doneStream(textAssistant("I do not know your name.")), nil
	}
	a := NewAgent(&AgentOptions{InitialState: &InitialAgentState{SystemPrompt: &prompt, Model: &model}, StreamFn: streamFn})
	if err := a.PromptText(context.Background(), "My name is Alice."); err != nil {
		t.Fatal(err)
	}
	if err := a.PromptText(context.Background(), "What is my name?"); err != nil {
		t.Fatal(err)
	}
	if s := a.State(); len(s.Messages) != 4 || !strings.Contains(strings.ToLower(textOfMessage(s.Messages[3])), "alice") {
		t.Fatalf("unexpected conversation %#v", s.Messages)
	}

	level := ai.ThinkingLevelLow
	thinking := assistantMsg([]ai.AssistantContent{ai.ThinkingContent{Type: ai.ContentTypeThinking, Thinking: "step by step"}, ai.TextContent{Type: ai.ContentTypeText, Text: "4"}}, ai.StopReasonStop)
	b := NewAgent(&AgentOptions{InitialState: &InitialAgentState{Model: &model, ThinkingLevel: &level}, StreamFn: streamFromMessages(thinking)})
	if err := b.PromptText(context.Background(), "What is 2+2?"); err != nil {
		t.Fatal(err)
	}
	got := b.State().Messages[1].(ai.AssistantMessage).Content
	if !reflect.DeepEqual(got, thinking.Content) {
		t.Fatalf("thinking content = %#v", got)
	}
}

func TestAgentE2EContinueValidationAndFromMessages(t *testing.T) {
	model := testModel()
	a := NewAgent(&AgentOptions{InitialState: &InitialAgentState{Model: &model}})
	if err := a.Continue(context.Background()); err == nil || !strings.Contains(err.Error(), "no messages to continue from") {
		t.Fatalf("expected no messages error, got %v", err)
	}
	a.state.Messages = []AgentMessage{textAssistant("Hello")}
	if err := a.Continue(context.Background()); err == nil || !strings.Contains(err.Error(), "cannot continue from message role: assistant") {
		t.Fatalf("expected assistant error, got %v", err)
	}

	b := NewAgent(&AgentOptions{InitialState: &InitialAgentState{Model: &model, Messages: []AgentMessage{userMsg("Say exactly: HELLO WORLD")}}, StreamFn: streamFromMessages(textAssistant("HELLO WORLD"))})
	if err := b.Continue(context.Background()); err != nil {
		t.Fatal(err)
	}
	if s := b.State(); s.IsStreaming || len(s.Messages) != 2 || !strings.Contains(strings.ToUpper(textOfMessage(s.Messages[1])), "HELLO WORLD") {
		t.Fatalf("unexpected user continue %#v", s)
	}

	toolResult := ai.ToolResultMessage{Role: ai.RoleToolResult, ToolCallID: "calc-1", ToolName: "calculate", Content: []ai.ToolResultContent{ai.TextContent{Type: ai.ContentTypeText, Text: "5 + 3 = 8"}}, Timestamp: nowMs()}
	call := assistantMsg([]ai.AssistantContent{ai.ToolCall{Type: ai.ContentTypeToolCall, Id: "calc-1", Name: "calculate", Args: map[string]any{"expression": "5 + 3"}}}, ai.StopReasonToolUse)
	c := NewAgent(&AgentOptions{InitialState: &InitialAgentState{Model: &model, Tools: []AgentTool[any, any]{calculateToolE2E()}, Messages: []AgentMessage{userMsg("What is 5 + 3?"), call, toolResult}}, StreamFn: streamFromMessages(textAssistant("The answer is 8."))})
	if err := c.Continue(context.Background()); err != nil {
		t.Fatal(err)
	}
	if s := c.State(); s.IsStreaming || len(s.Messages) < 4 || !strings.Contains(textOfMessage(s.Messages[len(s.Messages)-1]), "8") {
		t.Fatalf("unexpected tool continue %#v", s)
	}
}

func containsString(xs []string, x string) bool { return indexOf(xs, x) >= 0 }
func indexOf(xs []string, x string) int {
	for i, v := range xs {
		if v == x {
			return i
		}
	}
	return -1
}
func lastIndexOf(xs []string, x string) int {
	for i := len(xs) - 1; i >= 0; i-- {
		if xs[i] == x {
			return i
		}
	}
	return -1
}
