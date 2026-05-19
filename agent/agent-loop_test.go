package agent

import (
	"context"
	"reflect"
	"testing"
	"time"

	"github.com/ColeBurch/burrow/ai"
)

type customMessage struct {
	role, text string
	timestamp  int64
}

func (m customMessage) AgentMessageType() string { return m.role }

func testModel() ai.Model[ai.API] {
	return ai.Model[ai.API]{ID: "mock", Name: "mock", API: ai.APIOpenAIResponses, Provider: ai.ProviderOpenAI, BaseURL: "https://example.invalid", Input: []string{"text"}, ContextWindow: 8192, MaxTokens: 2048}
}
func nowMs() int64 { return time.Now().UnixMilli() }
func userMsg(text string) ai.UserMessage {
	return ai.UserMessage{Role: ai.RoleUser, Content: []ai.UserContent{ai.TextContent{Type: ai.ContentTypeText, Text: text}}, Timestamp: nowMs()}
}
func userText(m ai.UserMessage) string {
	if len(m.Content) == 0 {
		return ""
	}
	if t, ok := m.Content[0].(ai.TextContent); ok {
		return t.Text
	}
	return ""
}
func assistantMsg(content []ai.AssistantContent, stop ai.StopReason) ai.AssistantMessage {
	if stop == "" {
		stop = ai.StopReasonStop
	}
	return ai.AssistantMessage{Role: ai.RoleAssistant, Content: content, Api: ai.APIOpenAIResponses, Provider: ai.ProviderOpenAI, Model: "mock", StopReason: stop, Timestamp: nowMs()}
}
func textAssistant(text string) ai.AssistantMessage {
	return assistantMsg([]ai.AssistantContent{ai.TextContent{Type: ai.ContentTypeText, Text: text}}, ai.StopReasonStop)
}
func identity(messages []AgentMessage) []ai.Message {
	out := []ai.Message{}
	for _, m := range messages {
		if msg, ok := m.(ai.Message); ok {
			out = append(out, msg)
		}
	}
	return out
}
func baseConfig() AgentLoopConfig {
	return AgentLoopConfig{Model: testModel(), ConvertToLLM: identity, GetSteeringMessages: func(context.Context) []AgentMessage { return nil }, GetFollowUpMessages: func(context.Context) []AgentMessage { return nil }}
}
func doneStream(msg ai.AssistantMessage) *ai.AssistantMessageEventStream {
	s := ai.NewAssistantMessageEventStream(8)
	go s.Push(ai.DoneEvent{Type: "done", Reason: msg.StopReason, Message: msg})
	return s
}
func streamSeq(messages ...ai.AssistantMessage) StreamFn {
	i := 0
	return func(context.Context, ai.Model[ai.API], ai.ModelContext, *ai.SimpleStreamOptions) (*ai.AssistantMessageEventStream, error) {
		msg := messages[i]
		i++
		return doneStream(msg), nil
	}
}
func collectEvents(prompts []AgentMessage, ctx AgentContext, cfg AgentLoopConfig, sf StreamFn) ([]AgentMessage, []AgentEvent, error) {
	events := []AgentEvent{}
	msgs, err := RunAgentLoop(prompts, ctx, cfg, context.Background(), func(e AgentEvent) error { events = append(events, e); return nil }, &sf)
	return msgs, events, err
}
func eventTypes(events []AgentEvent) []string {
	r := []string{}
	for _, e := range events {
		r = append(r, e.EventType())
	}
	return r
}
func hasType(events []AgentEvent, typ string) bool {
	for _, e := range events {
		if e.EventType() == typ {
			return true
		}
	}
	return false
}
func boolPtr(b bool) *bool { return &b }

func TestAgentLoopEmitsEventsWithAgentMessageTypes(t *testing.T) {
	ctx := AgentContext{SystemPrompt: "You are helpful."}
	msgs, events, err := collectEvents([]AgentMessage{userMsg("Hello")}, ctx, baseConfig(), streamSeq(textAssistant("Hi there!")))
	if err != nil {
		t.Fatal(err)
	}
	if len(msgs) != 2 || msgs[0].AgentMessageType() != "user" || msgs[1].AgentMessageType() != "assistant" {
		t.Fatalf("unexpected messages: %#v", msgs)
	}
	for _, typ := range []string{"agent_start", "turn_start", "message_start", "message_end", "turn_end", "agent_end"} {
		if !hasType(events, typ) {
			t.Fatalf("missing event %s in %#v", typ, eventTypes(events))
		}
	}
}

func TestAgentLoopHandlesCustomMessagesViaConvertToLLM(t *testing.T) {
	converted := []ai.Message{}
	cfg := baseConfig()
	cfg.ConvertToLLM = func(messages []AgentMessage) []ai.Message {
		for _, m := range messages {
			if m.AgentMessageType() == "notification" {
				continue
			}
			if msg, ok := m.(ai.Message); ok {
				converted = append(converted, msg)
			}
		}
		return converted
	}
	_, _, err := collectEvents([]AgentMessage{userMsg("Hello")}, AgentContext{Messages: []AgentMessage{customMessage{role: "notification", text: "note", timestamp: nowMs()}}}, cfg, streamSeq(textAssistant("Response")))
	if err != nil {
		t.Fatal(err)
	}
	if len(converted) != 1 || converted[0].(ai.UserMessage).Role != ai.RoleUser {
		t.Fatalf("unexpected converted: %#v", converted)
	}
}

func TestAgentLoopAppliesTransformContextBeforeConvertToLLM(t *testing.T) {
	cfg := baseConfig()
	var transformed []AgentMessage
	var converted []ai.Message
	cfg.TransformContext = func(_ context.Context, messages []AgentMessage) []AgentMessage {
		transformed = messages[len(messages)-2:]
		return transformed
	}
	cfg.ConvertToLLM = func(messages []AgentMessage) []ai.Message { converted = identity(messages); return converted }
	ctx := AgentContext{Messages: []AgentMessage{userMsg("old1"), textAssistant("r1"), userMsg("old2"), textAssistant("r2")}}
	_, _, err := collectEvents([]AgentMessage{userMsg("new")}, ctx, cfg, streamSeq(textAssistant("Response")))
	if err != nil {
		t.Fatal(err)
	}
	if len(transformed) != 2 || len(converted) != 2 {
		t.Fatalf("transform/convert lengths = %d/%d", len(transformed), len(converted))
	}
}

func echoTool(executed *[]string) AgentTool[any, any] {
	return AgentTool[any, any]{Tool: ai.Tool{Name: "echo", Description: "Echo", Parameters: map[string]any{"type": "object", "properties": map[string]any{"value": map[string]any{"type": "string"}}, "required": []any{"value"}}}, Label: "Echo", Execute: func(_ context.Context, _ string, params any, _ AgentToolUpdateCallback[any]) (AgentToolResult[any], error) {
		v := params.(map[string]any)["value"].(string)
		if executed != nil {
			*executed = append(*executed, v)
		}
		return AgentToolResult[any]{Content: []ai.ToolResultContent{ai.TextContent{Type: ai.ContentTypeText, Text: "echoed: " + v}}, Details: map[string]any{"value": v}}, nil
	}}
}

func TestAgentLoopHandlesToolCallsAndResults(t *testing.T) {
	executed := []string{}
	ctx := AgentContext{Tools: []AgentTool[any, any]{echoTool(&executed)}}
	first := assistantMsg([]ai.AssistantContent{ai.ToolCall{Type: ai.ContentTypeToolCall, Id: "tool-1", Name: "echo", Args: map[string]any{"value": "hello"}}}, ai.StopReasonToolUse)
	_, events, err := collectEvents([]AgentMessage{userMsg("echo")}, ctx, baseConfig(), streamSeq(first, textAssistant("done")))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(executed, []string{"hello"}) {
		t.Fatalf("executed %#v", executed)
	}
	if !hasType(events, "tool_execution_start") || !hasType(events, "tool_execution_end") {
		t.Fatalf("missing tool events")
	}
}

func TestAgentLoopBeforeToolCallMutatedArgsAreExecutedWithoutRevalidation(t *testing.T) {
	executed := []any{}
	tool := echoTool(nil)
	tool.Execute = func(_ context.Context, _ string, params any, _ AgentToolUpdateCallback[any]) (AgentToolResult[any], error) {
		executed = append(executed, params.(map[string]any)["value"])
		return AgentToolResult[any]{Content: []ai.ToolResultContent{ai.TextContent{Type: ai.ContentTypeText, Text: "ok"}}}, nil
	}
	cfg := baseConfig()
	cfg.BeforeToolCall = func(_ context.Context, c BeforeToolCallContext) (*BeforeToolCallResult, error) {
		c.args.(map[string]any)["value"] = 123
		return nil, nil
	}
	first := assistantMsg([]ai.AssistantContent{ai.ToolCall{Type: ai.ContentTypeToolCall, Id: "tool-1", Name: "echo", Args: map[string]any{"value": "hello"}}}, ai.StopReasonToolUse)
	_, _, err := collectEvents([]AgentMessage{userMsg("echo")}, AgentContext{Tools: []AgentTool[any, any]{tool}}, cfg, streamSeq(first, textAssistant("done")))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(executed, []any{123}) {
		t.Fatalf("executed %#v", executed)
	}
}

func TestAgentLoopPrepareToolArgumentsForValidation(t *testing.T) {
	executed := [][]map[string]string{}
	tool := AgentTool[any, any]{Tool: ai.Tool{Name: "edit", Parameters: map[string]any{"type": "object", "properties": map[string]any{"edits": map[string]any{"type": "array", "items": map[string]any{"type": "object", "properties": map[string]any{"oldText": map[string]any{"type": "string"}, "newText": map[string]any{"type": "string"}}, "required": []any{"oldText", "newText"}}}}, "required": []any{"edits"}}}, PrepareArguments: func(args any) (any, error) {
		m := args.(map[string]any)
		return map[string]any{"edits": []any{map[string]any{"oldText": m["oldText"], "newText": m["newText"]}}}, nil
	}, Execute: func(_ context.Context, _ string, params any, _ AgentToolUpdateCallback[any]) (AgentToolResult[any], error) {
		arr := params.(map[string]any)["edits"].([]any)
		e := arr[0].(map[string]any)
		executed = append(executed, []map[string]string{{"oldText": e["oldText"].(string), "newText": e["newText"].(string)}})
		return AgentToolResult[any]{Content: []ai.ToolResultContent{ai.TextContent{Type: ai.ContentTypeText, Text: "ok"}}}, nil
	}}
	first := assistantMsg([]ai.AssistantContent{ai.ToolCall{Type: ai.ContentTypeToolCall, Id: "tool-1", Name: "edit", Args: map[string]any{"oldText": "before", "newText": "after"}}}, ai.StopReasonToolUse)
	_, _, err := collectEvents([]AgentMessage{userMsg("edit")}, AgentContext{Tools: []AgentTool[any, any]{tool}}, baseConfig(), streamSeq(first, textAssistant("done")))
	if err != nil {
		t.Fatal(err)
	}
	want := [][]map[string]string{{{"oldText": "before", "newText": "after"}}}
	if !reflect.DeepEqual(executed, want) {
		t.Fatalf("%#v", executed)
	}
}

func TestAgentLoopParallelCompletionOrderButSourceOrderResults(t *testing.T) {
	firstResolved, parallelObserved := false, false
	release := make(chan struct{})
	tool := echoTool(nil)
	tool.Execute = func(_ context.Context, _ string, params any, _ AgentToolUpdateCallback[any]) (AgentToolResult[any], error) {
		v := params.(map[string]any)["value"].(string)
		if v == "first" {
			<-release
			firstResolved = true
		}
		if v == "second" && !firstResolved {
			parallelObserved = true
		}
		return AgentToolResult[any]{Content: []ai.ToolResultContent{ai.TextContent{Type: ai.ContentTypeText, Text: v}}}, nil
	}
	cfg := baseConfig()
	mode := ToolExecutionModeParallel
	cfg.ToolExecution = &mode
	first := assistantMsg([]ai.AssistantContent{ai.ToolCall{Type: ai.ContentTypeToolCall, Id: "tool-1", Name: "echo", Args: map[string]any{"value": "first"}}, ai.ToolCall{Type: ai.ContentTypeToolCall, Id: "tool-2", Name: "echo", Args: map[string]any{"value": "second"}}}, ai.StopReasonToolUse)
	sf := streamSeq(first, textAssistant("done"))
	go func() { time.Sleep(20 * time.Millisecond); close(release) }()
	_, events, err := collectEvents([]AgentMessage{userMsg("both")}, AgentContext{Tools: []AgentTool[any, any]{tool}}, cfg, sf)
	if err != nil {
		t.Fatal(err)
	}
	endIDs, resultIDs := []string{}, []string{}
	for _, e := range events {
		switch ev := e.(type) {
		case ToolExecutionEndEvent:
			endIDs = append(endIDs, ev.ToolCallID)
		case MessageEndEvent:
			if tr, ok := ev.Message.(ai.ToolResultMessage); ok {
				resultIDs = append(resultIDs, tr.ToolCallID)
			}
		}
	}
	if !parallelObserved || !reflect.DeepEqual(endIDs, []string{"tool-2", "tool-1"}) || !reflect.DeepEqual(resultIDs, []string{"tool-1", "tool-2"}) {
		t.Fatalf("parallel=%v ends=%v results=%v", parallelObserved, endIDs, resultIDs)
	}
}

func TestAgentLoopInjectsQueuedMessagesAfterAllToolCallsComplete(t *testing.T) {
	executed := []string{}
	cfg := baseConfig()
	mode := ToolExecutionModeSequential
	cfg.ToolExecution = &mode
	delivered := false
	cfg.GetSteeringMessages = func(context.Context) []AgentMessage {
		if len(executed) >= 1 && !delivered {
			delivered = true
			return []AgentMessage{userMsg("interrupt")}
		}
		return nil
	}
	call := 0
	sawInterrupt := false
	sf := func(_ context.Context, _ ai.Model[ai.API], mc ai.ModelContext, _ *ai.SimpleStreamOptions) (*ai.AssistantMessageEventStream, error) {
		if call == 1 {
			for _, m := range mc.Messages {
				if u, ok := m.(ai.UserMessage); ok && userText(u) == "interrupt" {
					sawInterrupt = true
				}
			}
		}
		call++
		if call == 1 {
			return doneStream(assistantMsg([]ai.AssistantContent{ai.ToolCall{Type: ai.ContentTypeToolCall, Id: "tool-1", Name: "echo", Args: map[string]any{"value": "first"}}, ai.ToolCall{Type: ai.ContentTypeToolCall, Id: "tool-2", Name: "echo", Args: map[string]any{"value": "second"}}}, ai.StopReasonToolUse)), nil
		}
		return doneStream(textAssistant("done")), nil
	}
	_, events, err := collectEvents([]AgentMessage{userMsg("start")}, AgentContext{Tools: []AgentTool[any, any]{echoTool(&executed)}}, cfg, sf)
	if err != nil {
		t.Fatal(err)
	}
	seq := []string{}
	for _, e := range events {
		if ev, ok := e.(MessageStartEvent); ok {
			if tr, ok := ev.Message.(ai.ToolResultMessage); ok {
				seq = append(seq, "tool:"+tr.ToolCallID)
			}
			if u, ok := ev.Message.(ai.UserMessage); ok {
				seq = append(seq, userText(u))
			}
		}
	}
	idx := func(s string) int {
		for i, v := range seq {
			if v == s {
				return i
			}
		}
		return -1
	}
	if !reflect.DeepEqual(executed, []string{"first", "second"}) || !sawInterrupt || idx("interrupt") == -1 || idx("tool:tool-1") > idx("interrupt") || idx("tool:tool-2") > idx("interrupt") {
		t.Fatalf("executed=%v saw=%v seq=%v", executed, sawInterrupt, seq)
	}
}

func TestAgentLoopSequentialExecutionModes(t *testing.T) {
	for _, tc := range []struct {
		name           string
		slowSequential bool
		allParallel    bool
		wantParallel   bool
	}{{"tool forces sequential", true, false, false}, {"all parallel", false, true, true}} {
		t.Run(tc.name, func(t *testing.T) {
			firstResolved, parallelObserved := false, false
			release := make(chan struct{})
			tool := echoTool(nil)
			if tc.slowSequential {
				m := ToolExecutionModeSequential
				tool.ToolExecutionMode = &m
			} else if tc.allParallel {
				m := ToolExecutionModeParallel
				tool.ToolExecutionMode = &m
			}
			tool.Execute = func(_ context.Context, _ string, params any, _ AgentToolUpdateCallback[any]) (AgentToolResult[any], error) {
				v := params.(map[string]any)["value"].(string)
				if v == "first" {
					<-release
					firstResolved = true
				}
				if v == "second" && !firstResolved {
					parallelObserved = true
				}
				return AgentToolResult[any]{Content: []ai.ToolResultContent{ai.TextContent{Type: ai.ContentTypeText, Text: v}}}, nil
			}
			first := assistantMsg([]ai.AssistantContent{ai.ToolCall{Type: ai.ContentTypeToolCall, Id: "tool-1", Name: "echo", Args: map[string]any{"value": "first"}}, ai.ToolCall{Type: ai.ContentTypeToolCall, Id: "tool-2", Name: "echo", Args: map[string]any{"value": "second"}}}, ai.StopReasonToolUse)
			go func() { time.Sleep(20 * time.Millisecond); close(release) }()
			_, _, err := collectEvents([]AgentMessage{userMsg("run")}, AgentContext{Tools: []AgentTool[any, any]{tool}}, baseConfig(), streamSeq(first, textAssistant("done")))
			if err != nil {
				t.Fatal(err)
			}
			if parallelObserved != tc.wantParallel {
				t.Fatalf("parallelObserved=%v", parallelObserved)
			}
		})
	}
}

func TestAgentLoopSequentialModeOnOneOfMultipleToolsForcesBatchSequential(t *testing.T) {
	executionOrder := []string{}
	releaseSlow := make(chan struct{})
	seqMode := ToolExecutionModeSequential
	slow := echoTool(nil)
	slow.Name = "slow"
	slow.ToolExecutionMode = &seqMode
	slow.Execute = func(_ context.Context, _ string, params any, _ AgentToolUpdateCallback[any]) (AgentToolResult[any], error) {
		v := params.(map[string]any)["value"].(string)
		executionOrder = append(executionOrder, "slow:"+v)
		if v == "a" {
			<-releaseSlow
		}
		return AgentToolResult[any]{Content: []ai.ToolResultContent{ai.TextContent{Type: ai.ContentTypeText, Text: v}}}, nil
	}
	fast := echoTool(nil)
	fast.Name = "fast"
	fast.Execute = func(_ context.Context, _ string, params any, _ AgentToolUpdateCallback[any]) (AgentToolResult[any], error) {
		v := params.(map[string]any)["value"].(string)
		executionOrder = append(executionOrder, "fast:"+v)
		return AgentToolResult[any]{Content: []ai.ToolResultContent{ai.TextContent{Type: ai.ContentTypeText, Text: v}}}, nil
	}
	first := assistantMsg([]ai.AssistantContent{
		ai.ToolCall{Type: ai.ContentTypeToolCall, Id: "tool-1", Name: "slow", Args: map[string]any{"value": "a"}},
		ai.ToolCall{Type: ai.ContentTypeToolCall, Id: "tool-2", Name: "fast", Args: map[string]any{"value": "b"}},
	}, ai.StopReasonToolUse)
	go func() { time.Sleep(20 * time.Millisecond); close(releaseSlow) }()
	_, _, err := collectEvents([]AgentMessage{userMsg("run")}, AgentContext{Tools: []AgentTool[any, any]{slow, fast}}, baseConfig(), streamSeq(first, textAssistant("done")))
	if err != nil {
		t.Fatal(err)
	}
	if len(executionOrder) < 2 || executionOrder[0] != "slow:a" || executionOrder[1] != "fast:b" {
		t.Fatalf("executionOrder=%v", executionOrder)
	}
}

func TestAgentLoopStopsAfterCurrentTurnWhenShouldStopAfterTurn(t *testing.T) {
	executed := []string{}
	polls := 0
	follow := 0
	ids := []string{}
	roles := []string{}
	cfg := baseConfig()
	cfg.GetSteeringMessages = func(context.Context) []AgentMessage { polls++; return nil }
	cfg.GetFollowUpMessages = func(context.Context) []AgentMessage { follow++; return []AgentMessage{userMsg("follow")} }
	cfg.ShouldStopAfterTurn = func(c ShouldStopAfterTurnContext) bool {
		for _, tr := range c.ToolResults {
			ids = append(ids, tr.ToolCallID)
		}
		for _, m := range c.Context.Messages {
			roles = append(roles, m.AgentMessageType())
		}
		return true
	}
	first := assistantMsg([]ai.AssistantContent{ai.ToolCall{Type: ai.ContentTypeToolCall, Id: "tool-1", Name: "echo", Args: map[string]any{"value": "hello"}}}, ai.StopReasonToolUse)
	msgs, events, err := collectEvents([]AgentMessage{userMsg("echo")}, AgentContext{Tools: []AgentTool[any, any]{echoTool(&executed)}}, cfg, streamSeq(first, textAssistant("no")))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(executed, []string{"hello"}) || polls != 1 || follow != 0 || !reflect.DeepEqual(ids, []string{"tool-1"}) || !reflect.DeepEqual(roles, []string{"user", "assistant", "toolResult"}) || len(msgs) != 3 {
		t.Fatalf("bad stop state")
	}
	want := []string{"agent_start", "turn_start", "message_start", "message_end", "message_start", "message_end", "tool_execution_start", "tool_execution_end", "message_start", "message_end", "turn_end", "agent_end"}
	if !reflect.DeepEqual(eventTypes(events), want) {
		t.Fatalf("events %v", eventTypes(events))
	}
}

func TestAgentLoopToolBatchTermination(t *testing.T) {
	tool := echoTool(nil)
	term := true
	tool.Execute = func(_ context.Context, _ string, params any, _ AgentToolUpdateCallback[any]) (AgentToolResult[any], error) {
		return AgentToolResult[any]{Content: []ai.ToolResultContent{ai.TextContent{Type: ai.ContentTypeText, Text: "ok"}}, Terminate: &term}, nil
	}
	calls := 0
	sf := func(context.Context, ai.Model[ai.API], ai.ModelContext, *ai.SimpleStreamOptions) (*ai.AssistantMessageEventStream, error) {
		calls++
		return doneStream(assistantMsg([]ai.AssistantContent{ai.ToolCall{Type: ai.ContentTypeToolCall, Id: "tool-1", Name: "echo", Args: map[string]any{"value": "hello"}}}, ai.StopReasonToolUse)), nil
	}
	msgs, _, err := collectEvents([]AgentMessage{userMsg("echo")}, AgentContext{Tools: []AgentTool[any, any]{tool}}, baseConfig(), sf)
	if err != nil {
		t.Fatal(err)
	}
	if calls != 1 || len(msgs) != 3 {
		t.Fatalf("calls=%d msgs=%d", calls, len(msgs))
	}
}

func TestAgentLoopContinuesWhenNotAllParallelToolResultsTerminate(t *testing.T) {
	tool := echoTool(nil)
	tool.Execute = func(_ context.Context, _ string, params any, _ AgentToolUpdateCallback[any]) (AgentToolResult[any], error) {
		v := params.(map[string]any)["value"].(string)
		term := v == "first"
		return AgentToolResult[any]{Content: []ai.ToolResultContent{ai.TextContent{Type: ai.ContentTypeText, Text: v}}, Terminate: &term}, nil
	}
	cfg := baseConfig()
	mode := ToolExecutionModeParallel
	cfg.ToolExecution = &mode
	first := assistantMsg([]ai.AssistantContent{ai.ToolCall{Type: ai.ContentTypeToolCall, Id: "tool-1", Name: "echo", Args: map[string]any{"value": "first"}}, ai.ToolCall{Type: ai.ContentTypeToolCall, Id: "tool-2", Name: "echo", Args: map[string]any{"value": "second"}}}, ai.StopReasonToolUse)
	msgs, _, err := collectEvents([]AgentMessage{userMsg("both")}, AgentContext{Tools: []AgentTool[any, any]{tool}}, cfg, streamSeq(first, textAssistant("done")))
	if err != nil {
		t.Fatal(err)
	}
	roles := []string{}
	for _, m := range msgs {
		roles = append(roles, m.AgentMessageType())
	}
	want := []string{"user", "assistant", "toolResult", "toolResult", "assistant"}
	if !reflect.DeepEqual(roles, want) {
		t.Fatalf("%v", roles)
	}
}

func TestAgentLoopAfterToolCallCanTerminateBatch(t *testing.T) {
	cfg := baseConfig()
	cfg.AfterToolCall = func(context.Context, AfterToolCallContext[any]) (*AfterToolCallResult, error) {
		return &AfterToolCallResult{Terminate: boolPtr(true)}, nil
	}
	calls := 0
	sf := func(context.Context, ai.Model[ai.API], ai.ModelContext, *ai.SimpleStreamOptions) (*ai.AssistantMessageEventStream, error) {
		calls++
		return doneStream(assistantMsg([]ai.AssistantContent{ai.ToolCall{Type: ai.ContentTypeToolCall, Id: "tool-1", Name: "echo", Args: map[string]any{"value": "hello"}}}, ai.StopReasonToolUse)), nil
	}
	_, _, err := collectEvents([]AgentMessage{userMsg("echo")}, AgentContext{Tools: []AgentTool[any, any]{echoTool(nil)}}, cfg, sf)
	if err != nil {
		t.Fatal(err)
	}
	if calls != 1 {
		t.Fatalf("calls=%d", calls)
	}
}

func TestAgentLoopContinueRequiresMessages(t *testing.T) {
	_, err := AgentLoopContinue(AgentContext{}, baseConfig(), context.Background(), nil)
	if err == nil || err.Error() != "Cannot continue: no messages in context" {
		t.Fatalf("err=%v", err)
	}
}

func TestAgentLoopContinueFromExistingContext(t *testing.T) {
	cfg := baseConfig()
	events := []AgentEvent{}
	sf := streamSeq(textAssistant("Response"))
	msgs, err := RunAgentLoopContinue(AgentContext{Messages: []AgentMessage{userMsg("Hello")}}, cfg, context.Background(), func(e AgentEvent) error { events = append(events, e); return nil }, &sf)
	if err != nil {
		t.Fatal(err)
	}
	if len(msgs) != 1 || msgs[0].AgentMessageType() != "assistant" {
		t.Fatalf("msgs=%#v", msgs)
	}
	ends := 0
	for _, e := range events {
		if ev, ok := e.(MessageEndEvent); ok {
			ends++
			if ev.Message.AgentMessageType() != "assistant" {
				t.Fatalf("unexpected message end %#v", ev.Message)
			}
		}
	}
	if ends != 1 {
		t.Fatalf("ends=%d", ends)
	}
}

func TestAgentLoopContinueAllowsCustomLastMessage(t *testing.T) {
	cfg := baseConfig()
	cfg.ConvertToLLM = func(messages []AgentMessage) []ai.Message {
		out := []ai.Message{}
		for _, m := range messages {
			if cm, ok := m.(customMessage); ok {
				out = append(out, userMsg(cm.text))
				continue
			}
			if msg, ok := m.(ai.Message); ok {
				out = append(out, msg)
			}
		}
		return out
	}
	sf := streamSeq(textAssistant("Response to custom message"))
	msgs, err := RunAgentLoopContinue(AgentContext{Messages: []AgentMessage{customMessage{role: "custom", text: "Hook content", timestamp: nowMs()}}}, cfg, context.Background(), func(AgentEvent) error { return nil }, &sf)
	if err != nil {
		t.Fatal(err)
	}
	if len(msgs) != 1 || msgs[0].AgentMessageType() != "assistant" {
		t.Fatalf("msgs=%#v", msgs)
	}
}
