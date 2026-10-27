package baseAgent_test

import (
	"context"
	"reflect"
	"testing"

	"github.com/ColeBurch/burrow/agent"
	"github.com/ColeBurch/burrow/ai"
	"github.com/ColeBurch/burrow/ai/providertest"
	"github.com/ColeBurch/burrow/ai/providertest/openairesponses"
	"github.com/ColeBurch/burrow/baseAgent"
)

// requestToolNames are the names of the tools a Responses request declares.
func requestToolNames(t *testing.T, request providertest.Request) []string {
	t.Helper()
	names := []string{}
	tools, _ := request.JSON(t)["tools"].([]any)
	for _, raw := range tools {
		if tool, ok := raw.(map[string]any); ok {
			names = append(names, tool["name"].(string))
		}
	}
	return names
}

func registryTool(name string, execute func() error) agent.AgentTool[any, any] {
	return agent.AgentTool[any, any]{
		Tool: ai.Tool{Name: name, Description: "Tool " + name, Parameters: map[string]any{"type": "object", "properties": map[string]any{}}},
		Execute: func(context.Context, string, any, agent.AgentToolUpdateCallback[any]) (agent.AgentToolResult[any], error) {
			if execute != nil {
				if err := execute(); err != nil {
					return agent.AgentToolResult[any]{}, err
				}
			}
			return agent.AgentToolResult[any]{Content: []ai.ToolResultContent{ai.TextContent{Type: ai.ContentTypeText, Text: "ok"}}}, nil
		},
	}
}

// A tool registered while a prompt runs, as an MCP server that finished
// connecting does, is declared at the prompt's next turn.
func TestRegisterToolReachesNextTurn(t *testing.T) {
	h := newHTTPSessionHarness(t, openairesponses.Model(), providertest.Sequence(t,
		openairesponses.Reply(
			openairesponses.ToolCall("call_1", "fc_1", "connect", `{}`),
			openairesponses.Completed("resp_1", openairesponses.Usage{Input: 5, Output: 3}),
		),
		openairesponses.Reply(
			openairesponses.Text("msg_2", "done"),
			openairesponses.Completed("resp_2", openairesponses.Usage{Input: 9, Output: 2}),
		),
	))
	var session *baseAgent.AgentSession
	connect := registryTool("connect", func() error {
		return session.RegisterTool(registryTool("search", nil))
	})
	session = h.open(t, "", baseAgent.CreateAgentSessionOptions{Tools: []agent.AgentTool[any, any]{connect}})

	if err := session.Prompt(h.ctx, "connect", nil); err != nil {
		t.Fatal(err)
	}
	requests := h.server.Requests()
	if len(requests) != 2 {
		t.Fatalf("got %d requests", len(requests))
	}
	if got := requestToolNames(t, requests[0]); !reflect.DeepEqual(got, []string{"connect"}) {
		t.Fatalf("first request tools = %v", got)
	}
	if got := requestToolNames(t, requests[1]); !reflect.DeepEqual(got, []string{"connect", "search"}) {
		t.Fatalf("second request tools = %v", got)
	}
}

func TestToolRegistry(t *testing.T) {
	h := newHTTPSessionHarness(t, openairesponses.Model(), providertest.Sequence(t))
	readOnly := true
	first := registryTool("first", nil)
	first.Annotations = &agent.ToolAnnotations{ReadOnlyHint: &readOnly}
	session := h.open(t, "", baseAgent.CreateAgentSessionOptions{Tools: []agent.AgentTool[any, any]{first, registryTool("second", nil)}})
	active := func() []string {
		t.Helper()
		names, err := session.GetActiveToolNames()
		if err != nil {
			t.Fatal(err)
		}
		return names
	}

	if err := session.SetActiveToolsByName([]string{"second", "unknown", "second"}); err != nil {
		t.Fatal(err)
	}
	if got := active(); !reflect.DeepEqual(got, []string{"second"}) {
		t.Fatalf("active = %v", got)
	}

	// Registering again replaces the tool without activating it; a new tool
	// is activated.
	replaced := registryTool("first", nil)
	replaced.Description = "replaced"
	if err := session.RegisterTool(replaced); err != nil {
		t.Fatal(err)
	}
	if err := session.RegisterTool(registryTool("third", nil)); err != nil {
		t.Fatal(err)
	}
	if got := active(); !reflect.DeepEqual(got, []string{"second", "third"}) {
		t.Fatalf("active after registering = %v", got)
	}
	if err := session.SetActiveToolsByName([]string{"first"}); err != nil {
		t.Fatal(err)
	}
	if tools := session.Agent.State().Tools; len(tools) != 1 || tools[0].Description != "replaced" {
		t.Fatalf("active first = %#v", tools)
	}

	all := session.GetAllTools()
	var names []string
	for _, info := range all {
		names = append(names, info.Name)
	}
	if !reflect.DeepEqual(names, []string{"first", "second", "third"}) || all[0].Description != "replaced" || all[0].Annotations != nil {
		t.Fatalf("all tools = %#v", all)
	}
	if err := session.RegisterTool(registryTool("", nil)); err == nil {
		t.Fatal("registered a tool without a name")
	}

	// Registering an active tool again updates the agent's copy.
	again := registryTool("first", nil)
	again.Annotations = first.Annotations
	if err := session.RegisterTool(again); err != nil {
		t.Fatal(err)
	}
	if tools := session.Agent.State().Tools; len(tools) != 1 || tools[0].Annotations == nil || session.GetAllTools()[0].Annotations == nil {
		t.Fatalf("active first after registering again = %#v", tools)
	}
}
