package mcp

import (
	"context"
	"testing"
	"time"

	"github.com/ColeBurch/burrow/ai"
	"github.com/ColeBurch/burrow/ai/providertest"
	oair "github.com/ColeBurch/burrow/ai/providertest/openairesponses"
	"github.com/ColeBurch/burrow/baseAgent"
	"github.com/ColeBurch/burrow/baseAgent/authstoretest"
	"github.com/ColeBurch/burrow/baseAgent/sessionstoretest"
	sdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

// An Extension registers a server's tools on an agent session. The model
// calls one through the real OpenAI provider, and its image reaches the next
// request.
func TestExtensionInAgentSession(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 20*time.Second)
	defer cancel()

	mcpServer := sdk.NewServer(&sdk.Implementation{Name: "docs", Version: "1.0.0"}, nil)
	mcpServer.AddTool(&sdk.Tool{Name: "screenshot", Description: "Capture the page.", InputSchema: objectSchema},
		func(context.Context, *sdk.CallToolRequest) (*sdk.CallToolResult, error) {
			return &sdk.CallToolResult{Content: []sdk.Content{
				&sdk.TextContent{Text: "Captured"},
				&sdk.ImageContent{Data: []byte("png"), MIMEType: "image/png"},
			}}, nil
		})

	oair.Register()
	server := providertest.NewServer(t, providertest.Sequence(t,
		oair.Reply(
			oair.ToolCall("call_1", "fc_1", "mcp__docs__screenshot", `{}`),
			oair.Completed("resp_1", oair.Usage{Input: 5, Output: 3}),
		),
		oair.Reply(
			oair.Text("msg_2", "A page."),
			oair.Completed("resp_2", oair.Usage{Input: 9, Output: 2}),
		),
	))
	base := oair.Model()
	base.Input = []string{"text", "image"}
	model := server.Route(t, base)

	authStore := authstoretest.NewMemoryStore()
	credentials := baseAgent.ScopeAuthStore(authStore, "user-1")
	if _, err := credentials.Modify(ctx, model.Provider, func(*ai.Credential) (*ai.Credential, error) {
		return &ai.Credential{Type: ai.CredentialAPIKey, Key: "test-key"}, nil
	}); err != nil {
		t.Fatal(err)
	}
	registry := baseAgent.NewModelRegistry(&baseAgent.ModelRegistryOptions{
		Credentials: credentials,
		Models:      map[ai.Provider]map[string]ai.Model[ai.API]{model.Provider: {model.ID: model}},
	})
	sessionManager, err := baseAgent.NewSessionManager(ctx, nil, nil, true, sessionstoretest.NewMemoryStore(), nil)
	if err != nil {
		t.Fatal(err)
	}
	created, err := baseAgent.CreateAgentSession(ctx, baseAgent.CreateAgentSessionOptions{
		SessionManager: sessionManager,
		ModelRegistry:  registry,
	})
	if err != nil {
		t.Fatal(err)
	}
	memory := &inMemory{server: mcpServer}
	ext, err := NewExtension(ExtensionOptions{
		Servers: []ServerConnectionOptions{{Name: "docs", CreateTransport: memory.transport}},
		Tools:   created.Session,
	})
	if err != nil {
		t.Fatal(err)
	}
	ext.Start()
	defer ext.Close(ctx)
	if !ext.WaitForStartup(ctx) {
		t.Fatal("the server did not connect")
	}

	if err := created.Session.Prompt(ctx, "What is on the page?", nil); err != nil {
		t.Fatal(err)
	}

	requests := server.Requests()
	if len(requests) != 2 {
		t.Fatalf("got %d model requests, want 2", len(requests))
	}
	tools, _ := requests[0].JSON(t)["tools"].([]any)
	if len(tools) != 1 || tools[0].(map[string]any)["name"] != "mcp__docs__screenshot" {
		t.Fatalf("first request tools = %#v", tools)
	}
	outputs := oair.InputItems(t, requests[1], "function_call_output")
	if len(outputs) != 1 {
		t.Fatalf("function_call_output items = %#v", outputs)
	}
	parts, ok := outputs[0]["output"].([]any)
	if !ok || len(parts) != 2 {
		t.Fatalf("tool output = %#v, want text and image parts", outputs[0]["output"])
	}
	if image, _ := parts[1].(map[string]any); image["image_url"] != "data:image/png;base64,cG5n" {
		t.Fatalf("image part = %#v", parts[1])
	}

	messages, err := created.Session.GetMessages()
	if err != nil {
		t.Fatal(err)
	}
	var result *ai.ToolResultMessage
	for _, message := range messages {
		if m, ok := message.(ai.ToolResultMessage); ok {
			result = &m
		}
	}
	if result == nil || result.IsError || len(result.Content) != 2 {
		t.Fatalf("tool result = %#v", result)
	}
	if details, ok := result.Details.(ToolDetails); !ok || details.Server != "docs" || details.Tool != "screenshot" {
		t.Fatalf("tool result details = %#v", result.Details)
	}
}
