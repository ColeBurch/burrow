package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/ColeBurch/burrow/agent"
	"github.com/ColeBurch/burrow/ai"
	"github.com/modelcontextprotocol/go-sdk/jsonrpc"
	sdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

// newResourceServer offers a few resources, MCP App resources the tools
// leave out, a template, and a tool that returns a resource link.
func newResourceServer(opts *sdk.ServerOptions) *sdk.Server {
	server := sdk.NewServer(&sdk.Implementation{Name: "docs", Version: "1.0.0"}, opts)
	read := func(_ context.Context, req *sdk.ReadResourceRequest) (*sdk.ReadResourceResult, error) {
		uri := req.Params.URI
		switch uri {
		case "file:///a.txt":
			return &sdk.ReadResourceResult{Contents: []*sdk.ResourceContents{
				{URI: uri, MIMEType: "text/plain", Text: "alpha", Meta: sdk.Meta{"secret": true}},
			}}, nil
		case "file:///b.png":
			return &sdk.ReadResourceResult{Contents: []*sdk.ResourceContents{{URI: uri, MIMEType: "image/png", Blob: []byte("png")}}}, nil
		case "file:///dir":
			return &sdk.ReadResourceResult{Contents: []*sdk.ResourceContents{
				{URI: uri + "/1", Text: "one"},
				{URI: uri + "/2", Text: "two"},
			}}, nil
		default:
			return &sdk.ReadResourceResult{}, nil
		}
	}
	server.AddResource(&sdk.Resource{
		URI: "file:///a.txt", Name: "a.txt", MIMEType: "text/plain",
		Meta: sdk.Meta{"secret": true}, Icons: []sdk.Icon{{Source: "https://example.com/a.png"}},
	}, read)
	server.AddResource(&sdk.Resource{URI: "file:///b.png", Name: "b.png", MIMEType: "image/png"}, read)
	server.AddResource(&sdk.Resource{URI: "file:///dir", Name: "dir"}, read)
	server.AddResource(&sdk.Resource{URI: "file:///empty", Name: "empty"}, read)
	server.AddResource(&sdk.Resource{URI: "ui://widget", Name: "widget", MIMEType: "text/html"}, read)
	server.AddResource(&sdk.Resource{URI: "file:///app.html", Name: "app", MIMEType: `text/html; profile="mcp-app"`}, read)
	server.AddResourceTemplate(&sdk.ResourceTemplate{URITemplate: "file:///notes/{name}", Name: "note"}, read)
	server.AddTool(&sdk.Tool{Name: "find", InputSchema: objectSchema}, linkTool)
	return server
}

func linkTool(context.Context, *sdk.CallToolRequest) (*sdk.CallToolResult, error) {
	return &sdk.CallToolResult{Content: []sdk.Content{&sdk.ResourceLink{URI: "file:///a.txt", Name: "a.txt"}}}, nil
}

func toolNamed(t *testing.T, tools []agent.AgentTool[any, any], name string) agent.AgentTool[any, any] {
	t.Helper()
	for _, tool := range tools {
		if tool.Name == name {
			return tool
		}
	}
	t.Fatalf("no tool %s", name)
	return agent.AgentTool[any, any]{}
}

// listed decodes a listing from the model-facing JSON, and checks that the
// structured content says the same.
func listed(t *testing.T, result agent.AgentToolResult[any], key string) (items []map[string]any, next string, errs []any) {
	t.Helper()
	var payload map[string]any
	if err := json.Unmarshal([]byte(textOf(result.Content)), &payload); err != nil {
		t.Fatalf("listing is not JSON: %v: %s", err, textOf(result.Content))
	}
	structured, err := json.Marshal(result.Details.(ToolDetails).StructuredContent)
	if err != nil {
		t.Fatal(err)
	}
	if string(structured) != textOf(result.Content) {
		t.Fatalf("structured content %s differs from text %s", structured, textOf(result.Content))
	}
	raw, ok := payload[key].([]any)
	if !ok {
		t.Fatalf("listing has no %s array: %v", key, payload)
	}
	for _, item := range raw {
		items = append(items, item.(map[string]any))
	}
	next, _ = payload["nextCursor"].(string)
	errs, _ = payload["errors"].([]any)
	return items, next, errs
}

// unavailableServer is a ResourceServer whose requests fail.
type unavailableServer struct{ name string }

var errUnavailable = errors.New("unavailable")

func (s unavailableServer) Name() string           { return s.name }
func (s unavailableServer) Timeout() time.Duration { return time.Second }
func (s unavailableServer) ResourcesPage(context.Context, string, *RequestOptions) (*sdk.ListResourcesResult, error) {
	return nil, errUnavailable
}
func (s unavailableServer) ResourceTemplatesPage(context.Context, string, *RequestOptions) (*sdk.ListResourceTemplatesResult, error) {
	return nil, errUnavailable
}
func (s unavailableServer) AllResources(context.Context, *RequestOptions) ([]*sdk.Resource, error) {
	return nil, errUnavailable
}
func (s unavailableServer) AllResourceTemplates(context.Context, *RequestOptions) ([]*sdk.ResourceTemplate, error) {
	return nil, errUnavailable
}
func (s unavailableServer) ReadResource(context.Context, string, *RequestOptions) (*sdk.ReadResourceResult, error) {
	return nil, errUnavailable
}

func TestResourceTools(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	docs, _ := newConnection(t, newResourceServer(&sdk.ServerOptions{PageSize: 2}), ServerConnectionOptions{Name: "docs"})
	if _, err := docs.GetClient(ctx); err != nil {
		t.Fatal(err)
	}
	var servers []ResourceServer
	tools := CreateResourceToolDefinitions(func() []ResourceServer { return servers })
	var names []string
	for _, tool := range tools {
		names = append(names, tool.Name)
		if tool.Annotations == nil || tool.Annotations.ReadOnlyHint == nil || !*tool.Annotations.ReadOnlyHint {
			t.Errorf("%s is not marked read-only", tool.Name)
		}
	}
	if !slices.Equal(names, []string{ListResourcesTool, ListResourceTemplatesTool, ReadResourceTool}) {
		t.Fatalf("tools = %v", names)
	}
	list := toolNamed(t, tools, ListResourcesTool)
	execute := func(tool agent.AgentTool[any, any], args map[string]any) (agent.AgentToolResult[any], error) {
		return tool.Execute(ctx, "call", args, nil)
	}
	if _, err := execute(list, map[string]any{"server": "docs"}); err == nil || err.Error() != `MCP server "docs" has no resources` {
		t.Fatalf("no servers: %v", err)
	}
	// The servers are read at call time.
	servers = []ResourceServer{docs, unavailableServer{"broken"}}

	// Every server: visible resources tagged with their server, and the
	// server that failed.
	result, err := execute(list, map[string]any{})
	if err != nil {
		t.Fatal(err)
	}
	items, next, errs := listed(t, result, "resources")
	var uris []string
	for _, item := range items {
		uris = append(uris, item["uri"].(string))
		if item["server"] != "docs" {
			t.Errorf("item %v has server %v", item["uri"], item["server"])
		}
		if _, ok := item["_meta"]; ok {
			t.Errorf("item %v keeps _meta", item["uri"])
		}
		if _, ok := item["icons"]; ok {
			t.Errorf("item %v keeps icons", item["uri"])
		}
	}
	slices.Sort(uris)
	if want := []string{"file:///a.txt", "file:///b.png", "file:///dir", "file:///empty"}; !slices.Equal(uris, want) {
		t.Fatalf("listed %v, want %v", uris, want)
	}
	if next != "" || len(errs) != 1 || errs[0].(map[string]any)["server"] != "broken" {
		t.Fatalf("next %q, errors %v", next, errs)
	}

	// One server, page by page.
	var paged []string
	pages := 0
	for cursor := ""; pages == 0 || cursor != ""; pages++ {
		result, err := execute(list, map[string]any{"server": " docs ", "cursor": cursor})
		if err != nil {
			t.Fatal(err)
		}
		if result.Details.(ToolDetails).Server != "docs" {
			t.Fatalf("details = %#v", result.Details)
		}
		items, next, _ := listed(t, result, "resources")
		for _, item := range items {
			paged = append(paged, item["uri"].(string))
		}
		cursor = next
	}
	slices.Sort(paged)
	if !slices.Equal(paged, uris) || pages < 2 {
		t.Fatalf("paged %v in %d pages", paged, pages)
	}

	if _, err := execute(list, map[string]any{"cursor": "x"}); err == nil || !strings.Contains(err.Error(), "cursor can only be used") {
		t.Fatalf("cursor without server: %v", err)
	}
	if _, err := execute(list, map[string]any{"server": "plain"}); err == nil || !strings.Contains(err.Error(), "Servers with resources: docs, broken") {
		t.Fatalf("server without resources: %v", err)
	}
	if _, err := execute(list, map[string]any{"server": 3}); err == nil {
		t.Fatal("non-string server accepted")
	}
	if _, err := execute(list, map[string]any{"server": "broken"}); !errors.Is(err, errUnavailable) {
		t.Fatalf("failing server: %v", err)
	}

	result, err = execute(toolNamed(t, tools, ListResourceTemplatesTool), map[string]any{"server": "docs"})
	if err != nil {
		t.Fatal(err)
	}
	templates, _, _ := listed(t, result, "resourceTemplates")
	if len(templates) != 1 || templates[0]["uriTemplate"] != "file:///notes/{name}" || templates[0]["server"] != "docs" {
		t.Fatalf("templates = %v", templates)
	}
}

func TestReadResourceTool(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	docs, _ := newConnection(t, newResourceServer(nil), ServerConnectionOptions{Name: "docs"})
	tools := CreateResourceToolDefinitions(func() []ResourceServer { return []ResourceServer{docs} })
	read := toolNamed(t, tools, ReadResourceTool)
	readURI := func(uri string) agent.AgentToolResult[any] {
		t.Helper()
		result, err := read.Execute(ctx, "call", map[string]any{"server": "docs", "uri": uri}, nil)
		if err != nil {
			t.Fatal(err)
		}
		return result
	}

	text := readURI("file:///a.txt")
	if got := textOf(text.Content); got != "alpha" {
		t.Fatalf("text resource = %q", got)
	}
	details := text.Details.(ToolDetails)
	contents := details.StructuredContent.(map[string]any)["contents"].([]map[string]any)
	if details.Tool != ReadResourceTool || len(contents) != 1 || contents[0]["text"] != "alpha" || contents[0]["_meta"] != nil {
		t.Fatalf("details = %#v", details)
	}

	if image, ok := readURI("file:///b.png").Content[0].(ai.ImageContent); !ok || image.Image != "cG5n" {
		t.Fatalf("image resource = %#v", image)
	}
	if got := textOf(readURI("file:///dir").Content); got != "file:///dir/1:\none\nfile:///dir/2:\ntwo" {
		t.Fatalf("several contents = %q", got)
	}
	if _, err := read.Execute(ctx, "call", map[string]any{"server": "docs"}, nil); err == nil {
		t.Fatal("read without uri succeeded")
	}
}

// Resource links name read_mcp_resource only when the server's resources
// are readable.
func TestResourceLinksPointToReadTool(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	plain := sdk.NewServer(&sdk.Implementation{Name: "plain", Version: "1.0.0"}, nil)
	plain.AddTool(&sdk.Tool{Name: "find", InputSchema: objectSchema}, linkTool)

	for _, tc := range []struct {
		server   *sdk.Server
		readable bool
		hint     bool
	}{
		{newResourceServer(nil), true, true},
		{newResourceServer(nil), false, false},
		{plain, true, false},
	} {
		connection, _ := newConnection(t, tc.server, ServerConnectionOptions{})
		tool := connectionTools(t, ctx, connection)["mcp__s__find"]
		if !tc.readable {
			tool = CreateToolDefinition(ToolDefinitionOptions{
				Server: "s", Tool: connection.Tools()[0], Name: "mcp__s__find",
				GetClient: func(context.Context) (ToolCaller, error) { return connection, nil },
			})
		}
		result, err := tool.Execute(ctx, "call", map[string]any{}, nil)
		if err != nil {
			t.Fatal(err)
		}
		if hint := strings.Contains(textOf(result.Content), "Read it with read_mcp_resource"); hint != tc.hint {
			t.Errorf("readable %v: hint %v in %q", tc.readable, hint, textOf(result.Content))
		}
	}
}

func TestIsMethodNotFound(t *testing.T) {
	if !isMethodNotFound(fmt.Errorf("list: %w", &jsonrpc.Error{Code: jsonrpc.CodeMethodNotFound})) {
		t.Fatal("method not found not recognized")
	}
	if isMethodNotFound(&jsonrpc.Error{Code: jsonrpc.CodeInvalidParams}) || isMethodNotFound(nil) {
		t.Fatal("other errors recognized as method not found")
	}
}
