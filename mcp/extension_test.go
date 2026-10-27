package mcp

import (
	"context"
	"slices"
	"strings"
	"testing"
	"time"

	sdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

// newExtension returns an extension of servers on registry, closed when the
// test ends.
func newExtension(t *testing.T, registry ToolRegistry, servers ...ServerConnectionOptions) *Extension {
	t.Helper()
	ext, err := NewExtension(ExtensionOptions{Servers: servers, Tools: registry})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := ext.Close(ctx); err != nil {
			t.Errorf("close: %v", err)
		}
	})
	return ext
}

// replyTool answers every call with text.
func replyTool(text string) sdk.ToolHandler {
	return func(context.Context, *sdk.CallToolRequest) (*sdk.CallToolResult, error) {
		return textResult(text), nil
	}
}

func startExtension(t *testing.T, ctx context.Context, ext *Extension) {
	t.Helper()
	ext.Start()
	if !ext.WaitForStartup(ctx) {
		t.Fatal("the servers did not connect")
	}
}

func callRegistered(t *testing.T, ctx context.Context, registry *fakeRegistry, name string) (string, error) {
	t.Helper()
	result, err := registry.tool(t, name).Execute(ctx, "call", map[string]any{}, nil)
	if err != nil {
		return "", err
	}
	return textOf(result.Content), nil
}

// Tools the server adds are registered, tools it removes deactivated, and
// activated again when it offers them again, under the same names.
func TestExtensionFollowsToolChanges(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	server := sdk.NewServer(&sdk.Implementation{Name: "s", Version: "1.0.0"}, nil)
	server.AddTool(&sdk.Tool{Name: "a", InputSchema: objectSchema}, replyTool("a"))
	registry := newFakeRegistry()
	memory := &inMemory{server: server}
	ext := newExtension(t, registry, ServerConnectionOptions{Name: "s", CreateTransport: memory.transport})
	startExtension(t, ctx, ext)
	if names := registry.names(); !slices.Equal(names, []string{"mcp__s__a"}) || !registry.isActive("mcp__s__a") {
		t.Fatalf("registered %v", names)
	}

	server.AddTool(&sdk.Tool{Name: "b", InputSchema: objectSchema}, replyTool("b"))
	eventually(t, "b to be registered", func() bool { return registry.isActive("mcp__s__b") })

	server.RemoveTools("a")
	eventually(t, "a to be deactivated", func() bool { return !registry.isActive("mcp__s__a") })
	if !registry.isActive("mcp__s__b") {
		t.Fatal("b was deactivated")
	}
	// A turn that still holds the tool gets a clear error.
	if _, err := callRegistered(t, ctx, registry, "mcp__s__a"); err == nil || !strings.Contains(err.Error(), "no longer offers") {
		t.Fatalf("calling a removed tool: %v", err)
	}

	server.AddTool(&sdk.Tool{Name: "a", InputSchema: objectSchema}, replyTool("a again"))
	eventually(t, "a to be activated again", func() bool { return registry.isActive("mcp__s__a") })
	if got, err := callRegistered(t, ctx, registry, "mcp__s__a"); err != nil || got != "a again" {
		t.Fatalf("a = %q, %v", got, err)
	}
	if names := registry.names(); !slices.Equal(names, []string{"mcp__s__a", "mcp__s__b"}) {
		t.Fatalf("registered %v", names)
	}
	if memory.connects() != 1 {
		t.Fatalf("%d connects", memory.connects())
	}
}

// Tools whose names sanitize alike all get hashed names, within a server
// and across servers, and keep them across reconnects.
func TestExtensionToolNameCollisions(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	first := sdk.NewServer(&sdk.Implementation{Name: "first", Version: "1.0.0"}, nil)
	first.AddTool(&sdk.Tool{Name: "a-b", InputSchema: objectSchema}, replyTool("dash"))
	first.AddTool(&sdk.Tool{Name: "a.b", InputSchema: objectSchema}, replyTool("dot"))
	first.AddTool(&sdk.Tool{Name: "z", InputSchema: objectSchema}, replyTool("x-y z"))
	second := sdk.NewServer(&sdk.Implementation{Name: "second", Version: "1.0.0"}, nil)
	second.AddTool(&sdk.Tool{Name: "z", InputSchema: objectSchema}, replyTool("x.y z"))

	registry := newFakeRegistry()
	firstMemory, secondMemory := &inMemory{server: first}, &inMemory{server: second}
	ext := newExtension(t, registry,
		ServerConnectionOptions{Name: "x-y", CreateTransport: firstMemory.transport},
		ServerConnectionOptions{Name: "x.y", CreateTransport: secondMemory.transport},
	)
	// Connect in order, so the first server takes the plain name.
	if _, err := ext.Connection("x-y").GetClient(ctx); err != nil {
		t.Fatal(err)
	}
	startExtension(t, ctx, ext)

	names := registry.names()
	if len(names) != 4 {
		t.Fatalf("registered %v", names)
	}
	byText := map[string]string{}
	for _, name := range names {
		text, err := callRegistered(t, ctx, registry, name)
		if err != nil {
			t.Fatal(err)
		}
		byText[text] = name
	}
	for _, text := range []string{"dash", "dot"} {
		if name := byText[text]; !strings.HasPrefix(name, "mcp__x_y__a_b_") {
			t.Errorf("%s tool is named %q", text, name)
		}
	}
	if byText["x-y z"] != "mcp__x_y__z" || !strings.HasPrefix(byText["x.y z"], "mcp__x_y__z_") {
		t.Fatalf("names = %v", byText)
	}

	// The hashed server reconnects first this time; names stay.
	if err := ext.Reconnect(ctx, "x.y"); err != nil {
		t.Fatal(err)
	}
	if err := ext.Reconnect(ctx, "x-y"); err != nil {
		t.Fatal(err)
	}
	if got := registry.names(); !slices.Equal(got, names) {
		t.Fatalf("after reconnect %v, before %v", got, names)
	}
	for text, name := range byText {
		if got, err := callRegistered(t, ctx, registry, name); err != nil || got != text {
			t.Fatalf("%s = %q, %v; want %q", name, got, err, text)
		}
	}
}

// The resource tools are registered once a server offers resources, and
// active only while one does.
func TestExtensionResourceTools(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	plain := sdk.NewServer(&sdk.Implementation{Name: "plain", Version: "1.0.0"}, nil)
	plain.AddTool(&sdk.Tool{Name: "find", InputSchema: objectSchema}, linkTool)
	registry := newFakeRegistry()
	memory := &inMemory{server: plain}
	ext := newExtension(t, registry, ServerConnectionOptions{Name: "docs", CreateTransport: memory.transport})
	startExtension(t, ctx, ext)
	resourceTools := []string{ListResourcesTool, ListResourceTemplatesTool, ReadResourceTool}
	if names := registry.names(); !slices.Equal(names, []string{"mcp__docs__find"}) {
		t.Fatalf("registered %v without resources", names)
	}

	memory.server = newResourceServer(nil)
	if err := ext.Reconnect(ctx, "docs"); err != nil {
		t.Fatal(err)
	}
	for _, name := range resourceTools {
		if !registry.isActive(name) {
			t.Fatalf("%s is not active", name)
		}
	}
	result, err := registry.tool(t, ReadResourceTool).Execute(ctx, "call", map[string]any{"server": "docs", "uri": "file:///a.txt"}, nil)
	if err != nil || textOf(result.Content) != "alpha" {
		t.Fatalf("read = %v, %v", result.Content, err)
	}
	if got, err := callRegistered(t, ctx, registry, "mcp__docs__find"); err != nil || !strings.Contains(got, "read_mcp_resource") {
		t.Fatalf("link = %q, %v", got, err)
	}

	memory.server = plain
	if err := ext.Reconnect(ctx, "docs"); err != nil {
		t.Fatal(err)
	}
	for _, name := range resourceTools {
		if registry.isActive(name) {
			t.Fatalf("%s is active without resources", name)
		}
	}
	if !registry.isActive("mcp__docs__find") {
		t.Fatal("the server's tool was deactivated")
	}
}

// WaitForStartup waits a bounded time, once; a slow server's tools arrive
// when it connects.
func TestExtensionStartupWait(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	server := sdk.NewServer(&sdk.Implementation{Name: "slow", Version: "1.0.0"}, nil)
	server.AddTool(&sdk.Tool{Name: "a", InputSchema: objectSchema}, replyTool("a"))
	memory := &inMemory{server: server}
	gate := make(chan struct{})
	registry := newFakeRegistry()
	ext, err := NewExtension(ExtensionOptions{
		Servers: []ServerConnectionOptions{{Name: "slow", CreateTransport: func(oauth *OAuthHandler) (sdk.Transport, error) {
			<-gate
			return memory.transport(oauth)
		}}},
		Tools:       registry,
		StartupWait: 50 * time.Millisecond,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer ext.Close(ctx)
	if !ext.WaitForStartup(ctx) {
		t.Fatal("WaitForStartup before Start waited")
	}
	ext.Start()
	// The call before Start did not use up the wait.
	start := time.Now()
	if ext.WaitForStartup(ctx) || time.Since(start) < 40*time.Millisecond {
		t.Fatalf("first WaitForStartup after Start returned after %s", time.Since(start))
	}
	start = time.Now()
	if ext.WaitForStartup(ctx) || time.Since(start) > 40*time.Millisecond {
		t.Fatal("the second WaitForStartup waited")
	}
	close(gate)
	eventually(t, "the slow server's tools", func() bool { return registry.isActive("mcp__slow__a") })
	eventually(t, "startup", func() bool { return ext.WaitForStartup(ctx) })
}

func TestNewExtensionValidates(t *testing.T) {
	transport := (&inMemory{}).transport
	if _, err := NewExtension(ExtensionOptions{}); err == nil {
		t.Fatal("extension without a registry")
	}
	if _, err := NewExtension(ExtensionOptions{
		Tools:   newFakeRegistry(),
		Servers: []ServerConnectionOptions{{Name: "a", CreateTransport: transport}, {Name: "a", CreateTransport: transport}},
	}); err == nil || !strings.Contains(err.Error(), `two servers are named "a"`) {
		t.Fatalf("duplicate names: %v", err)
	}
	ext := newExtension(t, newFakeRegistry(), ServerConnectionOptions{Name: "a", CreateTransport: transport})
	ctx := context.Background()
	if ext.Connection("b") != nil || len(ext.Connections()) != 1 {
		t.Fatal("unexpected connections")
	}
	if err := ext.Reconnect(ctx, "b"); err == nil {
		t.Fatal("reconnected an unknown server")
	}
	if err := ext.SignIn(ctx, "a", SignInOptions{}); err == nil || !strings.Contains(err.Error(), "does not use OAuth") {
		t.Fatalf("sign-in without OAuth: %v", err)
	}
	if err := ext.Logout(ctx, "a"); err == nil {
		t.Fatal("logged out of a server without OAuth")
	}
}
