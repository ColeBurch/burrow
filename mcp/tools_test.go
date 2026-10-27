package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/ColeBurch/burrow/agent"
	"github.com/ColeBurch/burrow/ai"
	sdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

func TestCreateToolName(t *testing.T) {
	if got := CreateToolName("docs", "search", nil); got != "mcp__docs__search" {
		t.Fatalf("CreateToolName = %q", got)
	}
	if got := CreateToolName("my-server", "get.page", nil); got != "mcp__my_server__get_page" {
		t.Fatalf("sanitized CreateToolName = %q", got)
	}

	long := CreateToolName("server", strings.Repeat("x", 80), nil)
	if len(long) != maxToolNameLength {
		t.Fatalf("long name has length %d: %q", len(long), long)
	}
	if long != CreateToolName("server", strings.Repeat("x", 80), nil) {
		t.Fatal("long names are not stable")
	}
	if long == CreateToolName("server", strings.Repeat("x", 81), nil) {
		t.Fatal("different long tools got the same name")
	}

	taken := func(name string) bool { return name == "mcp__s__a_b" }
	got := CreateToolName("s", "a-b", taken)
	if !strings.HasPrefix(got, "mcp__s__a_b_") || len(got) != len("mcp__s__a_b_")+8 {
		t.Fatalf("taken name = %q", got)
	}
}

func TestToolParameters(t *testing.T) {
	got := toolParameters(map[string]any{"required": []any{"q"}})
	if got["type"] != "object" {
		t.Fatalf("type = %v", got["type"])
	}
	if _, ok := got["properties"].(map[string]any); !ok {
		t.Fatalf("properties = %#v", got["properties"])
	}
	if got := toolParameters(nil); got["type"] != "object" {
		t.Fatalf("nil schema = %#v", got)
	}
	typed := toolParameters(json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}}}`))
	if _, ok := typed["properties"].(map[string]any)["q"]; !ok {
		t.Fatalf("raw schema = %#v", typed)
	}
}

func TestConvertResultContent(t *testing.T) {
	size := int64(2048)
	result := &sdk.CallToolResult{Content: []sdk.Content{
		&sdk.TextContent{Text: "hello"},
		&sdk.ImageContent{Data: []byte("png"), MIMEType: "image/png"},
		&sdk.AudioContent{Data: []byte("wav"), MIMEType: "audio/wav"},
		&sdk.ResourceLink{URI: "file:///a.txt", Name: "a.txt", MIMEType: "text/plain", Size: &size, Description: "notes"},
		&sdk.ResourceLink{URI: "file:///b", Name: "b", Title: "Bee"},
		&sdk.EmbeddedResource{Resource: &sdk.ResourceContents{URI: "file:///t", Text: "embedded text"}},
		&sdk.EmbeddedResource{Resource: &sdk.ResourceContents{URI: "file:///j", MIMEType: "application/json", Blob: []byte(`{"a":1}`)}},
		&sdk.EmbeddedResource{Resource: &sdk.ResourceContents{URI: "file:///i", MIMEType: "image/jpeg", Blob: []byte("jpg")}},
		&sdk.EmbeddedResource{Resource: &sdk.ResourceContents{URI: "file:///z", Blob: []byte("zip")}},
	}}
	got := ConvertResult("s", "t", result, nil).Content
	want := []ai.ToolResultContent{
		ai.TextContent{Type: ai.ContentTypeText, Text: "hello"},
		ai.ImageContent{Type: ai.ContentTypeImage, Image: "cG5n", MimeType: "image/png"},
		ai.TextContent{Type: ai.ContentTypeText, Text: "[audio audio/wav omitted]"},
		ai.TextContent{Type: ai.ContentTypeText, Text: `[Resource file:///a.txt "a.txt" (text/plain, 2.0KB): notes]`},
		ai.TextContent{Type: ai.ContentTypeText, Text: `[Resource file:///b "Bee"]`},
		ai.TextContent{Type: ai.ContentTypeText, Text: "embedded text"},
		ai.TextContent{Type: ai.ContentTypeText, Text: `{"a":1}`},
		ai.ImageContent{Type: ai.ContentTypeImage, Image: "anBn", MimeType: "image/jpeg"},
		ai.TextContent{Type: ai.ContentTypeText, Text: "[Binary resource file:///z (unknown type, 3B) omitted]"},
	}
	if len(got) != len(want) {
		t.Fatalf("got %d blocks, want %d: %#v", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("block %d = %#v, want %#v", i, got[i], want[i])
		}
	}

	readable := ConvertResult("s", "t", result, &ConvertResultOptions{ReadableResources: true}).Content
	if text := readable[4].(ai.TextContent).Text; text != `[Resource file:///b "Bee". Read it with read_mcp_resource (server "s")]` {
		t.Fatalf("readable link = %q", text)
	}
}

func TestConvertResultStructuredAndErrors(t *testing.T) {
	structured := ConvertResult("s", "t", &sdk.CallToolResult{StructuredContent: map[string]any{"n": 1}}, nil)
	if text := textOf(structured.Content); text != "{\n  \"n\": 1\n}" {
		t.Fatalf("structured text = %q", text)
	}
	if details := structured.Details.(ToolDetails); details.StructuredContent == nil || details.Server != "s" || details.Tool != "t" {
		t.Fatalf("details = %#v", details)
	}

	failed := ConvertResult("s", "t", &sdk.CallToolResult{IsError: true}, nil)
	if text := textOf(failed.Content); text != "MCP tool s/t returned an error" || !failed.IsError {
		t.Fatalf("error result = %#v", failed)
	}
	withText := ConvertResult("s", "t", &sdk.CallToolResult{IsError: true, Content: []sdk.Content{&sdk.TextContent{Text: "bad input"}}}, nil)
	if text := textOf(withText.Content); text != "bad input" {
		t.Fatalf("error text = %q", text)
	}
}

func TestConvertResultTruncatesTextAndKeepsImages(t *testing.T) {
	long := strings.Repeat("a", 15000) + "\n" + strings.Repeat("b", 15000) + "\n"
	result := &sdk.CallToolResult{Content: []sdk.Content{
		&sdk.TextContent{Text: long},
		&sdk.ImageContent{Data: []byte("png"), MIMEType: "image/png"},
	}}
	got := ConvertResult("s", "t", result, nil).Content
	if len(got) != 2 {
		t.Fatalf("got %#v", got)
	}
	text := got[0].(ai.TextContent).Text
	wantPrefix := "Warning: truncated output (original token count: 7501)\nTotal output lines: 2\n\n" + strings.Repeat("a", OutputMaxBytes/2) + "…9522 chars truncated…b"
	if !strings.HasPrefix(text, wantPrefix) || !strings.HasSuffix(text, strings.Repeat("b", OutputMaxBytes/2-1)+"\n") {
		t.Fatalf("truncated text = %q…%q", text[:120], text[len(text)-40:])
	}
	if _, ok := got[1].(ai.ImageContent); !ok {
		t.Fatalf("image dropped: %#v", got[1])
	}
	short := &sdk.CallToolResult{Content: []sdk.Content{&sdk.TextContent{Text: "short"}, &sdk.ImageContent{Data: []byte("png"), MIMEType: "image/png"}}}
	if got := ConvertResult("s", "t", short, nil).Content; len(got) != 2 || got[0].(ai.TextContent).Text != "short" {
		t.Fatalf("short text changed: %#v", got)
	}
}

func TestTruncateMiddleCutsAtCharacterBoundaries(t *testing.T) {
	s := strings.Repeat("é", 20) // 2 bytes each
	got := truncateMiddle(s, 11)
	if !utf8.ValidString(got) {
		t.Fatalf("invalid UTF-8: %q", got)
	}
	// Head keeps 4 bytes (2 runes), tail keeps 6 bytes (3 runes).
	if got != "éé…15 chars truncated…ééé" {
		t.Fatalf("truncateMiddle = %q", got)
	}
}

// connect serves server over in-memory transports and returns the client
// session.
func connect(t *testing.T, server *sdk.Server, opts *sdk.ClientOptions) *sdk.ClientSession {
	t.Helper()
	ctx := context.Background()
	clientTransport, serverTransport := sdk.NewInMemoryTransports()
	serverSession, err := server.Connect(ctx, serverTransport, nil)
	if err != nil {
		t.Fatal(err)
	}
	client := sdk.NewClient(&sdk.Implementation{Name: "test-client", Version: "1.0.0"}, opts)
	session, err := client.Connect(ctx, clientTransport, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		session.Close()
		serverSession.Wait()
	})
	return session
}

// inMemory serves server over new in-memory transports at each connection
// attempt, and records the server sessions.
type inMemory struct {
	server   *sdk.Server
	mu       sync.Mutex
	sessions []*sdk.ServerSession
}

func (m *inMemory) transport(*OAuthHandler) (sdk.Transport, error) {
	clientTransport, serverTransport := sdk.NewInMemoryTransports()
	session, err := m.server.Connect(context.Background(), serverTransport, nil)
	if err != nil {
		return nil, err
	}
	m.mu.Lock()
	m.sessions = append(m.sessions, session)
	m.mu.Unlock()
	return clientTransport, nil
}

func (m *inMemory) connects() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.sessions)
}

// newConnection returns a connection to server named opts.Name, "s" when
// empty, and closes it when the test ends.
func newConnection(t *testing.T, server *sdk.Server, opts ServerConnectionOptions) (*ServerConnection, *inMemory) {
	t.Helper()
	memory := &inMemory{server: server}
	if opts.Name == "" {
		opts.Name = "s"
	}
	if opts.CreateTransport == nil {
		opts.CreateTransport = memory.transport
	}
	connection, err := NewServerConnection(opts)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := connection.Close(ctx); err != nil {
			t.Errorf("close: %v", err)
		}
	})
	return connection, memory
}

// connectionTools are the tool definitions of connection's tools.
func connectionTools(t *testing.T, ctx context.Context, connection *ServerConnection) map[string]agent.AgentTool[any, any] {
	t.Helper()
	if _, err := connection.GetClient(ctx); err != nil {
		t.Fatal(err)
	}
	tools := map[string]agent.AgentTool[any, any]{}
	for _, tool := range connection.Tools() {
		name := CreateToolName(connection.Name(), tool.Name, nil)
		tools[name] = CreateToolDefinition(ToolDefinitionOptions{
			Server:            connection.Name(),
			Tool:              tool,
			Name:              name,
			Timeout:           connection.Timeout(),
			GetClient:         func(context.Context) (ToolCaller, error) { return connection, nil },
			ReadableResources: connection.HasResources,
		})
	}
	return tools
}

// fakeRegistry is a ToolRegistry like AgentSession's.
type fakeRegistry struct {
	mu     sync.Mutex
	tools  map[string]agent.AgentTool[any, any]
	order  []string
	active []string
}

func newFakeRegistry() *fakeRegistry {
	return &fakeRegistry{tools: map[string]agent.AgentTool[any, any]{}}
}

func (r *fakeRegistry) RegisterTool(tool agent.AgentTool[any, any]) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if tool.Name == "" {
		return errors.New("tool name is required")
	}
	if _, ok := r.tools[tool.Name]; !ok {
		r.order = append(r.order, tool.Name)
		r.active = append(r.active, tool.Name)
	}
	r.tools[tool.Name] = tool
	return nil
}

func (r *fakeRegistry) GetActiveToolNames() ([]string, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return slices.Clone(r.active), nil
}

func (r *fakeRegistry) SetActiveToolsByName(names []string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.active = nil
	for _, name := range names {
		if _, ok := r.tools[name]; ok && !slices.Contains(r.active, name) {
			r.active = append(r.active, name)
		}
	}
	return nil
}

func (r *fakeRegistry) tool(t *testing.T, name string) agent.AgentTool[any, any] {
	t.Helper()
	r.mu.Lock()
	defer r.mu.Unlock()
	tool, ok := r.tools[name]
	if !ok {
		t.Fatalf("no tool %s in %v", name, r.order)
	}
	return tool
}

func (r *fakeRegistry) isActive(name string) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return slices.Contains(r.active, name)
}

func (r *fakeRegistry) names() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return slices.Clone(r.order)
}

// eventually fails the test unless cond holds within 5 seconds.
func eventually(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

var objectSchema = map[string]any{"type": "object", "properties": map[string]any{"q": map[string]any{"type": "string"}}}

func textResult(s string) *sdk.CallToolResult {
	return &sdk.CallToolResult{Content: []sdk.Content{&sdk.TextContent{Text: s}}}
}

func TestToolDefinitionCallsServer(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	server := sdk.NewServer(&sdk.Implementation{Name: "test-server", Version: "1.0.0"}, nil)
	server.AddTool(&sdk.Tool{Name: "echo", Description: "  Echo the query.  ", InputSchema: objectSchema},
		func(ctx context.Context, req *sdk.CallToolRequest) (*sdk.CallToolResult, error) {
			var args struct{ Q string }
			if err := json.Unmarshal(req.Params.Arguments, &args); err != nil {
				return nil, err
			}
			return textResult("echo: " + args.Q), nil
		})
	server.AddTool(&sdk.Tool{Name: "fail", Title: "Always fails", InputSchema: objectSchema},
		func(ctx context.Context, req *sdk.CallToolRequest) (*sdk.CallToolResult, error) {
			result := textResult("no such page")
			result.IsError = true
			return result, nil
		})

	connection, _ := newConnection(t, server, ServerConnectionOptions{Name: "docs"})
	byName := connectionTools(t, ctx, connection)
	echo, ok := byName["mcp__docs__echo"]
	if !ok {
		t.Fatalf("tools = %v", byName)
	}
	if echo.Description != "Echo the query." || echo.Label != "docs/echo" || echo.Parameters["type"] != "object" || echo.Annotations != nil {
		t.Fatalf("echo tool = %#v", echo)
	}
	if byName["mcp__docs__fail"].Description != "Always fails" {
		t.Fatalf("fail description = %q", byName["mcp__docs__fail"].Description)
	}

	result, err := echo.Execute(ctx, "call-1", map[string]any{"q": "hi"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if text := textOf(result.Content); text != "echo: hi" {
		t.Fatalf("echo result = %q", text)
	}
	// Parameters that are not a map are sent as their JSON object.
	result, err = echo.Execute(ctx, "call-2", struct {
		Q string `json:"q"`
	}{"typed"}, nil)
	if err != nil || textOf(result.Content) != "echo: typed" {
		t.Fatalf("typed parameters: %#v, %v", result, err)
	}

	failed, err := byName["mcp__docs__fail"].Execute(ctx, "call-3", map[string]any{}, nil)
	if err != nil || !failed.IsError || textOf(failed.Content) != "no such page" || failed.Details.(ToolDetails).Tool != "fail" {
		t.Fatalf("fail result = %#v, %v", failed, err)
	}

	broken := CreateToolDefinition(ToolDefinitionOptions{
		Server: "docs", Tool: &sdk.Tool{Name: "gone"}, Name: "mcp__docs__gone",
		GetClient: func(context.Context) (ToolCaller, error) { return nil, errors.New("not offered") },
	})
	if _, err := broken.Execute(ctx, "call-4", nil, nil); err == nil || err.Error() != "MCP tool docs/gone: not offered" {
		t.Fatalf("GetClient error = %v", err)
	}
}

func TestToolAnnotations(t *testing.T) {
	no, yes := false, true
	for _, tc := range []struct {
		hints *sdk.ToolAnnotations
		want  *agent.ToolAnnotations
	}{
		{nil, nil},
		{&sdk.ToolAnnotations{Title: "only a title"}, nil},
		{&sdk.ToolAnnotations{ReadOnlyHint: true, IdempotentHint: true}, &agent.ToolAnnotations{ReadOnlyHint: &yes, IdempotentHint: &yes}},
		{&sdk.ToolAnnotations{DestructiveHint: &no, OpenWorldHint: &yes}, &agent.ToolAnnotations{DestructiveHint: &no, OpenWorldHint: &yes}},
	} {
		got := toolAnnotations(&sdk.Tool{Name: "t", Annotations: tc.hints})
		gotJSON, _ := json.Marshal(got)
		wantJSON, _ := json.Marshal(tc.want)
		if string(gotJSON) != string(wantJSON) {
			t.Errorf("annotations of %#v = %s, want %s", tc.hints, gotJSON, wantJSON)
		}
	}
	// The hints are copies, not the server's.
	destructive := true
	tool := &sdk.Tool{Name: "t", Annotations: &sdk.ToolAnnotations{DestructiveHint: &destructive}}
	got := toolAnnotations(tool)
	destructive = false
	if !*got.DestructiveHint {
		t.Fatal("annotations alias the tool's hints")
	}
}

func TestToolProgressAndTimeout(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	updated := make(chan agent.AgentToolResult[any], 1)
	server := sdk.NewServer(&sdk.Implementation{Name: "test-server", Version: "1.0.0"}, nil)
	server.AddTool(&sdk.Tool{Name: "slow", InputSchema: objectSchema},
		func(ctx context.Context, req *sdk.CallToolRequest) (*sdk.CallToolResult, error) {
			err := req.Session.NotifyProgress(ctx, &sdk.ProgressNotificationParams{
				ProgressToken: req.Params.GetProgressToken(),
				Progress:      1,
				Total:         2,
			})
			if err != nil {
				return nil, err
			}
			// Answer once the client has seen the progress, so the update
			// arrives while the call runs.
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(5 * time.Second):
				return nil, errors.New("progress not delivered")
			case result := <-updated:
				updated <- result
			}
			return textResult("done"), nil
		})
	server.AddTool(&sdk.Tool{Name: "hang", InputSchema: objectSchema},
		func(ctx context.Context, req *sdk.CallToolRequest) (*sdk.CallToolResult, error) {
			<-ctx.Done()
			return nil, ctx.Err()
		})

	connection, _ := newConnection(t, server, ServerConnectionOptions{Timeout: 200 * time.Millisecond})
	tools := connectionTools(t, ctx, connection)
	slow, hang := tools["mcp__s__slow"], tools["mcp__s__hang"]

	result, err := slow.Execute(ctx, "call-1", nil, func(partial agent.AgentToolResult[any]) { updated <- partial })
	if err != nil {
		t.Fatal(err)
	}
	if text := textOf(result.Content); text != "done" {
		t.Fatalf("slow result = %q", text)
	}
	update := <-updated
	if text := textOf(update.Content); text != "Progress 1/2" {
		t.Fatalf("progress text = %q", text)
	}
	if details := update.Details.(ToolDetails); details.Server != "s" || details.Tool != "slow" {
		t.Fatalf("progress details = %#v", details)
	}
	connection.progress.mu.Lock()
	left := len(connection.progress.calls)
	connection.progress.mu.Unlock()
	if left != 0 {
		t.Fatalf("%d progress calls left registered", left)
	}

	start := time.Now()
	_, err = hang.Execute(ctx, "call-2", nil, nil)
	if err == nil || !strings.Contains(err.Error(), "timed out after 200ms") || !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("hang error = %v", err)
	}
	if elapsed := time.Since(start); elapsed > 5*time.Second {
		t.Fatalf("timeout took %s", elapsed)
	}
	// The connection survives a timed-out call.
	if result, err := slow.Execute(ctx, "call-3", nil, func(partial agent.AgentToolResult[any]) { updated <- partial }); err != nil || textOf(result.Content) != "done" {
		t.Fatalf("call after timeout: %#v, %v", result, err)
	}
}

// Progress restarts a call's timeout, as in pi: a tool that keeps reporting
// progress runs past it, even when the caller takes no updates, while one
// that stops reporting times out.
func TestToolProgressKeepsCallAlive(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	const timeout = 300 * time.Millisecond
	// report sends progress every 50ms for d, then answers, or with stall
	// waits for the call to be cancelled.
	report := func(d time.Duration, stall bool) sdk.ToolHandler {
		return func(ctx context.Context, req *sdk.CallToolRequest) (*sdk.CallToolResult, error) {
			token := req.Params.GetProgressToken()
			if token == nil {
				return nil, errors.New("the call did not ask for progress")
			}
			for i := 1; time.Duration(i)*50*time.Millisecond <= d; i++ {
				select {
				case <-ctx.Done():
					return nil, ctx.Err()
				case <-time.After(50 * time.Millisecond):
				}
				if err := req.Session.NotifyProgress(ctx, &sdk.ProgressNotificationParams{ProgressToken: token, Progress: float64(i)}); err != nil {
					return nil, err
				}
			}
			if stall {
				<-ctx.Done()
				return nil, ctx.Err()
			}
			return textResult("done"), nil
		}
	}
	server := sdk.NewServer(&sdk.Implementation{Name: "test-server", Version: "1.0.0"}, nil)
	server.AddTool(&sdk.Tool{Name: "long", InputSchema: objectSchema}, report(4*timeout, false))
	server.AddTool(&sdk.Tool{Name: "stalls", InputSchema: objectSchema}, report(timeout/2, true))
	connection, _ := newConnection(t, server, ServerConnectionOptions{Timeout: timeout})
	tools := connectionTools(t, ctx, connection)

	start := time.Now()
	result, err := tools["mcp__s__long"].Execute(ctx, "call-1", nil, nil)
	if err != nil {
		t.Fatalf("long call reporting progress: %v", err)
	}
	if text := textOf(result.Content); text != "done" || time.Since(start) < 4*timeout {
		t.Fatalf("result %q after %s", text, time.Since(start))
	}

	start = time.Now()
	_, err = tools["mcp__s__stalls"].Execute(ctx, "call-2", nil, nil)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("stalled call: %v", err)
	}
	// It timed out a timeout after its last progress, not after the start.
	if elapsed := time.Since(start); elapsed < timeout+timeout/2-20*time.Millisecond {
		t.Fatalf("stalled call timed out after %s", elapsed)
	}
}
