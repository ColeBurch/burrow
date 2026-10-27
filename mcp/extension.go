// Adapted from pi's packages/coding-agent/src/extensions/mcp/index.ts
// (MIT, see LICENSE).

package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"sync"
	"time"

	"github.com/ColeBurch/burrow/agent"
	"github.com/ColeBurch/burrow/baseAgent"
	sdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

// DefaultStartupWait is how long WaitForStartup waits for servers still
// connecting when ExtensionOptions.StartupWait is zero.
const DefaultStartupWait = 10 * time.Second

// ToolRegistry is where an [Extension] registers tools.
// [baseAgent.AgentSession] implements it: a tool registered during a prompt
// is declared from the prompt's next turn.
type ToolRegistry interface {
	// RegisterTool adds a tool, active, or replaces the tool of that name,
	// keeping whether it is active.
	RegisterTool(tool agent.AgentTool[any, any]) error
	GetActiveToolNames() ([]string, error)
	// SetActiveToolsByName activates the named tools and deactivates the
	// rest, ignoring unknown names.
	SetActiveToolsByName(names []string) error
}

var _ ToolRegistry = (*baseAgent.AgentSession)(nil)

// ExtensionOptions configures an [Extension].
type ExtensionOptions struct {
	// Servers are the servers to connect, with distinct names. The
	// extension sets their OnTools; their OnChange is called after the
	// extension's.
	Servers []ServerConnectionOptions
	// Tools receives the servers' tools and the resource tools. Required.
	Tools ToolRegistry
	// StartupWait is how long WaitForStartup waits for servers still
	// connecting. Zero means DefaultStartupWait.
	StartupWait time.Duration
}

// Extension connects an agent session to MCP servers, as pi's MCP
// extension does. Each server's tools are registered when it connects and
// again when they change. Names stay stable for the session, since its
// history calls tools by name. Tools a server no longer offers are
// deactivated, and activated again if it offers them again. The resource
// tools are registered once a server offers resources.
//
// Wire it into a session:
//
//	ext, err := mcp.NewExtension(mcp.ExtensionOptions{Servers: servers, Tools: session})
//	ext.Start()
//	defer ext.Close(ctx)
//	// Before each prompt:
//	ext.WaitForStartup(ctx)
//	ext.ReconnectSignedIn(ctx)
//	session.Prompt(ctx, text, nil)
type Extension struct {
	tools       ToolRegistry
	startupWait time.Duration
	connections []*ServerConnection
	// ctx is the extension's lifetime, cancelled by Close.
	ctx    context.Context
	cancel context.CancelFunc
	// startup is closed when the connects that Start began finished.
	startup chan struct{}

	mu sync.Mutex
	// toolOwners maps each tool name to the "<server>\x00<tool>" it was
	// assigned to, so names stay unique and stable.
	toolOwners map[string]string
	// serverTools are the tool names each server currently offers.
	serverTools map[string]map[string]bool
	// dropped are tool names deactivated because their server stopped
	// offering them.
	dropped map[string]bool
	// resourceTools records whether the resource tools are registered, and
	// active.
	resourceToolsRegistered, resourceToolsActive bool
	// tokensAtSignIn are the stored tokens of servers that need a sign-in,
	// as they were when it was needed.
	tokensAtSignIn   map[*ServerConnection]string
	started          bool
	waitedForStartup bool
}

// NewExtension creates the servers' connections. Call Start to connect
// them.
func NewExtension(opts ExtensionOptions) (*Extension, error) {
	if opts.Tools == nil {
		return nil, errors.New("mcp: ExtensionOptions.Tools is required")
	}
	e := &Extension{
		tools:          opts.Tools,
		startupWait:    opts.StartupWait,
		startup:        make(chan struct{}),
		toolOwners:     map[string]string{},
		serverTools:    map[string]map[string]bool{},
		dropped:        map[string]bool{},
		tokensAtSignIn: map[*ServerConnection]string{},
	}
	if e.startupWait == 0 {
		e.startupWait = DefaultStartupWait
	}
	e.ctx, e.cancel = context.WithCancel(context.Background())
	names := map[string]bool{}
	for _, server := range opts.Servers {
		if names[server.Name] {
			return nil, fmt.Errorf("mcp: two servers are named %q", server.Name)
		}
		names[server.Name] = true
		onChange := server.OnChange
		server.OnTools = e.registerTools
		server.OnChange = func(connection *ServerConnection) {
			e.onConnectionChange(connection)
			if onChange != nil {
				onChange(connection)
			}
		}
		connection, err := NewServerConnection(server)
		if err != nil {
			return nil, err
		}
		e.connections = append(e.connections, connection)
	}
	return e, nil
}

// Connections are the servers' connections, in the order configured.
func (e *Extension) Connections() []*ServerConnection {
	return slices.Clone(e.connections)
}

// Connection is the connection of the server named name, or nil.
func (e *Extension) Connection(name string) *ServerConnection {
	for _, connection := range e.connections {
		if connection.Name() == name {
			return connection
		}
	}
	return nil
}

// Start connects every server in the background. Failures show in each
// connection's Status. Calling it again does nothing.
func (e *Extension) Start() {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.started {
		return
	}
	e.started = true
	var wg sync.WaitGroup
	for _, connection := range e.connections {
		wg.Go(func() { _, _ = connection.GetClient(e.ctx) })
	}
	go func() {
		wg.Wait()
		close(e.startup)
	}()
}

// WaitForStartup lets the first prompt wait for the servers Start is still
// connecting, so their tools are declared in its first request, but not
// indefinitely: a slow or hanging server must not hold up the prompt. Its
// tools become available once it connects. Later calls return at once. It
// reports whether every server finished connecting.
func (e *Extension) WaitForStartup(ctx context.Context) bool {
	e.mu.Lock()
	started, waited := e.started, e.waitedForStartup
	// A call before Start does not use up the wait.
	e.waitedForStartup = started
	e.mu.Unlock()
	if !started {
		return true
	}
	if waited {
		select {
		case <-e.startup:
			return true
		default:
			return false
		}
	}
	timer := time.NewTimer(e.startupWait)
	defer timer.Stop()
	select {
	case <-e.startup:
		return true
	case <-timer.C:
	case <-ctx.Done():
	}
	return false
}

// ReconnectSignedIn reconnects the servers that need a sign-in when their
// credentials were stored since, for example by CompleteSignIn in another
// request or process. Call it before each prompt or turn.
func (e *Extension) ReconnectSignedIn(ctx context.Context) error {
	e.mu.Lock()
	waiting := make(map[*ServerConnection]string, len(e.tokensAtSignIn))
	for connection, tokens := range e.tokensAtSignIn {
		waiting[connection] = tokens
	}
	e.mu.Unlock()
	var signedIn []*ServerConnection
	for connection, tokens := range waiting {
		current, err := storedTokens(ctx, connection)
		if err != nil {
			return err
		}
		if current != tokens {
			signedIn = append(signedIn, connection)
		}
	}
	if len(signedIn) == 0 {
		return nil
	}
	e.mu.Lock()
	for _, connection := range signedIn {
		delete(e.tokensAtSignIn, connection)
	}
	e.mu.Unlock()
	var wg sync.WaitGroup
	for _, connection := range signedIn {
		// Failures show in the connection's status.
		wg.Go(func() { _ = connection.Reconnect(ctx) })
	}
	wg.Wait()
	return nil
}

// Reconnect reconnects the server named name.
func (e *Extension) Reconnect(ctx context.Context, name string) error {
	connection := e.Connection(name)
	if connection == nil {
		return fmt.Errorf("mcp: no server is named %q", name)
	}
	return connection.Reconnect(ctx)
}

// SignIn signs the user in to the server named name with [SignInMcpServer],
// then reconnects. Closing the extension cancels it.
func (e *Extension) SignIn(ctx context.Context, name string, opts SignInOptions) error {
	connection := e.Connection(name)
	if connection == nil {
		return fmt.Errorf("mcp: no server is named %q", name)
	}
	if connection.OAuth() == nil {
		return fmt.Errorf("mcp: server %q does not use OAuth", name)
	}
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	stop := context.AfterFunc(e.ctx, cancel)
	defer stop()
	if err := SignInMcpServer(ctx, connection.OAuth(), opts); err != nil {
		return err
	}
	if err := connection.Reconnect(ctx); err != nil {
		return fmt.Errorf("signed in, but %w", err)
	}
	return nil
}

// Logout deletes the stored credentials of the server named name and
// disconnects it.
func (e *Extension) Logout(ctx context.Context, name string) error {
	connection := e.Connection(name)
	if connection == nil {
		return fmt.Errorf("mcp: no server is named %q", name)
	}
	if connection.OAuth() == nil {
		return fmt.Errorf("mcp: server %q does not use OAuth", name)
	}
	if err := connection.OAuth().SignOut(ctx); err != nil {
		return err
	}
	return connection.SignOut(ctx)
}

// Close closes every connection. See [ServerConnection.Close].
func (e *Extension) Close(ctx context.Context) error {
	e.cancel()
	errs := make([]error, len(e.connections))
	var wg sync.WaitGroup
	for i, connection := range e.connections {
		wg.Go(func() { errs[i] = connection.Close(ctx) })
	}
	wg.Wait()
	return errors.Join(errs...)
}

// registerTools registers the tools connection lists. Like Codex, all tools
// whose names sanitize to the same name get the hash suffix, so which one
// would keep the plain name does not depend on the order of the list.
func (e *Extension) registerTools(connection *ServerConnection) {
	if e.ctx.Err() != nil {
		return
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	server := connection.Name()
	tools := connection.Tools()
	previous := e.serverTools[server]
	current := map[string]bool{}
	plain := map[string]int{}
	seen := map[string]bool{}
	for _, tool := range tools {
		if !seen[tool.Name] {
			seen[tool.Name] = true
			plain[CreateToolName(server, tool.Name, nil)]++
		}
	}
	var returned []string
	for _, tool := range tools {
		owner := server + "\x00" + tool.Name
		name := CreateToolName(server, tool.Name, func(candidate string) bool {
			existing, ok := e.toolOwners[candidate]
			return (ok && existing != owner) || current[candidate] || plain[candidate] > 1
		})
		e.toolOwners[name] = owner
		current[name] = true
		toolName := tool.Name
		// Registration fails only without an agent or a name, which cannot
		// happen here.
		_ = e.tools.RegisterTool(CreateToolDefinition(ToolDefinitionOptions{
			Server:  server,
			Tool:    tool,
			Name:    name,
			Timeout: connection.Timeout(),
			GetClient: func(context.Context) (ToolCaller, error) {
				if connection.Status().State == StateConnected && !offers(connection, toolName) {
					return nil, errors.New("the server no longer offers this tool")
				}
				return connection, nil
			},
			ReadableResources: connection.HasResources,
		}))
		if e.dropped[name] {
			delete(e.dropped, name)
			returned = append(returned, name)
		}
	}
	e.serverTools[server] = current
	// Tools cannot be unregistered, so tools the server dropped are
	// deactivated. When it offers them again they are activated above.
	var dropped []string
	for name := range previous {
		if !current[name] {
			e.dropped[name] = true
			dropped = append(dropped, name)
		}
	}
	e.updateActive(returned, dropped)
	e.syncResourceTools()
}

func offers(connection *ServerConnection, tool string) bool {
	return slices.ContainsFunc(connection.Tools(), func(offered *sdk.Tool) bool { return offered.Name == tool })
}

// resourceServers are the servers with resources, which the resource tools
// reach.
func (e *Extension) resourceServers() []ResourceServer {
	var servers []ResourceServer
	for _, connection := range e.connections {
		if connection.HasResources() {
			servers = append(servers, connection)
		}
	}
	return servers
}

// syncResourceTools registers the resource tools once a server offers
// resources, and keeps them active only while one does. e.mu is held.
func (e *Extension) syncResourceTools() {
	has := len(e.resourceServers()) > 0
	if !e.resourceToolsRegistered {
		if !has {
			return
		}
		for _, tool := range CreateResourceToolDefinitions(e.resourceServers) {
			_ = e.tools.RegisterTool(tool)
		}
		e.resourceToolsRegistered, e.resourceToolsActive = true, true
		return
	}
	if has == e.resourceToolsActive {
		return
	}
	e.resourceToolsActive = has
	names := []string{ListResourcesTool, ListResourceTemplatesTool, ReadResourceTool}
	if has {
		e.updateActive(names, nil)
	} else {
		e.updateActive(nil, names)
	}
}

// updateActive activates add and deactivates remove, keeping the rest.
func (e *Extension) updateActive(add, remove []string) {
	if len(add) == 0 && len(remove) == 0 {
		return
	}
	active, err := e.tools.GetActiveToolNames()
	if err != nil {
		return
	}
	next := slices.DeleteFunc(active, func(name string) bool { return slices.Contains(remove, name) })
	for _, name := range add {
		if !slices.Contains(next, name) {
			next = append(next, name)
		}
	}
	_ = e.tools.SetActiveToolsByName(next)
}

// onConnectionChange records the stored tokens of a server that started
// needing a sign-in, so ReconnectSignedIn notices when they change.
func (e *Extension) onConnectionChange(connection *ServerConnection) {
	if connection.Status().State != StateNeedsAuth {
		e.mu.Lock()
		delete(e.tokensAtSignIn, connection)
		e.mu.Unlock()
		return
	}
	e.mu.Lock()
	_, ok := e.tokensAtSignIn[connection]
	e.mu.Unlock()
	if ok || e.ctx.Err() != nil {
		return
	}
	ctx, cancel := context.WithTimeout(e.ctx, connection.Timeout())
	defer cancel()
	tokens, err := storedTokens(ctx, connection)
	if err != nil {
		return
	}
	e.mu.Lock()
	if _, ok := e.tokensAtSignIn[connection]; !ok {
		e.tokensAtSignIn[connection] = tokens
	}
	e.mu.Unlock()
}

// storedTokens are the server's stored tokens as JSON, "null" when there
// are none or the server does not use OAuth.
func storedTokens(ctx context.Context, connection *ServerConnection) (string, error) {
	if connection.OAuth() == nil {
		return "null", nil
	}
	state, err := connection.OAuth().load(ctx)
	if err != nil {
		return "", err
	}
	var tokens *oauthTokens
	if state != nil {
		tokens = state.Tokens
	}
	data, err := json.Marshal(tokens)
	return string(data), err
}
