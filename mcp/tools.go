// Adapted from pi's packages/coding-agent/src/extensions/mcp/tools.ts,
// packages/mcp/src/protocol/content.ts and truncateMiddle in
// packages/coding-agent/src/core/tools/truncate.ts (MIT, see LICENSE).

// Package mcp connects burrow agents to Model Context Protocol servers, as
// pi's MCP extension does. The protocol and transports come from
// github.com/modelcontextprotocol/go-sdk; this package keeps one
// [ServerConnection] per server (connected on first use, reconnected when a
// call finds it gone), registers the servers' tools on an agent session
// through an [Extension], and keeps OAuth credentials in a
// [baseAgent.AuthStore].
//
// Wiring an agent session to remote servers for one user:
//
//	handler, err := mcp.NewOAuthHandler(&mcp.OAuthOptions{
//		Store: store, OwnerID: userID, Server: "docs", ServerURL: serverURL,
//		HTTPClient: guarded, RedirectURL: "https://app.example/mcp/callback",
//	})
//	ext, err := mcp.NewExtension(mcp.ExtensionOptions{
//		Servers: []mcp.ServerConnectionOptions{{
//			Name: "docs", OAuth: handler,
//			CreateTransport: mcp.HTTPTransport(serverURL, &mcp.HTTPTransportOptions{HTTPClient: guarded}),
//		}},
//		Tools: session, // a *baseAgent.AgentSession
//	})
//	ext.Start()
//	defer ext.Close(ctx)
//	// Before each prompt:
//	ext.WaitForStartup(ctx)
//	ext.ReconnectSignedIn(ctx)
//
// Local servers use [StdioTransport], whose errors end with what the server
// last wrote to stderr.
//
// A server whose user must sign in shows [StateNeedsAuth] in its status,
// and its tools return an error result whose [ToolDetails] has
// SignInRequired set. A local app such as a TUI signs in with
// [Extension.SignIn], which uses a loopback callback and a pasted redirect
// URL as fallback. A web app sends the user to [OAuthHandler.StartSignIn]'s
// URL; its callback finds the sign-in with [FindPendingSignIn], for the
// owner of the app's own session, and passes the query to
// [OAuthHandler.CompleteSignIn]. ReconnectSignedIn then picks up the new
// tokens. Apps that host Client ID Metadata Documents set
// OAuthOptions.ClientMetadataDocument, for example with
// [HostedClientMetadataDocument], instead of registering clients.
//
// Close waits for a token refresh in progress, so rotated tokens are saved
// before the process exits.
package mcp

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/ColeBurch/burrow/agent"
	"github.com/ColeBurch/burrow/ai"
	sdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

// maxToolNameLength is the longest tool name providers accept: 64
// characters of [A-Za-z0-9_-].
const maxToolNameLength = 64

// OutputMaxBytes is the size of model-facing text beyond which a result is
// cut in the middle.
const OutputMaxBytes = 20 * 1024

// ToolDetails are the details of every MCP tool result and update.
type ToolDetails struct {
	Server string `json:"server"`
	Tool   string `json:"tool"`
	// StructuredContent is the result's structured content, if any. It is
	// never truncated.
	StructuredContent any `json:"structuredContent,omitempty"`
	// SignInRequired reports that the call failed because the user must sign
	// in to Server (see [ErrSignInRequired]); the result is an error result
	// saying so. Apps check it in tool events to offer sign-in. Details
	// restored from a session store are decoded JSON, not ToolDetails.
	SignInRequired bool `json:"signInRequired,omitempty"`
}

// RequestOptions are the options of one request to a server.
type RequestOptions struct {
	// Timeout bounds the request, restarting at each progress notification
	// when OnProgress is set. Zero leaves it bounded only by its context.
	Timeout time.Duration
	// OnProgress receives the server's progress notifications for the
	// request. Nil means the request does not ask for progress.
	OnProgress func(*sdk.ProgressNotificationParams)
}

// ToolCaller calls tools of one server. [ServerConnection] implements it.
type ToolCaller interface {
	CallTool(ctx context.Context, name string, args map[string]any, opts *RequestOptions) (*sdk.CallToolResult, error)
}

var invalidToolNameChars = regexp.MustCompile(`[^A-Za-z0-9_]`)

// CreateToolName is mcp__<server>__<tool>, sanitized and shortened with a
// hash suffix when too long. Like Codex, everything but [A-Za-z0-9_] becomes
// _. isTaken reports names used by a different MCP tool: sanitizing can map
// two tools to one name (a-b and a_b), which then get the hash suffix.
// isTaken may be nil.
func CreateToolName(server, tool string, isTaken func(name string) bool) string {
	name := invalidToolNameChars.ReplaceAllString("mcp__"+server+"__"+tool, "_")
	if len(name) <= maxToolNameLength && (isTaken == nil || !isTaken(name)) {
		return name
	}
	sum := sha256.Sum256([]byte(server + "\x00" + tool))
	hash := hex.EncodeToString(sum[:])[:8]
	return name[:min(len(name), maxToolNameLength-len(hash)-1)] + "_" + hash
}

// ToolDefinitionOptions configures [CreateToolDefinition].
type ToolDefinitionOptions struct {
	// Server names the server in labels and messages.
	Server string
	Tool   *sdk.Tool
	// Name is the agent tool name, from [CreateToolName].
	Name string
	// Timeout bounds each call, restarting at each progress notification.
	// Zero leaves calls bounded only by their context.
	Timeout time.Duration
	// GetClient returns the caller at execution time, so a tool outlives the
	// connection it was listed on.
	GetClient func(ctx context.Context) (ToolCaller, error)
	// ReadableResources reports whether read_mcp_resource can read the
	// server's resources, which resource links then name. Nil means it
	// cannot.
	ReadableResources func() bool
}

// CreateToolDefinition adapts one MCP tool to an agent tool. Progress
// notifications become tool updates. A call that fails because the user
// must sign in returns an error result with ToolDetails.SignInRequired set.
func CreateToolDefinition(opts ToolDefinitionOptions) agent.AgentTool[any, any] {
	server, tool := opts.Server, opts.Tool
	return agent.AgentTool[any, any]{
		Tool: ai.Tool{
			Name:        opts.Name,
			Description: toolDescription(server, tool),
			Parameters:  toolParameters(tool.InputSchema),
		},
		Label:       server + "/" + tool.Name,
		Annotations: toolAnnotations(tool),
		Execute: func(ctx context.Context, toolCallID string, params any, onUpdate agent.AgentToolUpdateCallback[any]) (agent.AgentToolResult[any], error) {
			result, err := callTool(ctx, opts, params, onUpdate)
			if err != nil {
				if errors.Is(err, ErrSignInRequired) {
					return signInResult(server, tool.Name), nil
				}
				return agent.AgentToolResult[any]{}, fmt.Errorf("MCP tool %s/%s: %w", server, tool.Name, err)
			}
			readable := opts.ReadableResources != nil && opts.ReadableResources()
			return ConvertResult(server, tool.Name, result, &ConvertResultOptions{ReadableResources: readable}), nil
		},
	}
}

func callTool(ctx context.Context, opts ToolDefinitionOptions, params any, onUpdate agent.AgentToolUpdateCallback[any]) (*sdk.CallToolResult, error) {
	client, err := opts.GetClient(ctx)
	if err != nil {
		return nil, err
	}
	// Calls always ask for progress, even without onUpdate, since progress
	// keeps a long-running call from timing out.
	request := &RequestOptions{Timeout: opts.Timeout, OnProgress: func(p *sdk.ProgressNotificationParams) {
		if onUpdate != nil {
			onUpdate(agent.AgentToolResult[any]{
				Content: []ai.ToolResultContent{text(progressText(p))},
				Details: ToolDetails{Server: opts.Server, Tool: opts.Tool.Name},
			})
		}
	}}
	return client.CallTool(ctx, opts.Tool.Name, toolArguments(params), request)
}

// toolArguments are the call's parameters as a JSON object.
func toolArguments(params any) map[string]any {
	if args, ok := params.(map[string]any); ok {
		return args
	}
	args := map[string]any{}
	if params != nil {
		if data, err := json.Marshal(params); err == nil {
			_ = json.Unmarshal(data, &args)
		}
	}
	return args
}

// toolAnnotations are the boolean hints of an MCP tool's annotations, or
// nil when it has none. The SDK decodes absent readOnlyHint and
// idempotentHint as false, their default, so only true is kept.
func toolAnnotations(tool *sdk.Tool) *agent.ToolAnnotations {
	hints := tool.Annotations
	if hints == nil {
		return nil
	}
	hint := func(value bool) *bool { return &value }
	var annotations agent.ToolAnnotations
	if hints.ReadOnlyHint {
		annotations.ReadOnlyHint = hint(true)
	}
	if hints.IdempotentHint {
		annotations.IdempotentHint = hint(true)
	}
	if hints.DestructiveHint != nil {
		annotations.DestructiveHint = hint(*hints.DestructiveHint)
	}
	if hints.OpenWorldHint != nil {
		annotations.OpenWorldHint = hint(*hints.OpenWorldHint)
	}
	if annotations == (agent.ToolAnnotations{}) {
		return nil
	}
	return &annotations
}

// signInResult tells the model, and the app through ToolDetails, that
// server needs the user to sign in.
func signInResult(server, tool string) agent.AgentToolResult[any] {
	return agent.AgentToolResult[any]{
		Content: []ai.ToolResultContent{text(fmt.Sprintf(
			"MCP server %q requires sign-in. The user must sign in before its tools can be used.", server,
		))},
		Details: ToolDetails{Server: server, Tool: tool, SignInRequired: true},
		IsError: true,
	}
}

// toolDescription is the trimmed description, else the title, else a
// generic description.
func toolDescription(server string, tool *sdk.Tool) string {
	if description := strings.TrimSpace(tool.Description); description != "" {
		return description
	}
	if tool.Title != "" {
		return tool.Title
	}
	if tool.Annotations != nil && tool.Annotations.Title != "" {
		return tool.Annotations.Title
	}
	return fmt.Sprintf("MCP tool %s from server %s", tool.Name, server)
}

// toolParameters makes an input schema an object schema with properties:
// servers may omit type, and some providers reject object schemas without
// properties.
func toolParameters(inputSchema any) map[string]any {
	schema, ok := inputSchema.(map[string]any)
	if !ok && inputSchema != nil {
		// Schemas set in Go, such as a *jsonschema.Schema, round-trip
		// through JSON.
		if data, err := json.Marshal(inputSchema); err == nil {
			_ = json.Unmarshal(data, &schema)
		}
	}
	parameters := make(map[string]any, len(schema)+2)
	for key, value := range schema {
		parameters[key] = value
	}
	if _, ok := parameters["type"]; !ok {
		parameters["type"] = "object"
	}
	if _, ok := parameters["properties"]; !ok {
		parameters["properties"] = map[string]any{}
	}
	return parameters
}

func progressText(p *sdk.ProgressNotificationParams) string {
	if p.Message != "" {
		return p.Message
	}
	if p.Total != 0 {
		return fmt.Sprintf("Progress %v/%v", p.Progress, p.Total)
	}
	return fmt.Sprintf("Progress %v", p.Progress)
}

// ConvertResultOptions configures [ConvertResult].
type ConvertResultOptions struct {
	// ReadableResources reports that read_mcp_resource can read the server's
	// resources, which resource links then name.
	ReadableResources bool
}

// ConvertResult converts a tool result to model-facing content, within
// [OutputMaxBytes]. A result without content blocks becomes its structured
// content as JSON. An error result keeps its content and is marked IsError,
// getting a generic message when it has no text. opts may be nil.
func ConvertResult(server, tool string, result *sdk.CallToolResult, opts *ConvertResultOptions) agent.AgentToolResult[any] {
	readFrom := ""
	if opts != nil && opts.ReadableResources {
		readFrom = server
	}
	var content []ai.ToolResultContent
	for _, block := range result.Content {
		content = append(content, blockContent(block, readFrom))
	}
	if len(content) == 0 && result.StructuredContent != nil {
		if data, err := json.MarshalIndent(result.StructuredContent, "", "  "); err == nil {
			content = append(content, text(string(data)))
		}
	}
	if result.IsError && textOf(content) == "" {
		content = append(content, text(fmt.Sprintf("MCP tool %s/%s returned an error", server, tool)))
	}
	return agent.AgentToolResult[any]{
		Content: limitContent(content),
		Details: ToolDetails{Server: server, Tool: tool, StructuredContent: result.StructuredContent},
		IsError: result.IsError,
	}
}

// blockContent is the model-facing content of one content block. Images
// pass through; audio and binary resources other than images are omitted.
// Resource links name read_mcp_resource when readFrom, their server, is set.
func blockContent(block sdk.Content, readFrom string) ai.ToolResultContent {
	switch block := block.(type) {
	case *sdk.TextContent:
		return text(block.Text)
	case *sdk.ImageContent:
		return image(block.Data, block.MIMEType)
	case *sdk.AudioContent:
		return text(fmt.Sprintf("[audio %s omitted]", block.MIMEType))
	case *sdk.ResourceLink:
		return text(resourceLinkText(block, readFrom))
	case *sdk.EmbeddedResource:
		resource := block.Resource
		if resource == nil {
			return text("[empty MCP resource]")
		}
		if resource.Blob == nil {
			return text(resource.Text)
		}
		if strings.HasPrefix(resource.MIMEType, "image/") {
			return image(resource.Blob, resource.MIMEType)
		}
		if isTextMIMEType(resource.MIMEType) {
			return text(string(resource.Blob))
		}
		mimeType := resource.MIMEType
		if mimeType == "" {
			mimeType = "unknown type"
		}
		return text(fmt.Sprintf("[Binary resource %s (%s, %s) omitted]", resource.URI, mimeType, formatSize(len(resource.Blob))))
	default:
		return text(fmt.Sprintf("[unsupported MCP content %T]", block))
	}
}

// resourceLinkText is [Resource <uri> "<title>" (<mime>, <size>): <description>],
// followed by how to read it when readFrom is set.
func resourceLinkText(link *sdk.ResourceLink, readFrom string) string {
	title := link.Title
	if title == "" {
		title = link.Name
	}
	var details []string
	if link.MIMEType != "" {
		details = append(details, link.MIMEType)
	}
	if link.Size != nil {
		details = append(details, formatSize(int(*link.Size)))
	}
	var b strings.Builder
	fmt.Fprintf(&b, "[Resource %s %q", link.URI, title)
	if len(details) > 0 {
		fmt.Fprintf(&b, " (%s)", strings.Join(details, ", "))
	}
	if link.Description != "" {
		b.WriteString(": " + link.Description)
	}
	if readFrom != "" {
		fmt.Fprintf(&b, ". Read it with %s (server %q)", ReadResourceTool, readFrom)
	}
	b.WriteString("]")
	return b.String()
}

// isTextMIMEType reports whether blobs of mimeType are shown as text.
func isTextMIMEType(mimeType string) bool {
	if mimeType == "" {
		return false
	}
	t, _, _ := strings.Cut(mimeType, ";")
	t = strings.ToLower(strings.TrimSpace(t))
	return strings.HasPrefix(t, "text/") || t == "application/json" || strings.HasSuffix(t, "+json") || strings.HasSuffix(t, "+xml")
}

func formatSize(bytes int) string {
	switch {
	case bytes < 1024:
		return fmt.Sprintf("%dB", bytes)
	case bytes < 1024*1024:
		return fmt.Sprintf("%.1fKB", float64(bytes)/1024)
	default:
		return fmt.Sprintf("%.1fMB", float64(bytes)/(1024*1024))
	}
}

// limitContent keeps model-facing text within OutputMaxBytes. Longer text
// becomes one text block in Codex's truncation format, followed by the
// images.
func limitContent(content []ai.ToolResultContent) []ai.ToolResultContent {
	combined := textOf(content)
	if len(combined) <= OutputMaxBytes {
		return content
	}
	truncated := truncateMiddle(combined, OutputMaxBytes)
	limited := []ai.ToolResultContent{text(fmt.Sprintf(
		"Warning: truncated output (original token count: %d)\nTotal output lines: %d\n\n%s",
		(len(combined)+3)/4, countLines(combined), truncated,
	))}
	for _, block := range content {
		if _, ok := block.(ai.ImageContent); ok {
			limited = append(limited, block)
		}
	}
	return limited
}

// truncateMiddle keeps the start and the end of s, half of maxBytes each,
// and replaces the middle with a …N chars truncated… marker. It cuts only at
// character boundaries.
func truncateMiddle(s string, maxBytes int) string {
	if len(s) <= maxBytes {
		return s
	}
	headEnd := maxBytes / 2
	for headEnd > 0 && !utf8.RuneStart(s[headEnd]) {
		headEnd--
	}
	tailStart := len(s) - (maxBytes - maxBytes/2)
	for tailStart < len(s) && !utf8.RuneStart(s[tailStart]) {
		tailStart++
	}
	removed := utf8.RuneCountInString(s[headEnd:tailStart])
	return fmt.Sprintf("%s…%d chars truncated…%s", s[:headEnd], removed, s[tailStart:])
}

// countLines counts lines like an editor does: a final newline does not
// start another line.
func countLines(s string) int {
	if s == "" {
		return 0
	}
	return strings.Count(strings.TrimSuffix(s, "\n"), "\n") + 1
}

func textOf(content []ai.ToolResultContent) string {
	var texts []string
	for _, block := range content {
		if block, ok := block.(ai.TextContent); ok {
			texts = append(texts, block.Text)
		}
	}
	return strings.Join(texts, "\n")
}

func text(s string) ai.TextContent {
	return ai.TextContent{Type: ai.ContentTypeText, Text: s}
}

func image(data []byte, mimeType string) ai.ImageContent {
	return ai.ImageContent{Type: ai.ContentTypeImage, Image: base64.StdEncoding.EncodeToString(data), MimeType: mimeType}
}
