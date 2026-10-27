// Adapted from pi's packages/coding-agent/src/extensions/mcp/resources.ts
// (MIT, see LICENSE).

package mcp

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/ColeBurch/burrow/agent"
	"github.com/ColeBurch/burrow/ai"
	"github.com/modelcontextprotocol/go-sdk/jsonrpc"
	sdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

// Names of the resource tools. Codex and opencode use the same tools, so
// models trained on them use these unchanged.
const (
	ListResourcesTool         = "list_mcp_resources"
	ListResourceTemplatesTool = "list_mcp_resource_templates"
	ReadResourceTool          = "read_mcp_resource"
)

// ResourceServer is a connected server that offers resources.
// [ServerConnection] implements it.
type ResourceServer interface {
	Name() string
	Timeout() time.Duration
	ResourcesPage(ctx context.Context, cursor string, opts *RequestOptions) (*sdk.ListResourcesResult, error)
	ResourceTemplatesPage(ctx context.Context, cursor string, opts *RequestOptions) (*sdk.ListResourceTemplatesResult, error)
	AllResources(ctx context.Context, opts *RequestOptions) ([]*sdk.Resource, error)
	AllResourceTemplates(ctx context.Context, opts *RequestOptions) ([]*sdk.ResourceTemplate, error)
	ReadResource(ctx context.Context, uri string, opts *RequestOptions) (*sdk.ReadResourceResult, error)
}

// CreateResourceToolDefinitions returns list_mcp_resources,
// list_mcp_resource_templates and read_mcp_resource. They take a server
// argument; servers returns the servers they reach, at call time. Listings
// are JSON, as in Codex. MCP App resources (ui:// URIs and profile=mcp-app
// HTML) are left out, since they are user interfaces for hosts that render
// them.
func CreateResourceToolDefinitions(servers func() []ResourceServer) []agent.AgentTool[any, any] {
	r := &resourceTools{servers: servers}
	return []agent.AgentTool[any, any]{r.listResourcesTool(), r.listTemplatesTool(), r.readResourceTool()}
}

type resourceTools struct {
	servers func() []ResourceServer
}

// readOnlyAnnotations mark the resource tools, which change nothing.
func readOnlyAnnotations() *agent.ToolAnnotations {
	readOnly := true
	return &agent.ToolAnnotations{ReadOnlyHint: &readOnly}
}

func stringProperty(description string) map[string]any {
	return map[string]any{"type": "string", "description": description}
}

func listParameters() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"server": stringProperty("MCP server name. Omit to list every server with resources."),
			"cursor": stringProperty("Opaque cursor from a previous call with the same server; omit for the first page."),
		},
		"additionalProperties": false,
	}
}

func readParameters() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"server": stringProperty("MCP server name exactly as configured. Must match the 'server' field returned by list_mcp_resources."),
			"uri":    stringProperty("Resource URI to read. Must be one of the URIs returned by list_mcp_resources."),
		},
		"required":             []any{"server", "uri"},
		"additionalProperties": false,
	}
}

// listingError is a server that could not be listed.
type listingError struct {
	Server string `json:"server"`
	Error  string `json:"error"`
}

type resourceListing struct {
	Server     string           `json:"server,omitempty"`
	Resources  []map[string]any `json:"resources"`
	NextCursor string           `json:"nextCursor,omitempty"`
	Errors     []listingError   `json:"errors,omitempty"`
}

type templateListing struct {
	Server            string           `json:"server,omitempty"`
	ResourceTemplates []map[string]any `json:"resourceTemplates"`
	NextCursor        string           `json:"nextCursor,omitempty"`
	Errors            []listingError   `json:"errors,omitempty"`
}

func (r *resourceTools) listResourcesTool() agent.AgentTool[any, any] {
	return r.listTool(ListResourcesTool,
		"Lists resources provided by MCP servers. Resources allow servers to share data that provides context to language models, such as files, database schemas, or application-specific information. Prefer resources over web search when possible.",
		func(ctx context.Context, server ResourceServer, cursor string, opts *RequestOptions) ([]any, string, error) {
			result, err := server.ResourcesPage(ctx, cursor, opts)
			if err != nil {
				return nil, "", err
			}
			return anySlice(result.Resources), result.NextCursor, nil
		},
		func(ctx context.Context, server ResourceServer, opts *RequestOptions) ([]any, error) {
			resources, err := server.AllResources(ctx, opts)
			return anySlice(resources), err
		},
		func(server string, items []map[string]any, next string, errs []listingError) any {
			return resourceListing{Server: server, Resources: items, NextCursor: next, Errors: errs}
		},
	)
}

func (r *resourceTools) listTemplatesTool() agent.AgentTool[any, any] {
	return r.listTool(ListResourceTemplatesTool,
		"Lists resource templates provided by MCP servers. Parameterized resource templates allow servers to share data that takes parameters and provides context to language models, such as files, database schemas, or application-specific information. Prefer resource templates over web search when possible.",
		func(ctx context.Context, server ResourceServer, cursor string, opts *RequestOptions) ([]any, string, error) {
			result, err := server.ResourceTemplatesPage(ctx, cursor, opts)
			if err != nil {
				return nil, "", err
			}
			return anySlice(result.ResourceTemplates), result.NextCursor, nil
		},
		func(ctx context.Context, server ResourceServer, opts *RequestOptions) ([]any, error) {
			templates, err := server.AllResourceTemplates(ctx, opts)
			return anySlice(templates), err
		},
		func(server string, items []map[string]any, next string, errs []listingError) any {
			return templateListing{Server: server, ResourceTemplates: items, NextCursor: next, Errors: errs}
		},
	)
}

// listTool lists one page of one server, or every page of every server.
func (r *resourceTools) listTool(
	name, description string,
	page func(ctx context.Context, server ResourceServer, cursor string, opts *RequestOptions) ([]any, string, error),
	all func(ctx context.Context, server ResourceServer, opts *RequestOptions) ([]any, error),
	listing func(server string, items []map[string]any, next string, errs []listingError) any,
) agent.AgentTool[any, any] {
	return agent.AgentTool[any, any]{
		Tool:        ai.Tool{Name: name, Description: description, Parameters: listParameters()},
		Label:       name,
		Annotations: readOnlyAnnotations(),
		Execute: func(ctx context.Context, toolCallID string, params any, onUpdate agent.AgentToolUpdateCallback[any]) (agent.AgentToolResult[any], error) {
			serverName, err := stringArgument(params, "server")
			if err != nil {
				return agent.AgentToolResult[any]{}, err
			}
			cursor, err := stringArgument(params, "cursor")
			if err != nil {
				return agent.AgentToolResult[any]{}, err
			}
			if serverName != "" {
				server, err := r.findServer(serverName)
				if err != nil {
					return agent.AgentToolResult[any]{}, err
				}
				items, next, err := page(ctx, server, cursor, &RequestOptions{Timeout: server.Timeout()})
				if err != nil {
					if errors.Is(err, ErrSignInRequired) {
						return signInResult(server.Name(), name), nil
					}
					return agent.AgentToolResult[any]{}, fmt.Errorf("MCP server %s: %w", server.Name(), err)
				}
				return jsonResult(name, server.Name(), listing(server.Name(), listedItems(server.Name(), items), next, nil)), nil
			}
			if cursor != "" {
				return agent.AgentToolResult[any]{}, errors.New("cursor can only be used when a server is specified")
			}
			servers := slices.SortedFunc(slices.Values(r.servers()), func(a, b ResourceServer) int { return cmp.Compare(a.Name(), b.Name()) })
			results := make([][]any, len(servers))
			errs := make([]error, len(servers))
			var wg sync.WaitGroup
			for i, server := range servers {
				wg.Go(func() {
					results[i], errs[i] = all(ctx, server, &RequestOptions{Timeout: server.Timeout()})
				})
			}
			wg.Wait()
			items := []map[string]any{}
			var failed []listingError
			for i, server := range servers {
				if errs[i] != nil {
					failed = append(failed, listingError{Server: server.Name(), Error: errs[i].Error()})
					continue
				}
				items = append(items, listedItems(server.Name(), results[i])...)
			}
			return jsonResult(name, "", listing("", items, "", failed)), nil
		},
	}
}

func (r *resourceTools) readResourceTool() agent.AgentTool[any, any] {
	return agent.AgentTool[any, any]{
		Tool: ai.Tool{
			Name:        ReadResourceTool,
			Description: "Read a specific resource from an MCP server given the server name and resource URI.",
			Parameters:  readParameters(),
		},
		Label:       ReadResourceTool,
		Annotations: readOnlyAnnotations(),
		Execute: func(ctx context.Context, toolCallID string, params any, onUpdate agent.AgentToolUpdateCallback[any]) (agent.AgentToolResult[any], error) {
			serverName, err := stringArgument(params, "server")
			if err != nil {
				return agent.AgentToolResult[any]{}, err
			}
			uri, err := stringArgument(params, "uri")
			if err != nil {
				return agent.AgentToolResult[any]{}, err
			}
			if serverName == "" {
				return agent.AgentToolResult[any]{}, errors.New("server must be provided")
			}
			if uri == "" {
				return agent.AgentToolResult[any]{}, errors.New("uri must be provided")
			}
			server, err := r.findServer(serverName)
			if err != nil {
				return agent.AgentToolResult[any]{}, err
			}
			result, err := server.ReadResource(ctx, uri, &RequestOptions{Timeout: server.Timeout()})
			if err != nil {
				if errors.Is(err, ErrSignInRequired) {
					return signInResult(server.Name(), ReadResourceTool), nil
				}
				return agent.AgentToolResult[any]{}, fmt.Errorf("MCP server %s: %w", server.Name(), err)
			}
			// Several contents (for example a directory) are labeled with
			// their URIs.
			var content []ai.ToolResultContent
			contents := []map[string]any{}
			for _, resource := range result.Contents {
				if resource == nil {
					continue
				}
				if len(result.Contents) > 1 {
					content = append(content, text(resource.URI+":"))
				}
				content = append(content, blockContent(&sdk.EmbeddedResource{Resource: resource}, ""))
				contents = append(contents, jsonObject(resource, "_meta"))
			}
			if len(content) == 0 {
				content = append(content, text(fmt.Sprintf("Resource %s is empty.", uri)))
			}
			return agent.AgentToolResult[any]{
				Content: limitContent(content),
				Details: ToolDetails{
					Server:            server.Name(),
					Tool:              ReadResourceTool,
					StructuredContent: map[string]any{"server": server.Name(), "uri": uri, "contents": contents},
				},
			}, nil
		},
	}
}

func (r *resourceTools) findServer(name string) (ResourceServer, error) {
	servers := r.servers()
	names := make([]string, len(servers))
	for i, server := range servers {
		if server.Name() == name {
			return server, nil
		}
		names[i] = server.Name()
	}
	if len(names) == 0 {
		return nil, fmt.Errorf("MCP server %q has no resources", name)
	}
	return nil, fmt.Errorf("MCP server %q has no resources. Servers with resources: %s", name, strings.Join(names, ", "))
}

// jsonResult is payload as compact JSON for the model, and as structured
// content for the app.
func jsonResult(tool, server string, payload any) agent.AgentToolResult[any] {
	data, err := json.Marshal(payload)
	if err != nil {
		data = fmt.Appendf(nil, "failed to encode %s result: %v", tool, err)
	}
	return agent.AgentToolResult[any]{
		Content: limitContent([]ai.ToolResultContent{text(string(data))}),
		Details: ToolDetails{Server: server, Tool: tool, StructuredContent: payload},
	}
}

// stringArgument is the trimmed string argument key, or "" when absent.
func stringArgument(params any, key string) (string, error) {
	args, _ := params.(map[string]any)
	value, ok := args[key]
	if !ok || value == nil {
		return "", nil
	}
	s, ok := value.(string)
	if !ok {
		return "", fmt.Errorf("%s must be a string", key)
	}
	return strings.TrimSpace(s), nil
}

var mcpAppProfile = regexp.MustCompile(`(?i);\s*profile\s*=\s*"?mcp-app"?`)

// listedItems are the listed resources or templates of server without MCP
// App resources, _meta and icons, each tagged with its server.
func listedItems(server string, items []any) []map[string]any {
	listed := []map[string]any{}
	for _, item := range items {
		object := jsonObject(item, "_meta", "icons")
		uri, _ := object["uri"].(string)
		if uri == "" {
			uri, _ = object["uriTemplate"].(string)
		}
		mimeType, _ := object["mimeType"].(string)
		if strings.HasPrefix(uri, "ui://") || mcpAppProfile.MatchString(mimeType) {
			continue
		}
		object["server"] = server
		listed = append(listed, object)
	}
	return listed
}

// jsonObject is v as a JSON object without the keys drop.
func jsonObject(v any, drop ...string) map[string]any {
	var object map[string]any
	if data, err := json.Marshal(v); err == nil {
		_ = json.Unmarshal(data, &object)
	}
	if object == nil {
		object = map[string]any{}
	}
	for _, key := range drop {
		delete(object, key)
	}
	return object
}

// isMethodNotFound reports a server without the method, such as one that
// does not implement resources/templates/list.
func isMethodNotFound(err error) bool {
	var wireErr *jsonrpc.Error
	return errors.As(err, &wireErr) && wireErr.Code == jsonrpc.CodeMethodNotFound
}

func anySlice[T any](items []T) []any {
	out := make([]any, len(items))
	for i, item := range items {
		out[i] = item
	}
	return out
}
