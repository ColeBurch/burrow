package ai

import (
	"reflect"
	"strings"
	"testing"
)

func createToolCallWithPlainSchema(schema map[string]any, value any) (Tool, ToolCall) {
	tool := Tool{
		Name:        "echo",
		Description: "Echo tool",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"value": schema,
			},
			"required": []any{"value"},
		},
	}

	toolCall := ToolCall{
		Type: ContentTypeToolCall,
		Id:   "tool-1",
		Name: "echo",
		Args: map[string]any{"value": value},
	}

	return tool, toolCall
}

func TestValidateToolArgumentsStillValidatesWithoutFunctionConstructor(t *testing.T) {
	// Go does not use JavaScript's Function constructor/code generation for schema
	// validation, but this mirrors the TypeScript regression case: validation and
	// conversion should still succeed through the non-codegen validator path.
	tool := Tool{
		Name:        "echo",
		Description: "Echo tool",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"count": map[string]any{"type": "number"},
			},
			"required": []any{"count"},
		},
	}
	toolCall := ToolCall{
		Type: ContentTypeToolCall,
		Id:   "tool-1",
		Name: "echo",
		Args: map[string]any{"count": "42"},
	}

	got, err := ValidateToolArguments(tool, toolCall)
	if err != nil {
		t.Fatalf("ValidateToolArguments returned error: %v", err)
	}

	expected := map[string]any{"count": float64(42)}
	if !reflect.DeepEqual(got, expected) {
		t.Fatalf("ValidateToolArguments = %#v, want %#v", got, expected)
	}
}

func TestValidateToolArgumentsCoercesSerializedPlainJSONSchemas(t *testing.T) {
	tests := []struct {
		name     string
		schema   map[string]any
		input    any
		expected any
	}{
		{name: "number from string", schema: map[string]any{"type": "number"}, input: "42", expected: float64(42)},
		{name: "number from true", schema: map[string]any{"type": "number"}, input: true, expected: float64(1)},
		{name: "number from null", schema: map[string]any{"type": "number"}, input: nil, expected: float64(0)},
		{name: "integer from string", schema: map[string]any{"type": "integer"}, input: "42", expected: float64(42)},
		{name: "boolean from true string", schema: map[string]any{"type": "boolean"}, input: "true", expected: true},
		{name: "boolean from false string", schema: map[string]any{"type": "boolean"}, input: "false", expected: false},
		{name: "boolean from one", schema: map[string]any{"type": "boolean"}, input: float64(1), expected: true},
		{name: "boolean from zero", schema: map[string]any{"type": "boolean"}, input: float64(0), expected: false},
		{name: "string from null", schema: map[string]any{"type": "string"}, input: nil, expected: ""},
		{name: "string from true", schema: map[string]any{"type": "string"}, input: true, expected: "true"},
		{name: "null from empty string", schema: map[string]any{"type": "null"}, input: "", expected: nil},
		{name: "null from zero", schema: map[string]any{"type": "null"}, input: float64(0), expected: nil},
		{name: "null from false", schema: map[string]any{"type": "null"}, input: false, expected: nil},
		{name: "union keeps already matching string", schema: map[string]any{"type": []any{"number", "string"}}, input: "1", expected: "1"},
		{name: "union coerces string to first matching primitive", schema: map[string]any{"type": []any{"boolean", "number"}}, input: "1", expected: float64(1)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tool, toolCall := createToolCallWithPlainSchema(tt.schema, tt.input)

			got, err := ValidateToolArguments(tool, toolCall)
			if err != nil {
				t.Fatalf("ValidateToolArguments returned error: %v", err)
			}

			expected := map[string]any{"value": tt.expected}
			if !reflect.DeepEqual(got, expected) {
				t.Fatalf("ValidateToolArguments = %#v, want %#v", got, expected)
			}
		})
	}
}

func TestValidateToolArgumentsRejectsInvalidCoercions(t *testing.T) {
	tests := []struct {
		name   string
		schema map[string]any
		input  any
	}{
		{name: "boolean rejects one string", schema: map[string]any{"type": "boolean"}, input: "1"},
		{name: "boolean rejects zero string", schema: map[string]any{"type": "boolean"}, input: "0"},
		{name: "null rejects null string", schema: map[string]any{"type": "null"}, input: "null"},
		{name: "integer rejects fractional string", schema: map[string]any{"type": "integer"}, input: "42.1"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tool, toolCall := createToolCallWithPlainSchema(tt.schema, tt.input)

			_, err := ValidateToolArguments(tool, toolCall)
			if err == nil {
				t.Fatal("ValidateToolArguments returned nil error, want validation error")
			}
			if !strings.Contains(err.Error(), "validation failed") {
				t.Fatalf("error = %q, want it to contain %q", err.Error(), "validation failed")
			}
		})
	}
}
