package ai

import (
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"strconv"
	"strings"
	"sync"

	"github.com/santhosh-tekuri/jsonschema/v6"
)

var validatorCache sync.Map // map[string]*jsonschema.Schema

func isRecord(value any) bool {
	if value == nil {
		return false
	}
	_, ok := value.(map[string]any)
	return ok
}

func isJSONSchemaObject(value any) bool {
	return isRecord(value)
}

func hasTypeBoxMetadata(schema any) bool {
	// JavaScript TypeBox metadata is stored on a Symbol. That concept does not
	// survive JSON/Go map conversion, so schemas represented by map[string]any are
	// treated as plain JSON Schema.
	return false
}

func getSchemaTypes(schema map[string]any) []string {
	switch typ := schema["type"].(type) {
	case string:
		return []string{typ}
	case []any:
		types := make([]string, 0, len(typ))
		for _, item := range typ {
			if s, ok := item.(string); ok {
				types = append(types, s)
			}
		}
		return types
	case []string:
		return append([]string(nil), typ...)
	default:
		return nil
	}
}

func matchesJSONType(value any, typ string) bool {
	switch typ {
	case "number":
		_, ok := toFloat64(value)
		return ok
	case "integer":
		n, ok := toFloat64(value)
		return ok && math.Trunc(n) == n
	case "boolean":
		_, ok := value.(bool)
		return ok
	case "string":
		_, ok := value.(string)
		return ok
	case "null":
		return value == nil
	case "array":
		_, ok := value.([]any)
		return ok
	case "object":
		_, ok := value.(map[string]any)
		return ok
	default:
		return false
	}
}

func isValidatorSchema(value any) bool {
	return isRecord(value)
}

func getSubSchemaValidator(schema map[string]any) *jsonschema.Schema {
	if !isValidatorSchema(schema) {
		return nil
	}
	validator, err := getValidator(schema)
	if err != nil {
		return nil
	}
	return validator
}

func coercePrimitiveByType(value any, typ string) any {
	switch typ {
	case "number":
		if value == nil {
			return float64(0)
		}
		if s, ok := value.(string); ok && strings.TrimSpace(s) != "" {
			parsed, err := strconv.ParseFloat(s, 64)
			if err == nil && !math.IsInf(parsed, 0) && !math.IsNaN(parsed) {
				return parsed
			}
		}
		if b, ok := value.(bool); ok {
			if b {
				return float64(1)
			}
			return float64(0)
		}
		return value
	case "integer":
		if value == nil {
			return float64(0)
		}
		if s, ok := value.(string); ok && strings.TrimSpace(s) != "" {
			parsed, err := strconv.ParseFloat(s, 64)
			if err == nil && !math.IsInf(parsed, 0) && !math.IsNaN(parsed) && math.Trunc(parsed) == parsed {
				return parsed
			}
		}
		if b, ok := value.(bool); ok {
			if b {
				return float64(1)
			}
			return float64(0)
		}
		return value
	case "boolean":
		if value == nil {
			return false
		}
		if s, ok := value.(string); ok {
			if s == "true" {
				return true
			}
			if s == "false" {
				return false
			}
		}
		if n, ok := toFloat64(value); ok {
			if n == 1 {
				return true
			}
			if n == 0 {
				return false
			}
		}
		return value
	case "string":
		if value == nil {
			return ""
		}
		switch v := value.(type) {
		case float64, float32, int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64, bool:
			return fmt.Sprint(v)
		}
		return value
	case "null":
		if value == "" || value == false {
			return nil
		}
		if n, ok := toFloat64(value); ok && n == 0 {
			return nil
		}
		return value
	default:
		return value
	}
}

func applySchemaObjectCoercion(value map[string]any, schema map[string]any) {
	properties, _ := schema["properties"].(map[string]any)
	definedKeys := map[string]struct{}{}

	for key, propertySchemaValue := range properties {
		definedKeys[key] = struct{}{}
		propertySchema, ok := propertySchemaValue.(map[string]any)
		if !ok {
			continue
		}
		propertyValue, exists := value[key]
		if !exists {
			continue
		}
		value[key] = coerceWithJSONSchema(propertyValue, propertySchema)
	}

	additionalProperties, ok := schema["additionalProperties"].(map[string]any)
	if !ok {
		return
	}
	for key, propertyValue := range value {
		if _, defined := definedKeys[key]; defined {
			continue
		}
		value[key] = coerceWithJSONSchema(propertyValue, additionalProperties)
	}
}

func applySchemaArrayCoercion(value []any, schema map[string]any) {
	switch items := schema["items"].(type) {
	case []any:
		for index := range value {
			if index >= len(items) {
				continue
			}
			itemSchema, ok := items[index].(map[string]any)
			if !ok {
				continue
			}
			value[index] = coerceWithJSONSchema(value[index], itemSchema)
		}
	case []map[string]any:
		for index := range value {
			if index >= len(items) {
				continue
			}
			value[index] = coerceWithJSONSchema(value[index], items[index])
		}
	case map[string]any:
		for index := range value {
			value[index] = coerceWithJSONSchema(value[index], items)
		}
	}
}

func coerceWithUnionSchema(value any, schemas []map[string]any) any {
	for _, schema := range schemas {
		candidate := cloneAny(value)
		coerced := coerceWithJSONSchema(candidate, schema)
		validator := getSubSchemaValidator(schema)
		if validator != nil && validator.Validate(coerced) == nil {
			return coerced
		}
	}
	return value
}

func coerceWithJSONSchema(value any, schema map[string]any) any {
	nextValue := value

	if allOf := schemaArray(schema["allOf"]); len(allOf) > 0 {
		for _, nested := range allOf {
			nextValue = coerceWithJSONSchema(nextValue, nested)
		}
	}
	if anyOf := schemaArray(schema["anyOf"]); len(anyOf) > 0 {
		nextValue = coerceWithUnionSchema(nextValue, anyOf)
	}
	if oneOf := schemaArray(schema["oneOf"]); len(oneOf) > 0 {
		nextValue = coerceWithUnionSchema(nextValue, oneOf)
	}

	schemaTypes := getSchemaTypes(schema)
	matchesUnionMember := false
	if len(schemaTypes) > 1 {
		for _, schemaType := range schemaTypes {
			if matchesJSONType(nextValue, schemaType) {
				matchesUnionMember = true
				break
			}
		}
	}
	if len(schemaTypes) > 0 && !matchesUnionMember {
		for _, schemaType := range schemaTypes {
			candidate := coercePrimitiveByType(nextValue, schemaType)
			if !reflect.DeepEqual(candidate, nextValue) {
				nextValue = candidate
				break
			}
		}
	}

	if containsString(schemaTypes, "object") {
		if obj, ok := nextValue.(map[string]any); ok {
			applySchemaObjectCoercion(obj, schema)
		}
	}
	if containsString(schemaTypes, "array") {
		if arr, ok := nextValue.([]any); ok {
			applySchemaArrayCoercion(arr, schema)
		}
	}

	return nextValue
}

func getValidator(schema map[string]any) (*jsonschema.Schema, error) {
	keyBytes, err := json.Marshal(schema)
	if err != nil {
		return nil, err
	}
	key := string(keyBytes)
	if cached, ok := validatorCache.Load(key); ok {
		return cached.(*jsonschema.Schema), nil
	}

	validator, err := compileSchema(schema)
	if err != nil {
		return nil, err
	}
	actual, _ := validatorCache.LoadOrStore(key, validator)
	return actual.(*jsonschema.Schema), nil
}

func formatValidationPathFromUnit(unit jsonschema.OutputUnit) string {
	path := strings.TrimPrefix(unit.InstanceLocation, "/")
	path = strings.ReplaceAll(path, "/", ".")
	if path == "" {
		return "root"
	}
	return path
}

// ValidateToolCall finds a tool by name and validates the tool call arguments against its schema.
func ValidateToolCall(tools []*Tool, toolCall ToolCall) (any, error) {
	for _, tool := range tools {
		if tool != nil && tool.Name == toolCall.Name {
			return ValidateToolArguments(*tool, toolCall)
		}
	}
	return nil, fmt.Errorf("tool %q not found", toolCall.Name)
}

// ValidateToolArguments validates tool call arguments against the tool's JSON schema.
func ValidateToolArguments(tool Tool, toolCall ToolCall) (any, error) {
	argsAny := cloneAny(toolCall.Args)
	args, ok := argsAny.(map[string]any)
	if !ok || args == nil {
		args = map[string]any{}
		argsAny = args
	}

	validator, err := getValidator(tool.Parameters)
	if err != nil {
		return nil, fmt.Errorf("invalid schema for tool %q: %w", tool.Name, err)
	}

	if !hasTypeBoxMetadata(tool.Parameters) && isJSONSchemaObject(tool.Parameters) {
		coerced := coerceWithJSONSchema(argsAny, tool.Parameters)
		if !reflect.DeepEqual(coerced, argsAny) {
			coercedRecord, coercedOK := coerced.(map[string]any)
			if coercedOK {
				for key := range args {
					delete(args, key)
				}
				for key, value := range coercedRecord {
					args[key] = value
				}
			} else if validator.Validate(coerced) == nil {
				return coerced, nil
			} else {
				return args, nil
			}
		}
	}

	if err := validator.Validate(args); err == nil {
		return args, nil
	} else {
		return nil, fmt.Errorf(
			"validation failed for tool %q:\n%s\n\nReceived arguments:\n%s",
			toolCall.Name,
			formatValidationError(err),
			mustPrettyJSON(toolCall.Args),
		)
	}
}

func compileSchema(schema map[string]any) (*jsonschema.Schema, error) {
	compiler := jsonschema.NewCompiler()
	if err := compiler.AddResource("tool-schema.json", cloneAny(schema)); err != nil {
		return nil, err
	}
	return compiler.Compile("tool-schema.json")
}

func formatValidationError(err error) string {
	if err == nil {
		return "Unknown validation error"
	}
	validationErr, ok := err.(*jsonschema.ValidationError)
	if !ok {
		return "  - " + err.Error()
	}

	var lines []string
	appendOutputErrors(&lines, *validationErr.DetailedOutput())
	if len(lines) == 0 {
		return "  - " + err.Error()
	}
	return strings.Join(lines, "\n")
}

func appendOutputErrors(lines *[]string, unit jsonschema.OutputUnit) {
	if unit.Error != nil {
		*lines = append(*lines, fmt.Sprintf("  - %s: %s", formatValidationPathFromUnit(unit), unit.Error))
	}
	for _, child := range unit.Errors {
		appendOutputErrors(lines, child)
	}
}

func mustPrettyJSON(v any) string {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return "<unmarshalable>"
	}
	return string(b)
}

func cloneAny(in any) any {
	b, err := json.Marshal(in)
	if err != nil {
		return in
	}
	var out any
	if err := json.Unmarshal(b, &out); err != nil {
		return in
	}
	return out
}

func schemaArray(value any) []map[string]any {
	switch schemas := value.(type) {
	case []any:
		out := make([]map[string]any, 0, len(schemas))
		for _, item := range schemas {
			if schema, ok := item.(map[string]any); ok {
				out = append(out, schema)
			}
		}
		return out
	case []map[string]any:
		return schemas
	default:
		return nil
	}
}

func containsString(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}

func toFloat64(value any) (float64, bool) {
	switch v := value.(type) {
	case float64:
		return v, true
	case float32:
		return float64(v), true
	case int:
		return float64(v), true
	case int8:
		return float64(v), true
	case int16:
		return float64(v), true
	case int32:
		return float64(v), true
	case int64:
		return float64(v), true
	case uint:
		return float64(v), true
	case uint8:
		return float64(v), true
	case uint16:
		return float64(v), true
	case uint32:
		return float64(v), true
	case uint64:
		return float64(v), true
	default:
		return 0, false
	}
}
