//go:build ignore

// Generate OpenAI model metadata from models.dev.
//
// Run with:
//
//	go run ./scripts/generate-models.go
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"go/format"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"
)

type modelsDevProvider struct {
	Models map[string]modelsDevModel `json:"models"`
}

type modelsDevModel struct {
	ID         string `json:"id"`
	Name       string `json:"name"`
	ToolCall   bool   `json:"tool_call"`
	Reasoning  bool   `json:"reasoning"`
	Limit      *struct {
		Context *int `json:"context"`
		Output  *int `json:"output"`
	} `json:"limit"`
	Cost *struct {
		Input      *float64 `json:"input"`
		Output     *float64 `json:"output"`
		CacheRead  *float64 `json:"cache_read"`
		CacheWrite *float64 `json:"cache_write"`
	} `json:"cost"`
	Modalities *struct {
		Input []string `json:"input"`
	} `json:"modalities"`
}

type model struct {
	ID               string
	Name             string
	API              string
	Provider         string
	BaseURL          string
	Reasoning        bool
	ThinkingLevelMap map[string]*string
	Input            []string
	Cost             cost
	ContextWindow    int
	MaxTokens        int
}

type cost struct {
	Input      float64
	Output     float64
	CacheRead  float64
	CacheWrite float64
}

const modelsDevURL = "https://models.dev/api.json"

func main() {
	models, err := loadOpenAIModels()
	if err != nil {
		fmt.Fprintf(os.Stderr, "generate models: %v\n", err)
		os.Exit(1)
	}

	addMissingOpenAIModels(&models)
	applyOpenAIOverrides(models)
	for i := range models {
		applyThinkingLevelMetadata(&models[i])
	}

	if err := writeGenerated(models); err != nil {
		fmt.Fprintf(os.Stderr, "write generated models: %v\n", err)
		os.Exit(1)
	}

	reasoning := 0
	for _, m := range models {
		if m.Reasoning {
			reasoning++
		}
	}
	fmt.Printf("Generated internal/ai/models.generated.go\n")
	fmt.Printf("OpenAI models: %d\n", len(models))
	fmt.Printf("Reasoning-capable models: %d\n", reasoning)
}

func loadOpenAIModels() ([]model, error) {
	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Get(modelsDevURL)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("models.dev returned %s", resp.Status)
	}

	var data map[string]modelsDevProvider
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return nil, err
	}

	openai := data["openai"]
	models := make([]model, 0, len(openai.Models))
	for modelID, raw := range openai.Models {
		if !raw.ToolCall {
			continue
		}

		name := raw.Name
		if name == "" {
			name = modelID
		}

		models = append(models, model{
			ID:            modelID,
			Name:          name,
			API:           "openai-responses",
			Provider:      "openai",
			BaseURL:       "https://api.openai.com/v1",
			Reasoning:     raw.Reasoning,
			Input:         inputModalities(raw),
			Cost:          modelCost(raw),
			ContextWindow: limitOr(raw.Limit, true, 4096),
			MaxTokens:     limitOr(raw.Limit, false, 4096),
		})
	}

	return dedupeAndSort(models), nil
}

func inputModalities(raw modelsDevModel) []string {
	input := []string{"text"}
	if raw.Modalities == nil {
		return input
	}
	for _, modality := range raw.Modalities.Input {
		if modality == "image" {
			return []string{"text", "image"}
		}
	}
	return input
}

func modelCost(raw modelsDevModel) cost {
	if raw.Cost == nil {
		return cost{}
	}
	return cost{
		Input:      floatPtrOr(raw.Cost.Input, 0),
		Output:     floatPtrOr(raw.Cost.Output, 0),
		CacheRead:  floatPtrOr(raw.Cost.CacheRead, 0),
		CacheWrite: floatPtrOr(raw.Cost.CacheWrite, 0),
	}
}

func limitOr(limit *struct {
	Context *int `json:"context"`
	Output  *int `json:"output"`
}, context bool, fallback int) int {
	if limit == nil {
		return fallback
	}
	if context && limit.Context != nil {
		return *limit.Context
	}
	if !context && limit.Output != nil {
		return *limit.Output
	}
	return fallback
}

func floatPtrOr(value *float64, fallback float64) float64 {
	if value == nil {
		return fallback
	}
	return *value
}

func addMissingOpenAIModels(models *[]model) {
	addIfMissing(models, model{
		ID: "gpt-5-chat-latest", Name: "GPT-5 Chat Latest", API: "openai-responses", Provider: "openai",
		BaseURL: "https://api.openai.com/v1", Reasoning: false, Input: []string{"text", "image"},
		Cost: cost{Input: 1.25, Output: 10, CacheRead: 0.125}, ContextWindow: 128000, MaxTokens: 16384,
	})
	addIfMissing(models, model{
		ID: "gpt-5.1-codex", Name: "GPT-5.1 Codex", API: "openai-responses", Provider: "openai",
		BaseURL: "https://api.openai.com/v1", Reasoning: true, Input: []string{"text", "image"},
		Cost: cost{Input: 1.25, Output: 5, CacheRead: 0.125, CacheWrite: 1.25}, ContextWindow: 400000, MaxTokens: 128000,
	})
	addIfMissing(models, model{
		ID: "gpt-5.1-codex-max", Name: "GPT-5.1 Codex Max", API: "openai-responses", Provider: "openai",
		BaseURL: "https://api.openai.com/v1", Reasoning: true, Input: []string{"text", "image"},
		Cost: cost{Input: 1.25, Output: 10, CacheRead: 0.125}, ContextWindow: 400000, MaxTokens: 128000,
	})
	addIfMissing(models, model{
		ID: "gpt-5.3-codex-spark", Name: "GPT-5.3 Codex Spark", API: "openai-responses", Provider: "openai",
		BaseURL: "https://api.openai.com/v1", Reasoning: true, Input: []string{"text"},
		Cost: cost{}, ContextWindow: 128000, MaxTokens: 16384,
	})
	addIfMissing(models, model{
		ID: "gpt-5.4", Name: "GPT-5.4", API: "openai-responses", Provider: "openai",
		BaseURL: "https://api.openai.com/v1", Reasoning: true, Input: []string{"text", "image"},
		Cost: cost{Input: 2.5, Output: 15, CacheRead: 0.25}, ContextWindow: 272000, MaxTokens: 128000,
	})
	*models = dedupeAndSort(*models)
}

func addIfMissing(models *[]model, candidate model) {
	for _, existing := range *models {
		if existing.ID == candidate.ID && existing.Provider == candidate.Provider {
			return
		}
	}
	*models = append(*models, candidate)
}

func applyOpenAIOverrides(models []model) {
	for i := range models {
		if models[i].Provider == "openai" && (models[i].ID == "gpt-5.4" || models[i].ID == "gpt-5.5") {
			models[i].ContextWindow = 272000
			models[i].MaxTokens = 128000
		}
	}
}

func applyThinkingLevelMetadata(m *model) {
	if m.API == "openai-responses" && strings.HasPrefix(m.ID, "gpt-5") {
		mergeThinkingLevelMap(m, "off", nil)
	}
	if supportsOpenAIXHigh(m.ID) {
		xhigh := "xhigh"
		mergeThinkingLevelMap(m, "xhigh", &xhigh)
	}
}

func supportsOpenAIXHigh(modelID string) bool {
	return strings.Contains(modelID, "gpt-5.2") ||
		strings.Contains(modelID, "gpt-5.3") ||
		strings.Contains(modelID, "gpt-5.4") ||
		strings.Contains(modelID, "gpt-5.5")
}

func mergeThinkingLevelMap(m *model, level string, value *string) {
	if m.ThinkingLevelMap == nil {
		m.ThinkingLevelMap = map[string]*string{}
	}
	m.ThinkingLevelMap[level] = value
}

func dedupeAndSort(models []model) []model {
	byID := make(map[string]model, len(models))
	for _, m := range models {
		if _, ok := byID[m.ID]; !ok {
			byID[m.ID] = m
		}
	}

	ids := make([]string, 0, len(byID))
	for id := range byID {
		ids = append(ids, id)
	}
	sort.Strings(ids)

	sorted := make([]model, 0, len(ids))
	for _, id := range ids {
		sorted = append(sorted, byID[id])
	}
	return sorted
}

func writeGenerated(models []model) error {
	var out bytes.Buffer
	out.WriteString("// Code generated by scripts/generate-models.go; DO NOT EDIT.\n\n")
	out.WriteString("package ai\n\n")
	out.WriteString("import \"github.com/govalues/decimal\"\n\n")
	out.WriteString("var Models = map[Provider]map[string]Model[API]{\n")
	out.WriteString("\tProviderOpenAI: {\n")

	for _, m := range models {
		writeModel(&out, m)
	}

	out.WriteString("\t},\n")
	out.WriteString("}\n\n")
	out.WriteString("func modelThinkingLevel(value string) *string { return &value }\n")

	formatted, err := format.Source(out.Bytes())
	if err != nil {
		return fmt.Errorf("format generated source: %w\n%s", err, out.String())
	}
	return os.WriteFile("internal/ai/models.generated.go", formatted, 0o644)
}

func writeModel(out *bytes.Buffer, m model) {
	fmt.Fprintf(out, "\t\t%s: {\n", quote(m.ID))
	fmt.Fprintf(out, "\t\t\tID: %s,\n", quote(m.ID))
	fmt.Fprintf(out, "\t\t\tName: %s,\n", quote(m.Name))
	fmt.Fprintf(out, "\t\t\tAPI: APIOpenAIResponses,\n")
	fmt.Fprintf(out, "\t\t\tProvider: ProviderOpenAI,\n")
	fmt.Fprintf(out, "\t\t\tBaseURL: %s,\n", quote(m.BaseURL))
	fmt.Fprintf(out, "\t\t\tReasoning: %t,\n", m.Reasoning)
	if len(m.ThinkingLevelMap) > 0 {
		writeThinkingLevelMap(out, m.ThinkingLevelMap)
	}
	fmt.Fprintf(out, "\t\t\tInput: []string{%s},\n", quotedStrings(m.Input))
	fmt.Fprintf(out, "\t\t\tCost: Cost{\n")
	fmt.Fprintf(out, "\t\t\t\tInput: decimal.MustParse(%s),\n", quote(decimalString(m.Cost.Input)))
	fmt.Fprintf(out, "\t\t\t\tOutput: decimal.MustParse(%s),\n", quote(decimalString(m.Cost.Output)))
	fmt.Fprintf(out, "\t\t\t\tCacheRead: decimal.MustParse(%s),\n", quote(decimalString(m.Cost.CacheRead)))
	fmt.Fprintf(out, "\t\t\t\tCacheWrite: decimal.MustParse(%s),\n", quote(decimalString(m.Cost.CacheWrite)))
	fmt.Fprintf(out, "\t\t\t},\n")
	fmt.Fprintf(out, "\t\t\tContextWindow: %d,\n", m.ContextWindow)
	fmt.Fprintf(out, "\t\t\tMaxTokens: %d,\n", m.MaxTokens)
	fmt.Fprintf(out, "\t\t},\n")
}

func writeThinkingLevelMap(out *bytes.Buffer, thinking map[string]*string) {
	fmt.Fprintf(out, "\t\t\tThinkingLevelMap: ThinkingLevelMap{\n")
	levels := make([]string, 0, len(thinking))
	for level := range thinking {
		levels = append(levels, level)
	}
	sort.Strings(levels)
	for _, level := range levels {
		value := thinking[level]
		if value == nil {
			fmt.Fprintf(out, "\t\t\t\t%s: nil,\n", thinkingLevelConstant(level))
		} else {
			fmt.Fprintf(out, "\t\t\t\t%s: modelThinkingLevel(%s),\n", thinkingLevelConstant(level), quote(*value))
		}
	}
	fmt.Fprintf(out, "\t\t\t},\n")
}

func thinkingLevelConstant(level string) string {
	switch level {
	case "off":
		return "ThinkingLevelOff"
	case "minimal":
		return "ThinkingLevelMinimal"
	case "low":
		return "ThinkingLevelLow"
	case "medium":
		return "ThinkingLevelMedium"
	case "high":
		return "ThinkingLevelHigh"
	case "xhigh":
		return "ThinkingLevelXHigh"
	default:
		panic("unknown thinking level: " + level)
	}
}

func quotedStrings(values []string) string {
	parts := make([]string, len(values))
	for i, value := range values {
		parts[i] = quote(value)
	}
	return strings.Join(parts, ", ")
}

func quote(value string) string {
	return strconv.Quote(value)
}

func decimalString(value float64) string {
	return strconv.FormatFloat(value, 'f', -1, 64)
}
