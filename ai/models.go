package ai

import (
	"sort"

	"github.com/govalues/decimal"
)

var extendedThinkingLevels = []ModelThinkingLevel{
	ThinkingLevelOff,
	ThinkingLevelMinimal,
	ThinkingLevelLow,
	ThinkingLevelMedium,
	ThinkingLevelHigh,
	ThinkingLevelXHigh,
}

func GetModel(provider Provider, modelID string) (Model[API], bool) {
	providerModels, ok := Models[provider]
	if !ok {
		return Model[API]{}, false
	}

	model, ok := providerModels[modelID]
	return model, ok
}

func GetProviders() []Provider {
	providers := make([]Provider, 0, len(Models))
	for provider := range Models {
		providers = append(providers, provider)
	}

	sort.Slice(providers, func(i, j int) bool {
		return providers[i] < providers[j]
	})

	return providers
}

func GetModels(provider Provider) []Model[API] {
	providerModels, ok := Models[provider]
	if !ok {
		return []Model[API]{}
	}

	models := make([]Model[API], 0, len(providerModels))
	for _, model := range providerModels {
		models = append(models, model)
	}

	sort.Slice(models, func(i, j int) bool {
		return models[i].ID < models[j].ID
	})

	return models
}

func CalculateCost(model Model[API], usage Usage) (Usage, error) {
	inputCostMilli, err := model.Cost.Input.Mul(decimal.MustNew(usage.Input, 0))
	if err != nil {
		return Usage{}, err
	}
	inputCost, err := inputCostMilli.Quo(decimal.MustNew(1_000_000, 0))
	if err != nil {
		return Usage{}, err
	}
	usage.Cost.Input = inputCost
	outputCostMilli, err := model.Cost.Output.Mul(decimal.MustNew(usage.Output, 0))
	if err != nil {
		return Usage{}, err
	}
	outputCost, err := outputCostMilli.Quo(decimal.MustNew(1_000_000, 0))
	if err != nil {
		return Usage{}, err
	}
	usage.Cost.Output = outputCost
	cacheReadCostMilli, err := model.Cost.CacheRead.Mul(decimal.MustNew(usage.CacheRead, 0))
	if err != nil {
		return Usage{}, err
	}
	cacheReadCost, err := cacheReadCostMilli.Quo(decimal.MustNew(1_000_000, 0))
	if err != nil {
		return Usage{}, err
	}
	usage.Cost.CacheRead = cacheReadCost
	cacheWriteCostMilli, err := model.Cost.CacheWrite.Mul(decimal.MustNew(usage.CacheWrite, 0))
	if err != nil {
		return Usage{}, err
	}
	cacheWriteCost, err := cacheWriteCostMilli.Quo(decimal.MustNew(1_000_000, 0))
	if err != nil {
		return Usage{}, err
	}
	usage.Cost.CacheWrite = cacheWriteCost
	totalCost, err := usage.Cost.Input.Add(usage.Cost.Output)
	if err != nil {
		return Usage{}, err
	}
	totalCost, err = totalCost.Add(usage.Cost.CacheRead)
	if err != nil {
		return Usage{}, err
	}
	totalCost, err = totalCost.Add(usage.Cost.CacheWrite)
	if err != nil {
		return Usage{}, err
	}
	usage.Cost.Total = totalCost
	return usage, nil
}

func GetSupportedThinkingLevels[TApi API](model Model[TApi]) []ModelThinkingLevel {
	if !model.Reasoning {
		return []ModelThinkingLevel{ThinkingLevelOff}
	}

	levels := make([]ModelThinkingLevel, 0, len(extendedThinkingLevels))
	for _, level := range extendedThinkingLevels {
		mapped, ok := model.ThinkingLevelMap[level]
		if ok && mapped == nil {
			continue
		}
		if level == ThinkingLevelXHigh && !ok {
			continue
		}
		levels = append(levels, level)
	}

	return levels
}

func ClampThinkingLevel[TApi API](model Model[TApi], level ModelThinkingLevel) ModelThinkingLevel {
	availableLevels := GetSupportedThinkingLevels(model)
	if containsThinkingLevel(availableLevels, level) {
		return level
	}

	requestedIndex := thinkingLevelIndex(level)
	if requestedIndex == -1 {
		return firstThinkingLevelOrOff(availableLevels)
	}

	for i := requestedIndex; i < len(extendedThinkingLevels); i++ {
		candidate := extendedThinkingLevels[i]
		if containsThinkingLevel(availableLevels, candidate) {
			return candidate
		}
	}
	for i := requestedIndex - 1; i >= 0; i-- {
		candidate := extendedThinkingLevels[i]
		if containsThinkingLevel(availableLevels, candidate) {
			return candidate
		}
	}

	return firstThinkingLevelOrOff(availableLevels)
}

func ModelsAreEqual[TApi API](a *Model[TApi], b *Model[TApi]) bool {
	if a == nil || b == nil {
		return false
	}
	return a.ID == b.ID && a.Provider == b.Provider
}

func containsThinkingLevel(levels []ModelThinkingLevel, level ModelThinkingLevel) bool {
	for _, candidate := range levels {
		if candidate == level {
			return true
		}
	}
	return false
}

func thinkingLevelIndex(level ModelThinkingLevel) int {
	for i, candidate := range extendedThinkingLevels {
		if candidate == level {
			return i
		}
	}
	return -1
}

func firstThinkingLevelOrOff(levels []ModelThinkingLevel) ModelThinkingLevel {
	if len(levels) == 0 {
		return ThinkingLevelOff
	}
	return levels[0]
}
