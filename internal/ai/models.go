package ai

import "github.com/govalues/decimal"

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
