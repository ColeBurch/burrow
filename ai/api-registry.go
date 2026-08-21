package ai

import (
	"fmt"
	"sync"
)

type ApiStreamFunction func(
	Model Model[API],
	ModelContext ModelContext,
	Options *StreamOptions,
) (*AssistantMessageEventStream, error)

type ApiStreamSimpleFunction func(
	model Model[API],
	ModelContext ModelContext,
	options *SimpleStreamOptions,
) (*AssistantMessageEventStream, error)

type StreamFunction[TApi API, TOptions any] func(
	model Model[TApi],
	ModelContext ModelContext,
	options *TOptions,
) (*AssistantMessageEventStream, error)

type ApiProvider[TApi API, TOptions any] struct {
	Api          TApi
	Stream       StreamFunction[TApi, TOptions]
	StreamSimple StreamFunction[TApi, SimpleStreamOptions]
}

type ApiProviderInternal struct {
	Api          API
	Stream       ApiStreamFunction
	StreamSimple ApiStreamSimpleFunction
}

type RegisteredApiProvider struct {
	Provider ApiProviderInternal
	SourceID *string
}

var (
	apiProviderRegistry = make(map[API]RegisteredApiProvider)
	apiProviderLock     = sync.RWMutex{}
)

func wrapStream[TApi API](
	api TApi,
	stream StreamFunction[TApi, StreamOptions],
) ApiStreamFunction {
	return func(
		model Model[API],
		modelContext ModelContext,
		options *StreamOptions,
	) (*AssistantMessageEventStream, error) {
		if model.API != API(api) {
			return nil, fmt.Errorf("mismatched api: %s expected %s", model.API, api)
		}
		if stream == nil {
			return nil, fmt.Errorf("API provider %s does not implement Stream", api)
		}

		typedModel := modelWithAPI(model, api)

		return stream(typedModel, modelContext, options)
	}
}

func wrapStreamSimple[TApi API](
	api TApi,
	streamSimple StreamFunction[TApi, SimpleStreamOptions],
) ApiStreamSimpleFunction {
	return func(
		model Model[API],
		modelContext ModelContext,
		options *SimpleStreamOptions,
	) (*AssistantMessageEventStream, error) {
		if model.API != API(api) {
			return nil, fmt.Errorf("mismatched api: %s expected %s", model.API, api)
		}
		if streamSimple == nil {
			return nil, fmt.Errorf("API provider %s does not implement StreamSimple", api)
		}

		typedModel := modelWithAPI(model, api)

		return streamSimple(typedModel, modelContext, options)
	}
}

func RegisterApiProvider[TApi API](provider ApiProvider[TApi, StreamOptions], sourceID *string) {
	apiProviderLock.Lock()
	defer apiProviderLock.Unlock()

	apiProviderRegistry[API(provider.Api)] = RegisteredApiProvider{
		Provider: ApiProviderInternal{
			Api:          API(provider.Api),
			Stream:       wrapStream(provider.Api, provider.Stream),
			StreamSimple: wrapStreamSimple(provider.Api, provider.StreamSimple),
		},
		SourceID: sourceID,
	}
}

func GetApiProvider(api API) (ApiProviderInternal, bool) {
	apiProviderLock.RLock()
	defer apiProviderLock.RUnlock()

	provider, ok := apiProviderRegistry[api]
	if !ok {
		return ApiProviderInternal{}, false
	}
	return provider.Provider, ok
}

func GetApiProviders() []ApiProviderInternal {
	apiProviderLock.RLock()
	defer apiProviderLock.RUnlock()

	var providers []ApiProviderInternal
	for _, provider := range apiProviderRegistry {
		providers = append(providers, provider.Provider)
	}
	return providers
}

func UnregisterApiProvider(sourceID string) {
	apiProviderLock.Lock()
	defer apiProviderLock.Unlock()

	for api, provider := range apiProviderRegistry {
		if provider.SourceID != nil && *provider.SourceID == sourceID {
			delete(apiProviderRegistry, api)
		}
	}
}

func ClearApiProviders() {
	apiProviderLock.Lock()
	defer apiProviderLock.Unlock()

	clear(apiProviderRegistry)
}
