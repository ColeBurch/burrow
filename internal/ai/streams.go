package ai

import "fmt"

func ResolveAPIProvider(api API) (ApiProviderInternal, error) {
	provider, ok := GetApiProvider(api)
	if !ok {
		return ApiProviderInternal{}, fmt.Errorf("No API provider found for: %s", api)
	}
	return provider, nil
}

func Stream(model Model[API], modelContext ModelContext, options *StreamOptions) (*AssistantMessageEventStream, error) {
	provider, err := ResolveAPIProvider(model.API)
	if err != nil {
		return nil, err
	}

	return provider.Stream(model, modelContext, options)
}

func Complete(model Model[API], modelContext ModelContext, options *StreamOptions) (AssistantMessage, error) {
	s, err := Stream(model, modelContext, options)
	if err != nil {
		return AssistantMessage{}, err
	}

	return s.stream.result.message, nil
}

func StreamSimple(model Model[API], modelContext ModelContext, options *SimpleStreamOptions) (*AssistantMessageEventStream, error) {
	provider, err := ResolveAPIProvider(model.API)
	if err != nil {
		return nil, err
	}

	return provider.StreamSimple(model, modelContext, options)
}

func CompleteSimple(model Model[API], modelContext ModelContext, options *SimpleStreamOptions) (AssistantMessage, error) {
	s, err := StreamSimple(model, modelContext, options)
	if err != nil {
		return AssistantMessage{}, err
	}

	return s.stream.result.message, nil
}
