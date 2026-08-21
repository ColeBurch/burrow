package baseAgent

import (
	"errors"
	"testing"

	"github.com/ColeBurch/burrow/ai"
)

func callProviderWithoutPanic(call func() error) (err error, panicValue any) {
	defer func() {
		panicValue = recover()
	}()

	err = call()
	return
}

func prepareDefaultProviderTest(t *testing.T) ai.Model[ai.API] {
	t.Helper()

	ai.ClearApiProviders()
	t.Cleanup(ai.ClearApiProviders)
	ensureDefaultAPIProvidersRegistered()

	return ai.Model[ai.API]{
		ID:       "test-model",
		API:      ai.APIOpenAIResponses,
		Provider: ai.ProviderOpenAI,
	}
}

func streamOptionsWithPayloadError(payloadErr error) *ai.StreamOptions {
	apiKey := "test-api-key"
	return &ai.StreamOptions{
		ApiKey: &apiKey,
		OnPayload: func(any, ai.Model[ai.API]) (any, error) {
			return nil, payloadErr
		},
	}
}

func TestDefaultAPIProviderStreamReturnsErrorWithoutPanic(t *testing.T) {
	model := prepareDefaultProviderTest(t)
	payloadErr := errors.New("stop before request")

	err, panicValue := callProviderWithoutPanic(func() error {
		_, err := ai.Stream(model, ai.ModelContext{}, streamOptionsWithPayloadError(payloadErr))
		return err
	})
	if panicValue != nil {
		t.Fatalf("ai.Stream() panicked: %v", panicValue)
	}
	if !errors.Is(err, payloadErr) {
		t.Fatalf("ai.Stream() error = %v, want %v", err, payloadErr)
	}
}

func TestDefaultAPIProviderCompleteReturnsErrorWithoutPanic(t *testing.T) {
	model := prepareDefaultProviderTest(t)
	payloadErr := errors.New("stop before request")

	err, panicValue := callProviderWithoutPanic(func() error {
		_, err := ai.Complete(model, ai.ModelContext{}, streamOptionsWithPayloadError(payloadErr))
		return err
	})
	if panicValue != nil {
		t.Fatalf("ai.Complete() panicked: %v", panicValue)
	}
	if !errors.Is(err, payloadErr) {
		t.Fatalf("ai.Complete() error = %v, want %v", err, payloadErr)
	}
}
