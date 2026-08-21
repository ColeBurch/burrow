package ai

import "testing"

func TestStreamReturnsErrorWhenProviderStreamIsMissing(t *testing.T) {
	testAPI := API("test-missing-stream")
	sourceID := "test-missing-stream"
	RegisterApiProvider(ApiProvider[API, StreamOptions]{
		Api: testAPI,
		StreamSimple: func(
			Model[API],
			ModelContext,
			*SimpleStreamOptions,
		) (*AssistantMessageEventStream, error) {
			return nil, nil
		},
	}, &sourceID)
	t.Cleanup(func() { UnregisterApiProvider(sourceID) })

	_, err := Stream(Model[API]{API: testAPI}, ModelContext{}, nil)
	if err == nil {
		t.Fatal("Stream() error = nil, want a missing Stream error")
	}
	if got, want := err.Error(), "API provider test-missing-stream does not implement Stream"; got != want {
		t.Fatalf("Stream() error = %q, want %q", got, want)
	}
}

func TestStreamSimpleReturnsErrorWhenProviderStreamSimpleIsMissing(t *testing.T) {
	testAPI := API("test-missing-stream-simple")
	sourceID := "test-missing-stream-simple"
	RegisterApiProvider(ApiProvider[API, StreamOptions]{
		Api: testAPI,
		Stream: func(
			Model[API],
			ModelContext,
			*StreamOptions,
		) (*AssistantMessageEventStream, error) {
			return nil, nil
		},
	}, &sourceID)
	t.Cleanup(func() { UnregisterApiProvider(sourceID) })

	_, err := StreamSimple(Model[API]{API: testAPI}, ModelContext{}, nil)
	if err == nil {
		t.Fatal("StreamSimple() error = nil, want a missing StreamSimple error")
	}
	if got, want := err.Error(), "API provider test-missing-stream-simple does not implement StreamSimple"; got != want {
		t.Fatalf("StreamSimple() error = %q, want %q", got, want)
	}
}
