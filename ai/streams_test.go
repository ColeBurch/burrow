package ai

import (
	"testing"
	"time"
)

type completionFunction func(Model[API], ModelContext) (AssistantMessage, error)

func TestCompleteDrainsEvents(t *testing.T) {
	testCompletionDrainsEvents(t, func(model Model[API], modelContext ModelContext) (AssistantMessage, error) {
		return Complete(model, modelContext, nil)
	})
}

func TestCompleteSimpleDrainsEvents(t *testing.T) {
	testCompletionDrainsEvents(t, func(model Model[API], modelContext ModelContext) (AssistantMessage, error) {
		return CompleteSimple(model, modelContext, nil)
	})
}

func testCompletionDrainsEvents(t *testing.T, complete completionFunction) {
	t.Helper()

	const bufferSize = 64

	testAPI := API("test-" + t.Name())
	sourceID := "test-" + t.Name()
	stream := NewAssistantMessageEventStream(bufferSize)
	bufferFilled := make(chan struct{})
	finalMessage := AssistantMessage{
		Role:       RoleAssistant,
		Api:        testAPI,
		Provider:   Provider("test"),
		Model:      "test-model",
		StopReason: StopReasonStop,
	}

	startProducer := func() {
		go func() {
			for range bufferSize {
				stream.Push(TextDeltaEvent{Type: "text_delta"})
			}
			close(bufferFilled)

			// This event blocks until the completion function consumes an event.
			stream.Push(TextDeltaEvent{Type: "text_delta"})
			stream.Push(DoneEvent{Type: "done", Message: finalMessage})
		}()
	}

	RegisterApiProvider(ApiProvider[API, StreamOptions]{
		Api: testAPI,
		Stream: func(
			Model[API],
			ModelContext,
			*StreamOptions,
		) (*AssistantMessageEventStream, error) {
			startProducer()
			return stream, nil
		},
		StreamSimple: func(
			Model[API],
			ModelContext,
			*SimpleStreamOptions,
		) (*AssistantMessageEventStream, error) {
			startProducer()
			return stream, nil
		},
	}, &sourceID)
	defer UnregisterApiProvider(sourceID)

	type completionResult struct {
		message AssistantMessage
		err     error
	}
	resultReady := make(chan completionResult, 1)
	go func() {
		message, err := complete(Model[API]{API: testAPI}, ModelContext{})
		resultReady <- completionResult{message: message, err: err}
	}()

	select {
	case <-bufferFilled:
	case <-time.After(time.Second):
		t.Fatal("the provider did not fill the event buffer")
	}

	select {
	case result := <-resultReady:
		if result.err != nil {
			t.Fatalf("completion function returned an error: %v", result.err)
		}
		if result.message.Model != finalMessage.Model {
			t.Fatalf("completion function returned model %q, want %q", result.message.Model, finalMessage.Model)
		}
	case <-time.After(time.Second):
		// Drain the stream so the blocked goroutines can exit after this failure.
		go func() {
			for range stream.Events() {
			}
		}()

		select {
		case <-resultReady:
		case <-time.After(time.Second):
			t.Fatal("completion function remained blocked after test cleanup")
		}

		t.Fatal("completion function blocked because it did not consume stream events")
	}
}
