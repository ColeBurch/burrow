package ai_test

import (
	"context"
	"testing"

	"github.com/ColeBurch/burrow/ai"
	"github.com/ColeBurch/burrow/ai/providertest"
)

// streamFromServer runs StreamOpenAIResponse against server and drains the
// stream. It fills in an API key and the server's client unless options sets
// them, and returns the event types with the stream's result.
func streamFromServer(
	t *testing.T,
	ctx context.Context,
	server *providertest.Server,
	model ai.Model[ai.API],
	modelContext ai.ModelContext,
	options *ai.OpenAIResponseOptions,
) ([]string, ai.AssistantMessage, error) {
	t.Helper()
	if options == nil {
		options = &ai.OpenAIResponseOptions{}
	}
	if options.ApiKey == nil {
		apiKey := "test-key"
		options.ApiKey = &apiKey
	}
	if options.HTTPClient == nil {
		options.HTTPClient = server.Client()
	}

	stream, err := ai.StreamOpenAIResponse(ctx, model, modelContext, options)
	if err != nil {
		t.Fatal(err)
	}
	var eventTypes []string
	for event := range stream.Events() {
		eventTypes = append(eventTypes, event.EventType())
	}
	msg, err := stream.Result()
	return eventTypes, msg, err
}

func userText(text string) ai.UserMessage {
	return ai.UserMessage{Role: ai.RoleUser, Content: []ai.UserContent{ai.TextContent{Type: ai.ContentTypeText, Text: text}}}
}
