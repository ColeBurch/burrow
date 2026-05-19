package agent

import (
	"context"
	"errors"
	"time"

	"github.com/ColeBurch/burrow/internal/ai"
)

// DefaultStreamFn adapts the lower-level ai.StreamSimple function to the
// agent-level StreamFn contract.
func DefaultStreamFn(
	ctx context.Context,
	model ai.Model[ai.API],
	modelContext ai.ModelContext,
	options *ai.SimpleStreamOptions,
) (*ai.AssistantMessageEventStream, error) {
	return ai.StreamSimple(model, modelContext, withStreamContext(ctx, options))
}

func withStreamContext(ctx context.Context, options *ai.SimpleStreamOptions) *ai.SimpleStreamOptions {
	if ctx == nil {
		return options
	}

	if options == nil {
		return &ai.SimpleStreamOptions{
			StreamOptions: ai.StreamOptions{Signal: ctx},
		}
	}

	copied := *options
	copied.StreamOptions = options.StreamOptions
	copied.Signal = ctx
	return &copied
}

// AssistantErrorMessage converts a StreamFn startup error into the assistant
// message shape expected by the agent loop. The loop should emit this as a
// normal message_start/message_end, followed by turn_end and agent_end.
func AssistantErrorMessage(ctx context.Context, model ai.Model[ai.API], err error) ai.AssistantMessage {
	reason := ai.StopReasonError
	if IsAbortError(ctx, err) {
		reason = ai.StopReasonAborted
	}

	message := ""
	if err != nil {
		message = err.Error()
	}

	return ai.AssistantMessage{
		Role:         ai.RoleAssistant,
		Content:      []ai.AssistantContent{},
		Api:          model.API,
		Provider:     model.Provider,
		Model:        model.ID,
		StopReason:   reason,
		ErrorMessage: message,
		Timestamp:    time.Now().UnixMilli(),
	}
}

func IsAbortError(ctx context.Context, err error) bool {
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return true
	}
	return ctx != nil && ctx.Err() != nil
}
