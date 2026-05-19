package ai

import (
	"errors"
	"sync"
)

type EventStream[T any, R any] struct {
	events        chan T
	isComplete    func(T) bool
	extractResult func(T) R
	mu            sync.Mutex
	done          bool
	result        R
	resultReady   chan struct{}
	closeResult   sync.Once
}

func NewEventStream[T any, R any](
	buffer int,
	isComplete func(T) bool,
	extractResult func(T) R,
) *EventStream[T, R] {
	return &EventStream[T, R]{
		events:        make(chan T, buffer),
		isComplete:    isComplete,
		extractResult: extractResult,
		resultReady:   make(chan struct{}),
	}
}

func (s *EventStream[T, R]) Push(event T) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.done {
		return
	}

	complete := s.isComplete(event)
	if complete {
		s.done = true
		s.result = s.extractResult(event)
		s.closeResult.Do(func() {
			close(s.resultReady)
		})
	}

	s.events <- event

	if complete {
		close(s.events)
	}
}

func (s *EventStream[T, R]) End(result R) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.done {
		return
	}

	s.done = true
	s.result = result
	s.closeResult.Do(func() {
		close(s.resultReady)
	})
	close(s.events)
}

func (s *EventStream[T, R]) Events() <-chan T {
	return s.events
}

func (s *EventStream[T, R]) Result() R {
	<-s.resultReady
	return s.result
}

type AssistantMessageEventType string

const (
	AssistantMessageEventText  AssistantMessageEventType = "text"
	AssistantMessageEventDone  AssistantMessageEventType = "done"
	AssistantMessageEventError AssistantMessageEventType = "error"
)

type assistantMessageResult struct {
	message AssistantMessage
	err     error
}

type AssistantMessageEventStream struct {
	stream *EventStream[AssistantMessageEvent, assistantMessageResult]
}

func NewAssistantMessageEventStream(buffer int) *AssistantMessageEventStream {
	return &AssistantMessageEventStream{
		stream: NewEventStream(
			buffer,
			func(event AssistantMessageEvent) bool {
				return AssistantMessageEventType(event.EventType()) == AssistantMessageEventDone ||
					AssistantMessageEventType(event.EventType()) == AssistantMessageEventError
			},
			func(event AssistantMessageEvent) assistantMessageResult {
				switch event := event.(type) {
				case DoneEvent:
					return assistantMessageResult{message: event.Message}
				case ErrorEvent:
					err := errors.New(event.Message.ErrorMessage)
					if event.Message.ErrorMessage == "" {
						err = errors.New("assistant message stream ended with an error")
					}
					return assistantMessageResult{
						message: event.Message,
						err:     err,
					}
				default:
					return assistantMessageResult{
						err: errors.New("unexpected event type for final result"),
					}
				}
			},
		),
	}
}

func (s *AssistantMessageEventStream) Push(event AssistantMessageEvent) {
	s.stream.Push(event)
}

func (s *AssistantMessageEventStream) End(err error) {
	s.stream.End(assistantMessageResult{err: err})
}

func (s *AssistantMessageEventStream) Events() <-chan AssistantMessageEvent {
	return s.stream.Events()
}

func (s *AssistantMessageEventStream) Result() (AssistantMessage, error) {
	result := s.stream.Result()
	return result.message, result.err
}
