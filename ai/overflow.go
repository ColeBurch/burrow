package ai

import "regexp"

var contextOverflowPatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)prompt is too long`),
	regexp.MustCompile(`(?i)request_too_large`),
	regexp.MustCompile(`(?i)input is too long for requested model`),
	regexp.MustCompile(`(?i)exceeds the context window`),
	regexp.MustCompile(`(?i)input token count.*exceeds the maximum`),
	regexp.MustCompile(`(?i)maximum prompt length is \d+`),
	regexp.MustCompile(`(?i)reduce the length of the messages`),
	regexp.MustCompile(`(?i)maximum context length is \d+ tokens`),
	regexp.MustCompile(`(?i)exceeds the limit of \d+`),
	regexp.MustCompile(`(?i)exceeds the available context size`),
	regexp.MustCompile(`(?i)greater than the context length`),
	regexp.MustCompile(`(?i)context window exceeds limit`),
	regexp.MustCompile(`(?i)exceeded model token limit`),
	regexp.MustCompile(`(?i)too large for model with \d+ maximum context length`),
	regexp.MustCompile(`(?i)model_context_window_exceeded`),
	regexp.MustCompile(`(?i)prompt too long; exceeded (?:max )?context length`),
	regexp.MustCompile(`(?i)context[_ ]length[_ ]exceeded`),
	regexp.MustCompile(`(?i)too many tokens`),
	regexp.MustCompile(`(?i)token limit exceeded`),
	regexp.MustCompile(`(?i)^4(?:00|13)\s*(?:status code)?\s*\(no body\)`),
}

// Throttling can mention "too many tokens" without indicating context overflow.
// Include raw provider exception codes as well as human-readable error messages.
var nonContextOverflowPattern = regexp.MustCompile(`(?i)throttl|^service unavailable:|rate[ _-]?limit|too many requests`)

// IsContextOverflow reports whether message indicates that the input exceeded the
// model's context window. It recognizes known provider errors, successful stops
// with Input + CacheRead greater than contextWindow, and length stops with zero
// output and Input + CacheRead filling at least 99% of contextWindow.
//
// A non-positive contextWindow disables usage-based checks, but not error matching.
// Rate-limit, throttling, and service-unavailable errors are not context overflow.
// Silent input truncation without one of these signals cannot be detected.
func IsContextOverflow(message AssistantMessage, contextWindow int) bool {
	if message.StopReason == StopReasonError && message.ErrorMessage != "" {
		if nonContextOverflowPattern.MatchString(message.ErrorMessage) {
			return false
		}
		for _, pattern := range contextOverflowPatterns {
			if pattern.MatchString(message.ErrorMessage) {
				return true
			}
		}
	}

	if contextWindow <= 0 {
		return false
	}

	inputTokens := message.Usage.Input + message.Usage.CacheRead
	window := int64(contextWindow)
	switch message.StopReason {
	case StopReasonStop:
		return inputTokens > window
	case StopReasonLength:
		// ceil(window * 0.99), without floating-point rounding or multiplication.
		return message.Usage.Output == 0 && inputTokens >= window-window/100
	default:
		return false
	}
}
