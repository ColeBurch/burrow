package ai

import (
	"strings"
	"testing"
)

func TestIsContextOverflowProviderErrors(t *testing.T) {
	tests := []struct {
		name string
		err  string
	}{
		{"anthropic tokens", "prompt is too long: 213462 tokens > 200000 maximum"},
		{"anthropic bytes", `413 {"error":{"type":"request_too_large","message":"Request exceeds the maximum size"}}`},
		{"bedrock", "ValidationException: Input is too long for requested model"},
		{"openai responses", "Your input exceeds the context window of this model"},
		{"google", "The input token count (1196265) exceeds the maximum number of tokens allowed (1048575)"},
		{"xai", "This model's maximum prompt length is 131072 but the request contains 537812 tokens"},
		{"groq", "Please reduce the length of the messages or completion"},
		{"openrouter and openai completions", "This endpoint's maximum context length is 8192 tokens. However, you requested about 9000 tokens"},
		{"copilot", "prompt token count of 9000 exceeds the limit of 8192"},
		{"llama cpp", "the request exceeds the available context size, try increasing it"},
		{"lm studio", "tokens to keep from the initial prompt is greater than the context length"},
		{"minimax", "invalid params, context window exceeds limit"},
		{"kimi", "Your request exceeded model token limit: 131072 (requested: 200000)"},
		{"mistral", "Prompt contains 200000 tokens and is too large for model with 131072 maximum context length"},
		{"z ai", "model_context_window_exceeded"},
		{"ollama max", "400 `prompt too long; exceeded max context length by 100918 tokens`"},
		{"ollama", "prompt too long; exceeded context length by 100 tokens"},
		{"generic code", `400 {"code":"context_length_exceeded"}`},
		{"generic text", "context length exceeded"},
		{"generic mixed separators", "context_length exceeded"},
		{"generic tokens", "Too many tokens in the prompt"},
		{"generic token limit", "Token limit exceeded"},
		{"cerebras 400", "400 (no body)"},
		{"cerebras 413", "413 (no body)"},
		{"cerebras 400 status", "400 status code (no body)"},
		{"cerebras 413 status", "413 status code (no body)"},
		{"cerebras whitespace", "413\tstatus code\n(no body)"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for _, text := range []string{tt.err, strings.ToUpper(tt.err)} {
				for _, window := range []int{0, -1, 8192} {
					message := AssistantMessage{StopReason: StopReasonError, ErrorMessage: text}
					if !IsContextOverflow(message, window) {
						t.Errorf("IsContextOverflow(error=%q, window=%d) = false, want true", text, window)
					}
				}
			}
		})
	}
}

func TestIsContextOverflowExcludesNonOverflowErrors(t *testing.T) {
	errors := []string{
		"",
		"500 `model runner crashed unexpectedly`",
		"Invalid API key",
		"Request timed out",
		"400 invalid request",
		"413 Request Entity Too Large",
		"400 status code (invalid request)",
		"500 status code (no body)",
		"429 status code (no body)",
		"1400 (no body)",
		"error: 400 (no body)",
		"Throttling error: Too many tokens, please wait before trying again.",
		"ThrottlingException: Too many tokens, please wait before trying again.",
		"Request throttled: token limit exceeded",
		"Service unavailable: The service is temporarily unavailable.",
		"Service unavailable: Too many tokens",
		"Rate limit exceeded, please retry after 30 seconds.",
		"Rate limit: Too many tokens",
		"Too many tokens: rate limit exceeded",
		`{"code":"rate_limit_exceeded","message":"token limit exceeded"}`,
		"Rate-limit exceeded: Too many tokens",
		"RateLimitExceeded: Too many tokens",
		"Too many requests. Please slow down.",
		"Too many requests: context_length_exceeded",
	}

	for _, text := range errors {
		t.Run(text, func(t *testing.T) {
			for _, variant := range []string{text, strings.ToUpper(text)} {
				message := AssistantMessage{
					StopReason:   StopReasonError,
					ErrorMessage: variant,
					Usage:        Usage{Input: 200000, CacheRead: 10000},
				}
				if IsContextOverflow(message, 8192) {
					t.Errorf("IsContextOverflow(error=%q) = true, want false even with excessive usage", variant)
				}
			}
		})
	}
}

func TestIsContextOverflowUsage(t *testing.T) {
	tests := []struct {
		name   string
		stop   StopReason
		usage  Usage
		window int
		want   bool
	}{
		{"successful input above window", StopReasonStop, Usage{Input: 10001, Output: 50}, 10000, true},
		{"successful cached input above window", StopReasonStop, Usage{Input: 5001, CacheRead: 5000}, 10000, true},
		{"successful cache alone above window", StopReasonStop, Usage{CacheRead: 10001}, 10000, true},
		{"successful input at window", StopReasonStop, Usage{Input: 10000}, 10000, false},
		{"successful cached input at window", StopReasonStop, Usage{Input: 5000, CacheRead: 5000}, 10000, false},
		{"successful input below window", StopReasonStop, Usage{Input: 9999}, 10000, false},
		{"successful empty usage", StopReasonStop, Usage{}, 10000, false},
		{"successful output excluded", StopReasonStop, Usage{Input: 9000, Output: 2000}, 10000, false},
		{"successful cache write excluded", StopReasonStop, Usage{Input: 9000, CacheWrite: 2000}, 10000, false},
		{"successful total excluded", StopReasonStop, Usage{Input: 9000, Total: 20000}, 10000, false},
		{"successful unknown window", StopReasonStop, Usage{Input: 10001}, 0, false},
		{"successful invalid window", StopReasonStop, Usage{Input: 10001}, -1, false},
		{"length input at window", StopReasonLength, Usage{Input: 10000}, 10000, true},
		{"length input above window", StopReasonLength, Usage{Input: 10001}, 10000, true},
		{"length input at threshold", StopReasonLength, Usage{Input: 9900}, 10000, true},
		{"length input below threshold", StopReasonLength, Usage{Input: 9899}, 10000, false},
		{"length cached input at threshold", StopReasonLength, Usage{Input: 4900, CacheRead: 5000}, 10000, true},
		{"length cached input below threshold", StopReasonLength, Usage{Input: 4899, CacheRead: 5000}, 10000, false},
		{"length cache alone at threshold", StopReasonLength, Usage{CacheRead: 9900}, 10000, true},
		{"xiaomi regression", StopReasonLength, Usage{Input: 58, CacheRead: 1048512}, 1048576, true},
		{"normal output limit", StopReasonLength, Usage{Input: 1000, Output: 4096}, 200000, false},
		{"length nonzero output at window", StopReasonLength, Usage{Input: 10000, Output: 1}, 10000, false},
		{"length nonzero output above window", StopReasonLength, Usage{Input: 10001, Output: 1}, 10000, false},
		{"length far below window", StopReasonLength, Usage{Input: 100}, 200000, false},
		{"length empty usage", StopReasonLength, Usage{}, 10000, false},
		{"length cache write excluded", StopReasonLength, Usage{Input: 100, CacheWrite: 10000}, 10000, false},
		{"length total excluded", StopReasonLength, Usage{Input: 100, Total: 10000}, 10000, false},
		{"length unknown window", StopReasonLength, Usage{Input: 10000}, 0, false},
		{"length invalid window", StopReasonLength, Usage{Input: 10000}, -1, false},
		{"fractional threshold rounded up", StopReasonLength, Usage{Input: 100}, 101, true},
		{"below fractional threshold", StopReasonLength, Usage{Input: 99}, 101, false},
		{"small window filled", StopReasonLength, Usage{Input: 1}, 1, true},
		{"small window empty", StopReasonLength, Usage{}, 1, false},
		{"error usage alone ignored", StopReasonError, Usage{Input: 10001}, 10000, false},
		{"aborted usage ignored", StopReasonAborted, Usage{Input: 10001}, 10000, false},
		{"tool use usage ignored", StopReasonToolUse, Usage{Input: 10001}, 10000, false},
		{"unset stop reason ignored", "", Usage{Input: 10001}, 10000, false},
		{"unknown stop reason ignored", "custom", Usage{Input: 10001}, 10000, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			message := AssistantMessage{StopReason: tt.stop, Usage: tt.usage}
			if got := IsContextOverflow(message, tt.window); got != tt.want {
				t.Errorf("IsContextOverflow(stop=%q, usage=%+v, window=%d) = %v, want %v", tt.stop, tt.usage, tt.window, got, tt.want)
			}
		})
	}
}

func TestIsContextOverflowRequiresErrorStopForErrorText(t *testing.T) {
	for _, stop := range []StopReason{StopReasonStop, StopReasonLength, StopReasonToolUse, StopReasonAborted, "", "custom"} {
		t.Run(string(stop), func(t *testing.T) {
			message := AssistantMessage{StopReason: stop, ErrorMessage: "prompt is too long"}
			if IsContextOverflow(message, 10000) {
				t.Fatal("IsContextOverflow = true, want false for error text without an error stop")
			}
		})
	}
}

func TestIsContextOverflowZeroMessage(t *testing.T) {
	if IsContextOverflow(AssistantMessage{}, 0) {
		t.Fatal("IsContextOverflow = true, want false for zero message and unknown window")
	}
}
