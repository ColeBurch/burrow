package ai

type DiagnosticErrorInfo struct {
	Name    *string `json:"name,omitempty"`
	Message string  `json:"message"`
	Stack   *string `json:"stack,omitempty"`
	Code    any     `json:"code,omitempty"`
}

type AssistantMessageDiagnostic struct {
	Type      string               `json:"type"`
	Timestamp int64                `json:"timestamp"`
	Error     *DiagnosticErrorInfo `json:"error,omitempty"`
	Details   any                  `json:"details,omitempty"`
}
