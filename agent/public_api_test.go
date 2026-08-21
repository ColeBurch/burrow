package agent_test

import (
	"testing"

	"github.com/ColeBurch/burrow/agent"
)

func TestToolCallHookContextsExportValidatedArguments(t *testing.T) {
	beforeArgs := &struct{ value string }{value: "before"}
	beforeCall := agent.BeforeToolCallContext{Args: beforeArgs}
	if beforeCall.Args != beforeArgs {
		t.Fatalf("BeforeToolCallContext.Args = %#v, want %#v", beforeCall.Args, beforeArgs)
	}

	afterArgs := &struct{ value string }{value: "after"}
	afterCall := agent.AfterToolCallContext[any]{Args: afterArgs}
	if afterCall.Args != afterArgs {
		t.Fatalf("AfterToolCallContext.Args = %#v, want %#v", afterCall.Args, afterArgs)
	}
}
