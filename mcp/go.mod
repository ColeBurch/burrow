module github.com/ColeBurch/burrow/mcp

go 1.27.3

require (
	github.com/ColeBurch/burrow/agent v0.7.0
	github.com/ColeBurch/burrow/ai v0.7.0
	github.com/ColeBurch/burrow/baseAgent v0.7.0
	github.com/modelcontextprotocol/go-sdk v1.8.0
	golang.org/x/oauth2 v0.35.0
)

require (
	github.com/google/jsonschema-go v0.4.3 // indirect
	github.com/govalues/decimal v0.1.36 // indirect
	github.com/openai/openai-go/v3 v3.35.0 // indirect
	github.com/santhosh-tekuri/jsonschema/v6 v6.0.2 // indirect
	github.com/segmentio/asm v1.1.3 // indirect
	github.com/segmentio/encoding v0.5.4 // indirect
	github.com/tidwall/gjson v1.18.0 // indirect
	github.com/tidwall/match v1.1.1 // indirect
	github.com/tidwall/pretty v1.2.1 // indirect
	github.com/tidwall/sjson v1.2.5 // indirect
	github.com/yosida95/uritemplate/v3 v3.0.2 // indirect
	golang.org/x/sync v0.22.0 // indirect
	golang.org/x/sys v0.41.0 // indirect
	golang.org/x/text v0.21.0 // indirect
	golang.org/x/time v0.15.0 // indirect
)

// Local development only; ignored when mcp is used as a dependency.
replace github.com/ColeBurch/burrow/agent => ../agent

// Local development only; ignored when mcp is used as a dependency.
replace github.com/ColeBurch/burrow/ai => ../ai

// Local development only; ignored when mcp is used as a dependency.
replace github.com/ColeBurch/burrow/baseAgent => ../baseAgent
