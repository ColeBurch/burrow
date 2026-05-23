module github.com/ColeBurch/burrow/baseAgent

go 1.26.3

require (
	github.com/ColeBurch/burrow/agent v0.1.3
	github.com/ColeBurch/burrow/ai v0.1.3
	github.com/google/uuid v1.6.0
)

require (
	github.com/govalues/decimal v0.1.36 // indirect
	github.com/openai/openai-go/v3 v3.35.0 // indirect
	github.com/santhosh-tekuri/jsonschema/v6 v6.0.2 // indirect
	github.com/tidwall/gjson v1.18.0 // indirect
	github.com/tidwall/match v1.1.1 // indirect
	github.com/tidwall/pretty v1.2.1 // indirect
	github.com/tidwall/sjson v1.2.5 // indirect
	golang.org/x/text v0.21.0 // indirect
)

// Local development only; ignored when baseAgent is used as a dependency.
replace github.com/ColeBurch/burrow/agent => ../agent

// Local development only; ignored when baseAgent is used as a dependency.
replace github.com/ColeBurch/burrow/ai => ../ai
