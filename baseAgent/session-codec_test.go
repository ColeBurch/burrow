package baseAgent

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/ColeBurch/burrow/agent"
	"github.com/ColeBurch/burrow/ai"
)

// codecTestMessage is a consumer-defined message role.
type codecTestMessage struct {
	Role   string `json:"role"`
	Ticket string `json:"ticket"`
}

func (codecTestMessage) AgentMessageType() string { return "codecTest" }

func init() {
	RegisterMessageDecoder("codecTest", func(data []byte) (agent.AgentMessage, error) {
		var message codecTestMessage
		err := json.Unmarshal(data, &message)
		return message, err
	})
}

// Built-in entries and messages round trip through the conformance suite;
// a registered role is only reachable from this package.
func TestSessionCodecRoundTripsRegisteredMessage(t *testing.T) {
	entry := &SessionMessageEntry{SessionEntryBase: SessionEntryBase{Type: EntryTypeMessage, ID: "id"}, Message: codecTestMessage{Role: "codecTest", Ticket: "T-1"}}
	data, err := MarshalSessionEntry(entry)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := UnmarshalSessionEntry(data)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(decoded, entry) {
		t.Fatalf("round trip = %#v, want %#v", decoded, entry)
	}
}

func TestSessionCodecAcceptsStringUserContent(t *testing.T) {
	data := `{"type":"message","id":"id","parentId":null,"timestamp":"t","message":{"role":"user","content":"plain text","timestamp":1}}`
	entry, err := UnmarshalSessionEntry([]byte(data))
	if err != nil {
		t.Fatal(err)
	}
	want := ai.UserMessage{Role: ai.RoleUser, Content: []ai.UserContent{ai.TextContent{Type: ai.ContentTypeText, Text: "plain text"}}, Timestamp: 1}
	if got := entry.(*SessionMessageEntry).Message; !reflect.DeepEqual(got, want) {
		t.Fatalf("message = %#v, want %#v", got, want)
	}
}

func TestMarshalSessionEntryRejectsUndecodableEntries(t *testing.T) {
	tests := []struct {
		name  string
		entry SessionEntry
		want  string
	}{
		{"type mismatch", &LabelEntry{SessionEntryBase: SessionEntryBase{Type: EntryTypeMessage, ID: "id"}}, `has type "message", want "label"`},
		{"missing ID", &LabelEntry{SessionEntryBase: SessionEntryBase{Type: EntryTypeLabel}}, "label entry has no ID"},
		{"nil message", &SessionMessageEntry{SessionEntryBase: SessionEntryBase{Type: EntryTypeMessage, ID: "id"}}, `message entry "id" has no message`},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if _, err := MarshalSessionEntry(test.entry); err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("MarshalSessionEntry() error = %v, want %q", err, test.want)
			}
		})
	}
}

// A partially decoded entry must not reach the agent's context, so anything
// the codec does not recognize is an error rather than a nil or empty value.
func TestUnmarshalSessionEntryRejectsUnknownData(t *testing.T) {
	message := func(body string) string {
		return `{"type":"message","id":"id","parentId":null,"timestamp":"t","message":` + body + `}`
	}
	tests := []struct {
		name string
		data string
		want string
	}{
		{"unknown entry type", `{"type":"mystery","id":"id"}`, "decode mystery entry: unknown session entry type"},
		{"missing ID", `{"type":"label","targetId":"x"}`, "decode label entry: missing id"},
		{"null message", message(`null`), "missing message"},
		{"unknown role", message(`{"role":"mystery"}`), `unknown message role "mystery"`},
		{"unknown user content", message(`{"role":"user","content":[{"type":"audio"}]}`), `content 0: unknown user content type "audio"`},
		{"unknown assistant content", message(`{"role":"assistant","content":[{"type":"image"}]}`), `unknown assistant content type "image"`},
		{"registered decoder failure", message(`{"role":"codecTest","ticket":7}`), `decode "codecTest" message`},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			entry, err := UnmarshalSessionEntry([]byte(test.data))
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("UnmarshalSessionEntry() = %#v, %v, want error containing %q", entry, err, test.want)
			}
		})
	}
}

func TestRegisterMessageDecoderPanics(t *testing.T) {
	decoder := func([]byte) (agent.AgentMessage, error) { return nil, nil }
	tests := []struct {
		name    string
		role    string
		decoder MessageDecoder
	}{
		{"built-in role", MessageRoleCompactionSummary, decoder},
		{"duplicate role", "codecTest", decoder},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("RegisterMessageDecoder() did not panic")
				}
			}()
			RegisterMessageDecoder(test.role, test.decoder)
		})
	}
}
