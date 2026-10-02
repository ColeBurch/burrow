package baseAgent

import (
	"encoding/json"
	"errors"
	"fmt"
	"sync"

	"github.com/ColeBurch/burrow/agent"
	"github.com/ColeBurch/burrow/ai"
)

// Session entry types, as stored in SessionEntryBase.Type.
const (
	EntryTypeMessage             = "message"
	EntryTypeThinkingLevelChange = "thinking_level_change"
	EntryTypeModelChange         = "model_change"
	EntryTypeCompaction          = "compaction"
	EntryTypeBranchSummary       = "branch_summary"
	EntryTypeCustom              = "custom"
	EntryTypeCustomMessage       = "custom_message"
	EntryTypeLabel               = "label"
	EntryTypeSessionInfo         = "session_info"
)

// MessageDecoder decodes the JSON of an agent message with a registered role.
type MessageDecoder func(data []byte) (agent.AgentMessage, error)

var (
	messageDecodersMu sync.RWMutex
	messageDecoders   = map[string]MessageDecoder{}
)

// RegisterMessageDecoder lets UnmarshalSessionEntry decode messages of a
// custom role, matched against the message's "role" field. Call it during
// initialization. It panics if role is empty, built in, or already registered.
func RegisterMessageDecoder(role string, decoder MessageDecoder) {
	if role == "" || decoder == nil {
		panic("baseAgent: RegisterMessageDecoder needs a role and a decoder")
	}
	if isBuiltinMessageRole(role) {
		panic(fmt.Sprintf("baseAgent: message role %q is built in", role))
	}
	messageDecodersMu.Lock()
	defer messageDecodersMu.Unlock()
	if _, ok := messageDecoders[role]; ok {
		panic(fmt.Sprintf("baseAgent: message role %q is already registered", role))
	}
	messageDecoders[role] = decoder
}

func isBuiltinMessageRole(role string) bool {
	switch role {
	case string(ai.RoleUser), string(ai.RoleAssistant), string(ai.RoleToolResult),
		MessageRoleCustom, MessageRoleBranchSummary, MessageRoleCompactionSummary:
		return true
	}
	return false
}

// MarshalSessionEntry encodes an entry for storage. It rejects entries that
// UnmarshalSessionEntry could not decode back into the same type.
func MarshalSessionEntry(entry SessionEntry) ([]byte, error) {
	if entry == nil {
		return nil, errors.New("session entry is nil")
	}
	base := entry.GetBase()
	want, err := sessionEntryType(entry)
	if err != nil {
		return nil, err
	}
	if base.Type != want {
		return nil, fmt.Errorf("session entry %q has type %q, want %q", base.ID, base.Type, want)
	}
	if base.ID == "" {
		return nil, fmt.Errorf("%s entry has no ID", base.Type)
	}
	if message, ok := entry.(*SessionMessageEntry); ok && message.Message == nil {
		return nil, fmt.Errorf("message entry %q has no message", base.ID)
	}
	return json.Marshal(entry)
}

func sessionEntryType(entry SessionEntry) (string, error) {
	switch entry.(type) {
	case *SessionMessageEntry:
		return EntryTypeMessage, nil
	case *ThinkingLevelChangeEntry:
		return EntryTypeThinkingLevelChange, nil
	case *ModelChangeEntry:
		return EntryTypeModelChange, nil
	case *CompactionEntry:
		return EntryTypeCompaction, nil
	case *BranchSummaryEntry:
		return EntryTypeBranchSummary, nil
	case *CustomEntry:
		return EntryTypeCustom, nil
	case *CustomMessageEntry:
		return EntryTypeCustomMessage, nil
	case *LabelEntry:
		return EntryTypeLabel, nil
	case *SessionInfoEntry:
		return EntryTypeSessionInfo, nil
	}
	return "", fmt.Errorf("unsupported session entry type %T", entry)
}

// UnmarshalSessionEntry decodes an entry encoded by MarshalSessionEntry.
// Unknown entry types, message roles, and content types are errors, so a
// partially decoded entry never reaches the agent's context.
func UnmarshalSessionEntry(data []byte) (SessionEntry, error) {
	var head struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(data, &head); err != nil {
		return nil, fmt.Errorf("decode session entry: %w", err)
	}
	entry, err := decodeSessionEntry(head.Type, data)
	if err != nil {
		return nil, fmt.Errorf("decode %s entry: %w", head.Type, err)
	}
	if entry.GetBase().ID == "" {
		return nil, fmt.Errorf("decode %s entry: missing id", head.Type)
	}
	return entry, nil
}

func decodeSessionEntry(entryType string, data []byte) (SessionEntry, error) {
	switch entryType {
	case EntryTypeMessage:
		var raw struct {
			SessionEntryBase
			Message json.RawMessage `json:"message"`
		}
		if err := json.Unmarshal(data, &raw); err != nil {
			return nil, err
		}
		message, err := UnmarshalAgentMessage(raw.Message)
		if err != nil {
			return nil, err
		}
		return &SessionMessageEntry{SessionEntryBase: raw.SessionEntryBase, Message: message}, nil
	case EntryTypeThinkingLevelChange:
		return unmarshalEntry[ThinkingLevelChangeEntry](data)
	case EntryTypeModelChange:
		return unmarshalEntry[ModelChangeEntry](data)
	case EntryTypeCompaction:
		return unmarshalEntry[CompactionEntry](data)
	case EntryTypeBranchSummary:
		return unmarshalEntry[BranchSummaryEntry](data)
	case EntryTypeCustom:
		return unmarshalEntry[CustomEntry](data)
	case EntryTypeCustomMessage:
		var raw struct {
			SessionEntryBase
			CustomType string          `json:"customType"`
			Content    json.RawMessage `json:"content"`
			Details    any             `json:"details,omitempty"`
			Display    bool            `json:"display"`
		}
		if err := json.Unmarshal(data, &raw); err != nil {
			return nil, err
		}
		content, err := unmarshalUserContents(raw.Content)
		if err != nil {
			return nil, err
		}
		return &CustomMessageEntry{
			SessionEntryBase: raw.SessionEntryBase,
			CustomType:       raw.CustomType,
			Content:          content,
			Details:          raw.Details,
			Display:          raw.Display,
		}, nil
	case EntryTypeLabel:
		return unmarshalEntry[LabelEntry](data)
	case EntryTypeSessionInfo:
		return unmarshalEntry[SessionInfoEntry](data)
	}
	return nil, errors.New("unknown session entry type")
}

func unmarshalEntry[T any, P interface {
	*T
	SessionEntry
}](data []byte) (SessionEntry, error) {
	var entry T
	if err := json.Unmarshal(data, &entry); err != nil {
		return nil, err
	}
	return P(&entry), nil
}

// UnmarshalAgentMessage decodes a built-in or registered agent message.
func UnmarshalAgentMessage(data []byte) (agent.AgentMessage, error) {
	if len(data) == 0 || string(data) == "null" {
		return nil, errors.New("missing message")
	}
	var head struct {
		Role string `json:"role"`
	}
	if err := json.Unmarshal(data, &head); err != nil {
		return nil, err
	}
	switch head.Role {
	case string(ai.RoleUser):
		var raw struct {
			Role      ai.Role         `json:"role"`
			Content   json.RawMessage `json:"content"`
			Timestamp int64           `json:"timestamp"`
		}
		if err := json.Unmarshal(data, &raw); err != nil {
			return nil, err
		}
		content, err := unmarshalUserContents(raw.Content)
		if err != nil {
			return nil, err
		}
		return ai.UserMessage{Role: raw.Role, Content: content, Timestamp: raw.Timestamp}, nil
	case string(ai.RoleAssistant):
		var raw struct {
			ai.AssistantMessage
			Content json.RawMessage `json:"content"`
		}
		if err := json.Unmarshal(data, &raw); err != nil {
			return nil, err
		}
		content, err := unmarshalContents(raw.Content, decodeAssistantContent)
		if err != nil {
			return nil, err
		}
		message := raw.AssistantMessage
		message.Content = content
		return message, nil
	case string(ai.RoleToolResult):
		var raw struct {
			ai.ToolResultMessage
			Content json.RawMessage `json:"content"`
		}
		if err := json.Unmarshal(data, &raw); err != nil {
			return nil, err
		}
		content, err := unmarshalContents(raw.Content, decodeToolResultContent)
		if err != nil {
			return nil, err
		}
		message := raw.ToolResultMessage
		message.Content = content
		return message, nil
	case MessageRoleCustom:
		var raw struct {
			CustomMessage
			Content json.RawMessage `json:"content"`
		}
		if err := json.Unmarshal(data, &raw); err != nil {
			return nil, err
		}
		content, err := unmarshalUserContents(raw.Content)
		if err != nil {
			return nil, err
		}
		message := raw.CustomMessage
		message.Content = content
		return message, nil
	case MessageRoleBranchSummary:
		var message BranchSummaryMessage
		return message, json.Unmarshal(data, &message)
	case MessageRoleCompactionSummary:
		var message CompactionSummaryMessage
		return message, json.Unmarshal(data, &message)
	}

	messageDecodersMu.RLock()
	decoder := messageDecoders[head.Role]
	messageDecodersMu.RUnlock()
	if decoder == nil {
		return nil, fmt.Errorf("unknown message role %q", head.Role)
	}
	message, err := decoder(data)
	if err != nil {
		return nil, fmt.Errorf("decode %q message: %w", head.Role, err)
	}
	if message == nil {
		return nil, fmt.Errorf("decode %q message: decoder returned nil", head.Role)
	}
	return message, nil
}

// unmarshalUserContents also accepts a plain string, which pi allows for user
// message content.
func unmarshalUserContents(data json.RawMessage) ([]ai.UserContent, error) {
	if len(data) > 0 && data[0] == '"' {
		var text string
		if err := json.Unmarshal(data, &text); err != nil {
			return nil, err
		}
		return []ai.UserContent{ai.TextContent{Type: ai.ContentTypeText, Text: text}}, nil
	}
	return unmarshalContents(data, decodeUserContent)
}

func unmarshalContents[T any](data json.RawMessage, decode func(ai.ContentType, []byte) (T, error)) ([]T, error) {
	if len(data) == 0 || string(data) == "null" {
		return nil, nil
	}
	var items []json.RawMessage
	if err := json.Unmarshal(data, &items); err != nil {
		return nil, err
	}
	contents := make([]T, 0, len(items))
	for i, item := range items {
		var head struct {
			Type ai.ContentType `json:"type"`
		}
		if err := json.Unmarshal(item, &head); err != nil {
			return nil, fmt.Errorf("content %d: %w", i, err)
		}
		content, err := decode(head.Type, item)
		if err != nil {
			return nil, fmt.Errorf("content %d: %w", i, err)
		}
		contents = append(contents, content)
	}
	return contents, nil
}

func decodeUserContent(contentType ai.ContentType, data []byte) (ai.UserContent, error) {
	switch contentType {
	case ai.ContentTypeText:
		return decodeContent[ai.TextContent](data)
	case ai.ContentTypeImage:
		return decodeContent[ai.ImageContent](data)
	}
	return nil, fmt.Errorf("unknown user content type %q", contentType)
}

func decodeAssistantContent(contentType ai.ContentType, data []byte) (ai.AssistantContent, error) {
	switch contentType {
	case ai.ContentTypeText:
		return decodeContent[ai.TextContent](data)
	case ai.ContentTypeThinking:
		return decodeContent[ai.ThinkingContent](data)
	case ai.ContentTypeToolCall:
		return decodeContent[ai.ToolCall](data)
	}
	return nil, fmt.Errorf("unknown assistant content type %q", contentType)
}

func decodeToolResultContent(contentType ai.ContentType, data []byte) (ai.ToolResultContent, error) {
	switch contentType {
	case ai.ContentTypeText:
		return decodeContent[ai.TextContent](data)
	case ai.ContentTypeImage:
		return decodeContent[ai.ImageContent](data)
	}
	return nil, fmt.Errorf("unknown tool result content type %q", contentType)
}

func decodeContent[T any](data []byte) (T, error) {
	var content T
	err := json.Unmarshal(data, &content)
	return content, err
}
