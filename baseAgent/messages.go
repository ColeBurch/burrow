package baseAgent

import (
	"github.com/ColeBurch/burrow/agent"
	"github.com/ColeBurch/burrow/ai"
)

const (
	MessageRoleCustom            = "custom"
	MessageRoleBranchSummary     = "branchSummary"
	MessageRoleCompactionSummary = "compactionSummary"
	BranchSummaryPrefix          = `The following is a summary of a branch that this conversation came back from:

<summary>
`
	BranchSummarySuffix = `</summary>`

	CompactionSummaryPrefix = `The conversation history before this point was compacted into the following summary:

<summary>
`
	CompactionSummarySuffix = `
</summary>`
)

type CustomMessage struct {
	Role       string           `json:"role"`
	CustomType string           `json:"customType"`
	Content    []ai.UserContent `json:"content"`
	Display    bool             `json:"display"`
	Details    any              `json:"details,omitempty"`
	Timestamp  int64            `json:"timestamp"`
}

func (CustomMessage) AgentMessageType() string {
	return MessageRoleCustom
}

type BranchSummaryMessage struct {
	Role      string `json:"role"`
	Summary   string `json:"summary"`
	FromID    string `json:"fromId"`
	Timestamp int64  `json:"timestamp"`
}

func (BranchSummaryMessage) AgentMessageType() string {
	return MessageRoleBranchSummary
}

type CompactionSummaryMessage struct {
	Role         string `json:"role"`
	Summary      string `json:"summary"`
	TokensBefore int64  `json:"tokensBefore"`
	Timestamp    int64  `json:"timestamp"`
}

func (CompactionSummaryMessage) AgentMessageType() string {
	return MessageRoleCompactionSummary
}

func CreateCustomMessage(customType string, content []ai.UserContent, display bool, details any, timestamp int64) CustomMessage {
	return CustomMessage{
		Role:       MessageRoleCustom,
		CustomType: customType,
		Content:    content,
		Display:    display,
		Details:    details,
		Timestamp:  timestamp,
	}
}

func CreateBranchSummaryMessage(summary, fromID string, timestamp int64) BranchSummaryMessage {
	return BranchSummaryMessage{
		Role:      MessageRoleBranchSummary,
		Summary:   summary,
		FromID:    fromID,
		Timestamp: timestamp,
	}
}

func CreateCompactionSummaryMessage(summary string, tokensBefore, timestamp int64) CompactionSummaryMessage {
	return CompactionSummaryMessage{
		Role:         MessageRoleCompactionSummary,
		Summary:      summary,
		TokensBefore: tokensBefore,
		Timestamp:    timestamp,
	}
}

func ConvertToLLM(messages []agent.AgentMessage) []ai.Message {
	result := make([]ai.Message, 0, len(messages))

	for _, msg := range messages {
		switch m := msg.(type) {
		case ai.UserMessage:
			result = append(result, m)

		case ai.AssistantMessage:
			result = append(result, m)

		case ai.ToolResultMessage:
			result = append(result, m)

		case CustomMessage:
			result = append(result, ai.UserMessage{
				Role:      ai.RoleUser,
				Content:   m.Content,
				Timestamp: m.Timestamp,
			})

		case BranchSummaryMessage:
			result = append(result, ai.UserMessage{
				Role: ai.RoleUser,
				Content: []ai.UserContent{
					ai.TextContent{
						Type: ai.ContentTypeText,
						Text: BranchSummaryPrefix + m.Summary + BranchSummarySuffix,
					},
				},
				Timestamp: m.Timestamp,
			})

		case CompactionSummaryMessage:
			result = append(result, ai.UserMessage{
				Role: ai.RoleUser,
				Content: []ai.UserContent{
					ai.TextContent{
						Type: ai.ContentTypeText,
						Text: CompactionSummaryPrefix + m.Summary + CompactionSummarySuffix,
					},
				},
				Timestamp: m.Timestamp,
			})
		}
	}

	return result
}
