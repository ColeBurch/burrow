package baseAgent

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"
	"unicode/utf16"

	"golang.org/x/sync/errgroup"

	"github.com/ColeBurch/burrow/agent"
	"github.com/ColeBurch/burrow/ai"
)

type CompactionSettings struct {
	Enabled          bool
	ReservedTokens   int64
	KeepRecentTokens int64
}

type CompactionResult struct {
	Summary          string
	FirstKeptEntryID string
	TokensBefore     int64
	// Details is stored on the CompactionEntry and must marshal to JSON. The next
	// compaction receives it as CompactionPreparation.PreviousDetails.
	Details any
}

var DEFAULT_COMPACTION_SETTINGS = CompactionSettings{
	Enabled:          true,
	ReservedTokens:   16384,
	KeepRecentTokens: 20000,
}

type contextUsageEstimate struct {
	tokens         int64
	usageTokens    int64
	trailingTokens int64
	lastUsageIndex *int
}

type CutPointResult struct {
	FirstKeptEntryIndex int
	TurnStartIndex      int
	IsSplitTurn         bool
}

type CompactionPreparation struct {
	FirstKeptEntryID    string
	MessagesToSummarize []agent.AgentMessage
	TurnPrefixMessages  []agent.AgentMessage
	IsSplitTurn         bool
	TokensBefore        int64
	PreviousSummary     *string
	// PreviousDetails is the previous compaction's Details as JSON, so it reads
	// the same before and after a reload. Decode it with DecodeCompactionDetails.
	PreviousDetails json.RawMessage
	Settings        CompactionSettings
}

// DecodeCompactionDetails decodes details stored by a previous compaction.
// ok is false when the previous compaction stored no details.
func DecodeCompactionDetails[T any](raw json.RawMessage) (details T, ok bool, err error) {
	if len(raw) == 0 || string(raw) == "null" {
		return details, false, nil
	}
	if err := json.Unmarshal(raw, &details); err != nil {
		return details, false, fmt.Errorf("decode compaction details: %w", err)
	}
	return details, true, nil
}

// Message Extraction

func getMessageFromEntry(entry SessionEntry) agent.AgentMessage {
	switch e := entry.(type) {
	case *SessionMessageEntry:
		return e.Message
	case *CustomMessageEntry:
		return CreateCustomMessage(e.CustomType, e.Content, e.Display, e.Details, parseMillis(e.Timestamp))
	case *BranchSummaryEntry:
		return CreateBranchSummaryMessage(e.Summary, e.FromID, parseMillis(e.Timestamp))
	case *CompactionEntry:
		return CreateCompactionSummaryMessage(e.Summary, e.TokensBefore, parseMillis(e.Timestamp))
	default:
		return nil
	}
}

func getMessageFromEntryForCompaction(entry SessionEntry) agent.AgentMessage {
	if _, ok := entry.(*CompactionEntry); ok {
		return nil
	}
	return getMessageFromEntry(entry)
}

// Token Calculation

/**
 * Calculate total context tokens from usage.
 * Uses the native totalTokens field when available, falls back to computing from components.
 */
func CalculateContextTokens(usage ai.Usage) int64 {
	if usage.Total != 0 {
		return usage.Total
	}
	return usage.Input + usage.Output + usage.CacheRead + usage.CacheWrite
}

/**
 * Get usage from an assistant message if available.
 * Skips aborted and error messages as they don't have valid usage data.
 */
func getAssistantUsage(msg agent.AgentMessage) *ai.Usage {
	assistantMsg, ok := msg.(ai.AssistantMessage)
	if !ok {
		return nil
	}
	if assistantMsg.StopReason == ai.StopReasonAborted || assistantMsg.StopReason == ai.StopReasonError {
		return nil
	}
	return &assistantMsg.Usage
}

/**
 * Find the last non-aborted assistant message usage from session entries.
 */
func GetLastAssistantUsage(entries []SessionEntry) *ai.Usage {
	for i := len(entries) - 1; i >= 0; i-- {
		entry, ok := entries[i].(*SessionMessageEntry)
		if !ok || entry == nil {
			continue
		}
		if usage := getAssistantUsage(entry.Message); usage != nil {
			return usage
		}
	}
	return nil
}

type assistantUsageInfo struct {
	usage ai.Usage
	index int
}

func getLastAssistantUsageInfo(messages []agent.AgentMessage) *assistantUsageInfo {
	for i := len(messages) - 1; i >= 0; i-- {
		if usage := getAssistantUsage(messages[i]); usage != nil {
			return &assistantUsageInfo{usage: *usage, index: i}
		}
	}
	return nil
}

/**
 * Estimate context tokens from messages, using the last assistant usage when available.
 * If there are messages after the last usage, estimate their tokens with estimateTokens.
 */
func EstimateContextTokens(messages []agent.AgentMessage) contextUsageEstimate {
	usageInfo := getLastAssistantUsageInfo(messages)

	if usageInfo == nil {
		var estimated int64
		for _, msg := range messages {
			estimated += EstimateTokens(msg)
		}
		return contextUsageEstimate{
			tokens:         estimated,
			usageTokens:    0,
			trailingTokens: estimated,
			lastUsageIndex: nil,
		}
	}

	usageTokens := CalculateContextTokens(usageInfo.usage)
	trailingTokens := int64(0)
	for i := usageInfo.index + 1; i < len(messages); i++ {
		trailingTokens += EstimateTokens(messages[i])
	}

	return contextUsageEstimate{
		tokens:         int64(usageTokens + trailingTokens),
		usageTokens:    int64(usageTokens),
		trailingTokens: trailingTokens,
		lastUsageIndex: &usageInfo.index,
	}
}

/**
 * Check if compaction should trigger based on context usage.
 * An unknown (non-positive) context window never triggers.
 */
func ShouldTriggerCompaction(contextTokens int64, contextWindow int64, settings CompactionSettings) bool {
	if !settings.Enabled || contextWindow <= 0 {
		return false
	}
	return contextTokens > contextWindow-settings.ReservedTokens
}

// Cut Point Detection

func EstimateTokens(message agent.AgentMessage) int64 {
	var chars int64

	switch msg := message.(type) {
	case ai.UserMessage:
		for _, content := range msg.Content {
			switch block := content.(type) {
			case ai.TextContent:
				chars += utf16Length(block.Text)
			case ai.ImageContent:
				chars += 4800
			}
		}

	case ai.AssistantMessage:
		for _, content := range msg.Content {
			switch block := content.(type) {
			case ai.TextContent:
				chars += utf16Length(block.Text)
			case ai.ThinkingContent:
				chars += utf16Length(block.Thinking)
			case ai.ToolCall:
				chars += utf16Length(block.Name)
				if args, err := marshalJSON(block.Args); err == nil {
					chars += utf16Length(string(args))
				}
			}
		}

	case CustomMessage:
		for _, content := range msg.Content {
			switch block := content.(type) {
			case ai.TextContent:
				chars += utf16Length(block.Text)
			case ai.ImageContent:
				chars += 4800
			}
		}

	case ai.ToolResultMessage:
		for _, content := range msg.Content {
			switch block := content.(type) {
			case ai.TextContent:
				chars += utf16Length(block.Text)
			case ai.ImageContent:
				chars += 4800
			}
		}

	case BranchSummaryMessage:
		chars = utf16Length(msg.Summary)

	case CompactionSummaryMessage:
		chars = utf16Length(msg.Summary)
	}

	return (chars + 3) / 4
}

// Using utf16 to match the original typescript count
func utf16Length(value string) int64 {
	var length int64
	for _, r := range value {
		length += int64(utf16.RuneLen(r))
	}
	return length
}

/**
 * Find valid cut points: indices of user, assistant, or custom.
 * Never cut at tool results (they must follow their tool call).
 * When we cut at an assistant message with tool calls, its tool results follow it
 * and will be kept.
 */
func findValidCutPoints(entries []SessionEntry, startIndex, endIndex int) []int {
	cutPoints := make([]int, 0)

	for i := startIndex; i < endIndex; i++ {
		switch entry := entries[i].(type) {
		case *SessionMessageEntry:
			if entry == nil || entry.Message == nil {
				continue
			}
			switch entry.Message.AgentMessageType() {
			case MessageRoleCustom,
				MessageRoleBranchSummary,
				MessageRoleCompactionSummary,
				string(ai.RoleUser),
				string(ai.RoleAssistant):
				cutPoints = append(cutPoints, i)
			}

		case *BranchSummaryEntry, *CustomMessageEntry:
			cutPoints = append(cutPoints, i)
		}
	}

	return cutPoints
}

/**
 * Find the user message that starts the turn containing the given entry index.
 * Returns -1 if no turn start found before the index.
 */
func FindTurnStartIndex(entries []SessionEntry, entryIndex, startIndex int) int {
	for i := entryIndex; i >= startIndex; i-- {
		switch entry := entries[i].(type) {
		case *BranchSummaryEntry, *CustomMessageEntry:
			return i
		case *SessionMessageEntry:
			if entry != nil && entry.Message != nil && entry.Message.AgentMessageType() == string(ai.RoleUser) {
				return i
			}
		}
	}
	return -1
}

// isUnseenAssistantMessage reports whether ai.TransformMessages drops message
// before it reaches the provider.
func isUnseenAssistantMessage(message agent.AgentMessage) bool {
	assistant, ok := message.(ai.AssistantMessage)
	return ok && (assistant.StopReason == ai.StopReasonError || assistant.StopReason == ai.StopReasonAborted)
}

// FindCutPoint finds the entry that keeps approximately keepRecentTokens.
func FindCutPoint(entries []SessionEntry, startIndex, endIndex int, keepRecentTokens int64) CutPointResult {
	cutPoints := findValidCutPoints(entries, startIndex, endIndex)
	if len(cutPoints) == 0 {
		return CutPointResult{
			FirstKeptEntryIndex: startIndex,
			TurnStartIndex:      -1,
			IsSplitTurn:         false,
		}
	}

	var accumulatedTokens int64
	cutIndex := cutPoints[0]

	for i := endIndex - 1; i >= startIndex; i-- {
		entry, ok := entries[i].(*SessionMessageEntry)
		if !ok || entry == nil || entry.Message == nil {
			continue
		}
		// Error and aborted replies never reach the model, so they must not use
		// the recent-token budget. They stay cut points for when nothing else fits.
		if isUnseenAssistantMessage(entry.Message) {
			continue
		}

		accumulatedTokens += EstimateTokens(entry.Message)
		if accumulatedTokens < keepRecentTokens {
			continue
		}

		for _, cutPoint := range cutPoints {
			if cutPoint >= i {
				cutIndex = cutPoint
				break
			}
		}
		break
	}

	for cutIndex > startIndex {
		previousEntry := entries[cutIndex-1]
		if _, ok := previousEntry.(*CompactionEntry); ok {
			break
		}
		if _, ok := previousEntry.(*SessionMessageEntry); ok {
			break
		}
		cutIndex--
	}

	isUserMessage := false
	if entry, ok := entries[cutIndex].(*SessionMessageEntry); ok && entry != nil && entry.Message != nil {
		isUserMessage = entry.Message.AgentMessageType() == string(ai.RoleUser)
	}

	turnStartIndex := -1
	if !isUserMessage {
		turnStartIndex = FindTurnStartIndex(entries, cutIndex, startIndex)
	}

	return CutPointResult{
		FirstKeptEntryIndex: cutIndex,
		TurnStartIndex:      turnStartIndex,
		IsSplitTurn:         !isUserMessage && turnStartIndex != -1,
	}
}

// Summarization

const SUMMARIZATION_SYSTEM_PROMPT = `You are a context summarization assistant. Your task is to read a conversation between a user and an AI assistant, then produce a structured summary following the exact format specified.

Do NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the structured summary.`

const SUMMARIZATION_PROMPT = `The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or "(none)" if not applicable]

Keep each section concise. Preserve exact names, identifiers, values, and error messages.`

const UPDATE_SUMMARIZATION_PROMPT = `The messages above are NEW conversation messages to incorporate into the existing summary provided in <previous-summary> tags.

Update the existing structured summary with new information. RULES:
- PRESERVE all existing information from the previous summary
- ADD new progress, decisions, and context from the new messages
- UPDATE the Progress section: move items from "In Progress" to "Done" when completed
- UPDATE "Next Steps" based on what was accomplished
- PRESERVE exact names, identifiers, values, and error messages
- If something is no longer relevant, you may remove it

Use this EXACT format:

## Goal
[Preserve existing goals, add new ones if the task expanded]

## Constraints & Preferences
- [Preserve existing, add new ones discovered]

## Progress
### Done
- [x] [Include previously done items AND newly completed items]

### In Progress
- [ ] [Current work - update based on progress]

### Blocked
- [Current blockers - remove if resolved]

## Key Decisions
- **[Decision]**: [Brief rationale] (preserve all previous, add new)

## Next Steps
1. [Update based on current state]

## Critical Context
- [Preserve important context, add new if needed]

Keep each section concise. Preserve exact names, identifiers, values, and error messages.`

// SummaryPrompts are the prompts used to generate compaction summaries. Empty
// fields use the defaults from DefaultSummaryPrompts.
type SummaryPrompts struct {
	System string
	// Initial asks for a new summary; Update merges into <previous-summary>.
	Initial    string
	Update     string
	TurnPrefix string
}

func DefaultSummaryPrompts() SummaryPrompts {
	return SummaryPrompts{
		System:     SUMMARIZATION_SYSTEM_PROMPT,
		Initial:    SUMMARIZATION_PROMPT,
		Update:     UPDATE_SUMMARIZATION_PROMPT,
		TurnPrefix: TURN_PREFIX_SUMMARIZATION_PROMPT,
	}
}

func (p SummaryPrompts) withDefaults() SummaryPrompts {
	defaults := DefaultSummaryPrompts()
	if p.System == "" {
		p.System = defaults.System
	}
	if p.Initial == "" {
		p.Initial = defaults.Initial
	}
	if p.Update == "" {
		p.Update = defaults.Update
	}
	if p.TurnPrefix == "" {
		p.TurnPrefix = defaults.TurnPrefix
	}
	return p
}

type GenerateSummaryOptions struct {
	Headers            map[string]string
	CustomInstructions string
	PreviousSummary    string
	ThinkingLevel      ai.ModelThinkingLevel
	Prompts            SummaryPrompts
}

/**
 * Generate a summary of the conversation using the LLM.
 * If previousSummary is provided, uses the update prompt to merge.
 */
func GenerateSummary(
	ctx context.Context,
	currentMessages []agent.AgentMessage,
	model ai.Model[ai.API],
	reserveTokens int64,
	apiKey string,
	options *GenerateSummaryOptions,
) (string, error) {
	maxTokens := reserveTokens * 4 / 5
	var headers map[string]string
	var previousSummary string
	var thinkingLevel ai.ModelThinkingLevel
	var prompts SummaryPrompts
	if options != nil {
		prompts = options.Prompts
	}
	prompts = prompts.withDefaults()
	basePrompt := prompts.Initial

	if options != nil {
		headers = options.Headers
		previousSummary = options.PreviousSummary
		if previousSummary != "" {
			basePrompt = prompts.Update
		}
		thinkingLevel = options.ThinkingLevel
		if options.CustomInstructions != "" {
			basePrompt += "\n\nAdditional focus: " + options.CustomInstructions
		}
	}

	conversationText := serializeConversation(ConvertToLLM(currentMessages))
	var prompt strings.Builder
	prompt.WriteString("<conversation>\n")
	prompt.WriteString(conversationText)
	prompt.WriteString("\n</conversation>\n\n")
	if previousSummary != "" {
		prompt.WriteString("<previous-summary>\n")
		prompt.WriteString(previousSummary)
		prompt.WriteString("\n</previous-summary>\n\n")
	}
	prompt.WriteString(basePrompt)

	systemPrompt := prompts.System
	summarizationMessage := ai.UserMessage{
		Role: ai.RoleUser,
		Content: []ai.UserContent{
			ai.TextContent{Type: ai.ContentTypeText, Text: prompt.String()},
		},
		Timestamp: time.Now().UnixMilli(),
	}
	completionOptions := &ai.SimpleStreamOptions{
		StreamOptions: ai.StreamOptions{
			MaxTokens: &maxTokens,
			Signal:    ctx,
			ApiKey:    &apiKey,
			Headers:   headers,
		},
	}
	if model.Reasoning && thinkingLevel != "" && thinkingLevel != ai.ThinkingLevelOff {
		completionOptions.Reasoning = &thinkingLevel
	}

	response, err := ai.CompleteSimple(
		model,
		ai.ModelContext{
			SystemPrompt: &systemPrompt,
			Messages:     []ai.Message{summarizationMessage},
		},
		completionOptions,
	)
	if err != nil {
		return "", fmt.Errorf("summarization failed: %w", err)
	}
	summary, err := summaryText(ctx, response)
	if err != nil {
		return "", fmt.Errorf("summarization failed: %w", err)
	}
	return summary, nil
}

// summaryText returns the text of a summarization response. A summary replaces
// the history it covers, so failed, aborted, truncated, and empty responses are
// errors rather than partial summaries.
func summaryText(ctx context.Context, response ai.AssistantMessage) (string, error) {
	switch response.StopReason {
	case ai.StopReasonError:
		if response.ErrorMessage == "" {
			return "", errors.New("unknown error")
		}
		return "", errors.New(response.ErrorMessage)
	case ai.StopReasonAborted:
		if err := ctx.Err(); err != nil {
			return "", err
		}
		return "", errors.New("response was aborted")
	case ai.StopReasonLength:
		return "", errors.New("summary was truncated at the output token limit")
	}

	textParts := make([]string, 0)
	for _, content := range response.Content {
		if text, ok := content.(ai.TextContent); ok {
			textParts = append(textParts, text.Text)
		}
	}
	summary := strings.Join(textParts, "\n")
	if strings.TrimSpace(summary) == "" {
		return "", errors.New("summary is empty")
	}
	return summary, nil
}

const TURN_PREFIX_SUMMARIZATION_PROMPT = `This is the PREFIX of a turn that was too large to keep. The SUFFIX (recent work) is retained.

Summarize the prefix to provide context for the retained suffix:

## Original Request
[What did the user ask for in this turn?]

## Early Progress
- [Key decisions and work done in the prefix]

## Context for Suffix
- [Information needed to understand the retained recent work]

Be concise. Focus on what's needed to understand the kept suffix.`

const toolResultMaxChars int64 = 2000

func serializeConversation(messages []ai.Message) string {
	parts := make([]string, 0)

	for _, message := range messages {
		switch msg := message.(type) {
		case ai.UserMessage:
			textParts := make([]string, 0)
			for _, content := range msg.Content {
				if text, ok := content.(ai.TextContent); ok {
					textParts = append(textParts, text.Text)
				}
			}
			if text := strings.Join(textParts, ""); text != "" {
				parts = append(parts, "[User]: "+text)
			}

		case ai.AssistantMessage:
			thinkingParts := make([]string, 0)
			textParts := make([]string, 0)
			toolCalls := make([]string, 0)
			for _, content := range msg.Content {
				switch block := content.(type) {
				case ai.ThinkingContent:
					thinkingParts = append(thinkingParts, block.Thinking)
				case ai.TextContent:
					textParts = append(textParts, block.Text)
				case ai.ToolCall:
					toolCalls = append(toolCalls, block.Name+"("+formatToolArguments(block.Args)+")")
				}
			}
			if len(thinkingParts) > 0 {
				parts = append(parts, "[Assistant thinking]: "+strings.Join(thinkingParts, "\n"))
			}
			if len(textParts) > 0 {
				parts = append(parts, "[Assistant]: "+strings.Join(textParts, "\n"))
			}
			if len(toolCalls) > 0 {
				parts = append(parts, "[Assistant tool calls]: "+strings.Join(toolCalls, "; "))
			}

		case ai.ToolResultMessage:
			textParts := make([]string, 0)
			for _, content := range msg.Content {
				if text, ok := content.(ai.TextContent); ok {
					textParts = append(textParts, text.Text)
				}
			}
			if text := strings.Join(textParts, ""); text != "" {
				parts = append(parts, "[Tool result]: "+truncateForSummary(text, toolResultMaxChars))
			}
		}
	}

	return strings.Join(parts, "\n\n")
}

func formatToolArguments(args any) string {
	values, ok := args.(map[string]any)
	if !ok {
		encoded, err := marshalJSON(args)
		if err != nil || json.Unmarshal(encoded, &values) != nil {
			return safeJSONString(args)
		}
	}

	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	parts := make([]string, 0, len(keys))
	for _, key := range keys {
		parts = append(parts, key+"="+safeJSONString(values[key]))
	}
	return strings.Join(parts, ", ")
}

func safeJSONString(value any) string {
	encoded, err := marshalJSON(value)
	if err != nil {
		return "[unserializable]"
	}
	return string(encoded)
}

// marshalJSON encodes like JSON.stringify. json.Marshal escapes <, > and & as
// \u003c, \u003e and \u0026, which inflates token estimates for code-heavy tool
// arguments and garbles them in summarization prompts.
func marshalJSON(value any) ([]byte, error) {
	var buffer bytes.Buffer
	encoder := json.NewEncoder(&buffer)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(value); err != nil {
		return nil, err
	}
	return bytes.TrimSuffix(buffer.Bytes(), []byte("\n")), nil
}

func truncateForSummary(text string, maxChars int64) string {
	length := utf16Length(text)
	if length <= maxChars {
		return text
	}

	var truncated strings.Builder
	var used int64
	for _, r := range text {
		runeLength := int64(utf16.RuneLen(r))
		if used+runeLength > maxChars {
			break
		}
		truncated.WriteRune(r)
		used += runeLength
	}
	return fmt.Sprintf("%s\n\n[... %d more characters truncated]", truncated.String(), length-maxChars)
}

// Compaction Preparation

func PrepareCompaction(
	pathEntries []SessionEntry,
	settings CompactionSettings,
) *CompactionPreparation {
	if len(pathEntries) == 0 {
		return nil
	}

	if _, ok := pathEntries[len(pathEntries)-1].(*CompactionEntry); ok {
		return nil
	}

	previousCompactionIndex := -1
	for i := len(pathEntries) - 1; i >= 0; i-- {
		if _, ok := pathEntries[i].(*CompactionEntry); ok {
			previousCompactionIndex = i
			break
		}
	}

	var previousSummary *string
	var previousDetails json.RawMessage
	boundaryStart := 0

	if previousCompactionIndex >= 0 {
		previousCompaction :=
			pathEntries[previousCompactionIndex].(*CompactionEntry)
		summary := previousCompaction.Summary
		previousSummary = &summary
		if previousCompaction.Details != nil {
			// AgentSession stores details as JSON already. Entries appended directly
			// with unmarshalable details are treated as having none.
			if raw, err := json.Marshal(previousCompaction.Details); err == nil {
				previousDetails = raw
			}
		}

		firstKeptIndex := -1
		for i, entry := range pathEntries {
			if entry.GetBase().ID ==
				previousCompaction.FirstKeptEntryID {
				firstKeptIndex = i
				break
			}
		}

		if firstKeptIndex >= 0 {
			boundaryStart = firstKeptIndex
		} else {
			boundaryStart = previousCompactionIndex + 1
		}
	}

	boundaryEnd := len(pathEntries)
	sessionContext := BuildSessionContext(pathEntries, nil, nil)
	tokensBefore := EstimateContextTokens(sessionContext.Messages).tokens

	cutPoint := FindCutPoint(
		pathEntries,
		boundaryStart,
		boundaryEnd,
		settings.KeepRecentTokens,
	)

	if cutPoint.FirstKeptEntryIndex < 0 ||
		cutPoint.FirstKeptEntryIndex >= len(pathEntries) {
		return nil
	}

	firstKeptEntry := pathEntries[cutPoint.FirstKeptEntryIndex]
	firstKeptEntryID := firstKeptEntry.GetBase().ID
	if firstKeptEntryID == "" {
		return nil
	}

	historyEnd := cutPoint.FirstKeptEntryIndex
	if cutPoint.IsSplitTurn {
		historyEnd = cutPoint.TurnStartIndex
	}

	messagesToSummarize := make([]agent.AgentMessage, 0)
	for i := boundaryStart; i < historyEnd; i++ {
		message := getMessageFromEntryForCompaction(pathEntries[i])
		if message != nil {
			messagesToSummarize = append(messagesToSummarize, message)
		}
	}

	turnPrefixMessages := make([]agent.AgentMessage, 0)
	if cutPoint.IsSplitTurn {
		for i := cutPoint.TurnStartIndex; i <
			cutPoint.FirstKeptEntryIndex; i++ {
			message := getMessageFromEntryForCompaction(pathEntries[i])
			if message != nil {
				turnPrefixMessages = append(turnPrefixMessages, message)
			}
		}
	}

	// Everything since the last boundary fits in KeepRecentTokens. Compacting
	// would add a summary without removing anything.
	if len(messagesToSummarize) == 0 && len(turnPrefixMessages) == 0 {
		return nil
	}

	return &CompactionPreparation{
		FirstKeptEntryID:    firstKeptEntryID,
		MessagesToSummarize: messagesToSummarize,
		TurnPrefixMessages:  turnPrefixMessages,
		IsSplitTurn:         cutPoint.IsSplitTurn,
		TokensBefore:        tokensBefore,
		PreviousSummary:     previousSummary,
		PreviousDetails:     previousDetails,
		Settings:            settings,
	}
}

// CompactOptions contains the optional parameters for Compact.
type CompactOptions struct {
	Headers            map[string]string
	CustomInstructions string
	ThinkingLevel      ai.ModelThinkingLevel
	Prompts            SummaryPrompts
}

// Compact generates summaries from prepared compaction data.
func Compact(
	ctx context.Context,
	preparation *CompactionPreparation,
	model ai.Model[ai.API],
	apiKey string,
	options *CompactOptions,
) (*CompactionResult, error) {
	if preparation == nil {
		return nil, errors.New("compaction preparation is nil")
	}

	var headers map[string]string
	var customInstructions string
	var thinkingLevel ai.ModelThinkingLevel
	var prompts SummaryPrompts
	if options != nil {
		headers = options.Headers
		customInstructions = options.CustomInstructions
		thinkingLevel = options.ThinkingLevel
		prompts = options.Prompts
	}
	prompts = prompts.withDefaults()

	previousSummary := ""
	if preparation.PreviousSummary != nil {
		previousSummary = *preparation.PreviousSummary
	}

	generateOptions := &GenerateSummaryOptions{
		Headers:            headers,
		CustomInstructions: customInstructions,
		PreviousSummary:    previousSummary,
		ThinkingLevel:      thinkingLevel,
		Prompts:            prompts,
	}

	var summary string
	if preparation.IsSplitTurn && len(preparation.TurnPrefixMessages) > 0 {
		// With nothing new before the split turn, the previous summary is still
		// the history. Only the latest compaction is kept in context, so dropping
		// it here would lose everything it covers.
		historySummary := "No prior history."
		if previousSummary != "" {
			historySummary = previousSummary
		}
		var turnPrefixSummary string
		// Both summaries are required, so the first failure cancels the other
		// request and is the error returned.
		group, groupCtx := errgroup.WithContext(ctx)

		if len(preparation.MessagesToSummarize) > 0 {
			group.Go(func() error {
				var err error
				historySummary, err = GenerateSummary(
					groupCtx,
					preparation.MessagesToSummarize,
					model,
					preparation.Settings.ReservedTokens,
					apiKey,
					generateOptions,
				)
				return err
			})
		}

		group.Go(func() error {
			var err error
			turnPrefixSummary, err = generateTurnPrefixSummary(
				groupCtx,
				preparation.TurnPrefixMessages,
				model,
				preparation.Settings.ReservedTokens,
				apiKey,
				headers,
				thinkingLevel,
				prompts,
			)
			return err
		})

		if err := group.Wait(); err != nil {
			return nil, err
		}

		summary = historySummary + "\n\n---\n\n**Turn Context (split turn):**\n\n" + turnPrefixSummary
	} else {
		var err error
		summary, err = GenerateSummary(
			ctx,
			preparation.MessagesToSummarize,
			model,
			preparation.Settings.ReservedTokens,
			apiKey,
			generateOptions,
		)
		if err != nil {
			return nil, err
		}
	}

	if preparation.FirstKeptEntryID == "" {
		return nil, errors.New("first kept entry has no UUID - session may need migration")
	}

	return &CompactionResult{
		Summary:          summary,
		FirstKeptEntryID: preparation.FirstKeptEntryID,
		TokensBefore:     preparation.TokensBefore,
	}, nil
}

// generateTurnPrefixSummary generates a summary for the prefix of a split turn.
func generateTurnPrefixSummary(
	ctx context.Context,
	messages []agent.AgentMessage,
	model ai.Model[ai.API],
	reserveTokens int64,
	apiKey string,
	headers map[string]string,
	thinkingLevel ai.ModelThinkingLevel,
	prompts SummaryPrompts,
) (string, error) {
	maxTokens := reserveTokens / 2
	conversationText := serializeConversation(ConvertToLLM(messages))
	promptText := "<conversation>\n" + conversationText + "\n</conversation>\n\n" + prompts.TurnPrefix
	systemPrompt := prompts.System
	summarizationMessage := ai.UserMessage{
		Role: ai.RoleUser,
		Content: []ai.UserContent{
			ai.TextContent{Type: ai.ContentTypeText, Text: promptText},
		},
		Timestamp: time.Now().UnixMilli(),
	}

	completionOptions := &ai.SimpleStreamOptions{
		StreamOptions: ai.StreamOptions{
			MaxTokens: &maxTokens,
			Signal:    ctx,
			ApiKey:    &apiKey,
			Headers:   headers,
		},
	}
	if model.Reasoning && thinkingLevel != "" && thinkingLevel != ai.ThinkingLevelOff {
		completionOptions.Reasoning = &thinkingLevel
	}

	response, err := ai.CompleteSimple(
		model,
		ai.ModelContext{
			SystemPrompt: &systemPrompt,
			Messages:     []ai.Message{summarizationMessage},
		},
		completionOptions,
	)
	if err != nil {
		return "", fmt.Errorf("turn prefix summarization failed: %w", err)
	}
	summary, err := summaryText(ctx, response)
	if err != nil {
		return "", fmt.Errorf("turn prefix summarization failed: %w", err)
	}
	return summary, nil
}
