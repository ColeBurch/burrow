package baseAgent

import (
	"context"
	"errors"
	"sort"
	"strings"
	"time"

	"github.com/ColeBurch/burrow/agent"
	"github.com/ColeBurch/burrow/ai"
	"github.com/google/uuid"
)

const CurrentSessionVersion = 1

type SessionHeader struct {
	Type          string         `json:"type"`
	Version       int            `json:"version,omitempty"`
	ID            string         `json:"id"`
	Timestamp     string         `json:"timestamp"`
	ParentSession *string        `json:"parentSession,omitempty"`
	Name          *string        `json:"name,omitempty"`
	MessageCount  int64          `json:"messageCount,omitempty"`
	FirstMessage  *string        `json:"firstMessage,omitempty"`
	Metadata      map[string]any `json:"metadata,omitempty"`
}

type SessionEntryBase struct {
	Type      string  `json:"type"`
	ID        string  `json:"id"`
	ParentID  *string `json:"parentId"`
	Timestamp string  `json:"timestamp"`
}

type SessionMessageEntry struct {
	SessionEntryBase
	Message agent.AgentMessage `json:"message"`
}

type ThinkingLevelChangeEntry struct {
	SessionEntryBase
	ThinkingLevel string `json:"thinkingLevel"`
}

type ModelChangeEntry struct {
	SessionEntryBase
	Provider string `json:"provider"`
	ModelID  string `json:"modelId"`
}

type CompactionEntry struct {
	SessionEntryBase
	Summary          string `json:"summary"`
	FirstKeptEntryID string `json:"firstKeptEntryId"`
	TokensBefore     int64  `json:"tokensBefore"`
	Details          any    `json:"details,omitempty"`
	FromHook         *bool  `json:"fromHook,omitempty"`
}

type BranchSummaryEntry struct {
	SessionEntryBase
	FromID   string `json:"fromId"`
	Summary  string `json:"summary"`
	Details  any    `json:"details,omitempty"`
	FromHook *bool  `json:"fromHook,omitempty"`
}

type CustomEntry struct {
	SessionEntryBase
	CustomType string `json:"customType"`
	Data       any    `json:"data,omitempty"`
}

type LabelEntry struct {
	SessionEntryBase
	TargetID string  `json:"targetId"`
	Label    *string `json:"label,omitempty"`
}

type SessionInfoEntry struct {
	SessionEntryBase
	Name *string `json:"name,omitempty"`
}

type CustomMessageEntry struct {
	SessionEntryBase
	CustomType string           `json:"customType"`
	Content    []ai.UserContent `json:"content"`
	Details    any              `json:"details,omitempty"`
	Display    bool             `json:"display"`
}

type SessionEntry interface{ GetBase() *SessionEntryBase }

func (e *SessionMessageEntry) GetBase() *SessionEntryBase      { return &e.SessionEntryBase }
func (e *ThinkingLevelChangeEntry) GetBase() *SessionEntryBase { return &e.SessionEntryBase }
func (e *ModelChangeEntry) GetBase() *SessionEntryBase         { return &e.SessionEntryBase }
func (e *CompactionEntry) GetBase() *SessionEntryBase          { return &e.SessionEntryBase }
func (e *BranchSummaryEntry) GetBase() *SessionEntryBase       { return &e.SessionEntryBase }
func (e *CustomEntry) GetBase() *SessionEntryBase              { return &e.SessionEntryBase }
func (e *CustomMessageEntry) GetBase() *SessionEntryBase       { return &e.SessionEntryBase }
func (e *LabelEntry) GetBase() *SessionEntryBase               { return &e.SessionEntryBase }
func (e *SessionInfoEntry) GetBase() *SessionEntryBase         { return &e.SessionEntryBase }

type SessionTreeNode struct {
	Entry          SessionEntry       `json:"entry"`
	Children       []*SessionTreeNode `json:"children"`
	Label          *string            `json:"label,omitempty"`
	LabelTimestamp *string            `json:"labelTimestamp,omitempty"`
}
type SessionContext struct {
	Messages      []agent.AgentMessage
	ThinkingLevel string
	Model         *SessionModel
}
type SessionModel struct{ Provider, ModelID string }

type SessionInfo struct {
	ID                      string
	Name, ParentSessionPath *string
	Created, Modified       time.Time
	MessageCount            int
	FirstMessage            string
}
type SessionListProgress func(loaded, total int)

type SessionStore interface {
	CreateSession(ctx context.Context, header *SessionHeader) error
	LoadSession(ctx context.Context, sessionID string) (*SessionHeader, []SessionEntry, error)
	AppendEntry(ctx context.Context, sessionID string, entry SessionEntry) error
	ListSessions(ctx context.Context, metadata map[string]any) ([]SessionInfo, error)
	FindMostRecentSession(ctx context.Context, metadata map[string]any) (*SessionInfo, error)
	CreateBranch(ctx context.Context, fromLeafID string, newHeader SessionHeader) error
}

type SessionManager struct {
	sessionID string
	persist   bool

	Header  *SessionHeader
	Entries []SessionEntry

	byID                map[string]SessionEntry
	labelsByID          map[string]string
	labelTimestampsByID map[string]string
	leafID              *string

	Store SessionStore
}

func createSessionID() string { return uuid.NewString() }

func generateID() string {
	id, err := uuid.NewV7()
	if err != nil {
		return uuid.NewString()
	}
	return id.String()
}

func nowISO() string { return time.Now().UTC().Format(time.RFC3339Nano) }

func GetLatestCompactionEntry(entries []SessionEntry) *CompactionEntry {
	for i := len(entries) - 1; i >= 0; i-- {
		if e, ok := entries[i].(*CompactionEntry); ok {
			return e
		}
	}
	return nil
}

func BuildSessionContext(entries []SessionEntry, leafID *string, byID map[string]SessionEntry) SessionContext {
	if byID == nil {
		byID = map[string]SessionEntry{}
		for _, e := range entries {
			byID[e.GetBase().ID] = e
		}
	}
	if leafID != nil && *leafID == "" {
		return SessionContext{Messages: []agent.AgentMessage{}, ThinkingLevel: "off"}
	}
	var leaf SessionEntry
	if leafID != nil {
		leaf = byID[*leafID]
	}
	if leaf == nil && len(entries) > 0 {
		leaf = entries[len(entries)-1]
	}
	if leaf == nil {
		return SessionContext{Messages: []agent.AgentMessage{}, ThinkingLevel: "off"}
	}
	var path []SessionEntry
	for cur := leaf; cur != nil; {
		path = append([]SessionEntry{cur}, path...)
		if cur.GetBase().ParentID == nil {
			break
		}
		cur = byID[*cur.GetBase().ParentID]
	}
	thinking := "off"
	var model *SessionModel
	var comp *CompactionEntry
	for _, e := range path {
		switch x := e.(type) {
		case *ThinkingLevelChangeEntry:
			thinking = x.ThinkingLevel
		case *ModelChangeEntry:
			model = &SessionModel{x.Provider, x.ModelID}
		case *SessionMessageEntry:
			if m, ok := x.Message.(ai.AssistantMessage); ok {
				model = &SessionModel{string(m.Provider), m.Model}
			}
		case *CompactionEntry:
			comp = x
		}
	}
	msgs := []agent.AgentMessage{}
	appendMsg := func(e SessionEntry) {
		switch x := e.(type) {
		case *SessionMessageEntry:
			msgs = append(msgs, x.Message)
		case *CustomMessageEntry:
			msgs = append(msgs, CreateCustomMessage(x.CustomType, x.Content, x.Display, x.Details, parseMillis(x.Timestamp)))
		case *BranchSummaryEntry:
			if x.Summary != "" {
				msgs = append(msgs, CreateBranchSummaryMessage(x.Summary, x.FromID, parseMillis(x.Timestamp)))
			}
		}
	}
	if comp != nil {
		msgs = append(msgs, CreateCompactionSummaryMessage(comp.Summary, comp.TokensBefore, parseMillis(comp.Timestamp)))
		ci := 0
		for i, e := range path {
			if e.GetBase().ID == comp.ID {
				ci = i
				break
			}
		}
		found := false
		for i := 0; i < ci; i++ {
			if path[i].GetBase().ID == comp.FirstKeptEntryID {
				found = true
			}
			if found {
				appendMsg(path[i])
			}
		}
		for i := ci + 1; i < len(path); i++ {
			appendMsg(path[i])
		}
	} else {
		for _, e := range path {
			appendMsg(e)
		}
	}
	return SessionContext{msgs, thinking, model}
}

func NewDefaultSessionHeader(parentSession *string, metadata map[string]any) *SessionHeader {
	return &SessionHeader{
		Type:          "session",
		Version:       CurrentSessionVersion,
		ID:            createSessionID(),
		Timestamp:     time.Now().Format(time.RFC3339),
		ParentSession: parentSession,
		Name:          nil,
		MessageCount:  0,
		FirstMessage:  nil,
		Metadata:      metadata,
	}
}

func NewSessionManager(ctx context.Context, header *SessionHeader, sessionID *string, persist bool, store SessionStore, metadata map[string]any) (*SessionManager, error) {
	sm := &SessionManager{persist: persist, byID: map[string]SessionEntry{}, labelsByID: map[string]string{}, labelTimestampsByID: map[string]string{}, Store: store}

	if sessionID != nil {
		if store == nil {
			return nil, errors.New("store required when loading session")
		}
		if err := sm.LoadSession(ctx, *sessionID); err != nil {
			return nil, err
		}
		return sm, nil
	}

	if header == nil {
		header = NewDefaultSessionHeader(nil, metadata)
	}

	if err := sm.NewSession(ctx, header, metadata); err != nil {
		return nil, err
	}

	return sm, nil
}

func (s *SessionManager) LoadSession(ctx context.Context, id string) error {
	header, entries, err := s.Store.LoadSession(ctx, id)
	if err != nil {
		return err
	}

	s.sessionID = id
	s.Header = header
	s.Entries = entries
	s.buildIndex()

	return nil
}

func (s *SessionManager) NewSession(ctx context.Context, header *SessionHeader, metadata map[string]any) error {
	if header == nil {
		header = NewDefaultSessionHeader(nil, metadata)
	}

	s.sessionID = header.ID
	s.Header = header
	s.Entries = []SessionEntry{}
	s.byID = map[string]SessionEntry{}
	s.labelsByID = map[string]string{}
	s.labelTimestampsByID = map[string]string{}
	s.leafID = nil

	if s.persist {
		if s.Store == nil {
			return errors.New("store required for persistent session")
		}

		if err := s.Store.CreateSession(ctx, header); err != nil {
			return err
		}
	}

	return nil
}

func (s *SessionManager) buildIndex() {
	s.byID = map[string]SessionEntry{}
	s.labelsByID = map[string]string{}
	s.labelTimestampsByID = map[string]string{}
	s.leafID = nil
	for _, fe := range s.Entries {
		e, ok := fe.(SessionEntry)
		if !ok {
			continue
		}
		id := e.GetBase().ID
		s.byID[id] = e
		s.leafID = &e.GetBase().ID
		if l, ok := e.(*LabelEntry); ok {
			if l.Label != nil && *l.Label != "" {
				s.labelsByID[l.TargetID] = *l.Label
				s.labelTimestampsByID[l.TargetID] = l.Timestamp
			} else {
				delete(s.labelsByID, l.TargetID)
				delete(s.labelTimestampsByID, l.TargetID)
			}
		}
	}
}

func (s *SessionManager) IsPersisted() bool { return s.persist }

func (s *SessionManager) GetSessionID() string { return s.sessionID }

func (s *SessionManager) appendEntry(ctx context.Context, e SessionEntry) error {
	if s.persist {
		if s.Store == nil {
			return errors.New("store required for persistent session")
		}
		if err := s.Store.AppendEntry(ctx, s.sessionID, e); err != nil {
			return err
		}
	}
	s.Entries = append(s.Entries, e)
	s.byID[e.GetBase().ID] = e
	s.leafID = &e.GetBase().ID

	return nil
}

func (s *SessionManager) base(t string) SessionEntryBase {
	return SessionEntryBase{t, generateID(), s.leafID, nowISO()}
}

func (s *SessionManager) AppendMessage(ctx context.Context, m agent.AgentMessage) (string, error) {
	e := &SessionMessageEntry{s.base("message"), m}
	if err := s.appendEntry(ctx, e); err != nil {
		return "", err
	}
	return e.ID, nil
}

func (s *SessionManager) AppendThinkingLevelChange(ctx context.Context, v string) (string, error) {
	e := &ThinkingLevelChangeEntry{s.base("thinking_level_change"), v}
	if err := s.appendEntry(ctx, e); err != nil {
		return "", err
	}
	return e.ID, nil
}

func (s *SessionManager) AppendModelChange(ctx context.Context, p, m string) (string, error) {
	e := &ModelChangeEntry{s.base("model_change"), p, m}
	if err := s.appendEntry(ctx, e); err != nil {
		return "", err
	}
	return e.ID, nil
}

func (s *SessionManager) AppendCompaction(ctx context.Context, summary, first string, tokens int64, details any, fromHook *bool) (string, error) {
	e := &CompactionEntry{s.base("compaction"), summary, first, tokens, details, fromHook}
	if err := s.appendEntry(ctx, e); err != nil {
		return "", err
	}
	return e.ID, nil
}

func (s *SessionManager) AppendCustomEntry(ctx context.Context, t string, data any) (string, error) {
	e := &CustomEntry{SessionEntryBase: s.base("custom"), CustomType: t, Data: data}
	if err := s.appendEntry(ctx, e); err != nil {
		return "", err
	}
	return e.ID, nil
}

func (s *SessionManager) AppendSessionInfo(ctx context.Context, name string) (string, error) {
	n := strings.TrimSpace(name)
	e := &SessionInfoEntry{s.base("session_info"), &n}
	if err := s.appendEntry(ctx, e); err != nil {
		return "", err
	}
	return e.ID, nil
}

func (s *SessionManager) GetSessionName() *string {
	es := s.GetEntries()
	for i := len(es) - 1; i >= 0; i-- {
		if e, ok := es[i].(*SessionInfoEntry); ok {
			if e.Name == nil || strings.TrimSpace(*e.Name) == "" {
				return nil
			}
			n := strings.TrimSpace(*e.Name)
			return &n
		}
	}
	return nil
}

func (s *SessionManager) AppendCustomMessageEntry(ctx context.Context, t string, c []ai.UserContent, display bool, details any) (string, error) {
	e := &CustomMessageEntry{SessionEntryBase: s.base("custom_message"), CustomType: t, Content: c, Details: details, Display: display}
	if err := s.appendEntry(ctx, e); err != nil {
		return "", err
	}
	return e.ID, nil
}

func (s *SessionManager) GetLeafID() *string { return s.leafID }

func (s *SessionManager) GetLeafEntry() SessionEntry {
	if s.leafID == nil {
		return nil
	}
	return s.byID[*s.leafID]
}

func (s *SessionManager) GetEntry(id string) SessionEntry { return s.byID[id] }

func (s *SessionManager) GetChildren(parentID string) []SessionEntry {
	var out []SessionEntry
	for _, e := range s.byID {
		if e.GetBase().ParentID != nil && *e.GetBase().ParentID == parentID {
			out = append(out, e)
		}
	}
	return out
}

func (s *SessionManager) GetLabel(id string) *string {
	if v, ok := s.labelsByID[id]; ok {
		return &v
	}
	return nil
}

func (s *SessionManager) AppendLabelChange(ctx context.Context, target string, label *string) (string, error) {
	if _, ok := s.byID[target]; !ok {
		return "", errors.New("entry not found")
	}
	e := &LabelEntry{s.base("label"), target, label}
	if err := s.appendEntry(ctx, e); err != nil {
		return "", err
	}
	if label != nil && *label != "" {
		s.labelsByID[target] = *label
		s.labelTimestampsByID[target] = e.Timestamp
	} else {
		delete(s.labelsByID, target)
		delete(s.labelTimestampsByID, target)
	}
	return e.ID, nil
}

func (s *SessionManager) GetBranch(from ...string) []SessionEntry {
	var start *string = s.leafID
	if len(from) > 0 {
		start = &from[0]
	}
	var path []SessionEntry
	for cur := func() SessionEntry {
		if start == nil {
			return nil
		}
		return s.byID[*start]
	}(); cur != nil; {
		path = append([]SessionEntry{cur}, path...)
		if cur.GetBase().ParentID == nil {
			break
		}
		cur = s.byID[*cur.GetBase().ParentID]
	}
	return path
}

func (s *SessionManager) BuildSessionContext() SessionContext {
	return BuildSessionContext(s.GetEntries(), s.leafID, s.byID)
}

func (s *SessionManager) GetHeader() *SessionHeader { return s.Header }

func (s *SessionManager) GetEntries() []SessionEntry {
	out := []SessionEntry{}
	for _, fe := range s.Entries {
		if e, ok := fe.(SessionEntry); ok {
			out = append(out, e)
		}
	}
	return out
}

func (s *SessionManager) GetTree() []*SessionTreeNode {
	entries := s.GetEntries()
	nodes := map[string]*SessionTreeNode{}
	var roots []*SessionTreeNode
	for _, e := range entries {
		id := e.GetBase().ID
		var lab, lt *string
		if v, ok := s.labelsByID[id]; ok {
			lab = &v
		}
		if v, ok := s.labelTimestampsByID[id]; ok {
			lt = &v
		}
		nodes[id] = &SessionTreeNode{Entry: e, Children: []*SessionTreeNode{}, Label: lab, LabelTimestamp: lt}
	}
	for _, e := range entries {
		n := nodes[e.GetBase().ID]
		if e.GetBase().ParentID == nil || *e.GetBase().ParentID == e.GetBase().ID {
			roots = append(roots, n)
		} else if p := nodes[*e.GetBase().ParentID]; p != nil {
			p.Children = append(p.Children, n)
		} else {
			roots = append(roots, n)
		}
	}
	var stack []*SessionTreeNode = append([]*SessionTreeNode{}, roots...)
	for len(stack) > 0 {
		n := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		sort.Slice(n.Children, func(i, j int) bool {
			return n.Children[i].Entry.GetBase().Timestamp < n.Children[j].Entry.GetBase().Timestamp
		})
		stack = append(stack, n.Children...)
	}
	return roots
}

func (s *SessionManager) SetLeaf(id string) error {
	if _, ok := s.byID[id]; !ok {
		return errors.New("entry not found")
	}
	s.leafID = &id
	return nil
}

func (s *SessionManager) ResetLeaf() { s.leafID = nil }

/*
func (s *SessionManager) BranchWithSummary(ctx context.Context, from *string, summary string, details any, fromHook *bool) (string, error) {
	if from != nil {
		if _, ok := s.byID[*from]; !ok {
			return "", errors.New("entry not found")
		}
	}
	s.leafID = from
	fid := "root"
	if from != nil {
		fid = *from
	}
	e := &BranchSummaryEntry{s.base("branch_summary"), fid, summary, details, fromHook}
	e.ParentID = from
	if err := s.appendEntry(ctx, e); err != nil {
		return "", err
	}
	return e.ID, nil
}
*/

func InMemorySessionManager(ctx context.Context) (*SessionManager, error) {
	sm, err := NewSessionManager(ctx, nil, nil, false, nil, nil)
	if err != nil {
		return nil, err
	}
	return sm, nil
}

func (s *SessionManager) CreateBranchedSession(ctx context.Context, leafID string) (*SessionManager, error) {
	path := s.GetBranch(leafID)
	if len(path) == 0 {
		return nil, errors.New("leaf session not found")
	}

	if s.persist && s.Store == nil {
		return nil, errors.New("cannot persist session: Store is nil")
	}

	parentSessionID := s.sessionID
	id, ts := createSessionID(), nowISO()

	metadata := map[string]any(nil)
	if s.Header.Metadata != nil {
		metadata = make(map[string]any, len(s.Header.Metadata))
		for k, v := range s.Header.Metadata {
			metadata[k] = v
		}
	}

	newHeader := &SessionHeader{
		Type:          "session",
		Version:       CurrentSessionVersion,
		ID:            id,
		Timestamp:     ts,
		ParentSession: &parentSessionID,
		Name:          s.Header.Name,
		MessageCount:  0,
		FirstMessage:  s.Header.FirstMessage,
		Metadata:      metadata,
	}

	entries := make([]SessionEntry, 0, len(path))
	var lastKeptID *string

	for _, e := range path {
		base := e.GetBase()

		// Skip labels, but re-parent following entries so the branch chain remains valid.
		if base.Type == "label" {
			continue
		}

		cloned := cloneSessionEntry(e)
		cloned.GetBase().ParentID = lastKeptID

		entries = append(entries, cloned)

		idCopy := cloned.GetBase().ID
		lastKeptID = &idCopy

		if _, ok := cloned.(*SessionMessageEntry); ok {
			newHeader.MessageCount++
		}
	}

	if s.persist {
		if err := s.Store.CreateBranch(ctx, leafID, *newHeader); err != nil {
			return nil, err
		}
	}

	branched := &SessionManager{
		sessionID: id,
		persist:   s.persist,
		Header:    newHeader,
		Entries:   entries,
		Store:     s.Store,
	}

	branched.buildIndex()

	return branched, nil
}

func cloneSessionEntry(e SessionEntry) SessionEntry {
	switch x := e.(type) {
	case *SessionMessageEntry:
		c := *x
		return &c
	case *ThinkingLevelChangeEntry:
		c := *x
		return &c
	case *ModelChangeEntry:
		c := *x
		return &c
	case *CompactionEntry:
		c := *x
		return &c
	case *BranchSummaryEntry:
		c := *x
		return &c
	case *CustomEntry:
		c := *x
		return &c
	case *CustomMessageEntry:
		c := *x
		return &c
	case *SessionInfoEntry:
		c := *x
		return &c
	case *LabelEntry:
		c := *x
		return &c
	default:
		return e
	}
}

func parseMillis(s string) int64 {
	if t, err := time.Parse(time.RFC3339Nano, s); err == nil {
		return t.UnixMilli()
	}
	return 0
}

/*
 * TODO: reimplement this logic on the session store
 *
func parseFileEntry(b []byte) (FileEntry, error) {
	var m map[string]json.RawMessage
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, err
	}
	var typ string
	_ = json.Unmarshal(m["type"], &typ)
	switch typ {
	case "session":
		var h SessionHeader
		return &h, json.Unmarshal(b, &h)
	case "message":
		var e SessionMessageEntry
		if err := unmarshalBase(b, &e.SessionEntryBase); err != nil {
			return nil, err
		}
		e.Message = parseAgentMessage(m["message"])
		return &e, nil
	case "thinking_level_change":
		var e ThinkingLevelChangeEntry
		return &e, json.Unmarshal(b, &e)
	case "model_change":
		var e ModelChangeEntry
		return &e, json.Unmarshal(b, &e)
	case "compaction":
		var e CompactionEntry
		return &e, json.Unmarshal(b, &e)
	case "branch_summary":
		var e BranchSummaryEntry
		return &e, json.Unmarshal(b, &e)
	case "custom":
		var e CustomEntry
		return &e, json.Unmarshal(b, &e)
	case "custom_message":
		var e CustomMessageEntry
		unmarshalBase(b, &e.SessionEntryBase)
		var tmp struct {
			CustomType string
			Content    json.RawMessage
			Details    any
			Display    bool
		}
		_ = json.Unmarshal(b, &tmp)
		e.CustomType = tmp.CustomType
		e.Content = parseUserContents(tmp.Content)
		e.Details = tmp.Details
		e.Display = tmp.Display
		return &e, nil
	case "label":
		var e LabelEntry
		return &e, json.Unmarshal(b, &e)
	case "session_info":
		var e SessionInfoEntry
		return &e, json.Unmarshal(b, &e)
	}
	return nil, errors.New("unknown entry")
}

func unmarshalBase(b []byte, base *SessionEntryBase) error { return json.Unmarshal(b, base) }

func parseAgentMessage(b json.RawMessage) agent.AgentMessage {
	var h struct {
		Role string `json:"role"`
	}
	_ = json.Unmarshal(b, &h)
	switch h.Role {
	case "user":
		var tmp struct {
			Role      ai.Role
			Content   json.RawMessage
			Timestamp int64
		}
		_ = json.Unmarshal(b, &tmp)
		return ai.UserMessage{Role: tmp.Role, Content: parseUserContents(tmp.Content), Timestamp: tmp.Timestamp}
	case "assistant":
		var m ai.AssistantMessage
		_ = json.Unmarshal(b, &m)
		return m
	case "toolResult":
		var m ai.ToolResultMessage
		_ = json.Unmarshal(b, &m)
		return m
	case MessageRoleCustom:
		var m CustomMessage
		_ = json.Unmarshal(b, &m)
		return m
	case MessageRoleBranchSummary:
		var m BranchSummaryMessage
		_ = json.Unmarshal(b, &m)
		return m
	case MessageRoleCompactionSummary:
		var m CompactionSummaryMessage
		_ = json.Unmarshal(b, &m)
		return m
	}
	return nil
}

func parseUserContents(b json.RawMessage) []ai.UserContent {
	if len(b) == 0 {
		return nil
	}
	if b[0] == '"' {
		var s string
		_ = json.Unmarshal(b, &s)
		return []ai.UserContent{ai.TextContent{Type: ai.ContentTypeText, Text: s}}
	}
	var raws []json.RawMessage
	_ = json.Unmarshal(b, &raws)
	out := []ai.UserContent{}
	for _, r := range raws {
		var h struct{ Type ai.ContentType }
		_ = json.Unmarshal(r, &h)
		if h.Type == ai.ContentTypeImage {
			var x ai.ImageContent
			_ = json.Unmarshal(r, &x)
			out = append(out, x)
		} else {
			var x ai.TextContent
			_ = json.Unmarshal(r, &x)
			out = append(out, x)
		}
	}
	return out
}

*/
