package baseAgent

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strings"
	"sync"
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

// SessionManager holds a session's entry tree and current position, and saves
// changes through a SessionStore when persisted. It is safe for concurrent use.
//
// Its methods only lock mu and delegate to state. Writes hold mu across their
// store call, so each entry's parent is the position the previous write left,
// and reads wait for an in-flight write rather than see the state before it.
type SessionManager struct {
	mu    sync.RWMutex
	state sessionState
}

// sessionState is everything SessionManager.mu guards. Its methods never lock
// and cannot reach SessionManager's, so code holding the lock cannot deadlock
// by calling back into a locking method.
type sessionState struct {
	sessionID string
	header    *SessionHeader
	entries   []SessionEntry

	byID                map[string]SessionEntry
	labelsByID          map[string]string
	labelTimestampsByID map[string]string
	leafID              *string

	persist bool
	store   SessionStore
}

// ReadonlySessionManager is the read side of SessionManager, for code such as
// hooks that may inspect the session but must not write to it.
type ReadonlySessionManager interface {
	GetSessionID() string
	GetHeader() *SessionHeader
	GetSessionName() *string
	GetLeafID() *string
	GetLeafEntry() SessionEntry
	GetEntry(id string) SessionEntry
	GetChildren(parentID string) []SessionEntry
	GetLabel(id string) *string
	GetBranch(from ...string) []SessionEntry
	GetBranches() []SessionBranch
	GetEntries() []SessionEntry
	GetTree() []*SessionTreeNode
	BuildSessionContext() SessionContext
}

var _ ReadonlySessionManager = (*SessionManager)(nil)

// SessionBranch is one branch of a session's entry tree: the path from the
// root to a leaf, an entry with no children.
type SessionBranch struct {
	// LeafID is the branch's last entry. SetLeaf(LeafID) switches to it.
	LeafID string
	// BranchPointID is the nearest entry above the leaf with more than one
	// child, where this branch splits from its siblings. It is nil when the
	// branch splits from another root, or the session has a single branch.
	BranchPointID *string
	// FirstEntryID is the first entry after BranchPointID. The branch's own
	// entries run from it to LeafID.
	FirstEntryID string
	// Label is the label nearest the leaf on the branch's own entries.
	Label *string
	// FirstMessage is the text of the first user message on the branch's own
	// entries. If they have none, as when a reply was regenerated, it is the
	// last user message before them.
	FirstMessage string
	// Active reports whether the current position is on the branch's own
	// entries. At a branch point, before the next append, no branch is active.
	Active bool
}

var errStoreRequired = errors.New("store required for persistent session")

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

// NewDefaultSessionHeader returns a header for a new session with a fresh ID.
func NewDefaultSessionHeader(parentSession *string, metadata map[string]any) *SessionHeader {
	header := completeHeader(&SessionHeader{ParentSession: parentSession}, metadata)
	return &header
}

// completeHeader returns a copy of header with its empty fields filled in, so
// a caller can pass only the fields it cares about, such as a custom ID.
func completeHeader(header *SessionHeader, metadata map[string]any) SessionHeader {
	var h SessionHeader
	if header != nil {
		h = *header
	}
	if h.Type == "" {
		h.Type = "session"
	}
	if h.Version == 0 {
		h.Version = CurrentSessionVersion
	}
	if h.ID == "" {
		h.ID = createSessionID()
	}
	if h.Timestamp == "" {
		h.Timestamp = nowISO()
	}
	if h.Metadata == nil {
		h.Metadata = metadata
	}
	return h
}

// NewSessionManager loads the stored session sessionID, or creates a new
// session when sessionID is nil. A new session uses header, with any empty
// fields filled in: pass &SessionHeader{ID: id} for a custom session ID.
// metadata is used when header has none.
func NewSessionManager(ctx context.Context, header *SessionHeader, sessionID *string, persist bool, store SessionStore, metadata map[string]any) (*SessionManager, error) {
	sm := &SessionManager{state: sessionState{persist: persist, store: store}}

	if sessionID != nil {
		if store == nil {
			return nil, errors.New("store required when loading session")
		}
		if err := sm.LoadSession(ctx, *sessionID); err != nil {
			return nil, err
		}
		return sm, nil
	}

	if err := sm.NewSession(ctx, header, metadata); err != nil {
		return nil, err
	}
	return sm, nil
}

func InMemorySessionManager(ctx context.Context) (*SessionManager, error) {
	return NewSessionManager(ctx, nil, nil, false, nil, nil)
}

// ============================================================================
// Loading and Creating
// ============================================================================

// LoadSession switches the manager to a stored session, resuming at its saved
// position. On error the manager is unchanged.
func (s *SessionManager) LoadSession(ctx context.Context, id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.state.load(ctx, id)
}

// Reload replaces the manager's state with the stored session, such as after
// ErrSessionConflict reports that another writer changed it.
func (s *SessionManager) Reload(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.state.persist {
		return errors.New("cannot reload a session that is not persisted")
	}
	return s.state.load(ctx, s.state.sessionID)
}

// NewSession switches the manager to a new, empty session. See NewSessionManager
// for header and metadata. On error the manager is unchanged.
func (s *SessionManager) NewSession(ctx context.Context, header *SessionHeader, metadata map[string]any) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.state.create(ctx, completeHeader(header, metadata))
}

func (s *SessionManager) IsPersisted() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.state.persist
}

func (st *sessionState) load(ctx context.Context, id string) error {
	if st.store == nil {
		return errStoreRequired
	}
	loaded, err := st.store.LoadSession(ctx, id)
	if err != nil {
		return err
	}
	// The stored position, not the last entry: after a branch the newest
	// entry can be on another branch.
	if current := loaded.CurrentEntryID; current != nil && !slices.ContainsFunc(loaded.Entries, func(e SessionEntry) bool {
		return e.GetBase().ID == *current
	}) {
		return fmt.Errorf("session %q: current entry %q not found", id, *current)
	}
	st.reset(id, loaded.Header, loaded.Entries, loaded.CurrentEntryID)
	return nil
}

func (st *sessionState) create(ctx context.Context, header SessionHeader) error {
	if st.persist {
		if st.store == nil {
			return errStoreRequired
		}
		if err := st.store.CreateSession(ctx, header, nil); err != nil {
			return err
		}
	}
	st.reset(header.ID, header, nil, nil)
	return nil
}

// reset replaces the session and rebuilds the indexes.
func (st *sessionState) reset(sessionID string, header SessionHeader, entries []SessionEntry, leafID *string) {
	st.sessionID = sessionID
	st.header = &header
	st.entries = entries
	st.byID = map[string]SessionEntry{}
	st.labelsByID = map[string]string{}
	st.labelTimestampsByID = map[string]string{}
	for _, e := range entries {
		st.index(e)
	}
	st.leafID = leafID
}

func (st *sessionState) index(e SessionEntry) {
	st.byID[e.GetBase().ID] = e
	if l, ok := e.(*LabelEntry); ok {
		if l.Label != nil && *l.Label != "" {
			st.labelsByID[l.TargetID] = *l.Label
			st.labelTimestampsByID[l.TargetID] = l.Timestamp
		} else {
			delete(st.labelsByID, l.TargetID)
			delete(st.labelTimestampsByID, l.TargetID)
		}
	}
}

// ============================================================================
// Appending
// ============================================================================

// add appends an entry built on the current position and returns its ID.
func (s *SessionManager) add(ctx context.Context, entryType string, build func(SessionEntryBase) SessionEntry) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.state.append(ctx, entryType, build)
}

func (s *SessionManager) AppendMessage(ctx context.Context, m agent.AgentMessage) (string, error) {
	return s.AppendMessageWithID(ctx, "", m)
}

// AppendMessageWithID appends a message with the given entry ID, or a fresh
// one when id is empty.
func (s *SessionManager) AppendMessageWithID(ctx context.Context, id string, m agent.AgentMessage) (string, error) {
	return s.add(ctx, EntryTypeMessage, func(b SessionEntryBase) SessionEntry {
		if id != "" {
			b.ID = id
		}
		return &SessionMessageEntry{b, m}
	})
}

func (s *SessionManager) AppendThinkingLevelChange(ctx context.Context, v string) (string, error) {
	return s.add(ctx, EntryTypeThinkingLevelChange, func(b SessionEntryBase) SessionEntry {
		return &ThinkingLevelChangeEntry{b, v}
	})
}

func (s *SessionManager) AppendModelChange(ctx context.Context, p, m string) (string, error) {
	return s.add(ctx, EntryTypeModelChange, func(b SessionEntryBase) SessionEntry {
		return &ModelChangeEntry{b, p, m}
	})
}

func (s *SessionManager) AppendCompaction(ctx context.Context, summary, first string, tokens int64, details any, fromHook *bool) (string, error) {
	return s.add(ctx, EntryTypeCompaction, func(b SessionEntryBase) SessionEntry {
		return &CompactionEntry{b, summary, first, tokens, details, fromHook}
	})
}

func (s *SessionManager) AppendCustomEntry(ctx context.Context, t string, data any) (string, error) {
	return s.add(ctx, EntryTypeCustom, func(b SessionEntryBase) SessionEntry {
		return &CustomEntry{SessionEntryBase: b, CustomType: t, Data: data}
	})
}

func (s *SessionManager) AppendSessionInfo(ctx context.Context, name string) (string, error) {
	n := strings.TrimSpace(name)
	return s.add(ctx, EntryTypeSessionInfo, func(b SessionEntryBase) SessionEntry {
		return &SessionInfoEntry{b, &n}
	})
}

func (s *SessionManager) AppendCustomMessageEntry(ctx context.Context, t string, c []ai.UserContent, display bool, details any) (string, error) {
	return s.add(ctx, EntryTypeCustomMessage, func(b SessionEntryBase) SessionEntry {
		return &CustomMessageEntry{SessionEntryBase: b, CustomType: t, Content: c, Details: details, Display: display}
	})
}

// AppendLabelChange sets the label of an entry, or clears it when label is nil
// or empty.
func (s *SessionManager) AppendLabelChange(ctx context.Context, target string, label *string) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.state.appendLabel(ctx, target, label)
}

func (st *sessionState) appendLabel(ctx context.Context, target string, label *string) (string, error) {
	if st.byID[target] == nil {
		return "", fmt.Errorf("entry %q not found", target)
	}
	return st.append(ctx, EntryTypeLabel, func(b SessionEntryBase) SessionEntry {
		return &LabelEntry{b, target, label}
	})
}

// append saves an entry built on the current position and makes it the
// current position.
func (st *sessionState) append(ctx context.Context, entryType string, build func(SessionEntryBase) SessionEntry) (string, error) {
	e := build(SessionEntryBase{Type: entryType, ID: generateID(), ParentID: st.leafID, Timestamp: nowISO()})
	if st.persist {
		if st.store == nil {
			return "", errStoreRequired
		}
		if err := st.store.AppendEntry(ctx, st.sessionID, e); err != nil {
			return "", err
		}
	}
	st.entries = append(st.entries, e)
	st.index(e)
	st.leafID = &e.GetBase().ID
	return e.GetBase().ID, nil
}

// ============================================================================
// Reading
// ============================================================================

func (s *SessionManager) GetSessionID() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.state.sessionID
}

// GetHeader returns a copy of the session header.
func (s *SessionManager) GetHeader() *SessionHeader {
	s.mu.RLock()
	defer s.mu.RUnlock()
	header := *s.state.header
	return &header
}

func (s *SessionManager) GetSessionName() *string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.state.sessionName()
}

func (s *SessionManager) GetLeafID() *string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.state.leafID
}

func (s *SessionManager) GetLeafEntry() SessionEntry {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.state.leafID == nil {
		return nil
	}
	return s.state.byID[*s.state.leafID]
}

func (s *SessionManager) GetEntry(id string) SessionEntry {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.state.byID[id]
}

// GetEntries returns every entry across all branches, in append order.
func (s *SessionManager) GetEntries() []SessionEntry {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return slices.Clone(s.state.entries)
}

// GetChildren returns the entries whose parent is parentID, in append order.
func (s *SessionManager) GetChildren(parentID string) []SessionEntry {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.state.children(parentID)
}

func (s *SessionManager) GetLabel(id string) *string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if v, ok := s.state.labelsByID[id]; ok {
		return &v
	}
	return nil
}

// GetBranch returns the path from the root to from, or to the current position
// when from is omitted.
func (s *SessionManager) GetBranch(from ...string) []SessionEntry {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if len(from) > 0 {
		return s.state.path(&from[0])
	}
	return s.state.path(s.state.leafID)
}

func (s *SessionManager) BuildSessionContext() SessionContext {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.state.leafID == nil {
		return SessionContext{
			Messages:      []agent.AgentMessage{},
			ThinkingLevel: "off",
			Model:         nil,
		}
	}
	return BuildSessionContext(s.state.entries, s.state.leafID, s.state.byID)
}

// GetTree returns the entry tree, with children in append order.
func (s *SessionManager) GetTree() []*SessionTreeNode {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.state.tree()
}

// GetBranches returns every branch of the entry tree, in tree order: depth
// first, with children in append order.
func (s *SessionManager) GetBranches() []SessionBranch {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.state.branches()
}

func (st *sessionState) sessionName() *string {
	for i := len(st.entries) - 1; i >= 0; i-- {
		if e, ok := st.entries[i].(*SessionInfoEntry); ok {
			if e.Name == nil || strings.TrimSpace(*e.Name) == "" {
				return nil
			}
			n := strings.TrimSpace(*e.Name)
			return &n
		}
	}
	return nil
}

func (st *sessionState) children(parentID string) []SessionEntry {
	var out []SessionEntry
	for _, e := range st.entries {
		if p := e.GetBase().ParentID; p != nil && *p == parentID {
			out = append(out, e)
		}
	}
	return out
}

// path returns the entries from the root to id.
func (st *sessionState) path(id *string) []SessionEntry {
	if id == nil {
		return nil
	}
	var path []SessionEntry
	// The length check stops on a parent cycle in corrupt data.
	for cur := st.byID[*id]; cur != nil && len(path) <= len(st.entries); {
		path = append(path, cur)
		parent := cur.GetBase().ParentID
		if parent == nil {
			break
		}
		cur = st.byID[*parent]
	}
	slices.Reverse(path)
	return path
}

func (st *sessionState) tree() []*SessionTreeNode {
	nodes := map[string]*SessionTreeNode{}
	var roots []*SessionTreeNode
	for _, e := range st.entries {
		id := e.GetBase().ID
		var lab, lt *string
		if v, ok := st.labelsByID[id]; ok {
			lab = &v
		}
		if v, ok := st.labelTimestampsByID[id]; ok {
			lt = &v
		}
		nodes[id] = &SessionTreeNode{Entry: e, Children: []*SessionTreeNode{}, Label: lab, LabelTimestamp: lt}
	}
	for _, e := range st.entries {
		n := nodes[e.GetBase().ID]
		if e.GetBase().ParentID == nil || *e.GetBase().ParentID == e.GetBase().ID {
			roots = append(roots, n)
		} else if p := nodes[*e.GetBase().ParentID]; p != nil {
			p.Children = append(p.Children, n)
		} else {
			roots = append(roots, n)
		}
	}
	return roots
}

func (st *sessionState) branches() []SessionBranch {
	// Roots are children of "", which no entry ID is.
	children := map[string][]SessionEntry{}
	for _, e := range st.entries {
		parent := ""
		if p := e.GetBase().ParentID; p != nil && *p != e.GetBase().ID && st.byID[*p] != nil {
			parent = *p
		}
		children[parent] = append(children[parent], e)
	}

	var branches []SessionBranch
	stack := slices.Clone(children[""])
	slices.Reverse(stack)
	for len(stack) > 0 {
		e := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		id := e.GetBase().ID
		if next := children[id]; len(next) > 0 {
			next = slices.Clone(next)
			slices.Reverse(next)
			stack = append(stack, next...)
			continue
		}
		branches = append(branches, st.branch(st.path(&id), children))
	}
	return branches
}

// branch describes the branch ending at the last entry of path.
func (st *sessionState) branch(path []SessionEntry, children map[string][]SessionEntry) SessionBranch {
	start := 0
	for i := len(path) - 1; i > 0; i-- {
		if len(children[path[i-1].GetBase().ID]) > 1 {
			start = i
			break
		}
	}
	own := path[start:]
	branch := SessionBranch{
		LeafID:       path[len(path)-1].GetBase().ID,
		FirstEntryID: own[0].GetBase().ID,
	}
	if start > 0 {
		branch.BranchPointID = &path[start-1].GetBase().ID
	}
	for i := len(own) - 1; i >= 0 && branch.Label == nil; i-- {
		if label, ok := st.labelsByID[own[i].GetBase().ID]; ok {
			branch.Label = &label
		}
	}
	for _, e := range own {
		if st.leafID != nil && e.GetBase().ID == *st.leafID {
			branch.Active = true
		}
		if branch.FirstMessage == "" {
			branch.FirstMessage = entryUserText(e)
		}
	}
	for i := start - 1; i >= 0 && branch.FirstMessage == ""; i-- {
		branch.FirstMessage = entryUserText(path[i])
	}
	return branch
}

func entryUserText(e SessionEntry) string {
	if m, ok := e.(*SessionMessageEntry); ok {
		if user, ok := m.Message.(ai.UserMessage); ok {
			return UserMessageText(user)
		}
	}
	return ""
}

// ============================================================================
// Branching and Forking
// ============================================================================

// SetLeaf moves the current position to an existing entry, so the next append
// starts a branch from it. A persistent session saves the position first.
func (s *SessionManager) SetLeaf(ctx context.Context, id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.state.setLeaf(ctx, &id)
}

// ResetLeaf moves the current position before the first entry, so the next
// append starts a new root.
func (s *SessionManager) ResetLeaf(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.state.setLeaf(ctx, nil)
}

// BranchWithSummary moves the current position to from, or before the first
// entry when from is nil, and appends a summary there of the branch being left.
func (s *SessionManager) BranchWithSummary(ctx context.Context, from *string, summary string, details any, fromHook *bool) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.state.branchWithSummary(ctx, from, summary, details, fromHook)
}

// Fork creates a new session from the path from the root to leafID, which then
// continues independently of this one. The entries keep their IDs. Labels on
// the path are carried over as new label entries after it, keeping their
// timestamps. A nil leafID forks from the root, giving a session with no
// entries. The fork is saved through this manager's store when persisted.
func (s *SessionManager) Fork(ctx context.Context, leafID *string) (*SessionManager, error) {
	s.mu.RLock()
	forked, err := s.state.fork(leafID)
	s.mu.RUnlock()
	if err != nil {
		return nil, err
	}
	// The fork is a new session, so this one need not stay locked while it
	// is saved.
	if forked.persist {
		if err := forked.store.CreateSession(ctx, *forked.header, forked.entries); err != nil {
			return nil, err
		}
	}
	return &SessionManager{state: forked}, nil
}

// setLeaf moves the current position to id, which must be an entry or nil.
func (st *sessionState) setLeaf(ctx context.Context, id *string) error {
	if id != nil && st.byID[*id] == nil {
		return fmt.Errorf("entry %q not found", *id)
	}
	if st.persist {
		if st.store == nil {
			return errStoreRequired
		}
		if err := st.store.SetCurrentEntry(ctx, st.sessionID, st.leafID, id); err != nil {
			return err
		}
	}
	st.leafID = id
	return nil
}

func (st *sessionState) branchWithSummary(ctx context.Context, from *string, summary string, details any, fromHook *bool) (string, error) {
	fromID := "root"
	if from != nil {
		fromID = *from
	}
	if err := st.setLeaf(ctx, from); err != nil {
		return "", err
	}
	return st.append(ctx, EntryTypeBranchSummary, func(b SessionEntryBase) SessionEntry {
		return &BranchSummaryEntry{b, fromID, summary, details, fromHook}
	})
}

// fork builds the state of a fork from leafID without saving it.
func (st *sessionState) fork(leafID *string) (sessionState, error) {
	if st.persist && st.store == nil {
		return sessionState{}, errStoreRequired
	}
	var path []SessionEntry
	if leafID != nil {
		if st.byID[*leafID] == nil {
			return sessionState{}, fmt.Errorf("entry %q not found", *leafID)
		}
		path = st.path(leafID)
	}
	parentSession := st.sessionID
	header := SessionHeader{ParentSession: &parentSession, Metadata: maps.Clone(st.header.Metadata)}
	if st.header.Name != nil {
		name := *st.header.Name
		header.Name = &name
	}

	entries := make([]SessionEntry, 0, len(path))
	var lastID *string
	var labeled []string
	for _, e := range path {
		// Labels are re-added below from the current label of each entry, so
		// skip them and re-parent the entries after them.
		if _, ok := e.(*LabelEntry); ok {
			continue
		}
		cloned := cloneSessionEntry(e)
		cloned.GetBase().ParentID = lastID
		entries = append(entries, cloned)
		lastID = &cloned.GetBase().ID
		if _, ok := st.labelsByID[cloned.GetBase().ID]; ok {
			labeled = append(labeled, cloned.GetBase().ID)
		}
	}
	for _, target := range labeled {
		label := st.labelsByID[target]
		e := &LabelEntry{
			SessionEntryBase: SessionEntryBase{Type: EntryTypeLabel, ID: generateID(), ParentID: lastID, Timestamp: st.labelTimestampsByID[target]},
			TargetID:         target,
			Label:            &label,
		}
		entries = append(entries, e)
		lastID = &e.ID
	}

	h := completeHeader(&header, nil)
	forked := sessionState{persist: st.persist, store: st.store}
	forked.reset(h.ID, h, entries, lastID)
	return forked, nil
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
