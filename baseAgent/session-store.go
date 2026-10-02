package baseAgent

import (
	"context"
	"errors"
	"strings"
	"time"

	"github.com/ColeBurch/burrow/ai"
)

var (
	// ErrSessionNotFound is returned when a session ID has no stored session.
	ErrSessionNotFound = errors.New("session not found")
	// ErrSessionExists is returned by CreateSession when the header ID is taken.
	ErrSessionExists = errors.New("session already exists")
	// ErrSessionConflict is returned when another writer moved the session's
	// current position. Reload the session before writing again.
	ErrSessionConflict = errors.New("session changed by another writer")
)

// LoadedSession is a stored session with every entry it has.
type LoadedSession struct {
	Header SessionHeader
	// Entries holds every branch, in append order.
	Entries []SessionEntry
	// CurrentEntryID is the current position: the entry the next append
	// attaches to. Nil with entries present means "before the first entry",
	// so the next append starts a new root.
	CurrentEntryID *string
}

// SessionQuery selects sessions for ListSessions.
type SessionQuery struct {
	// Metadata matches sessions whose header metadata has every key with an
	// equal value. Nil or empty matches every session.
	Metadata map[string]any
	// Limit caps the number of sessions returned. Zero means no limit.
	Limit int
}

// SessionStore persists sessions. baseAgent ships no implementation, so any
// database can back it. sessionstoretest.Run checks an implementation against
// the rules below, and MarshalSessionEntry and UnmarshalSessionEntry encode
// entries for storage.
//
// Ordering and scope:
//   - Entries come back in append order. Use an insertion sequence, not entry
//     timestamps or IDs, which do not sort in append order.
//   - LoadSession returns all of a session's entries across every branch.
//     baseAgent works out the active path itself.
//   - Entry IDs are unique per session, not globally: key entries by
//     (session ID, entry ID). Forks keep their source entries' IDs.
//
// Position and conflicts:
//   - Every session has a current position, which AppendEntry and
//     SetCurrentEntry move. It is how baseAgent detects another writer.
//   - AppendEntry is safe to retry. Appending an entry whose ID the session
//     already has succeeds without changing anything; check this before the
//     conflict check, so a retry of a write that landed is not a conflict.
//   - Otherwise AppendEntry returns ErrSessionConflict unless entry.ParentID
//     equals the current position (both nil counts as equal). On success the
//     entry becomes the current position.
//   - SetCurrentEntry returns ErrSessionConflict unless expected equals the
//     current position. Branching within a session has no method of its own:
//     it is SetCurrentEntry followed by AppendEntry.
//   - The conflict check is a safety net, not a scheduler. Deployments should
//     keep one live AgentSession per session, through sticky routing or a lease.
//
// Creating and forking:
//   - CreateSession is all-or-nothing. It returns ErrSessionExists if the
//     header ID is taken, and fails without storing anything if any entry
//     cannot be stored, such as one with a duplicate ID. The current position
//     is the last entry, or nil without entries.
//   - A fork is CreateSession with header.ParentSession set and the copied
//     entries, which keep their IDs. Its entries are empty for a fork from the
//     root. A fork is independent of its source once created.
//
// Session info: stores derive SessionInfo themselves, as BuildSessionInfo does.
// ListSessions sorts by Modified, newest first.
//
// Methods given an unknown session ID return an error wrapping
// ErrSessionNotFound. Stores must not modify the headers and entries passed to them.
type SessionStore interface {
	CreateSession(ctx context.Context, header SessionHeader, entries []SessionEntry) error
	AppendEntry(ctx context.Context, sessionID string, entry SessionEntry) error
	SetCurrentEntry(ctx context.Context, sessionID string, expected, entryID *string) error
	LoadSession(ctx context.Context, sessionID string) (LoadedSession, error)
	ListSessions(ctx context.Context, query SessionQuery) ([]SessionInfo, error)
}

// SessionInfo summarizes a stored session for listing.
type SessionInfo struct {
	ID string
	// Name is the latest session_info entry's name, or nil if that entry
	// cleared it. Without any session_info entry a store may supply its own
	// name, such as one generated from the first message.
	Name            *string
	ParentSessionID *string
	// Created is the header timestamp.
	Created time.Time
	// Modified is when the store last appended an entry, or Created if it
	// has none.
	Modified time.Time
	// MessageCount counts message entries across every branch.
	MessageCount int
	// FirstMessage is the text of the first user message with any text, in
	// append order, or "" if there is none.
	FirstMessage string
	Metadata     map[string]any
}

// FindMostRecentSession returns the most recently modified session matching
// metadata, or nil if there is none.
func FindMostRecentSession(ctx context.Context, store SessionStore, metadata map[string]any) (*SessionInfo, error) {
	sessions, err := store.ListSessions(ctx, SessionQuery{Metadata: metadata, Limit: 1})
	if err != nil || len(sessions) == 0 {
		return nil, err
	}
	return &sessions[0], nil
}

// BuildSessionInfo derives SessionInfo from a session's header and entries in
// append order. modified is the store's time of the last append; pass the zero
// time when there are no entries. Stores that keep these values in columns
// should update them the same way on each append.
func BuildSessionInfo(header SessionHeader, entries []SessionEntry, modified time.Time) SessionInfo {
	info := SessionInfo{
		ID:              header.ID,
		ParentSessionID: header.ParentSession,
		Metadata:        header.Metadata,
	}
	if created, err := time.Parse(time.RFC3339Nano, header.Timestamp); err == nil {
		info.Created = created
	}
	info.Modified = modified
	if modified.IsZero() {
		info.Modified = info.Created
	}
	for _, entry := range entries {
		switch e := entry.(type) {
		case *SessionInfoEntry:
			info.Name = nil
			if e.Name != nil {
				if name := strings.TrimSpace(*e.Name); name != "" {
					info.Name = &name
				}
			}
		case *SessionMessageEntry:
			info.MessageCount++
			if info.FirstMessage == "" {
				if message, ok := e.Message.(ai.UserMessage); ok {
					info.FirstMessage = UserMessageText(message)
				}
			}
		}
	}
	return info
}

// UserMessageText joins a user message's text blocks with spaces.
func UserMessageText(message ai.UserMessage) string {
	var parts []string
	for _, content := range message.Content {
		if text, ok := content.(ai.TextContent); ok && text.Text != "" {
			parts = append(parts, text.Text)
		}
	}
	return strings.Join(parts, " ")
}
