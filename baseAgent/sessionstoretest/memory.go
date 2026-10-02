package sessionstoretest

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"sort"
	"sync"
	"time"

	"github.com/ColeBurch/burrow/baseAgent"
)

// MemoryStore is an in-memory baseAgent.SessionStore. It keeps entries encoded
// with baseAgent.MarshalSessionEntry, as a database would, so it doubles as a
// reference implementation of the contract and a store for tests.
type MemoryStore struct {
	mu       sync.Mutex
	sessions map[string]*memorySession
}

type memorySession struct {
	header   []byte
	entries  [][]byte
	ids      map[string]bool
	current  *string
	modified time.Time
}

var _ baseAgent.SessionStore = (*MemoryStore)(nil)

// NewMemoryStore returns an empty MemoryStore.
func NewMemoryStore() *MemoryStore {
	return &MemoryStore{sessions: map[string]*memorySession{}}
}

func (s *MemoryStore) CreateSession(ctx context.Context, header baseAgent.SessionHeader, entries []baseAgent.SessionEntry) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	encodedHeader, err := json.Marshal(header)
	if err != nil {
		return err
	}
	// Encode everything before storing anything, so a failure stores nothing.
	session := &memorySession{header: encodedHeader, ids: map[string]bool{}}
	for _, entry := range entries {
		encoded, err := baseAgent.MarshalSessionEntry(entry)
		if err != nil {
			return err
		}
		id := entry.GetBase().ID
		if session.ids[id] {
			return fmt.Errorf("duplicate entry ID %q", id)
		}
		session.ids[id] = true
		session.entries = append(session.entries, encoded)
		session.current = &id
	}
	if len(entries) > 0 {
		session.modified = time.Now()
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.sessions[header.ID]; ok {
		return fmt.Errorf("create session %q: %w", header.ID, baseAgent.ErrSessionExists)
	}
	s.sessions[header.ID] = session
	return nil
}

func (s *MemoryStore) AppendEntry(ctx context.Context, sessionID string, entry baseAgent.SessionEntry) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	encoded, err := baseAgent.MarshalSessionEntry(entry)
	if err != nil {
		return err
	}
	base := entry.GetBase()

	s.mu.Lock()
	defer s.mu.Unlock()
	session, err := s.session(sessionID)
	if err != nil {
		return err
	}
	if session.ids[base.ID] {
		return nil
	}
	if !equalIDs(base.ParentID, session.current) {
		return fmt.Errorf("append to session %q: %w", sessionID, baseAgent.ErrSessionConflict)
	}
	id := base.ID
	session.ids[id] = true
	session.entries = append(session.entries, encoded)
	session.current = &id
	session.modified = time.Now()
	return nil
}

func (s *MemoryStore) SetCurrentEntry(ctx context.Context, sessionID string, expected, entryID *string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	session, err := s.session(sessionID)
	if err != nil {
		return err
	}
	if !equalIDs(expected, session.current) {
		return fmt.Errorf("set current entry of session %q: %w", sessionID, baseAgent.ErrSessionConflict)
	}
	if entryID != nil && !session.ids[*entryID] {
		return fmt.Errorf("set current entry of session %q: entry %q not found", sessionID, *entryID)
	}
	if entryID == nil {
		session.current = nil
	} else {
		id := *entryID
		session.current = &id
	}
	return nil
}

func (s *MemoryStore) LoadSession(ctx context.Context, sessionID string) (baseAgent.LoadedSession, error) {
	if err := ctx.Err(); err != nil {
		return baseAgent.LoadedSession{}, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	session, err := s.session(sessionID)
	if err != nil {
		return baseAgent.LoadedSession{}, err
	}
	header, entries, err := session.decode()
	if err != nil {
		return baseAgent.LoadedSession{}, err
	}
	loaded := baseAgent.LoadedSession{Header: header, Entries: entries}
	if session.current != nil {
		id := *session.current
		loaded.CurrentEntryID = &id
	}
	return loaded, nil
}

func (s *MemoryStore) ListSessions(ctx context.Context, query baseAgent.SessionQuery) ([]baseAgent.SessionInfo, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	want, err := normalizeJSON(query.Metadata)
	if err != nil {
		return nil, err
	}
	wantMetadata, _ := want.(map[string]any)

	s.mu.Lock()
	defer s.mu.Unlock()
	var infos []baseAgent.SessionInfo
	for _, session := range s.sessions {
		header, entries, err := session.decode()
		if err != nil {
			return nil, err
		}
		if !matchesMetadata(header.Metadata, wantMetadata) {
			continue
		}
		infos = append(infos, baseAgent.BuildSessionInfo(header, entries, session.modified))
	}
	sort.Slice(infos, func(i, j int) bool {
		if !infos[i].Modified.Equal(infos[j].Modified) {
			return infos[i].Modified.After(infos[j].Modified)
		}
		return infos[i].ID < infos[j].ID
	})
	if query.Limit > 0 && len(infos) > query.Limit {
		infos = infos[:query.Limit]
	}
	return infos, nil
}

func (s *MemoryStore) session(sessionID string) (*memorySession, error) {
	session, ok := s.sessions[sessionID]
	if !ok {
		return nil, fmt.Errorf("session %q: %w", sessionID, baseAgent.ErrSessionNotFound)
	}
	return session, nil
}

func (session *memorySession) decode() (baseAgent.SessionHeader, []baseAgent.SessionEntry, error) {
	var header baseAgent.SessionHeader
	if err := json.Unmarshal(session.header, &header); err != nil {
		return header, nil, err
	}
	entries := make([]baseAgent.SessionEntry, 0, len(session.entries))
	for _, encoded := range session.entries {
		entry, err := baseAgent.UnmarshalSessionEntry(encoded)
		if err != nil {
			return header, nil, err
		}
		entries = append(entries, entry)
	}
	return header, entries, nil
}

func equalIDs(a, b *string) bool {
	if a == nil || b == nil {
		return a == nil && b == nil
	}
	return *a == *b
}

func matchesMetadata(metadata, want map[string]any) bool {
	for key, value := range want {
		got, ok := metadata[key]
		if !ok || !reflect.DeepEqual(got, value) {
			return false
		}
	}
	return true
}

// normalizeJSON round-trips a value through JSON, so values compare the way
// they come back from storage.
func normalizeJSON(value any) (any, error) {
	if value == nil {
		return nil, nil
	}
	encoded, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	var normalized any
	if err := json.Unmarshal(encoded, &normalized); err != nil {
		return nil, fmt.Errorf("normalize JSON: %w", err)
	}
	return normalized, nil
}
