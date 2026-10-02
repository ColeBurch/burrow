// Package sessionstoretest checks baseAgent.SessionStore implementations
// against the contract documented on baseAgent.SessionStore.
//
// Call Run from a test in the store's package:
//
//	func TestConformance(t *testing.T) {
//		sessionstoretest.Run(t, func(t *testing.T) baseAgent.SessionStore {
//			return newTestStore(t)
//		})
//	}
package sessionstoretest

import (
	"context"
	"errors"
	"reflect"
	"sort"
	"sync"
	"testing"
	"time"

	"github.com/ColeBurch/burrow/ai"
	"github.com/ColeBurch/burrow/baseAgent"
	"github.com/google/uuid"
	"github.com/govalues/decimal"
)

// modifiedGap separates appends whose Modified order a test depends on. It
// leaves room for stores with coarse clocks.
const modifiedGap = 20 * time.Millisecond

// Run checks the store returned by newStore against the SessionStore contract.
// newStore is called once per subtest. The store may share a database with
// other tests: every session uses fresh UUIDs, and listing tests filter on a
// metadata value unique to the subtest.
func Run(t *testing.T, newStore func(t *testing.T) baseAgent.SessionStore) {
	t.Helper()
	tests := []struct {
		name string
		run  func(*testing.T, *suite)
	}{
		{"CreateSession", testCreateSession},
		{"CreateIsAllOrNothing", testCreateIsAllOrNothing},
		{"ForkKeepsEntryIDs", testForkKeepsEntryIDs},
		{"AppendKeepsOrderAndEveryBranch", testAppendKeepsOrderAndEveryBranch},
		{"AppendRetryAndConflict", testAppendRetryAndConflict},
		{"ConcurrentAppendsConflict", testConcurrentAppendsConflict},
		{"SetCurrentEntry", testSetCurrentEntry},
		{"UnknownSession", testUnknownSession},
		{"SessionInfo", testSessionInfo},
		{"ListSessionsOrderLimitAndFilter", testListSessionsOrderLimitAndFilter},
		{"EntriesRoundTrip", testEntriesRoundTrip},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			s := &suite{
				ctx:   context.Background(),
				store: newStore(t),
				run:   uuid.NewString(),
				clock: time.Now().UTC().Truncate(time.Millisecond),
			}
			test.run(t, s)
		})
	}
}

type suite struct {
	ctx   context.Context
	store baseAgent.SessionStore
	// run is a metadata value unique to the subtest, for filtering lists.
	run   string
	clock time.Time
}

func (s *suite) header(parent *string, metadata map[string]any) baseAgent.SessionHeader {
	all := map[string]any{"conformanceRun": s.run}
	for key, value := range metadata {
		all[key] = value
	}
	return baseAgent.SessionHeader{
		Type:          "session",
		Version:       baseAgent.CurrentSessionVersion,
		ID:            uuid.NewString(),
		Timestamp:     time.Now().UTC().Truncate(time.Millisecond).Format(time.RFC3339Nano),
		ParentSession: parent,
		Metadata:      all,
	}
}

func (s *suite) create(t *testing.T, header baseAgent.SessionHeader, entries ...baseAgent.SessionEntry) {
	t.Helper()
	if err := s.store.CreateSession(s.ctx, header, entries); err != nil {
		t.Fatalf("CreateSession(%s) error = %v", header.ID, err)
	}
}

func (s *suite) append(t *testing.T, sessionID string, entries ...baseAgent.SessionEntry) {
	t.Helper()
	for _, entry := range entries {
		if err := s.store.AppendEntry(s.ctx, sessionID, entry); err != nil {
			t.Fatalf("AppendEntry(%s) error = %v", entry.GetBase().ID, err)
		}
	}
}

func (s *suite) setCurrent(t *testing.T, sessionID string, expected, entryID *string) {
	t.Helper()
	if err := s.store.SetCurrentEntry(s.ctx, sessionID, expected, entryID); err != nil {
		t.Fatalf("SetCurrentEntry(%s) error = %v", describeID(entryID), err)
	}
}

func (s *suite) load(t *testing.T, sessionID string) baseAgent.LoadedSession {
	t.Helper()
	loaded, err := s.store.LoadSession(s.ctx, sessionID)
	if err != nil {
		t.Fatalf("LoadSession(%s) error = %v", sessionID, err)
	}
	return loaded
}

func (s *suite) list(t *testing.T, query baseAgent.SessionQuery) []baseAgent.SessionInfo {
	t.Helper()
	if query.Metadata == nil {
		query.Metadata = map[string]any{}
	}
	query.Metadata["conformanceRun"] = s.run
	infos, err := s.store.ListSessions(s.ctx, query)
	if err != nil {
		t.Fatalf("ListSessions() error = %v", err)
	}
	return infos
}

// ids returns n fresh UUIDs that sort in descending order, so a store that
// orders entries by ID instead of append order is caught.
func ids(n int) []string {
	out := make([]string, n)
	for i := range out {
		out[i] = uuid.NewString()
	}
	sort.Sort(sort.Reverse(sort.StringSlice(out)))
	return out
}

// timestamp returns entry timestamps that run backwards in append order, so a
// store that orders entries by timestamp is caught.
func (s *suite) timestamp(i int) string {
	return s.clock.Add(-time.Duration(i) * time.Minute).Format(time.RFC3339Nano)
}

func (s *suite) userEntry(id string, parent *string, i int, text string) *baseAgent.SessionMessageEntry {
	return &baseAgent.SessionMessageEntry{
		SessionEntryBase: baseAgent.SessionEntryBase{Type: baseAgent.EntryTypeMessage, ID: id, ParentID: parent, Timestamp: s.timestamp(i)},
		Message: ai.UserMessage{
			Role:      ai.RoleUser,
			Content:   []ai.UserContent{ai.TextContent{Type: ai.ContentTypeText, Text: text}},
			Timestamp: s.clock.UnixMilli() + int64(i),
		},
	}
}

// chain links entries so each one's parent is the entry before it.
func chain(entries ...baseAgent.SessionEntry) []baseAgent.SessionEntry {
	for i := 1; i < len(entries); i++ {
		parent := entries[i-1].GetBase().ID
		entries[i].GetBase().ParentID = &parent
	}
	return entries
}

func (s *suite) userChain(n int) []baseAgent.SessionEntry {
	entries := make([]baseAgent.SessionEntry, n)
	for i, id := range ids(n) {
		entries[i] = s.userEntry(id, nil, i, "message "+id)
	}
	return chain(entries...)
}

func ptr[T any](value T) *T { return &value }

func assertHeader(t *testing.T, got, want baseAgent.SessionHeader) {
	t.Helper()
	gotTime, gotErr := time.Parse(time.RFC3339Nano, got.Timestamp)
	wantTime, _ := time.Parse(time.RFC3339Nano, want.Timestamp)
	if gotErr != nil || !gotTime.Equal(wantTime) {
		t.Errorf("header timestamp = %q, want the instant %q", got.Timestamp, want.Timestamp)
	}
	got.Timestamp, want.Timestamp = "", ""
	assertJSONEqual(t, "header", got, want)
}

func assertEntries(t *testing.T, got, want []baseAgent.SessionEntry) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("entries = %v, want %v", entryIDs(got), entryIDs(want))
	}
	for i := range want {
		if got[i] == nil {
			t.Fatalf("entry %d is nil", i)
		}
		assertJSONEqual(t, "entry "+want[i].GetBase().ID, got[i], want[i])
	}
}

func assertCurrent(t *testing.T, loaded baseAgent.LoadedSession, want *string) {
	t.Helper()
	if !equalIDs(loaded.CurrentEntryID, want) {
		t.Errorf("CurrentEntryID = %s, want %s", describeID(loaded.CurrentEntryID), describeID(want))
	}
}

func assertJSONEqual(t *testing.T, what string, got, want any) {
	t.Helper()
	if reflect.TypeOf(got) != reflect.TypeOf(want) {
		t.Errorf("%s has type %T, want %T", what, got, want)
		return
	}
	gotJSON, err := normalizeJSON(got)
	if err != nil {
		t.Fatalf("%s: %v", what, err)
	}
	wantJSON, err := normalizeJSON(want)
	if err != nil {
		t.Fatalf("%s: %v", what, err)
	}
	if !reflect.DeepEqual(gotJSON, wantJSON) {
		t.Errorf("%s differs after storage:\ngot  %v\nwant %v", what, gotJSON, wantJSON)
	}
}

func entryIDs(entries []baseAgent.SessionEntry) []string {
	out := make([]string, len(entries))
	for i, entry := range entries {
		if entry != nil {
			out[i] = entry.GetBase().ID
		}
	}
	return out
}

func describeID(id *string) string {
	if id == nil {
		return "nil"
	}
	return *id
}

// ============================================================================
// CreateSession
// ============================================================================

func testCreateSession(t *testing.T, s *suite) {
	empty := s.header(nil, map[string]any{"user": "u-1", "count": 3.0, "enabled": true})
	empty.Name = ptr("header name")
	s.create(t, empty)
	loaded := s.load(t, empty.ID)
	assertHeader(t, loaded.Header, empty)
	assertEntries(t, loaded.Entries, nil)
	assertCurrent(t, loaded, nil)

	header := s.header(nil, nil)
	entries := s.userChain(2)
	s.create(t, header, entries...)
	loaded = s.load(t, header.ID)
	assertHeader(t, loaded.Header, header)
	assertEntries(t, loaded.Entries, entries)
	assertCurrent(t, loaded, &entries[1].GetBase().ID)

	duplicate := header
	duplicate.Name = ptr("replacement")
	if err := s.store.CreateSession(s.ctx, duplicate, s.userChain(1)); !errors.Is(err, baseAgent.ErrSessionExists) {
		t.Fatalf("CreateSession(duplicate) error = %v, want ErrSessionExists", err)
	}
	loaded = s.load(t, header.ID)
	assertHeader(t, loaded.Header, header)
	assertEntries(t, loaded.Entries, entries)
}

// A failed create stores nothing, so a clean retry does not collide with it.
func testCreateIsAllOrNothing(t *testing.T, s *suite) {
	header := s.header(nil, nil)
	entries := s.userChain(3)
	repeated := s.userEntry(entries[1].GetBase().ID, &entries[1].GetBase().ID, 2, "repeated ID")
	if err := s.store.CreateSession(s.ctx, header, []baseAgent.SessionEntry{entries[0], entries[1], repeated}); err == nil {
		t.Fatal("CreateSession() with a duplicate entry ID succeeded")
	}
	if _, err := s.store.LoadSession(s.ctx, header.ID); !errors.Is(err, baseAgent.ErrSessionNotFound) {
		t.Fatalf("LoadSession() after a failed create error = %v, want ErrSessionNotFound", err)
	}
	s.create(t, header, entries...)
	assertEntries(t, s.load(t, header.ID).Entries, entries)
}

// A fork is a new session holding copies of the source's entries under the
// same IDs. Neither session sees the other's later appends.
func testForkKeepsEntryIDs(t *testing.T, s *suite) {
	source := s.header(nil, nil)
	entries := s.userChain(3)
	s.create(t, source, entries...)

	fork := s.header(&source.ID, nil)
	var copied []baseAgent.SessionEntry
	for _, entry := range entries[:2] {
		original := *entry.(*baseAgent.SessionMessageEntry)
		copied = append(copied, &original)
	}
	s.create(t, fork, copied...)
	next := s.userEntry(uuid.NewString(), &copied[1].GetBase().ID, 3, "fork only")
	s.append(t, fork.ID, next)

	forked := s.load(t, fork.ID)
	assertHeader(t, forked.Header, fork)
	assertEntries(t, forked.Entries, append(copied, next))
	original := s.load(t, source.ID)
	assertEntries(t, original.Entries, entries)
	assertCurrent(t, original, &entries[2].GetBase().ID)
}

// ============================================================================
// AppendEntry and SetCurrentEntry
// ============================================================================

// Entries load in append order, including abandoned branches and extra roots.
func testAppendKeepsOrderAndEveryBranch(t *testing.T, s *suite) {
	header := s.header(nil, nil)
	s.create(t, header)
	id := ids(5)
	a := s.userEntry(id[0], nil, 0, "a")
	b := s.userEntry(id[1], &a.ID, 1, "b")
	c := s.userEntry(id[2], &b.ID, 2, "c")
	s.append(t, header.ID, a, b, c)

	s.setCurrent(t, header.ID, &c.ID, &a.ID)
	d := s.userEntry(id[3], &a.ID, 3, "d")
	s.append(t, header.ID, d)
	s.setCurrent(t, header.ID, &d.ID, nil)
	e := s.userEntry(id[4], nil, 4, "e")
	s.append(t, header.ID, e)

	loaded := s.load(t, header.ID)
	assertEntries(t, loaded.Entries, []baseAgent.SessionEntry{a, b, c, d, e})
	assertCurrent(t, loaded, &e.ID)
}

// Retrying an append that already succeeded changes nothing, even after the
// position moved on. An append from any other position is a conflict.
func testAppendRetryAndConflict(t *testing.T, s *suite) {
	header := s.header(nil, nil)
	s.create(t, header)
	entries := s.userChain(2)
	if err := s.store.AppendEntry(s.ctx, header.ID, s.userEntry(uuid.NewString(), ptr(uuid.NewString()), 0, "unknown parent")); !errors.Is(err, baseAgent.ErrSessionConflict) {
		t.Fatalf("AppendEntry(parent set, no position) error = %v, want ErrSessionConflict", err)
	}
	s.append(t, header.ID, entries...)
	s.append(t, header.ID, entries[0])

	stale := s.userEntry(uuid.NewString(), &entries[0].GetBase().ID, 2, "stale parent")
	if err := s.store.AppendEntry(s.ctx, header.ID, stale); !errors.Is(err, baseAgent.ErrSessionConflict) {
		t.Fatalf("AppendEntry(stale parent) error = %v, want ErrSessionConflict", err)
	}
	root := s.userEntry(uuid.NewString(), nil, 2, "new root")
	if err := s.store.AppendEntry(s.ctx, header.ID, root); !errors.Is(err, baseAgent.ErrSessionConflict) {
		t.Fatalf("AppendEntry(nil parent, position set) error = %v, want ErrSessionConflict", err)
	}
	loaded := s.load(t, header.ID)
	assertEntries(t, loaded.Entries, entries)
	assertCurrent(t, loaded, &entries[1].GetBase().ID)
}

func testConcurrentAppendsConflict(t *testing.T, s *suite) {
	header := s.header(nil, nil)
	root := s.userChain(1)
	s.create(t, header, root...)
	parent := root[0].GetBase().ID

	const writers = 8
	candidates := make([]*baseAgent.SessionMessageEntry, writers)
	for i := range candidates {
		candidates[i] = s.userEntry(uuid.NewString(), &parent, i+1, "writer")
	}
	errs := make([]error, writers)
	start := make(chan struct{})
	var wg sync.WaitGroup
	for i := range candidates {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			errs[i] = s.store.AppendEntry(s.ctx, header.ID, candidates[i])
		}()
	}
	close(start)
	wg.Wait()

	var winner *baseAgent.SessionMessageEntry
	for i, err := range errs {
		switch {
		case err == nil:
			if winner != nil {
				t.Fatalf("two appends with the same parent both succeeded: %s and %s", winner.ID, candidates[i].ID)
			}
			winner = candidates[i]
		case !errors.Is(err, baseAgent.ErrSessionConflict):
			t.Fatalf("AppendEntry() error = %v, want nil or ErrSessionConflict", err)
		}
	}
	if winner == nil {
		t.Fatal("no concurrent append succeeded")
	}
	loaded := s.load(t, header.ID)
	assertEntries(t, loaded.Entries, append(root, winner))
	assertCurrent(t, loaded, &winner.ID)
}

func testSetCurrentEntry(t *testing.T, s *suite) {
	header := s.header(nil, nil)
	entries := s.userChain(2)
	s.create(t, header, entries...)
	first, last := entries[0].GetBase().ID, entries[1].GetBase().ID

	if err := s.store.SetCurrentEntry(s.ctx, header.ID, &first, &first); !errors.Is(err, baseAgent.ErrSessionConflict) {
		t.Fatalf("SetCurrentEntry(stale expected) error = %v, want ErrSessionConflict", err)
	}
	if err := s.store.SetCurrentEntry(s.ctx, header.ID, nil, &first); !errors.Is(err, baseAgent.ErrSessionConflict) {
		t.Fatalf("SetCurrentEntry(nil expected, position set) error = %v, want ErrSessionConflict", err)
	}
	if err := s.store.SetCurrentEntry(s.ctx, header.ID, &last, ptr(uuid.NewString())); err == nil {
		t.Fatal("SetCurrentEntry(unknown entry) succeeded")
	}
	assertCurrent(t, s.load(t, header.ID), &last)

	s.setCurrent(t, header.ID, &last, &first)
	assertCurrent(t, s.load(t, header.ID), &first)
	s.setCurrent(t, header.ID, &first, nil)
	loaded := s.load(t, header.ID)
	assertCurrent(t, loaded, nil)
	assertEntries(t, loaded.Entries, entries)
}

func testUnknownSession(t *testing.T, s *suite) {
	missing := uuid.NewString()
	if _, err := s.store.LoadSession(s.ctx, missing); !errors.Is(err, baseAgent.ErrSessionNotFound) {
		t.Errorf("LoadSession() error = %v, want ErrSessionNotFound", err)
	}
	if err := s.store.AppendEntry(s.ctx, missing, s.userChain(1)[0]); !errors.Is(err, baseAgent.ErrSessionNotFound) {
		t.Errorf("AppendEntry() error = %v, want ErrSessionNotFound", err)
	}
	if err := s.store.SetCurrentEntry(s.ctx, missing, nil, nil); !errors.Is(err, baseAgent.ErrSessionNotFound) {
		t.Errorf("SetCurrentEntry() error = %v, want ErrSessionNotFound", err)
	}
}

// ============================================================================
// ListSessions
// ============================================================================

func testSessionInfo(t *testing.T, s *suite) {
	parent := uuid.NewString()
	header := s.header(&parent, map[string]any{"user": "u-1"})
	s.create(t, header)
	id := ids(5)
	entries := []baseAgent.SessionEntry{
		// A user message without text does not count as the first message.
		&baseAgent.SessionMessageEntry{
			SessionEntryBase: baseAgent.SessionEntryBase{Type: baseAgent.EntryTypeMessage, ID: id[0], Timestamp: s.timestamp(0)},
			Message:          ai.UserMessage{Role: ai.RoleUser, Content: []ai.UserContent{ai.ImageContent{Type: ai.ContentTypeImage, Image: "aGk=", MimeType: "image/png"}}},
		},
		&baseAgent.SessionMessageEntry{
			SessionEntryBase: baseAgent.SessionEntryBase{Type: baseAgent.EntryTypeMessage, ID: id[1], Timestamp: s.timestamp(1)},
			Message: ai.UserMessage{Role: ai.RoleUser, Content: []ai.UserContent{
				ai.TextContent{Type: ai.ContentTypeText, Text: "first"},
				ai.TextContent{Type: ai.ContentTypeText, Text: "question"},
			}},
		},
		&baseAgent.SessionInfoEntry{SessionEntryBase: baseAgent.SessionEntryBase{Type: baseAgent.EntryTypeSessionInfo, ID: id[2], Timestamp: s.timestamp(2)}, Name: ptr("Old name")},
		&baseAgent.SessionInfoEntry{SessionEntryBase: baseAgent.SessionEntryBase{Type: baseAgent.EntryTypeSessionInfo, ID: id[3], Timestamp: s.timestamp(3)}, Name: ptr("  New name  ")},
		s.userEntry(id[4], nil, 4, "later question"),
	}
	s.append(t, header.ID, chain(entries...)...)

	infos := s.list(t, baseAgent.SessionQuery{})
	if len(infos) != 1 {
		t.Fatalf("ListSessions() = %d sessions, want 1", len(infos))
	}
	info := infos[0]
	created, _ := time.Parse(time.RFC3339Nano, header.Timestamp)
	want := baseAgent.BuildSessionInfo(header, entries, info.Modified)
	if info.ID != header.ID || !info.Created.Equal(created) || !equalIDs(info.ParentSessionID, &parent) {
		t.Errorf("info identity = %s created %v parent %s, want %s created %v parent %s",
			info.ID, info.Created, describeID(info.ParentSessionID), header.ID, created, parent)
	}
	if info.Name == nil || *info.Name != "New name" {
		t.Errorf("Name = %s, want the latest session_info name %q", describeID(info.Name), "New name")
	}
	if info.MessageCount != want.MessageCount || info.FirstMessage != want.FirstMessage {
		t.Errorf("MessageCount, FirstMessage = %d, %q, want %d, %q", info.MessageCount, info.FirstMessage, want.MessageCount, want.FirstMessage)
	}
	assertJSONEqual(t, "metadata", info.Metadata, header.Metadata)

	// Clearing the name leaves none, and every append moves Modified forward.
	time.Sleep(modifiedGap)
	cleared := &baseAgent.SessionInfoEntry{SessionEntryBase: baseAgent.SessionEntryBase{Type: baseAgent.EntryTypeSessionInfo, ID: uuid.NewString(), ParentID: &id[4], Timestamp: s.timestamp(5)}, Name: ptr("")}
	s.append(t, header.ID, cleared)
	updated := s.list(t, baseAgent.SessionQuery{})[0]
	if updated.Name != nil {
		t.Errorf("Name after clearing = %q, want nil", *updated.Name)
	}
	if !updated.Modified.After(info.Modified) {
		t.Errorf("Modified after append = %v, want after %v", updated.Modified, info.Modified)
	}
}

func testListSessionsOrderLimitAndFilter(t *testing.T, s *suite) {
	a := s.header(nil, map[string]any{"group": "x"})
	b := s.header(nil, map[string]any{"group": "y"})
	c := s.header(nil, map[string]any{"group": "x"})
	tips := map[string]*string{}
	for _, header := range []baseAgent.SessionHeader{a, b, c} {
		s.create(t, header)
		entry := s.userChain(1)[0]
		s.append(t, header.ID, entry)
		tips[header.ID] = &entry.GetBase().ID
		time.Sleep(modifiedGap)
	}
	s.append(t, a.ID, s.userEntry(uuid.NewString(), tips[a.ID], 1, "latest"))

	assertOrder := func(what string, infos []baseAgent.SessionInfo, want ...string) {
		t.Helper()
		var got []string
		for _, info := range infos {
			got = append(got, info.ID)
		}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("%s = %v, want %v (newest Modified first)", what, got, want)
		}
	}
	assertOrder("ListSessions()", s.list(t, baseAgent.SessionQuery{}), a.ID, c.ID, b.ID)
	assertOrder("ListSessions(Limit: 2)", s.list(t, baseAgent.SessionQuery{Limit: 2}), a.ID, c.ID)
	assertOrder("ListSessions(group x)", s.list(t, baseAgent.SessionQuery{Metadata: map[string]any{"group": "x"}}), a.ID, c.ID)
	assertOrder("ListSessions(no match)", s.list(t, baseAgent.SessionQuery{Metadata: map[string]any{"group": "z"}}))
}

// ============================================================================
// Entry storage
// ============================================================================

// Every entry type and message role comes back as it was appended.

func testEntriesRoundTrip(t *testing.T, s *suite) {
	header := s.header(nil, nil)
	s.create(t, header)
	id := ids(14)
	base := func(i int, entryType string) baseAgent.SessionEntryBase {
		return baseAgent.SessionEntryBase{Type: entryType, ID: id[i], Timestamp: s.timestamp(i)}
	}
	image := ai.ImageContent{Type: ai.ContentTypeImage, Image: "aGVsbG8=", MimeType: "image/png"}
	entries := chain(
		&baseAgent.SessionMessageEntry{SessionEntryBase: base(0, baseAgent.EntryTypeMessage), Message: ai.UserMessage{
			Role:      ai.RoleUser,
			Content:   []ai.UserContent{ai.TextContent{Type: ai.ContentTypeText, Text: "look <here> & there"}, image},
			Timestamp: 1700000000000,
		}},
		&baseAgent.SessionMessageEntry{SessionEntryBase: base(1, baseAgent.EntryTypeMessage), Message: ai.AssistantMessage{
			Role: ai.RoleAssistant,
			Content: []ai.AssistantContent{
				ai.ThinkingContent{Type: ai.ContentTypeThinking, Thinking: "plan", ThinkingSignature: ptr("sig"), Redacted: ptr(false)},
				ai.TextContent{Type: ai.ContentTypeText, Text: "running", TextSignature: ptr("text-sig")},
				ai.ToolCall{Type: ai.ContentTypeToolCall, Id: "call-1", Name: "bash", Args: map[string]any{"command": "ls -la", "timeout": 30.0}, ThoughtSignature: ptr("thought")},
			},
			Api:           ai.APIOpenAIResponses,
			Provider:      ai.ProviderOpenAI,
			Model:         "test-model",
			ResponseModel: ptr("test-model-2026"),
			ResponseID:    ptr("resp-1"),
			Diagnostics:   []ai.AssistantMessageDiagnostic{{Type: "retry", Timestamp: 1700000000001, Details: map[string]any{"attempt": 2.0}}},
			Usage: ai.Usage{Input: 1200, Output: 300, CacheRead: 50, CacheWrite: 10, Total: 1560, Cost: ai.Cost{
				Input: decimal.MustParse("0.0012"), Output: decimal.MustParse("0.0090"), Total: decimal.MustParse("0.0102"),
			}},
			StopReason: ai.StopReasonToolUse,
			Timestamp:  1700000000002,
		}},
		&baseAgent.SessionMessageEntry{SessionEntryBase: base(2, baseAgent.EntryTypeMessage), Message: ai.ToolResultMessage{
			Role:       ai.RoleToolResult,
			ToolCallID: "call-1",
			ToolName:   "bash",
			Content:    []ai.ToolResultContent{ai.TextContent{Type: ai.ContentTypeText, Text: "file.txt"}, image},
			Details:    map[string]any{"exitCode": 0.0},
			IsError:    true,
			Timestamp:  1700000000003,
		}},
		&baseAgent.SessionMessageEntry{SessionEntryBase: base(3, baseAgent.EntryTypeMessage), Message: ai.AssistantMessage{
			Role:         ai.RoleAssistant,
			Content:      []ai.AssistantContent{},
			StopReason:   ai.StopReasonError,
			ErrorMessage: "provider failed",
		}},
		&baseAgent.SessionMessageEntry{SessionEntryBase: base(4, baseAgent.EntryTypeMessage), Message: baseAgent.CreateCustomMessage("notice", []ai.UserContent{ai.TextContent{Type: ai.ContentTypeText, Text: "custom"}}, true, map[string]any{"k": "v"}, 1700000000004)},
		&baseAgent.SessionMessageEntry{SessionEntryBase: base(5, baseAgent.EntryTypeMessage), Message: baseAgent.CreateBranchSummaryMessage("branch summary", id[1], 1700000000005)},
		&baseAgent.SessionMessageEntry{SessionEntryBase: base(6, baseAgent.EntryTypeMessage), Message: baseAgent.CreateCompactionSummaryMessage("compaction summary", 4096, 1700000000006)},
		&baseAgent.ThinkingLevelChangeEntry{SessionEntryBase: base(7, baseAgent.EntryTypeThinkingLevelChange), ThinkingLevel: "high"},
		&baseAgent.ModelChangeEntry{SessionEntryBase: base(8, baseAgent.EntryTypeModelChange), Provider: "openai", ModelID: "test-model"},
		&baseAgent.CompactionEntry{SessionEntryBase: base(9, baseAgent.EntryTypeCompaction), Summary: "summary", FirstKeptEntryID: id[1], TokensBefore: 90000, Details: map[string]any{"files": []any{"a.go"}}, FromHook: ptr(true)},
		&baseAgent.BranchSummaryEntry{SessionEntryBase: base(10, baseAgent.EntryTypeBranchSummary), FromID: id[2], Summary: "branch", Details: []any{"x"}, FromHook: ptr(false)},
		&baseAgent.CustomEntry{SessionEntryBase: base(11, baseAgent.EntryTypeCustom), CustomType: "state", Data: map[string]any{"nested": map[string]any{"ok": true}}},
		&baseAgent.CustomMessageEntry{SessionEntryBase: base(12, baseAgent.EntryTypeCustomMessage), CustomType: "reminder", Content: []ai.UserContent{ai.TextContent{Type: ai.ContentTypeText, Text: "remember"}, image}, Details: "detail", Display: true},
		&baseAgent.LabelEntry{SessionEntryBase: base(13, baseAgent.EntryTypeLabel), TargetID: id[0], Label: ptr("start")},
	)
	s.append(t, header.ID, entries...)

	loaded := s.load(t, header.ID)
	assertEntries(t, loaded.Entries, entries)
	for i, entry := range loaded.Entries {
		if message, ok := entry.(*baseAgent.SessionMessageEntry); ok {
			want := entries[i].(*baseAgent.SessionMessageEntry).Message
			if reflect.TypeOf(message.Message) != reflect.TypeOf(want) {
				t.Errorf("entry %s message type = %T, want %T", message.ID, message.Message, want)
			}
		}
	}
}
