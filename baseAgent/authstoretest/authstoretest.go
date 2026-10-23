// Package authstoretest checks baseAgent.AuthStore implementations against
// the contract documented on baseAgent.AuthStore.
//
// Call Run from a test in the store's package, with two stores over the same
// database:
//
//	func TestConformance(t *testing.T) {
//		authstoretest.Run(t, func(t *testing.T) baseAgent.AuthStore {
//			return newTestStore(t)
//		})
//	}
//
// Run it with go test -timeout: a broken lock usually deadlocks rather than
// failing.
package authstoretest

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
	"uuid"

	"github.com/ColeBurch/burrow/ai"
	"github.com/ColeBurch/burrow/baseAgent"
)

// waitTimeout bounds every wait, so a broken store fails instead of hanging.
const waitTimeout = 5 * time.Second

// Run checks the store returned by newStore against the AuthStore contract.
// newStore is called twice per subtest, and the subtest uses both stores,
// often at once. They must share one database, as two processes would, so a
// lock held only in memory fails. Every subtest uses fresh owner IDs, so the
// stores may share a database with other tests.
func Run(t *testing.T, newStore func(t *testing.T) baseAgent.AuthStore) {
	t.Helper()
	tests := []struct {
		name string
		run  func(*testing.T, *suite)
	}{
		{"RoundTrip", testRoundTrip},
		{"Missing", testMissing},
		{"OwnersAreIsolated", testOwnersAreIsolated},
		{"NamespacesAreIsolated", testNamespacesAreIsolated},
		{"ListNames", testListNames},
		{"NamesAreOpaque", testNamesAreOpaque},
		{"ModifySeesCurrent", testModifySeesCurrent},
		{"ModifyErrorStoresNothing", testModifyErrorStoresNothing},
		{"ModifyIsExclusive", testModifyIsExclusive},
		{"ModifyFirstWriteIsExclusive", testModifyFirstWriteIsExclusive},
		{"DeleteWaitsForModify", testDeleteWaitsForModify},
		{"CancelledBeforeLock", testCancelledBeforeLock},
		{"NoAliasing", testNoAliasing},
		{"ScopedCredentials", testScopedCredentials},
		{"RefreshesOnce", testRefreshesOnce},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			s := &suite{
				ctx: context.Background(),
				a:   newStore(t),
				b:   newStore(t),
				run: uuid.New().String(),
			}
			test.run(t, s)
		})
	}
}

// namespace is where subtests store values, apart from provider credentials.
const namespace = "authstoretest"

type suite struct {
	ctx context.Context
	// a and b are two stores over one database, like two processes.
	a, b baseAgent.AuthStore
	// run is unique to the subtest, for owner IDs.
	run string
}

// owner returns an owner ID unique to the subtest and label.
func (s *suite) owner(label string) string {
	return "authstoretest-" + s.run + "-" + label
}

func (s *suite) put(t *testing.T, store baseAgent.AuthStore, ownerID, namespace, name, value string) {
	t.Helper()
	_, err := store.Modify(s.ctx, ownerID, namespace, name, func([]byte) ([]byte, error) { return []byte(value), nil })
	if err != nil {
		t.Fatalf("Modify(%q, %q, %q) error = %v", ownerID, namespace, name, err)
	}
}

func (s *suite) get(t *testing.T, store baseAgent.AuthStore, ownerID, namespace, name string) string {
	t.Helper()
	value, err := store.Get(s.ctx, ownerID, namespace, name)
	if err != nil {
		t.Fatalf("Get(%q, %q, %q) error = %v", ownerID, namespace, name, err)
	}
	return string(value)
}

// names returns List's names, sorted, since stores may return any order.
func (s *suite) names(t *testing.T, store baseAgent.AuthStore, ownerID, namespace string) []string {
	t.Helper()
	names, err := store.List(s.ctx, ownerID, namespace)
	if err != nil {
		t.Fatalf("List(%q, %q) error = %v", ownerID, namespace, err)
	}
	slices.Sort(names)
	return names
}

func (s *suite) assertValue(t *testing.T, store baseAgent.AuthStore, ownerID, namespace, name, want string) {
	t.Helper()
	if got := s.get(t, store, ownerID, namespace, name); got != want {
		t.Errorf("Get(%q, %q, %q) = %q, want %q", ownerID, namespace, name, got, want)
	}
}

func assertNames(t *testing.T, got []string, want ...string) {
	t.Helper()
	slices.Sort(want)
	if len(got) == 0 && len(want) == 0 {
		return
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("List = %q, want %q", got, want)
	}
}

// holdLock runs a Modify on store whose fn blocks, holding the lock on the
// key until release is called. release waits for the Modify to return.
func (s *suite) holdLock(t *testing.T, store baseAgent.AuthStore, ownerID, namespace, name string, write string) (release func()) {
	t.Helper()
	held, unblock, done := make(chan struct{}), make(chan struct{}), make(chan error, 1)
	go func() {
		_, err := store.Modify(s.ctx, ownerID, namespace, name, func([]byte) ([]byte, error) {
			close(held)
			<-unblock
			return []byte(write), nil
		})
		done <- err
	}()
	wait(t, held, "the lock-holding Modify to start fn")
	var once sync.Once
	release = func() {
		t.Helper()
		once.Do(func() {
			close(unblock)
			if err := receive(t, done, "the lock-holding Modify to return"); err != nil {
				t.Errorf("lock-holding Modify error = %v", err)
			}
		})
	}
	t.Cleanup(release)
	return release
}

func wait(t *testing.T, ch <-chan struct{}, what string) {
	t.Helper()
	select {
	case <-ch:
	case <-time.After(waitTimeout):
		t.Fatalf("timed out waiting for %s", what)
	}
}

func receive[T any](t *testing.T, ch <-chan T, what string) T {
	t.Helper()
	select {
	case value := <-ch:
		return value
	case <-time.After(waitTimeout):
		t.Fatalf("timed out waiting for %s", what)
		panic("unreachable")
	}
}

func testRoundTrip(t *testing.T, s *suite) {
	owner := s.owner("owner")
	binary := make([]byte, 256)
	for i := range binary {
		binary[i] = byte(i)
	}
	large := make([]byte, 64<<10)
	for i := range large {
		large[i] = byte(i*7 + i>>8)
	}
	values := map[string][]byte{
		"json":   []byte(`{"v":1,"type":"api_key","key":"sk-test"}`),
		"binary": binary,
		"large":  large,
	}
	for name, value := range values {
		post, err := s.a.Modify(s.ctx, owner, namespace, name, func([]byte) ([]byte, error) { return value, nil })
		if err != nil {
			t.Fatalf("Modify(%s) error = %v", name, err)
		}
		if !bytes.Equal(post, value) {
			t.Errorf("Modify(%s) returned %d bytes, want the %d written", name, len(post), len(value))
		}
		got, err := s.b.Get(s.ctx, owner, namespace, name)
		if err != nil {
			t.Fatalf("Get(%s) error = %v", name, err)
		}
		if !bytes.Equal(got, value) {
			t.Errorf("Get(%s) returned %d bytes, not the %d written", name, len(got), len(value))
		}
	}
}

func testMissing(t *testing.T, s *suite) {
	owner := s.owner("owner")
	if value, err := s.a.Get(s.ctx, owner, namespace, "missing"); len(value) != 0 || err != nil {
		t.Errorf("Get(missing) = %q, %v; want nil, nil", value, err)
	}
	assertNames(t, s.names(t, s.a, owner, namespace))
	if err := s.a.Delete(s.ctx, owner, namespace, "missing"); err != nil {
		t.Errorf("Delete(missing) error = %v", err)
	}

	var current []byte
	post, err := s.a.Modify(s.ctx, owner, namespace, "missing", func(c []byte) ([]byte, error) {
		current = c
		return nil, nil
	})
	if err != nil || len(current) != 0 || len(post) != 0 {
		t.Errorf("Modify(missing) saw %q, returned %q, %v; want nothing", current, post, err)
	}
	assertNames(t, s.names(t, s.b, owner, namespace))
}

func testOwnersAreIsolated(t *testing.T, s *suite) {
	// The "" owner is shared with everything else in the database, so use a
	// namespace unique to the subtest.
	ns := namespace + "-" + s.run
	owners := map[string]string{s.owner("a"): "value a", s.owner("b"): "value b", "": "value of empty owner"}
	for owner, value := range owners {
		s.put(t, s.a, owner, ns, "openai", value)
	}
	t.Cleanup(func() { s.a.Delete(context.Background(), "", ns, "openai") })

	for owner, value := range owners {
		s.assertValue(t, s.b, owner, ns, "openai", value)
		assertNames(t, s.names(t, s.b, owner, ns), "openai")
	}

	if err := s.a.Delete(s.ctx, s.owner("a"), ns, "openai"); err != nil {
		t.Fatal(err)
	}
	s.assertValue(t, s.b, s.owner("a"), ns, "openai", "")
	s.assertValue(t, s.b, s.owner("b"), ns, "openai", "value b")
	s.assertValue(t, s.b, "", ns, "openai", "value of empty owner")
}

func testNamespacesAreIsolated(t *testing.T, s *suite) {
	owner := s.owner("owner")
	s.put(t, s.a, owner, "provider", "linear", "provider value")
	s.put(t, s.a, owner, "provider", "openai", "openai value")
	s.put(t, s.a, owner, "mcp", "linear", "mcp value")

	s.assertValue(t, s.b, owner, "provider", "linear", "provider value")
	s.assertValue(t, s.b, owner, "mcp", "linear", "mcp value")
	assertNames(t, s.names(t, s.b, owner, "provider"), "linear", "openai")
	assertNames(t, s.names(t, s.b, owner, "mcp"), "linear")

	if err := s.a.Delete(s.ctx, owner, "mcp", "linear"); err != nil {
		t.Fatal(err)
	}
	s.assertValue(t, s.b, owner, "provider", "linear", "provider value")
	assertNames(t, s.names(t, s.b, owner, "mcp"))
}

func testListNames(t *testing.T, s *suite) {
	owner, other := s.owner("owner"), s.owner("other")
	for _, name := range []string{"zeta", "alpha", "mid"} {
		s.put(t, s.a, owner, namespace, name, "value "+name)
	}
	s.put(t, s.a, other, namespace, "other", "value")
	assertNames(t, s.names(t, s.b, owner, namespace), "alpha", "mid", "zeta")

	if err := s.a.Delete(s.ctx, owner, namespace, "mid"); err != nil {
		t.Fatal(err)
	}
	assertNames(t, s.names(t, s.b, owner, namespace), "alpha", "zeta")
}

func testNamesAreOpaque(t *testing.T, s *suite) {
	owner := s.owner("owner")
	names := []string{
		"linear|https://mcp.linear.app/sse?workspace=a_b%20c",
		"a/b",
		"../escape",
		"a%",
		"a_",
		"Linear", // a case-insensitive collation folds these two together
		"linear",
		"linéar",
		" padded ",
		strings.Repeat("n", 1024),
	}
	for i, name := range names {
		s.put(t, s.a, owner, namespace, name, "value "+strconv.Itoa(i))
	}
	for i, name := range names {
		if got := s.get(t, s.b, owner, namespace, name); got != "value "+strconv.Itoa(i) {
			t.Errorf("Get(%.40q) = %q, want %q", name, got, "value "+strconv.Itoa(i))
		}
	}
	assertNames(t, s.names(t, s.b, owner, namespace), names...)
}

func testModifySeesCurrent(t *testing.T, s *suite) {
	owner := s.owner("owner")
	s.put(t, s.a, owner, namespace, "key", "one")

	for _, unchanged := range [][]byte{nil, {}} {
		var current []byte
		post, err := s.b.Modify(s.ctx, owner, namespace, "key", func(c []byte) ([]byte, error) {
			current = c
			return unchanged, nil
		})
		if err != nil || string(current) != "one" || string(post) != "one" {
			t.Errorf("Modify returning %#v saw %q and returned %q, %v; want one and one", unchanged, current, post, err)
		}
	}

	var current []byte
	post, err := s.b.Modify(s.ctx, owner, namespace, "key", func(c []byte) ([]byte, error) {
		current = c
		return []byte("two"), nil
	})
	if err != nil || string(current) != "one" || string(post) != "two" {
		t.Errorf("Modify saw %q and returned %q, %v; want one and two", current, post, err)
	}
	s.assertValue(t, s.a, owner, namespace, "key", "two")
}

// fnError is an error type fn returns, to check errors.As through Modify.
type fnError struct{ msg string }

func (e *fnError) Error() string { return e.msg }

func testModifyErrorStoresNothing(t *testing.T, s *suite) {
	owner := s.owner("owner")
	s.put(t, s.a, owner, namespace, "existing", "original")

	for _, name := range []string{"existing", "missing"} {
		sentinel := &fnError{"refresh failed for " + name}
		_, err := s.b.Modify(s.ctx, owner, namespace, name, func([]byte) ([]byte, error) {
			return []byte("written anyway"), sentinel
		})
		var got *fnError
		if !errors.As(err, &got) || got != sentinel {
			t.Errorf("Modify(%s) error = %v; want fn's error, findable with errors.As", name, err)
		}
	}
	s.assertValue(t, s.a, owner, namespace, "existing", "original")
	s.assertValue(t, s.a, owner, namespace, "missing", "")
}

func testModifyIsExclusive(t *testing.T, s *suite) { s.checkExclusive(t, true) }

// Locking an existing row is not enough: when nothing is stored yet, there is
// no row to lock, and SELECT ... FOR UPDATE locks nothing.
func testModifyFirstWriteIsExclusive(t *testing.T, s *suite) { s.checkExclusive(t, false) }

// checkExclusive runs concurrent Modify calls across both stores, each
// incrementing a counter, and fails if two fn calls overlap or an update is
// lost.
func (s *suite) checkExclusive(t *testing.T, seed bool) {
	const writers = 20
	owner := s.owner("owner")
	if seed {
		s.put(t, s.a, owner, namespace, "counter", "0")
	}

	var inside atomic.Int32
	var overlapped atomic.Bool
	errs := make(chan error, writers)
	var wg sync.WaitGroup
	for i := range writers {
		store := s.a
		if i%2 == 1 {
			store = s.b
		}
		wg.Go(func() {
			_, err := store.Modify(s.ctx, owner, namespace, "counter", func(current []byte) ([]byte, error) {
				if inside.Add(1) != 1 {
					overlapped.Store(true)
				}
				defer inside.Add(-1)
				n := 0
				if len(current) > 0 {
					var err error
					if n, err = strconv.Atoi(string(current)); err != nil {
						return nil, err
					}
				}
				time.Sleep(5 * time.Millisecond) // widen the window for a race
				return []byte(strconv.Itoa(n + 1)), nil
			})
			if err != nil {
				errs <- err
			}
		})
	}
	done := make(chan struct{})
	go func() { wg.Wait(); close(done) }()
	wait(t, done, "concurrent Modify calls; a lock may be deadlocked")
	close(errs)
	for err := range errs {
		t.Errorf("concurrent Modify error = %v", err)
	}
	if overlapped.Load() {
		t.Error("two fn calls ran at once; Modify must hold a lock that both stores respect")
	}
	if got := s.get(t, s.b, owner, namespace, "counter"); got != strconv.Itoa(writers) {
		t.Errorf("counter = %q, want %d; concurrent Modify calls lost updates", got, writers)
	}
}

func testDeleteWaitsForModify(t *testing.T, s *suite) {
	owner := s.owner("owner")
	s.put(t, s.a, owner, namespace, "key", "original")
	release := s.holdLock(t, s.a, owner, namespace, "key", "refreshed")

	deleted := make(chan error, 1)
	go func() { deleted <- s.b.Delete(s.ctx, owner, namespace, "key") }()
	select {
	case err := <-deleted:
		t.Fatalf("Delete returned %v while a Modify held the lock; it must wait", err)
	case <-time.After(50 * time.Millisecond):
	}

	release()
	if err := receive(t, deleted, "Delete after the Modify finished"); err != nil {
		t.Fatalf("Delete error = %v", err)
	}
	// Delete ran after the Modify's write, so a logout is never undone.
	s.assertValue(t, s.a, owner, namespace, "key", "")
}

func testCancelledBeforeLock(t *testing.T, s *suite) {
	owner := s.owner("owner")
	s.put(t, s.a, owner, namespace, "key", "original")
	var ran atomic.Bool
	fn := func([]byte) ([]byte, error) {
		ran.Store(true)
		return []byte("changed"), nil
	}

	ctx, cancel := context.WithCancel(s.ctx)
	cancel()
	if _, err := s.a.Modify(ctx, owner, namespace, "key", fn); !errors.Is(err, context.Canceled) {
		t.Errorf("Modify with a cancelled ctx error = %v, want context.Canceled", err)
	}
	if ran.Load() {
		t.Error("Modify ran fn with a cancelled ctx")
	}

	release := s.holdLock(t, s.a, owner, namespace, "key", "")
	ctx, cancel = context.WithTimeout(s.ctx, 50*time.Millisecond)
	defer cancel()
	done := make(chan error, 1)
	go func() {
		_, err := s.b.Modify(ctx, owner, namespace, "key", fn)
		done <- err
	}()
	select {
	case err := <-done:
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Errorf("Modify waiting for the lock error = %v, want context.DeadlineExceeded", err)
		}
	case <-time.After(waitTimeout):
		release()
		<-done
		t.Error("Modify kept waiting for the lock after its ctx was done")
	}
	release()
	if ran.Load() {
		t.Error("Modify ran fn after its ctx was done while waiting for the lock")
	}
	s.assertValue(t, s.a, owner, namespace, "key", "original")
}

func testNoAliasing(t *testing.T, s *suite) {
	owner := s.owner("owner")
	written := []byte("original")
	post, err := s.a.Modify(s.ctx, owner, namespace, "key", func([]byte) ([]byte, error) { return written, nil })
	if err != nil {
		t.Fatal(err)
	}
	written[0] = 'X'
	post[0] = 'Y'

	got, err := s.a.Get(s.ctx, owner, namespace, "key")
	if err != nil {
		t.Fatal(err)
	}
	got[0] = 'Z'

	post, err = s.a.Modify(s.ctx, owner, namespace, "key", func(current []byte) ([]byte, error) {
		current[0] = 'W'
		return nil, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	post[0] = 'V'

	if got := s.get(t, s.a, owner, namespace, "key"); got != "original" {
		t.Errorf("stored value = %q, want original; the store kept or returned a slice it shares with callers", got)
	}
}

func testScopedCredentials(t *testing.T, s *suite) {
	ownerA, ownerB := s.owner("a"), s.owner("b")
	viaA := baseAgent.ScopeAuthStore(s.a, ownerA)
	viaB := baseAgent.ScopeAuthStore(s.b, ownerA)
	otherOwner := baseAgent.ScopeAuthStore(s.a, ownerB)

	apiKey := &ai.Credential{Type: ai.CredentialAPIKey, Key: "sk-a", Env: map[string]string{"ACCOUNT": "a"}}
	oauth := &ai.Credential{Type: ai.CredentialOAuth, OAuth: &ai.OAuthToken{
		Access:  "access",
		Refresh: "refresh",
		Expires: time.Now().Add(time.Hour).UTC().Truncate(time.Millisecond),
	}}
	for provider, credential := range map[ai.Provider]*ai.Credential{"openai": apiKey, "zeta": oauth} {
		if _, err := viaA.Modify(s.ctx, provider, func(*ai.Credential) (*ai.Credential, error) { return credential, nil }); err != nil {
			t.Fatalf("Modify(%s) error = %v", provider, err)
		}
	}
	if _, err := otherOwner.Modify(s.ctx, "openai", func(*ai.Credential) (*ai.Credential, error) {
		return &ai.Credential{Type: ai.CredentialAPIKey, Key: "sk-b"}, nil
	}); err != nil {
		t.Fatal(err)
	}
	// Other namespaces of the same owner are not credentials.
	s.put(t, s.a, ownerA, "mcp", "linear", "mcp state")

	infos, err := viaB.List(s.ctx)
	want := []ai.CredentialInfo{{Provider: "openai", Type: ai.CredentialAPIKey}, {Provider: "zeta", Type: ai.CredentialOAuth}}
	if err != nil || !reflect.DeepEqual(infos, want) {
		t.Errorf("List = %+v, %v; want %+v", infos, err, want)
	}

	if got, err := viaB.Read(s.ctx, "openai"); err != nil || !reflect.DeepEqual(got, apiKey) {
		t.Errorf("Read(openai) = %+v, %v; want %+v", got, err, apiKey)
	}
	got, err := viaB.Read(s.ctx, "zeta")
	if err != nil || got == nil || got.OAuth == nil || got.OAuth.Access != "access" || !got.OAuth.Expires.Equal(oauth.OAuth.Expires) {
		t.Errorf("Read(zeta) = %+v, %v; want %+v", got, err, oauth)
	}

	if err := viaA.Delete(s.ctx, "openai"); err != nil {
		t.Fatal(err)
	}
	if got, err := viaB.Read(s.ctx, "openai"); got != nil || err != nil {
		t.Errorf("Read after Delete = %+v, %v; want nil, nil", got, err)
	}
	if got, err := otherOwner.Read(s.ctx, "openai"); err != nil || got == nil || got.Key != "sk-b" {
		t.Errorf("other owner's Read = %+v, %v; want its own key", got, err)
	}
}

// testRefreshesOnce checks the failure the lock exists for: concurrent
// requests in two processes find an OAuth token expired, and a provider that
// rotates refresh tokens rejects the second refresh, losing the grant.
func testRefreshesOnce(t *testing.T, s *suite) {
	const requests = 10
	const provider ai.Provider = "openai"
	owner := s.owner("owner")

	expired := &ai.Credential{Type: ai.CredentialOAuth, OAuth: &ai.OAuthToken{
		Access: "access-1", Refresh: "refresh-1", Expires: time.Now().Add(-time.Minute),
	}}
	if _, err := baseAgent.ScopeAuthStore(s.a, owner).Modify(s.ctx, provider, func(*ai.Credential) (*ai.Credential, error) {
		return expired, nil
	}); err != nil {
		t.Fatal(err)
	}

	var calls atomic.Int32
	var mu sync.Mutex
	used := map[string]bool{}
	auth := ai.ProviderAuth{OAuth: &ai.OAuthAuth{
		Refresh: func(ctx context.Context, c *ai.Credential) (*ai.Credential, error) {
			calls.Add(1)
			mu.Lock()
			reused := used[c.OAuth.Refresh]
			used[c.OAuth.Refresh] = true
			mu.Unlock()
			if reused {
				return nil, errors.New("invalid_grant: refresh token reused")
			}
			time.Sleep(20 * time.Millisecond)
			return &ai.Credential{Type: ai.CredentialOAuth, OAuth: &ai.OAuthToken{
				Access: "access-2", Refresh: "refresh-2", Expires: time.Now().Add(time.Hour),
			}}, nil
		},
		ToAuth: func(c *ai.Credential) (ai.ModelAuth, error) {
			access := c.OAuth.Access
			return ai.ModelAuth{APIKey: &access}, nil
		},
	}}

	errs := make(chan error, requests)
	var wg sync.WaitGroup
	for i := range requests {
		store := s.a
		if i%2 == 1 {
			store = s.b
		}
		wg.Go(func() {
			result, err := ai.ResolveProviderAuth(s.ctx, provider, auth, baseAgent.ScopeAuthStore(store, owner), ai.NoAmbientAuth(), nil)
			switch {
			case err != nil:
				errs <- err
			case result == nil || result.Auth.APIKey == nil || *result.Auth.APIKey != "access-2":
				errs <- fmt.Errorf("resolved %+v, want the refreshed token", result)
			}
		})
	}
	done := make(chan struct{})
	go func() { wg.Wait(); close(done) }()
	wait(t, done, "concurrent auth resolution; a lock may be deadlocked")
	close(errs)
	for err := range errs {
		t.Errorf("ResolveProviderAuth error = %v", err)
	}
	if n := calls.Load(); n != 1 {
		t.Errorf("Refresh called %d times, want 1", n)
	}
	stored, err := baseAgent.ScopeAuthStore(s.b, owner).Read(s.ctx, provider)
	if err != nil || stored == nil || stored.OAuth.Refresh != "refresh-2" {
		t.Errorf("stored credential = %+v, %v; want the rotated refresh token", stored, err)
	}
}
