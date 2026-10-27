package mcp

import (
	"context"
	"io"
	"net/http"
	"net/url"
	"strings"
	"testing"
	"time"
)

func listenCallback(t *testing.T, opts OAuthCallbackServerOptions) *OAuthCallbackServer {
	t.Helper()
	server, err := ListenOAuthCallback(opts)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { server.Close() })
	return server
}

// visit opens the redirect URL with path and query, as the browser does.
func visit(t *testing.T, server *OAuthCallbackServer, path string, query url.Values) (int, string) {
	t.Helper()
	u, err := url.Parse(server.RedirectURL())
	if err != nil {
		t.Fatal(err)
	}
	u.Path, u.RawQuery = path, query.Encode()
	resp, err := http.Get(u.String())
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.Header.Get("Cache-Control") != "no-store" {
		t.Errorf("Cache-Control = %q", resp.Header.Get("Cache-Control"))
	}
	return resp.StatusCode, string(body)
}

func TestOAuthCallbackServer(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	server := listenCallback(t, OAuthCallbackServerOptions{RedirectHost: "localhost", ExtraPaths: []string{"/callback/abc"}})
	if u, _ := url.Parse(server.RedirectURL()); u.Hostname() != "localhost" || u.Port() == "" || u.Path != "/callback" {
		t.Fatalf("redirect URL = %q", server.RedirectURL())
	}

	if status, _ := visit(t, server, "/other", url.Values{"state": {"s"}}); status != http.StatusNotFound {
		t.Fatalf("other path: %d", status)
	}
	if status, body := visit(t, server, "/callback", url.Values{"state": {"unknown"}, "code": {"c"}}); status != http.StatusBadRequest || !strings.Contains(body, "Invalid or expired") {
		t.Fatalf("unknown state: %d %q", status, body)
	}

	wait, err := server.WaitForCallback("s1", "/callback")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := server.WaitForCallback("s1", "/callback"); err == nil {
		t.Fatal("waited twice for one state")
	}
	if status, body := visit(t, server, "/callback", url.Values{"state": {"s1"}, "code": {"c1"}, "iss": {"https://as"}}); status != http.StatusOK || !strings.Contains(body, "Authorization complete") {
		t.Fatalf("callback: %d %q", status, body)
	}
	callback, err := wait(ctx)
	if err != nil || *callback != (OAuthCallback{Code: "c1", State: "s1", Iss: "https://as"}) {
		t.Fatalf("callback = %+v, %v", callback, err)
	}
	// A state is answered once.
	if status, _ := visit(t, server, "/callback", url.Values{"state": {"s1"}, "code": {"c1"}}); status != http.StatusBadRequest {
		t.Fatalf("replayed callback: %d", status)
	}

	for _, tc := range []struct {
		name, path string
		query      url.Values
		status     int
		err        string
	}{
		{"error", "/callback", url.Values{"error": {"access_denied"}, "error_description": {"The user said no"}}, http.StatusOK, "The user said no"},
		{"error code", "/callback", url.Values{"error": {"access_denied"}}, http.StatusOK, "access_denied"},
		{"no code", "/callback", url.Values{}, http.StatusBadRequest, "did not include an authorization code"},
		// The response belongs on the server-specific path.
		{"wrong path", "/callback/abc", url.Values{"code": {"c"}}, http.StatusBadRequest, "another redirect URI"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			wait, err := server.WaitForCallback(tc.name, "/callback")
			if err != nil {
				t.Fatal(err)
			}
			tc.query.Set("state", tc.name)
			if status, _ := visit(t, server, tc.path, tc.query); status != tc.status {
				t.Fatalf("status %d, want %d", status, tc.status)
			}
			if _, err := wait(ctx); err == nil || !strings.Contains(err.Error(), tc.err) {
				t.Fatalf("err = %v, want %q", err, tc.err)
			}
		})
	}

	// The extra path receives responses expected there.
	wait, err = server.WaitForCallback("s2", "/callback/abc")
	if err != nil {
		t.Fatal(err)
	}
	visit(t, server, "/callback/abc", url.Values{"state": {"s2"}, "code": {"c2"}})
	if callback, err := wait(ctx); err != nil || callback.Code != "c2" {
		t.Fatalf("extra path: %+v, %v", callback, err)
	}

	// A caller that stops waiting frees the state.
	wait, err = server.WaitForCallback("s3", "")
	if err != nil {
		t.Fatal(err)
	}
	stopped, stop := context.WithCancel(ctx)
	stop()
	if _, err := wait(stopped); err != context.Canceled {
		t.Fatalf("cancelled wait: %v", err)
	}
	if status, _ := visit(t, server, "/callback", url.Values{"state": {"s3"}, "code": {"c"}}); status != http.StatusBadRequest {
		t.Fatalf("callback after the wait stopped: %d", status)
	}
}

func TestOAuthCallbackServerTimeoutAndClose(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	server := listenCallback(t, OAuthCallbackServerOptions{Timeout: 20 * time.Millisecond, RenderPage: func(page OAuthCallbackPage) string {
		return "<p>" + page.Message + "</p>"
	}})
	wait, err := server.WaitForCallback("late", "")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := wait(ctx); err == nil || !strings.Contains(err.Error(), "timed out") {
		t.Fatalf("err = %v", err)
	}
	if _, body := visit(t, server, "/callback", url.Values{"state": {"late"}}); body != "<p>Invalid or expired OAuth state</p>" {
		t.Fatalf("rendered page = %q", body)
	}

	wait, err = server.WaitForCallback("open", "")
	if err != nil {
		t.Fatal(err)
	}
	if err := server.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := wait(ctx); err == nil || !strings.Contains(err.Error(), "closed") {
		t.Fatalf("wait after Close: %v", err)
	}
	if _, err := server.WaitForCallback("after", ""); err == nil {
		t.Fatal("waited on a closed server")
	}
	if _, err := http.Get(server.RedirectURL()); err == nil {
		t.Fatal("closed server still serves")
	}
}
