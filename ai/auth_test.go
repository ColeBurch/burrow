package ai

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"
)

func TestCredentialClone(t *testing.T) {
	original := &Credential{
		Type: CredentialOAuth,
		Env:  map[string]string{"ACCOUNT": "a"},
		OAuth: &OAuthToken{
			Access:  "access",
			Refresh: "refresh",
			Expires: time.Unix(1700000000, 0),
			Extra:   map[string]json.RawMessage{"accountId": json.RawMessage(`"acct"`)},
		},
	}

	cloned := original.Clone()
	if !reflect.DeepEqual(cloned, original) {
		t.Fatalf("clone = %+v, want %+v", cloned, original)
	}

	cloned.Env["ACCOUNT"] = "changed"
	cloned.OAuth.Access = "changed"
	cloned.OAuth.Extra["accountId"][1] = 'X'
	cloned.OAuth.Extra["new"] = json.RawMessage(`1`)

	if original.Env["ACCOUNT"] != "a" {
		t.Error("clone shares Env")
	}
	if original.OAuth.Access != "access" {
		t.Error("clone shares OAuth")
	}
	if string(original.OAuth.Extra["accountId"]) != `"acct"` {
		t.Error("clone shares Extra values")
	}
	if _, ok := original.OAuth.Extra["new"]; ok {
		t.Error("clone shares the Extra map")
	}
}

func TestCredentialCloneNil(t *testing.T) {
	var c *Credential
	if c.Clone() != nil {
		t.Fatal("Clone of nil is not nil")
	}

	apiKey := &Credential{Type: CredentialAPIKey, Key: "k"}
	cloned := apiKey.Clone()
	if cloned.Env != nil || cloned.OAuth != nil || cloned.Key != "k" {
		t.Fatalf("clone = %+v, want nil Env and OAuth kept nil", cloned)
	}
}

func TestDefaultAuthContextEnv(t *testing.T) {
	t.Setenv("BURROW_TEST_SET", "value")
	t.Setenv("BURROW_TEST_BLANK", "  ")

	env := DefaultAuthContext()
	if value, ok := env.Env("BURROW_TEST_SET"); !ok || value != "value" {
		t.Errorf("Env(set) = %q, %v", value, ok)
	}
	if value, ok := env.Env("BURROW_TEST_BLANK"); ok || value != "" {
		t.Errorf("Env(blank) = %q, %v; want unset", value, ok)
	}
	if _, ok := env.Env("BURROW_TEST_UNSET"); ok {
		t.Error("Env(unset) reported set")
	}
}

func TestDefaultAuthContextFileExists(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	if err := os.MkdirAll(filepath.Join(home, ".config"), 0o700); err != nil {
		t.Fatal(err)
	}
	file := filepath.Join(home, ".config", "creds.json")
	if err := os.WriteFile(file, []byte("{}"), 0o600); err != nil {
		t.Fatal(err)
	}

	env := DefaultAuthContext()
	for path, want := range map[string]bool{
		file:                           true,
		"~/.config/creds.json":         true,
		"~":                            true,
		"~/.config/missing.json":       false,
		filepath.Join(home, "missing"): false,
	} {
		if got := env.FileExists(path); got != want {
			t.Errorf("FileExists(%q) = %v, want %v", path, got, want)
		}
	}
}

func TestNoAmbientAuth(t *testing.T) {
	t.Setenv("BURROW_TEST_SET", "value")
	env := NoAmbientAuth()
	if _, ok := env.Env("BURROW_TEST_SET"); ok {
		t.Error("NoAmbientAuth read the environment")
	}
	if env.FileExists(os.TempDir()) {
		t.Error("NoAmbientAuth reported a file")
	}
}
