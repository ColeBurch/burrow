package baseAgent

import (
	"encoding/json"
	"reflect"
	"testing"
	"time"

	"github.com/ColeBurch/burrow/ai"
)

// assertCredential compares credentials, with Expires compared as instants.
func assertCredential(t *testing.T, got, want *ai.Credential) {
	t.Helper()
	if got == nil || want == nil {
		if got != want {
			t.Fatalf("credential = %+v, want %+v", got, want)
		}
		return
	}
	got, want = got.Clone(), want.Clone()
	if got.OAuth != nil && want.OAuth != nil {
		if !got.OAuth.Expires.Equal(want.OAuth.Expires) {
			t.Fatalf("Expires = %v, want %v", got.OAuth.Expires, want.OAuth.Expires)
		}
		got.OAuth.Expires, want.OAuth.Expires = time.Time{}, time.Time{}
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("credential = %+v, want %+v", got, want)
	}
}

// The golden bytes pin the stored format: stores hold credentials written by
// older builds, so a change here needs a new format version.
func TestCredentialCodecGolden(t *testing.T) {
	tests := []struct {
		name   string
		cred   *ai.Credential
		golden string
	}{
		{
			"api_key",
			&ai.Credential{Type: ai.CredentialAPIKey, Key: "sk-test", Env: map[string]string{"ACCOUNT": "a"}},
			`{"v":1,"type":"api_key","key":"sk-test","env":{"ACCOUNT":"a"}}`,
		},
		{
			"oauth",
			&ai.Credential{Type: ai.CredentialOAuth, OAuth: &ai.OAuthToken{
				Access:  "acc",
				Refresh: "ref",
				Expires: time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC),
				Extra:   map[string]json.RawMessage{"accountId": json.RawMessage(`"acct_1"`)},
			}},
			`{"v":1,"type":"oauth","access":"acc","refresh":"ref","expires":1767225600000,"extra":{"accountId":"acct_1"}}`,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			data, err := MarshalCredential(tc.cred)
			if err != nil {
				t.Fatal(err)
			}
			if string(data) != tc.golden {
				t.Fatalf("encoded\n%s\nwant\n%s", data, tc.golden)
			}
			decoded, err := UnmarshalCredential([]byte(tc.golden))
			if err != nil {
				t.Fatal(err)
			}
			assertCredential(t, decoded, tc.cred)
		})
	}
}

func TestCredentialCodecRoundTrip(t *testing.T) {
	est := time.FixedZone("EST", -5*60*60)
	tests := []struct {
		name string
		in   *ai.Credential
		want *ai.Credential
	}{
		{
			"api key with Env only",
			&ai.Credential{Type: ai.CredentialAPIKey, Env: map[string]string{"ACCOUNT": "a"}},
			&ai.Credential{Type: ai.CredentialAPIKey, Env: map[string]string{"ACCOUNT": "a"}},
		},
		{
			"empty maps come back nil",
			&ai.Credential{Type: ai.CredentialAPIKey, Key: "k", Env: map[string]string{}},
			&ai.Credential{Type: ai.CredentialAPIKey, Key: "k"},
		},
		{
			"Expires to the millisecond",
			&ai.Credential{Type: ai.CredentialOAuth, OAuth: &ai.OAuthToken{Access: "a", Expires: time.Date(2026, 3, 4, 5, 6, 7, 891234567, est)}},
			&ai.Credential{Type: ai.CredentialOAuth, OAuth: &ai.OAuthToken{Access: "a", Expires: time.Date(2026, 3, 4, 10, 6, 7, 891000000, time.UTC)}},
		},
		{
			"zero Expires stays zero",
			&ai.Credential{Type: ai.CredentialOAuth, OAuth: &ai.OAuthToken{Access: "a"}},
			&ai.Credential{Type: ai.CredentialOAuth, OAuth: &ai.OAuthToken{Access: "a"}},
		},
		{
			"Extra keeps any JSON, compacted",
			&ai.Credential{Type: ai.CredentialOAuth, OAuth: &ai.OAuthToken{Access: "a", Extra: map[string]json.RawMessage{
				"n": json.RawMessage(`42`), "obj": json.RawMessage(`{ "scopes": ["read", "write"] }`), "null": json.RawMessage(`null`),
			}}},
			&ai.Credential{Type: ai.CredentialOAuth, OAuth: &ai.OAuthToken{Access: "a", Extra: map[string]json.RawMessage{
				"n": json.RawMessage(`42`), "obj": json.RawMessage(`{"scopes":["read","write"]}`), "null": json.RawMessage(`null`),
			}}},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			data, err := MarshalCredential(tc.in)
			if err != nil {
				t.Fatal(err)
			}
			decoded, err := UnmarshalCredential(data)
			if err != nil {
				t.Fatal(err)
			}
			assertCredential(t, decoded, tc.want)
			if decoded.OAuth != nil && decoded.OAuth.Expires.Location() != time.UTC {
				t.Errorf("Expires location = %v, want UTC", decoded.OAuth.Expires.Location())
			}
		})
	}
}

func TestUnmarshalCredentialNothingStored(t *testing.T) {
	for _, data := range [][]byte{nil, {}} {
		if c, err := UnmarshalCredential(data); c != nil || err != nil {
			t.Fatalf("UnmarshalCredential(%#v) = %+v, %v; want nil, nil", data, c, err)
		}
	}
}

func TestMarshalCredentialRejectsLossyCredentials(t *testing.T) {
	tests := []struct {
		name string
		cred *ai.Credential
	}{
		{"nil", nil},
		{"unknown type", &ai.Credential{Type: "password", Key: "k"}},
		{"oauth without token", &ai.Credential{Type: ai.CredentialOAuth}},
		{"oauth with key", &ai.Credential{Type: ai.CredentialOAuth, Key: "k", OAuth: &ai.OAuthToken{Access: "a"}}},
		{"oauth with Env", &ai.Credential{Type: ai.CredentialOAuth, Env: map[string]string{"A": "a"}, OAuth: &ai.OAuthToken{Access: "a"}}},
		{"api_key with token", &ai.Credential{Type: ai.CredentialAPIKey, Key: "k", OAuth: &ai.OAuthToken{Access: "a"}}},
		{"invalid Extra JSON", &ai.Credential{Type: ai.CredentialOAuth, OAuth: &ai.OAuthToken{Extra: map[string]json.RawMessage{"bad": json.RawMessage(`{`)}}}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if data, err := MarshalCredential(tc.cred); err == nil {
				t.Fatalf("encoded %s, want an error", data)
			}
		})
	}
}

func TestUnmarshalCredentialRejectsUnknownData(t *testing.T) {
	tests := []struct {
		name string
		data string
	}{
		{"not JSON", `sk-raw-key`},
		{"missing version", `{"type":"api_key","key":"k"}`},
		{"newer version", `{"v":2,"type":"api_key","key":"k"}`},
		{"unknown type", `{"v":1,"type":"password","key":"k"}`},
		{"unknown field", `{"v":1,"type":"api_key","key":"k","scope":"all"}`},
		{"api_key with OAuth fields", `{"v":1,"type":"api_key","key":"k","access":"a"}`},
		{"oauth with api_key fields", `{"v":1,"type":"oauth","access":"a","key":"k"}`},
		{"data after the credential", `{"v":1,"type":"api_key","key":"k"}{}`},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if c, err := UnmarshalCredential([]byte(tc.data)); err == nil {
				t.Fatalf("decoded %+v, want an error", c)
			}
		})
	}
}
