package baseAgent

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/ColeBurch/burrow/ai"
)

// CredentialNamespace is the AuthStore namespace of provider credentials.
// Names in it are provider IDs, and values are encoded with MarshalCredential.
const CredentialNamespace = "provider"

// credentialFormatVersion is the "v" of encoded credentials. Bump it for any
// change an older binary cannot read without losing data.
const credentialFormatVersion = 1

// credentialJSON is the stored form of an ai.Credential.
type credentialJSON struct {
	V    int               `json:"v"`
	Type ai.CredentialType `json:"type"`

	Key string            `json:"key,omitempty"`
	Env map[string]string `json:"env,omitempty"`

	Access  string `json:"access,omitempty"`
	Refresh string `json:"refresh,omitempty"`
	// Expires is Unix milliseconds, as in pi. It is a pointer so that a zero
	// time stays zero instead of becoming 1970.
	Expires *int64                     `json:"expires,omitempty"`
	Extra   map[string]json.RawMessage `json:"extra,omitempty"`
}

// MarshalCredential encodes a credential for storage. It rejects credentials
// that UnmarshalCredential could not decode back without losing data. Expires
// is kept to the millisecond and comes back in UTC, Extra values come back
// compacted, and empty maps come back nil.
func MarshalCredential(c *ai.Credential) ([]byte, error) {
	if c == nil {
		return nil, errors.New("credential is nil")
	}
	stored := credentialJSON{V: credentialFormatVersion, Type: c.Type}
	switch c.Type {
	case ai.CredentialAPIKey:
		if c.OAuth != nil {
			return nil, errors.New("api_key credential has an OAuth token")
		}
		stored.Key = c.Key
		stored.Env = c.Env
	case ai.CredentialOAuth:
		if c.OAuth == nil {
			return nil, errors.New("oauth credential has no OAuth token")
		}
		if c.Key != "" || len(c.Env) > 0 {
			return nil, errors.New("oauth credential has a key or Env, which only api_key credentials store")
		}
		stored.Access = c.OAuth.Access
		stored.Refresh = c.OAuth.Refresh
		if !c.OAuth.Expires.IsZero() {
			expires := c.OAuth.Expires.UnixMilli()
			stored.Expires = &expires
		}
		stored.Extra = c.OAuth.Extra
	default:
		return nil, fmt.Errorf("unknown credential type %q", c.Type)
	}
	data, err := json.Marshal(stored)
	if err != nil {
		return nil, fmt.Errorf("encode %s credential: %w", c.Type, err)
	}
	return data, nil
}

// UnmarshalCredential decodes a credential encoded by MarshalCredential. Nil
// or empty data means nothing is stored and returns nil, nil. Unknown
// versions, types and fields are errors, so an older binary never decodes part
// of a newer credential and writes the rest back lost.
func UnmarshalCredential(data []byte) (*ai.Credential, error) {
	if len(data) == 0 {
		return nil, nil
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	var stored credentialJSON
	if err := decoder.Decode(&stored); err != nil {
		return nil, fmt.Errorf("decode credential: %w", err)
	}
	if decoder.More() {
		return nil, errors.New("decode credential: data after the credential")
	}
	if stored.V != credentialFormatVersion {
		return nil, fmt.Errorf("decode credential: unsupported version %d", stored.V)
	}

	switch stored.Type {
	case ai.CredentialAPIKey:
		if stored.Access != "" || stored.Refresh != "" || stored.Expires != nil || stored.Extra != nil {
			return nil, errors.New("decode api_key credential: has OAuth fields")
		}
		return &ai.Credential{Type: ai.CredentialAPIKey, Key: stored.Key, Env: stored.Env}, nil
	case ai.CredentialOAuth:
		if stored.Key != "" || stored.Env != nil {
			return nil, errors.New("decode oauth credential: has api_key fields")
		}
		token := &ai.OAuthToken{Access: stored.Access, Refresh: stored.Refresh, Extra: stored.Extra}
		if stored.Expires != nil {
			token.Expires = time.UnixMilli(*stored.Expires).UTC()
		}
		return &ai.Credential{Type: ai.CredentialOAuth, OAuth: token}, nil
	default:
		return nil, fmt.Errorf("decode credential: unknown type %q", stored.Type)
	}
}
