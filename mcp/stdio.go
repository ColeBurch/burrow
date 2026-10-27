// Adapted from pi's packages/mcp/src/transports/stdio.ts and
// packages/coding-agent/src/extensions/mcp/runtime.ts (MIT, see LICENSE).

package mcp

import (
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"

	sdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

const (
	// stderrTailChars bounds the end of a stdio server's stderr that errors
	// show.
	stderrTailChars = 2000
	// maxStderrBytes bounds the stderr a stdio server's transport keeps.
	maxStderrBytes = 64 << 10
)

// stderrDrainTimeout bounds the wait for a server's stderr to end, which a
// child process it started may hold open.
var stderrDrainTimeout = time.Second

// StdioTransport runs a stdio server. command returns the process to start
// for each connection attempt; set its Dir and Env there. The end of what
// the server writes to stderr is added to the error when it fails to connect
// or exits. A Stderr the command sets still receives all of it.
func StdioTransport(command func() *exec.Cmd) TransportFactory {
	return func(*OAuthHandler) (sdk.Transport, error) {
		return &stdioTransport{command: sdk.CommandTransport{Command: command()}, done: make(chan struct{})}, nil
	}
}

// stdioTransport is an sdk.CommandTransport that keeps the end of the
// server's stderr.
type stdioTransport struct {
	command sdk.CommandTransport
	// done is closed when the server's stderr ends.
	done chan struct{}

	mu     sync.Mutex
	stderr []byte
}

func (t *stdioTransport) Connect(ctx context.Context) (sdk.Connection, error) {
	// The server writes to a pipe of its own, rather than one exec copies
	// from until Wait, so the transport knows when stderr has ended.
	r, w, err := os.Pipe()
	if err != nil {
		close(t.done)
		return nil, err
	}
	cmd := t.command.Command
	tee := cmd.Stderr
	cmd.Stderr = w
	conn, err := t.command.Connect(ctx)
	// The server has its own copy of w; closing ours lets r end when the
	// server exits.
	w.Close()
	if err != nil {
		r.Close()
		close(t.done)
		return nil, err
	}
	go t.readStderr(r, tee)
	return conn, nil
}

func (t *stdioTransport) readStderr(r *os.File, tee io.Writer) {
	defer close(t.done)
	defer r.Close()
	buf := make([]byte, 4096)
	for {
		n, err := r.Read(buf)
		if n > 0 {
			t.mu.Lock()
			t.stderr = append(t.stderr, buf[:n]...)
			if len(t.stderr) > maxStderrBytes {
				t.stderr = t.stderr[len(t.stderr)-maxStderrBytes:]
			}
			t.mu.Unlock()
			if tee != nil {
				_, _ = tee.Write(buf[:n])
			}
		}
		if err != nil {
			return
		}
	}
}

// stderrTail returns the end of the server's stderr, once it has ended or
// stderrDrainTimeout has passed.
func (t *stdioTransport) stderrTail() string {
	select {
	case <-t.done:
	case <-time.After(stderrDrainTimeout):
	}
	t.mu.Lock()
	tail := strings.TrimSpace(string(t.stderr))
	t.mu.Unlock()
	if runes := []rune(tail); len(runes) > stderrTailChars {
		tail = string(runes[len(runes)-stderrTailChars:])
	}
	return tail
}

// stderrOf returns the end of the stderr of transport's server, if it is a
// stdio server.
func stderrOf(transport sdk.Transport) string {
	if stdio, ok := transport.(*stdioTransport); ok {
		return stdio.stderrTail()
	}
	return ""
}

// withStderr adds the end of a stdio server's stderr to err, which is often
// only that the connection closed.
func withStderr(transport sdk.Transport, err error) error {
	if tail := stderrOf(transport); tail != "" {
		return fmt.Errorf("%w\n%s", err, tail)
	}
	return err
}
