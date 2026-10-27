package mcp

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"

	sdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

// stdioServerEnv makes the test binary act as a stdio server: "crash" exits
// before answering, "serve" serves a tool "die" that exits.
const stdioServerEnv = "BURROW_MCP_TEST_STDIO_SERVER"

func TestMain(m *testing.M) {
	switch os.Getenv(stdioServerEnv) {
	case "crash":
		fmt.Fprint(os.Stderr, "early output\n"+strings.Repeat("x", 3000)+"\nmissing API key\n\n")
		os.Exit(1)
	case "serve":
		server := sdk.NewServer(&sdk.Implementation{Name: "stdio", Version: "1.0.0"}, nil)
		server.AddTool(&sdk.Tool{Name: "die", InputSchema: objectSchema},
			func(context.Context, *sdk.CallToolRequest) (*sdk.CallToolResult, error) {
				fmt.Fprintln(os.Stderr, "out of memory")
				os.Exit(3)
				return nil, nil
			})
		fmt.Fprintln(os.Stderr, "serving")
		_ = server.Run(context.Background(), &sdk.StdioTransport{})
		os.Exit(0)
	}
	os.Exit(m.Run())
}

// waitForStderr makes the transport wait for stderr to end, so a test fails
// if it does not.
func waitForStderr(t *testing.T) {
	saved := stderrDrainTimeout
	stderrDrainTimeout = time.Minute
	t.Cleanup(func() { stderrDrainTimeout = saved })
}

func stdioServer(mode string, stderr *bytes.Buffer) func() *exec.Cmd {
	return func() *exec.Cmd {
		cmd := exec.Command(os.Args[0])
		cmd.Env = append(os.Environ(), stdioServerEnv+"="+mode)
		if stderr != nil {
			cmd.Stderr = stderr
		}
		return cmd
	}
}

// A stdio server that fails to connect shows the end of its stderr, as in
// pi, and a Stderr the command sets gets all of it.
func TestStdioServerStderrOnConnectFailure(t *testing.T) {
	waitForStderr(t)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	var stderr bytes.Buffer
	connection, _ := newConnection(t, nil, ServerConnectionOptions{Name: "local", CreateTransport: StdioTransport(stdioServer("crash", &stderr))})
	_, err := connection.GetClient(ctx)
	if err == nil || !strings.Contains(err.Error(), `MCP server "local" failed to connect`) {
		t.Fatalf("GetClient: %v", err)
	}
	status := connection.Status()
	if status.State != StateFailed || !strings.HasSuffix(err.Error(), status.Error) {
		t.Fatalf("status %+v, error %v", status, err)
	}
	_, tail, _ := strings.Cut(status.Error, "\n")
	if len(tail) != stderrTailChars || !strings.HasSuffix(tail, "x\nmissing API key") {
		t.Fatalf("stderr tail of %d characters: ...%q", len(tail), tail[max(0, len(tail)-40):])
	}
	if !strings.HasPrefix(stderr.String(), "early output\n") {
		t.Fatalf("the command's Stderr got %q", stderr.String()[:min(40, stderr.Len())])
	}
}

// A stdio server that exits leaves its stderr in the status, and the next
// call starts it again.
func TestStdioServerStderrWhenItExits(t *testing.T) {
	waitForStderr(t)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	connection, _ := newConnection(t, nil, ServerConnectionOptions{Name: "local", CreateTransport: StdioTransport(stdioServer("serve", nil))})
	if _, err := connection.CallTool(ctx, "die", nil, nil); err == nil {
		t.Fatal("the server answered after exiting")
	}
	eventually(t, "the disconnect", func() bool { return connection.Status().State == StateDisconnected })
	if got := connection.Status().Error; got != "Connection closed\nserving\nout of memory" {
		t.Fatalf("status error = %q", got)
	}
	if _, err := connection.GetClient(ctx); err != nil || connection.Status().State != StateConnected {
		t.Fatalf("reconnect: %v, %+v", err, connection.Status())
	}
}

func TestStdioTransportWithoutStderr(t *testing.T) {
	transport := &stdioTransport{done: make(chan struct{})}
	close(transport.done)
	if tail := transport.stderrTail(); tail != "" {
		t.Fatalf("tail = %q", tail)
	}
	transport.stderr = []byte("  \n" + strings.Repeat("é", stderrTailChars+5) + "\n")
	if tail := transport.stderrTail(); tail != strings.Repeat("é", stderrTailChars) {
		t.Fatalf("tail of %d bytes", len(tail))
	}
}
