package server

import (
	"context"
	"github.com/amw720386/forge/api/internal/metrics"
	"github.com/amw720386/forge/api/internal/queue"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestTerminalTokenIsDelivered(t *testing.T) {
	for _, stream := range []bool{false, true} {
		ctx, cancel := context.WithCancel(context.Background())
		req := &queue.Request{ReceivedAt: time.Now(), ID: "test", Ctx: ctx, Cancel: cancel, Tokens: make(chan queue.TokenEvent, 2)}
		req.Tokens <- queue.TokenEvent{Text: "first"}
		req.Tokens <- queue.TokenEvent{Text: "final", Done: true}
		close(req.Tokens)
		w := httptest.NewRecorder()
		s := &Server{metrics: metrics.New()}
		if stream {
			s.handleStream(w, req)
		} else {
			s.handleNonStream(w, req)
		}
		cancel()
		if !strings.Contains(w.Body.String(), "final") {
			t.Fatalf("stream=%v final token dropped: %s", stream, w.Body.String())
		}
		if stream && !strings.Contains(w.Body.String(), "[DONE]") {
			t.Fatal("missing stream termination")
		}
	}
}
