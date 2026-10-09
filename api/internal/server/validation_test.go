package server

import (
	"context"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/amw720386/forge/api/internal/queue"
)

func TestRejectMalformedRequests(t *testing.T) {
	s := &Server{}
	for _, body := range []string{`{`, `{"prompt":"hi"} {}`, `null`, `{"prompt":" "}`, `{"prompt":"hi","max_tokens":-1}`, `{"prompt":"hi","temperature":-1}`, `{"prompt":"\u0000"}`} {
		w := httptest.NewRecorder()
		s.handleCompletions(w, httptest.NewRequest("POST", "/v1/completions", strings.NewReader(body)))
		if w.Code != 400 {
			t.Fatalf("body=%s status=%d", body, w.Code)
		}
	}
}

func TestNonStreamCancellationDoesNotWaitForScheduler(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	req := &queue.Request{Ctx: ctx, Tokens: make(chan queue.TokenEvent)}
	done := make(chan struct{})
	go func() { (&Server{}).handleNonStream(httptest.NewRecorder(), req); close(done) }()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("cancelled HTTP request still waiting")
	}
}
