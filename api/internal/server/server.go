// Package server hosts the HTTP API.
package server

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/amw720386/forge/api/internal/metrics"
	"github.com/amw720386/forge/api/internal/queue"
	"github.com/amw720386/forge/api/internal/scheduler"
	"github.com/amw720386/forge/api/internal/streaming"
)

type Config struct {
	Addr          string
	QueueCap      int
	DefaultMaxTok int
}

type Server struct {
	cfg        Config
	q          *queue.Queue
	sched      *scheduler.Scheduler
	metrics    *metrics.Collector
	httpServer *http.Server
	wg         sync.WaitGroup
}

type completionRequest struct {
	Prompt      string  `json:"prompt"`
	MaxTokens   int     `json:"max_tokens"`
	Temperature float32 `json:"temperature"`
	Stream      bool    `json:"stream"`
}

type errorBody struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code"`
	} `json:"error"`
	RequestID string `json:"request_id"`
}

func New(cfg Config, q *queue.Queue, sched *scheduler.Scheduler, m *metrics.Collector) *Server {
	if cfg.DefaultMaxTok <= 0 {
		cfg.DefaultMaxTok = 64
	}
	s := &Server{cfg: cfg, q: q, sched: sched, metrics: m}
	mux := http.NewServeMux()
	mux.HandleFunc("/health", s.handleHealth)
	mux.HandleFunc("/metrics", s.handleMetrics)
	mux.HandleFunc("/v1/completions", s.handleCompletions)
	s.httpServer = &http.Server{
		Addr:              cfg.Addr,
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
	}
	return s
}

func (s *Server) Start() error {
	listener, err := net.Listen("tcp", s.cfg.Addr)
	if err != nil {
		return fmt.Errorf("listen on %s: %w", s.cfg.Addr, err)
	}
	s.sched.Start()
	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		log.Printf("forge-server listening on %s", s.cfg.Addr)
		if err := s.httpServer.Serve(listener); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Printf("server error: %v", err)
		}
	}()
	return nil
}

func (s *Server) Shutdown(ctx context.Context) error {
	s.q.Close()
	s.sched.Stop()
	err := s.httpServer.Shutdown(ctx)
	s.wg.Wait()
	return err
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"status":    "ok",
		"queue_len": s.q.Len(),
		"queue_cap": s.q.Cap(),
	})
}

func (s *Server) handleMetrics(w http.ResponseWriter, r *http.Request) {
	s.metrics.SetQueueLen(s.q.Len())
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write(s.metrics.JSON())
}

func (s *Server) handleCompletions(w http.ResponseWriter, r *http.Request) {
	received := time.Now()
	if r.Method != http.MethodPost {
		s.writeError(w, http.StatusMethodNotAllowed, "", "method_not_allowed", "POST required")
		return
	}
	var body completionRequest
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20))
	if err := decoder.Decode(&body); err != nil {
		s.writeError(w, http.StatusBadRequest, "", "invalid_request", "invalid JSON body")
		return
	}
	if decoder.Decode(new(any)) != io.EOF {
		s.writeError(w, http.StatusBadRequest, "", "invalid_request", "body must contain one JSON object")
		return
	}
	if strings.TrimSpace(body.Prompt) == "" || strings.ContainsRune(body.Prompt, 0) {
		s.writeError(w, http.StatusBadRequest, "", "invalid_request", "prompt is required")
		return
	}
	if body.MaxTokens < 0 || body.Temperature < 0 {
		s.writeError(w, http.StatusBadRequest, "", "invalid_request", "max_tokens and temperature must be nonnegative")
		return
	}
	if body.MaxTokens == 0 {
		body.MaxTokens = s.cfg.DefaultMaxTok
	}
	if body.MaxTokens > 2048 {
		body.MaxTokens = 2048
	}

	s.metrics.ObserveStage("http_parse", time.Since(received))
	defer func() { s.metrics.ObserveStage("http_completion", time.Since(received)) }()
	id := newRequestID()
	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()
	req := &queue.Request{
		ReceivedAt:  received,
		ID:          id,
		Prompt:      formatPrompt(body.Prompt),
		MaxTokens:   body.MaxTokens,
		Temperature: body.Temperature,
		Stream:      body.Stream,
		Ctx:         ctx,
		Cancel:      cancel,
		Tokens:      make(chan queue.TokenEvent, 64),
	}
	s.metrics.IncTotal()

	enqueueStart := time.Now()
	if err := s.q.TryEnqueue(req); err != nil {
		cancel()
		if errors.Is(err, queue.ErrFull) {
			s.writeError(w, http.StatusServiceUnavailable, id, "queue_full", "admission queue is full")
			return
		}
		s.writeError(w, http.StatusInternalServerError, id, "queue_error", err.Error())
		return
	}

	s.metrics.ObserveStage("queue_admission", time.Since(enqueueStart))
	if !body.Stream {
		s.handleNonStream(w, req)
		return
	}
	s.handleStream(w, req)
}

func (s *Server) handleStream(w http.ResponseWriter, req *queue.Request) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		s.writeError(w, http.StatusInternalServerError, req.ID, "internal", "streaming unsupported")
		req.Cancel()
		return
	}
	streaming.WriteHeaders(w)
	w.Header().Set("X-Request-ID", req.ID)
	flusher.Flush()

	first := true
	for {
		select {
		case <-req.Ctx.Done():
			req.Cancel()
			streaming.WriteError(w, flusher, "client disconnected")
			return
		case ev, ok := <-req.Tokens:
			if !ok {
				if req.Err != nil {
					streaming.WriteError(w, flusher, req.Err.Error())
				} else {
					_ = streaming.WriteDone(w, flusher)
				}
				return
			}
			writeStart := time.Now()
			if first {
				s.metrics.ObserveStage("server_ttft", time.Since(req.ReceivedAt))
				first = false
			}
			if ev.Done {
				// A terminal event carries the final sampled token too.
				if err := streaming.WriteToken(w, flusher, req.ID, ev.Text); err != nil {
					req.Cancel()
					return
				}
				_ = streaming.WriteDone(w, flusher)
				s.metrics.ObserveStage("sse_write_flush", time.Since(writeStart))
				return
			}
			if err := streaming.WriteToken(w, flusher, req.ID, ev.Text); err != nil {
				req.Cancel()
				return
			}
			s.metrics.ObserveStage("sse_write_flush", time.Since(writeStart))
		}
	}
}

func (s *Server) handleNonStream(w http.ResponseWriter, req *queue.Request) {
	var text string
reading:
	for {
		select {
		case <-req.Ctx.Done():
			return
		case ev, ok := <-req.Tokens:
			if !ok {
				break reading
			}
			text += ev.Text
			// Wait for channel closure: scheduler owns completion/error state.
		}
	}
	if req.Err != nil {
		code := http.StatusInternalServerError
		if strings.Contains(req.Err.Error(), "formatted prompt exceeds context limit") {
			code = http.StatusBadRequest
		}
		s.writeError(w, code, req.ID, "generation_failed", req.Err.Error())
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Request-ID", req.ID)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"id":     req.ID,
		"object": "text_completion",
		"choices": []map[string]any{{
			"index": 0,
			"text":  text,
		}},
	})
}

func (s *Server) writeError(w http.ResponseWriter, code int, id, typ, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	var body errorBody
	body.Error.Message = msg
	body.Error.Type = typ
	body.Error.Code = fmt.Sprintf("%d", code)
	body.RequestID = id
	_ = json.NewEncoder(w).Encode(body)
}

func newRequestID() string {
	var b [8]byte
	_, _ = rand.Read(b[:])
	return "req_" + hex.EncodeToString(b[:])
}

// formatPrompt wraps plain user text in the Qwen chat template when needed.
func formatPrompt(prompt string) string {
	if containsIMStart(prompt) {
		return prompt
	}
	return "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n" +
		"<|im_start|>user\n" + prompt + "<|im_end|>\n" +
		"<|im_start|>assistant\n"
}

func containsIMStart(s string) bool {
	return strings.Contains(s, "<|im_start|>")
}
