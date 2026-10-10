package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/amw720386/forge/api/internal/queue"
	"github.com/amw720386/forge/api/internal/streaming"
	forge "github.com/amw720386/forge/bindings/go"
)

type chatMessage struct {
	Role    string  `json:"role"`
	Content *string `json:"content"`
}
type chatRequest struct {
	Model         string        `json:"model"`
	Messages      []chatMessage `json:"messages"`
	Stream        bool          `json:"stream"`
	MaxTokens     *int          `json:"max_tokens"`
	Temperature   *float64      `json:"temperature"`
	TopP          *float64      `json:"top_p"`
	TopK          *int          `json:"top_k"`
	N             *int          `json:"n"`
	StreamOptions *struct {
		IncludeUsage bool `json:"include_usage"`
	} `json:"stream_options"`
}

func parseChat(data []byte, model string, defaultLimit int) (chatRequest, []forge.Message, error) {
	var body chatRequest
	bad := func(msg string) (chatRequest, []forge.Message, error) { return body, nil, errors.New(msg) }
	if !utf8.Valid(data) {
		return bad("body must be valid UTF-8")
	}
	// Reject explicit nulls rather than treating an unsupported value as omission.
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil || fields == nil {
		return bad("body must be one JSON object")
	}
	for key, value := range fields {
		if bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
			return bad(key + " cannot be null")
		}
	}
	if raw, ok := fields["stream_options"]; ok {
		var options map[string]json.RawMessage
		if err := json.Unmarshal(raw, &options); err != nil || options == nil {
			return bad("stream_options must be an object")
		}
		for key, value := range options {
			if bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
				return bad("stream_options." + key + " cannot be null")
			}
		}
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&body); err != nil {
		return bad("invalid request: " + err.Error())
	}
	if decoder.Decode(new(any)) != io.EOF {
		return bad("body must contain one JSON object")
	}
	if body.Model == "" || body.Model != model {
		return bad("unknown model; use the ID returned by /v1/models")
	}
	if body.MaxTokens == nil {
		body.MaxTokens = &defaultLimit
	}
	if *body.MaxTokens < 1 || *body.MaxTokens > 2048 {
		return bad("max_tokens must be between 1 and 2048")
	}
	if body.Temperature != nil && *body.Temperature != 0 {
		return bad("only greedy generation is supported: temperature must be 0")
	}
	if body.TopP != nil && *body.TopP != 1 {
		return bad("top_p sampling is unsupported; only top_p=1 is accepted")
	}
	if body.TopK != nil && *body.TopK != 1 {
		return bad("top_k sampling is unsupported; only top_k=1 is accepted")
	}
	if body.N != nil && *body.N != 1 {
		return bad("only n=1 is supported")
	}
	if body.StreamOptions != nil && !body.Stream {
		return bad("stream_options requires stream=true")
	}
	if len(body.Messages) == 0 {
		return bad("messages must not be empty")
	}
	messages := make([]forge.Message, 0, len(body.Messages))
	expected := "user"
	for i, m := range body.Messages {
		if m.Content == nil || strings.ContainsRune(*m.Content, 0) {
			return bad("each message must have string content without NUL")
		}
		if i == 0 && m.Role == "system" {
			messages = append(messages, forge.Message{Role: m.Role, Content: *m.Content})
			continue
		}
		if m.Role != expected {
			return bad("messages require an optional initial system message, then alternating user/assistant turns ending with user")
		}
		messages = append(messages, forge.Message{Role: m.Role, Content: *m.Content})
		if expected == "user" {
			expected = "assistant"
		} else {
			expected = "user"
		}
	}
	if expected != "assistant" {
		return bad("conversation must end with a user message")
	}
	return body, messages, nil
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		s.writeError(w, 405, "", "method_not_allowed", "GET required")
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"object": "list", "data": []any{map[string]any{
		"id": s.cfg.ModelID, "object": "model", "created": s.modelCreated, "owned_by": "local",
	}}})
}

func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	received := time.Now()
	if r.Method != http.MethodPost {
		s.writeError(w, 405, "", "method_not_allowed", "POST required")
		return
	}
	data, err := io.ReadAll(http.MaxBytesReader(w, r.Body, 1<<20))
	if err != nil {
		s.writeError(w, 400, "", "invalid_request_error", "body exceeds limit or could not be read")
		return
	}
	body, messages, err := parseChat(data, s.cfg.ModelID, s.cfg.DefaultMaxTok)
	if err != nil {
		s.writeError(w, 400, "", "invalid_request_error", err.Error())
		return
	}
	prepared, err := s.sched.PrepareChat(messages)
	if err != nil {
		s.writeError(w, 400, "", "invalid_request_error", err.Error())
		return
	}
	if prepared.PromptTokens < 1 || prepared.PromptTokens+*body.MaxTokens > prepared.ContextSize {
		s.writeError(w, 400, "", "context_length_exceeded", fmt.Sprintf("formatted input (%d tokens) plus max_tokens (%d) exceeds context (%d)", prepared.PromptTokens, *body.MaxTokens, prepared.ContextSize))
		return
	}
	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()
	req := &queue.Request{ID: "chatcmpl-" + strings.TrimPrefix(newRequestID(), "req_"), ReceivedAt: received,
		Prompt: prepared.Prompt, PromptTokens: prepared.PromptTokens, EOSTokenID: prepared.EOSTokenID,
		MaxTokens: *body.MaxTokens, Chat: true, Stream: body.Stream, Ctx: ctx, Cancel: cancel, Tokens: make(chan queue.TokenEvent, 64)}
	s.metrics.IncTotal()
	if err := s.q.TryEnqueue(req); err != nil {
		code := 500
		if errors.Is(err, queue.ErrFull) {
			code = 503
		}
		s.writeError(w, code, req.ID, "queue_error", err.Error())
		return
	}
	created := received.Unix()
	response := func(object string, choices any) map[string]any {
		result := map[string]any{
			"id": req.ID, "object": object, "created": created, "model": s.cfg.ModelID, "choices": choices,
		}
		if body.StreamOptions != nil && body.StreamOptions.IncludeUsage {
			result["usage"] = nil
		}
		return result
	}
	var flusher http.Flusher
	var writeChunk func(any) bool
	if body.Stream {
		var ok bool
		flusher, ok = w.(http.Flusher)
		if !ok {
			s.writeError(w, 500, req.ID, "internal_error", "streaming unsupported")
			return
		}
		streaming.WriteHeaders(w)
		w.Header().Set("X-Request-ID", req.ID)
		writeChunk = func(value any) bool {
			encoded, err := json.Marshal(value)
			if err != nil {
				return false
			}
			if _, err = fmt.Fprintf(w, "data: %s\n\n", encoded); err != nil {
				return false
			}
			flusher.Flush()
			return true
		}
		if !writeChunk(response("chat.completion.chunk", []any{map[string]any{"index": 0, "delta": map[string]any{"role": "assistant", "content": ""}, "finish_reason": nil}})) {
			return
		}
	}
	var text strings.Builder
reading:
	for {
		select {
		case <-ctx.Done():
			return
		case ev, ok := <-req.Tokens:
			if !ok {
				break reading
			}
			if body.Stream {
				if ev.Text != "" && !writeChunk(response("chat.completion.chunk", []any{map[string]any{"index": 0, "delta": map[string]any{"content": ev.Text}, "finish_reason": nil}})) {
					return
				}
			} else {
				text.WriteString(ev.Text)
			}
		}
	}
	// Closure synchronizes scheduler-owned final accounting and error state.
	if req.Err != nil || req.FinishReason == "" {
		msg := "generation ended without a completion"
		if req.Err != nil {
			msg = req.Err.Error()
		}
		if body.Stream {
			writeChunk(map[string]any{"error": map[string]any{"message": msg, "type": "generation_failed", "code": "500"}})
		} else {
			s.writeError(w, 500, req.ID, "generation_failed", msg)
		}
		return
	}
	usage := map[string]int{"prompt_tokens": req.PromptTokens, "completion_tokens": req.CompletionTokens, "total_tokens": req.PromptTokens + req.CompletionTokens}
	if body.Stream {
		if !writeChunk(response("chat.completion.chunk", []any{map[string]any{"index": 0, "delta": map[string]any{}, "finish_reason": req.FinishReason}})) {
			return
		}
		if body.StreamOptions != nil && body.StreamOptions.IncludeUsage {
			final := response("chat.completion.chunk", []any{})
			final["usage"] = usage
			if !writeChunk(final) {
				return
			}
		}
		_ = streaming.WriteDone(w, flusher)
		return
	}
	result := response("chat.completion", []any{map[string]any{"index": 0, "message": map[string]any{"role": "assistant", "content": text.String()}, "finish_reason": req.FinishReason}})
	result["usage"] = usage
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Request-ID", req.ID)
	_ = json.NewEncoder(w).Encode(result)
}
