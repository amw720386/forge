package server

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestChatValidation(t *testing.T) {
	base := map[string]any{"model": "qwen", "messages": []any{map[string]any{"role": "user", "content": ""}}}
	check := func(body map[string]any, valid bool) {
		t.Helper()
		data, _ := json.Marshal(body)
		_, _, err := parseChat(data, "qwen", 64)
		if (err == nil) != valid {
			t.Fatalf("body %s: %v", data, err)
		}
	}
	check(base, true)
	for key, value := range map[string]any{"temperature": 0.7, "top_p": 0.8, "top_k": 40, "n": 2, "max_tokens": 0, "tools": []any{}, "logprobs": true, "stop": "x", "stream_options": map[string]any{"include_usage": true}, "model": "other", "stream": nil} {
		body := make(map[string]any)
		for k, v := range base {
			body[k] = v
		}
		body[key] = value
		check(body, false)
	}
	for _, messages := range []any{
		[]any{}, []any{map[string]any{"role": "tool", "content": "hi"}},
		[]any{map[string]any{"role": "user", "content": nil}},
		[]any{map[string]any{"role": "user", "content": []any{}}},
		[]any{map[string]any{"role": "user", "content": "hi", "name": "extra"}},
		[]any{map[string]any{"role": "user", "content": "hi"}, map[string]any{"role": "assistant", "content": "hi"}},
		[]any{map[string]any{"role": "system", "content": "hi"}},
		[]any{map[string]any{"role": "user", "content": "hi"}, map[string]any{"role": "system", "content": "hi"}},
	} {
		check(map[string]any{"model": "qwen", "messages": messages}, false)
	}
	for _, data := range []string{`null`, `[]`, `{`, `{} {}`, "\xff"} {
		if _, _, err := parseChat([]byte(data), "qwen", 64); err == nil {
			t.Fatal("accepted", data)
		}
	}
	check(map[string]any{"model": "qwen", "temperature": 0, "top_p": 1, "top_k": 1, "n": 1, "stream": true, "stream_options": map[string]any{"include_usage": true}, "messages": []any{
		map[string]any{"role": "system", "content": "help"}, map[string]any{"role": "user", "content": "hello"}, map[string]any{"role": "assistant", "content": "hi"}, map[string]any{"role": "user", "content": "日本語 😀"},
	}}, true)
}

func TestChatCompatibleParameters(t *testing.T) {
	for _, tc := range []struct {
		name, fields, errorText string
		limit                   int
	}{
		{"default", ``, "", 64},
		{"legacy limit", `,"max_tokens":7`, "", 7},
		{"new limit", `,"max_completion_tokens":9`, "", 9},
		{"equal limits", `,"max_tokens":8,"max_completion_tokens":8`, "", 8},
		{"conflict", `,"max_tokens":8,"max_completion_tokens":9`, "must match", 0},
		{"neutral", `,"frequency_penalty":0,"presence_penalty":0,"repetition_penalty":1,"response_format":{"type":"text"}`, "", 64},
		{"frequency", `,"frequency_penalty":0.1`, "frequency_penalty=0", 0},
		{"presence", `,"presence_penalty":-1`, "presence_penalty=0", 0},
		{"repetition", `,"repetition_penalty":0`, "repetition_penalty=1", 0},
		{"json", `,"response_format":{"type":"json_object"}`, "type=text", 0},
		{"schema", `,"response_format":{"type":"json_schema"}`, "type=text", 0},
		{"missing format", `,"response_format":{}`, "type=text", 0},
		{"null format type", `,"response_format":{"type":null}`, "type=text", 0},
		{"extra format field", `,"response_format":{"type":"text","extra":true}`, "unknown field", 0},
		{"unknown", `,"seed":1`, "unknown field", 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			data := []byte(`{"model":"qwen","messages":[{"role":"user","content":"hello"}]` + tc.fields + `}`)
			body, _, err := parseChat(data, "qwen", 64)
			if tc.errorText != "" {
				if err == nil || !strings.Contains(err.Error(), tc.errorText) {
					t.Fatalf("expected %q, got %v", tc.errorText, err)
				}
			} else if err != nil || *body.MaxTokens != tc.limit {
				t.Fatalf("limit %d: body=%+v error=%v", tc.limit, body, err)
			}
		})
	}
	for _, key := range []string{"max_tokens", "max_completion_tokens"} {
		for _, value := range []string{"0", "-1", "2049", "1.5", `"8"`, "null"} {
			data := []byte(`{"model":"qwen","messages":[{"role":"user","content":"hi"}],"` + key + `":` + value + `}`)
			if _, _, err := parseChat(data, "qwen", 64); err == nil {
				t.Fatalf("accepted %s=%s", key, value)
			}
		}
	}
	for _, key := range []string{"frequency_penalty", "presence_penalty", "repetition_penalty", "response_format"} {
		data := []byte(`{"model":"qwen","messages":[{"role":"user","content":"hi"}],"` + key + `":null}`)
		if _, _, err := parseChat(data, "qwen", 64); err == nil {
			t.Fatalf("accepted null %s", key)
		}
	}
}
