package server

import (
	"encoding/json"
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
