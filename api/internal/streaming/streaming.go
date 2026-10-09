// Package streaming implements SSE token delivery.
package streaming

import (
	"encoding/json"
	"fmt"
	"net/http"
)

type Chunk struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Choices []struct {
		Index int    `json:"index"`
		Text  string `json:"text"`
	} `json:"choices"`
}

func WriteHeaders(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
}

func WriteToken(w http.ResponseWriter, flusher http.Flusher, id, text string) error {
	ch := Chunk{
		ID:     id,
		Object: "text.completion.chunk",
	}
	ch.Choices = []struct {
		Index int    `json:"index"`
		Text  string `json:"text"`
	}{{Index: 0, Text: text}}
	b, err := json.Marshal(ch)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w, "data: %s\n\n", b); err != nil {
		return err
	}
	flusher.Flush()
	return nil
}

func WriteDone(w http.ResponseWriter, flusher http.Flusher) error {
	if _, err := fmt.Fprintf(w, "data: [DONE]\n\n"); err != nil {
		return err
	}
	flusher.Flush()
	return nil
}

func WriteError(w http.ResponseWriter, flusher http.Flusher, msg string) {
	payload, _ := json.Marshal(map[string]string{"error": msg})
	fmt.Fprintf(w, "event: error\ndata: %s\n\n", payload)
	flusher.Flush()
}
