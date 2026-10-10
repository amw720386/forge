// Package queue provides a bounded admission queue for inference requests.
package queue

import (
	"context"
	"errors"
	"sync"
	"time"
)

var (
	ErrFull   = errors.New("admission queue full")
	ErrClosed = errors.New("queue closed")
)

// State is the request lifecycle state.
type State string

const (
	StateQueued    State = "queued"
	StateRunning   State = "running"
	StateStreaming State = "streaming"
	StateCompleted State = "completed"
	StateCancelled State = "cancelled"
	StateFailed    State = "failed"
)

// Request is a queued inference job.
type Request struct {
	PromptTokens     int
	CompletionTokens int
	FinishReason     string
	EOSTokenID       uint32
	Chat             bool
	ReceivedAt       time.Time
	ID               string
	Prompt           string
	MaxTokens        int
	Temperature      float32
	Stream           bool
	State            State
	EnqueuedAt       time.Time
	StartedAt        time.Time
	Ctx              context.Context
	Cancel           context.CancelFunc
	Tokens           chan TokenEvent
	Err              error
}

// TokenEvent is one streamed token.
type TokenEvent struct {
	TokenID uint32
	Text    string
	Done    bool
}

// Queue is a bounded FIFO of *Request.
type Queue struct {
	mu     sync.Mutex
	ch     chan *Request
	closed bool
}

func New(capacity int) *Queue {
	if capacity < 1 {
		capacity = 1
	}
	return &Queue{ch: make(chan *Request, capacity)}
}

func (q *Queue) TryEnqueue(r *Request) error {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.closed {
		return ErrClosed
	}
	r.State = StateQueued
	r.EnqueuedAt = time.Now()
	select {
	case q.ch <- r:
		return nil
	default:
		return ErrFull
	}
}

func (q *Queue) Chan() <-chan *Request {
	return q.ch
}

func (q *Queue) Close() {
	q.mu.Lock()
	defer q.mu.Unlock()
	if !q.closed {
		q.closed = true
		close(q.ch)
	}
}

func (q *Queue) Cap() int {
	return cap(q.ch)
}

func (q *Queue) Len() int {
	return len(q.ch)
}
