// Package forge provides Go bindings to the Forge Rust runtime C ABI.
package forge

import (
	"encoding/json"
	"errors"
	"fmt"
	"sync"
)

// Status mirrors forge_status from the C ABI.
type Status int32

const (
	StatusOK                Status = 0
	StatusInvalidArgument   Status = 1
	StatusDimensionMismatch Status = 2
	StatusOutOfMemory       Status = 3
	StatusInternal          Status = 4
	StatusBusy              Status = 5
	StatusNotFound          Status = 6
	StatusFinished          Status = 7
)

func (s Status) Error() string {
	switch s {
	case StatusOK:
		return "ok"
	case StatusInvalidArgument:
		return "invalid argument"
	case StatusDimensionMismatch:
		return "dimension mismatch"
	case StatusOutOfMemory:
		return "out of memory"
	case StatusInternal:
		return "internal error"
	case StatusBusy:
		return "busy"
	case StatusNotFound:
		return "not found"
	case StatusFinished:
		return "finished"
	default:
		return fmt.Sprintf("forge status %d", int(s))
	}
}

// Version returns the runtime version string.
func Version() string {
	return nativeVersion()
}

// MatmulF32 computes C = A(m×k) * B(k×n) in row-major FP32.
func MatmulF32(m, n, k int, a, b, c []float32) error {
	if m <= 0 || n <= 0 || k <= 0 {
		return StatusInvalidArgument
	}
	if len(a) < m*k || len(b) < k*n || len(c) < m*n {
		return errors.New("slice too short for given dimensions")
	}
	st := nativeMatmul(m, n, k, a, b, c)
	if st != StatusOK {
		return st
	}
	return nil
}

// MatvecF32 computes y = A(m×n) * x(n) in row-major FP32.
func MatvecF32(m, n int, a, x, y []float32) error {
	if m <= 0 || n <= 0 {
		return StatusInvalidArgument
	}
	if len(a) < m*n || len(x) < n || len(y) < m {
		return errors.New("slice too short for given dimensions")
	}
	st := nativeMatvec(m, n, a, x, y)
	if st != StatusOK {
		return st
	}
	return nil
}

// BatchEvent is one token from a continuous-batching step.
type BatchEvent struct {
	RequestID uint64
	TokenID   uint32
	Text      string
	Done      bool
}

// Runtime owns a loaded model and continuous batcher.
type Runtime struct {
	mu   sync.Mutex
	ptr  uintptr
	path string
}

// DeviceCPU / DeviceCUDA / DeviceCUDABF16 select the execution backend
// (CUDA requires a CUDA-enabled runtime build).
const (
	DeviceCPU      = 0
	DeviceCUDA     = 1 // CUDA FP32 weights
	DeviceCUDABF16 = 2 // CUDA BF16 weights, FP32 activations
)

// Load opens a Hugging Face model directory on CPU.
func Load(modelDir string, maxSeq int) (*Runtime, error) {
	return LoadDevice(modelDir, maxSeq, DeviceCPU)
}

// LoadDevice loads with an explicit device (see DeviceCPU / DeviceCUDA / DeviceCUDABF16).
func LoadDevice(modelDir string, maxSeq int, device int) (*Runtime, error) {
	if maxSeq <= 0 {
		maxSeq = 2048
	}
	ptr, err := nativeLoadEx(modelDir, maxSeq, device)
	if err != nil {
		return nil, err
	}
	return &Runtime{ptr: ptr, path: modelDir}, nil
}

// Close releases the runtime.
func (r *Runtime) Close() {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.ptr != 0 {
		nativeFree(r.ptr)
		r.ptr = 0
	}
}

// GenerateBegin starts a generation job (blocks during prefill).
func (r *Runtime) GenerateBegin(prompt string, maxTokens int, temperature float32) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.ptr == 0 {
		return errors.New("runtime closed")
	}
	milli := int32(temperature * 1000)
	return nativeGenBegin(r.ptr, prompt, maxTokens, milli)
}

// GenerateNext returns the next token piece. done=true when finished.
func (r *Runtime) GenerateNext() (tokenID uint32, piece string, done bool, err error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.ptr == 0 {
		return 0, "", true, errors.New("runtime closed")
	}
	return nativeGenNext(r.ptr)
}

// GenerateCancel aborts the current generation.
func (r *Runtime) GenerateCancel() {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.ptr != 0 {
		nativeGenCancel(r.ptr)
	}
}

// BatchCapacity returns max concurrent sequences.
func (r *Runtime) BatchCapacity() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.ptr == 0 {
		return 0
	}
	return nativeBatchCapacity(r.ptr)
}

// BatchFreeSlots returns available admission slots.
func (r *Runtime) BatchFreeSlots() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.ptr == 0 {
		return 0
	}
	return nativeBatchFreeSlots(r.ptr)
}

// BatchActive returns currently active sequences.
func (r *Runtime) BatchActive() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.ptr == 0 {
		return 0
	}
	return nativeBatchActive(r.ptr)
}

// BatchAdmit prefills a new request into the continuous batcher.
// Returns StatusBusy when no free slot (backpressure).
func (r *Runtime) BatchAdmit(prompt string, maxTokens int, temperature float32) (uint64, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.ptr == 0 {
		return 0, errors.New("runtime closed")
	}
	milli := int32(temperature * 1000)
	return nativeBatchAdmit(r.ptr, prompt, maxTokens, milli)
}

// BatchCancel removes a request without affecting siblings.
func (r *Runtime) BatchCancel(requestID uint64) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.ptr == 0 {
		return errors.New("runtime closed")
	}
	return nativeBatchCancel(r.ptr, requestID)
}

// BatchStep runs one shared continuous-batching decode step.
func (r *Runtime) BatchStep() ([]BatchEvent, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.ptr == 0 {
		return nil, errors.New("runtime closed")
	}
	return nativeBatchStep(r.ptr)
}

// BatchSchedStats is scheduler / KV observability (no pointers).
type BatchSchedStats struct {
	ActiveDecode           uint64
	ActivePrefill          uint64
	LastScheduledDecode    uint64
	LastScheduledPrefill   uint64
	KvNumBlocks            uint64
	KvBlockSize            uint64
	KvFreeBlocks           uint64
	KvAllocatedBlocks      uint64
	PrefillStallNs         uint64
}

// BatchSchedStats returns the latest scheduler/KV counters (zeros on CPU / unsupported).
func (r *Runtime) BatchSchedStats() (BatchSchedStats, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.ptr == 0 {
		return BatchSchedStats{}, errors.New("runtime closed")
	}
	return nativeBatchSchedStats(r.ptr)
}

// Message is a text-only conversation turn.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// PreparedChat describes the exact rendered input and loaded runtime limits.
type PreparedChat struct {
	Prompt       string `json:"prompt"`
	PromptTokens int    `json:"prompt_tokens"`
	ContextSize  int    `json:"context_size"`
	EOSTokenID   uint32 `json:"eos_token_id"`
}

func (r *Runtime) PrepareChat(messages []Message) (PreparedChat, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	var prepared PreparedChat
	if r.ptr == 0 {
		return prepared, errors.New("runtime closed")
	}
	input, err := json.Marshal(messages)
	if err != nil {
		return prepared, err
	}
	output, err := nativeChatPrepare(r.ptr, string(input))
	if err != nil {
		return prepared, err
	}
	err = json.Unmarshal([]byte(output), &prepared)
	return prepared, err
}
