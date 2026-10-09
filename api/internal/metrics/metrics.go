// Package metrics collects request lifecycle stats.
package metrics

import (
	"encoding/json"
	"sync"
	"sync/atomic"
	"time"
)

type StageTiming struct {
	Count   uint64 `json:"count"`
	TotalNs uint64 `json:"total_ns"`
	MaxNs   uint64 `json:"max_ns"`
}

type Snapshot struct {
	Stages            map[string]StageTiming `json:"stages"`
	RequestsTotal     uint64                 `json:"requests_total"`
	RequestsCompleted uint64                 `json:"requests_completed"`
	RequestsFailed    uint64                 `json:"requests_failed"`
	RequestsCancelled uint64                 `json:"requests_cancelled"`
	TokensGenerated   uint64                 `json:"tokens_generated"`
	QueueLen          int                    `json:"queue_len"`
	Busy              bool                   `json:"inference_busy"`
	AvgQueueWaitMs    float64                `json:"avg_queue_wait_ms"`
	AvgGenMs          float64                `json:"avg_generation_ms"`
	// Phase 10 scheduler / KV (optional; zero when not reported).
	ActiveDecode           uint64 `json:"active_decode"`
	ActivePrefill          uint64 `json:"active_prefill"`
	ScheduledDecodeTokens  uint64 `json:"scheduled_decode_tokens"`
	ScheduledPrefillTokens uint64 `json:"scheduled_prefill_tokens"`
	KvNumBlocks            uint64 `json:"kv_num_blocks"`
	KvBlockSize            uint64 `json:"kv_block_size"`
	KvFreeBlocks           uint64 `json:"kv_free_blocks"`
	KvAllocatedBlocks      uint64 `json:"kv_allocated_blocks"`
	PrefillStallNs         uint64 `json:"prefill_stall_ns"`
}

type Collector struct {
	requestsTotal     atomic.Uint64
	requestsCompleted atomic.Uint64
	requestsFailed    atomic.Uint64
	requestsCancelled atomic.Uint64
	tokensGenerated   atomic.Uint64

	stages       map[string]StageTiming
	mu           sync.Mutex
	queueWaitSum time.Duration
	genSum       time.Duration
	waitN        uint64
	genN         uint64
	queueLen     int
	busy         bool

	activeDecode           uint64
	activePrefill          uint64
	scheduledDecodeTokens  uint64
	scheduledPrefillTokens uint64
	kvNumBlocks            uint64
	kvBlockSize            uint64
	kvFreeBlocks           uint64
	kvAllocatedBlocks      uint64
	prefillStallNs         uint64
}

func New() *Collector { return &Collector{stages: make(map[string]StageTiming)} }

// ObserveStage records host wall time. GPU execution requires separate device timing.
func (c *Collector) ObserveStage(name string, d time.Duration) {
	if d < 0 {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.stages == nil {
		c.stages = make(map[string]StageTiming)
	}
	v := c.stages[name]
	v.Count++
	v.TotalNs += uint64(d)
	if uint64(d) > v.MaxNs {
		v.MaxNs = uint64(d)
	}
	c.stages[name] = v
}

func (c *Collector) IncTotal()          { c.requestsTotal.Add(1) }
func (c *Collector) IncCompleted()      { c.requestsCompleted.Add(1) }
func (c *Collector) IncFailed()         { c.requestsFailed.Add(1) }
func (c *Collector) IncCancelled()      { c.requestsCancelled.Add(1) }
func (c *Collector) AddTokens(n uint64) { c.tokensGenerated.Add(n) }

func (c *Collector) ObserveQueueWait(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.queueWaitSum += d
	c.waitN++
}

func (c *Collector) ObserveGeneration(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.genSum += d
	c.genN++
}

func (c *Collector) SetQueueLen(n int) {
	c.mu.Lock()
	c.queueLen = n
	c.mu.Unlock()
}

func (c *Collector) SetBusy(b bool) {
	c.mu.Lock()
	c.busy = b
	c.mu.Unlock()
}

// SetSchedStats records Phase 10 batcher/KV counters (safe to call every step).
func (c *Collector) SetSchedStats(
	activeDecode, activePrefill, schedDecode, schedPrefill uint64,
	kvNum, kvBlock, kvFree, kvAlloc, stallNs uint64,
) {
	c.mu.Lock()
	c.activeDecode = activeDecode
	c.activePrefill = activePrefill
	c.scheduledDecodeTokens = schedDecode
	c.scheduledPrefillTokens = schedPrefill
	c.kvNumBlocks = kvNum
	c.kvBlockSize = kvBlock
	c.kvFreeBlocks = kvFree
	c.kvAllocatedBlocks = kvAlloc
	c.prefillStallNs = stallNs
	c.mu.Unlock()
}

func (c *Collector) Snapshot() Snapshot {
	c.mu.Lock()
	defer c.mu.Unlock()
	stages := make(map[string]StageTiming, len(c.stages))
	for k, v := range c.stages {
		stages[k] = v
	}
	s := Snapshot{
		Stages:                 stages,
		RequestsTotal:          c.requestsTotal.Load(),
		RequestsCompleted:      c.requestsCompleted.Load(),
		RequestsFailed:         c.requestsFailed.Load(),
		RequestsCancelled:      c.requestsCancelled.Load(),
		TokensGenerated:        c.tokensGenerated.Load(),
		QueueLen:               c.queueLen,
		Busy:                   c.busy,
		ActiveDecode:           c.activeDecode,
		ActivePrefill:          c.activePrefill,
		ScheduledDecodeTokens:  c.scheduledDecodeTokens,
		ScheduledPrefillTokens: c.scheduledPrefillTokens,
		KvNumBlocks:            c.kvNumBlocks,
		KvBlockSize:            c.kvBlockSize,
		KvFreeBlocks:           c.kvFreeBlocks,
		KvAllocatedBlocks:      c.kvAllocatedBlocks,
		PrefillStallNs:         c.prefillStallNs,
	}
	if c.waitN > 0 {
		s.AvgQueueWaitMs = float64(c.queueWaitSum) / float64(time.Millisecond) / float64(c.waitN)
	}
	if c.genN > 0 {
		s.AvgGenMs = float64(c.genSum) / float64(time.Millisecond) / float64(c.genN)
	}
	return s
}

func (c *Collector) JSON() []byte {
	b, _ := json.MarshalIndent(c.Snapshot(), "", "  ")
	return b
}
