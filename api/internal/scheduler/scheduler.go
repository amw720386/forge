// Package scheduler owns continuous-batching inference execution.
//
// Multiple HTTP clients may be active concurrently. The scheduler admits
// requests into free Rust batch slots (backpressure when full) and runs
// shared forge.BatchStep decode rounds — not one Rust generate loop per
// request in parallel.
package scheduler

import (
	"errors"
	"log"
	"runtime"
	"strings"
	"sync"
	"time"

	forge "github.com/amw720386/forge/bindings/go"

	"github.com/amw720386/forge/api/internal/metrics"
	"github.com/amw720386/forge/api/internal/queue"
)

type liveReq struct {
	req      *queue.Request
	nTokens  uint64
	genStart time.Time
}

type Scheduler struct {
	rt       *forge.Runtime
	q        *queue.Queue
	metrics  *metrics.Collector
	stop     chan struct{}
	done     chan struct{}
	stopOnce sync.Once

	mu   sync.Mutex
	live map[uint64]*liveReq // runtime request id → HTTP request
}

func New(rt *forge.Runtime, q *queue.Queue, m *metrics.Collector) *Scheduler {
	return &Scheduler{
		rt:      rt,
		q:       q,
		metrics: m,
		stop:    make(chan struct{}),
		done:    make(chan struct{}),
		live:    make(map[uint64]*liveReq),
	}
}

func (s *Scheduler) Start() {
	go s.loop()
}

func (s *Scheduler) Stop() {
	s.stopOnce.Do(func() { close(s.stop) })
	<-s.done
}

func (s *Scheduler) loop() {
	defer close(s.done)
	// CUDA kernels and graph capture use the per-thread default stream. Keep
	// admissions, steps, and cancellation on the same OS thread so a Go
	// goroutine migration cannot reorder work across independent CUDA streams.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	for {
		select {
		case <-s.stop:
			s.cancelAll()
			return
		default:
		}

		iterationStart := time.Now()
		s.admitAvailable()
		s.reapCancelled()

		s.mu.Lock()
		nLive := len(s.live)
		s.mu.Unlock()

		if nLive == 0 {
			// Block until a queued request arrives or stop.
			select {
			case <-s.stop:
				s.cancelAll()
				return
			case req, ok := <-s.q.Chan():
				if !ok {
					return
				}
				s.tryAdmit(req)
			}
			continue
		}

		// Shared continuous-batching step for all active sequences.
		s.metrics.SetBusy(true)
		stepStart := time.Now()
		events, err := s.rt.BatchStep()
		stepDuration := time.Since(stepStart)
		s.metrics.ObserveStage("runtime_step", stepDuration)
		if st, stErr := s.rt.BatchSchedStats(); stErr == nil {
			s.metrics.SetSchedStats(
				st.ActiveDecode, st.ActivePrefill,
				st.LastScheduledDecode, st.LastScheduledPrefill,
				st.KvNumBlocks, st.KvBlockSize, st.KvFreeBlocks, st.KvAllocatedBlocks,
				st.PrefillStallNs,
			)
		}
		s.metrics.SetBusy(false)
		if err != nil {
			log.Printf("batch step: %v", err)
			s.failAll(errors.New("generation failed; check server logs"))
			continue
		}
		dispatchStart := time.Now()
		s.dispatch(events)
		s.metrics.ObserveStage("token_dispatch", time.Since(dispatchStart))
		s.metrics.ObserveStage("scheduler_iteration", time.Since(iterationStart))
		s.metrics.ObserveStage("scheduler_host", time.Since(iterationStart)-stepDuration)
		s.metrics.SetQueueLen(s.q.Len())
	}
}

func (s *Scheduler) admitAvailable() {
	for s.rt.BatchFreeSlots() > 0 {
		select {
		case req, ok := <-s.q.Chan():
			if !ok {
				return
			}
			if !s.tryAdmit(req) {
				return
			}
		default:
			return
		}
	}
}

func (s *Scheduler) tryAdmit(req *queue.Request) bool {
	select {
	case <-req.Ctx.Done():
		req.State = queue.StateCancelled
		s.metrics.IncCancelled()
		close(req.Tokens)
		return true
	default:
	}

	wait := time.Since(req.EnqueuedAt)
	s.metrics.ObserveQueueWait(wait)
	s.metrics.ObserveStage("queue_wait", wait)
	req.State = queue.StateRunning
	req.StartedAt = time.Now()

	admitStart := time.Now()
	id, err := s.rt.BatchAdmit(req.Prompt, req.MaxTokens, req.Temperature)
	s.metrics.ObserveStage("runtime_admit_tokenize", time.Since(admitStart))
	if err != nil {
		req.State = queue.StateFailed
		log.Printf("admit request %s: %v", req.ID, err)
		if strings.Contains(err.Error(), "prompt exceeds max_seq") {
			req.Err = errors.New("formatted prompt exceeds context limit; shorten prompt or increase -max-seq")
		} else {
			req.Err = errors.New("model could not admit request; check server logs")
		}
		s.metrics.IncFailed()
		close(req.Tokens)
		// Busy means backpressure raced; stop draining the queue this round.
		return !errors.Is(err, forge.StatusBusy)
	}

	req.State = queue.StateStreaming
	s.mu.Lock()
	s.live[id] = &liveReq{
		req:      req,
		genStart: time.Now(),
	}
	s.mu.Unlock()
	s.metrics.SetQueueLen(s.q.Len())
	return true
}

func (s *Scheduler) reapCancelled() {
	s.mu.Lock()
	defer s.mu.Unlock()
	for id, lr := range s.live {
		select {
		case <-lr.req.Ctx.Done():
			_ = s.rt.BatchCancel(id)
			lr.req.State = queue.StateCancelled
			s.metrics.IncCancelled()
			close(lr.req.Tokens)
			delete(s.live, id)
		default:
		}
	}
}

func (s *Scheduler) dispatch(events []forge.BatchEvent) {
	for _, ev := range events {
		s.mu.Lock()
		lr, ok := s.live[ev.RequestID]
		s.mu.Unlock()
		if !ok {
			continue
		}

		lr.req.CompletionTokens++
		if ev.Done && lr.req.Chat {
			if ev.TokenID == lr.req.EOSTokenID {
				lr.req.FinishReason = "stop"
			} else {
				lr.req.FinishReason = "length"
			}
		}
		tokEv := queue.TokenEvent{TokenID: ev.TokenID, Text: ev.Text, Done: ev.Done}
		select {
		case lr.req.Tokens <- tokEv:
		case <-s.stop:
			return
		case <-lr.req.Ctx.Done():
			_ = s.rt.BatchCancel(ev.RequestID)
			s.mu.Lock()
			if cur, ok := s.live[ev.RequestID]; ok {
				cur.req.State = queue.StateCancelled
				s.metrics.IncCancelled()
				close(cur.req.Tokens)
				delete(s.live, ev.RequestID)
			}
			s.mu.Unlock()
			continue
		}

		if !ev.Done {
			lr.nTokens++
			continue
		}

		s.mu.Lock()
		if cur, ok := s.live[ev.RequestID]; ok {
			cur.req.State = queue.StateCompleted
			s.metrics.IncCompleted()
			s.metrics.AddTokens(cur.nTokens + 1)
			s.metrics.ObserveGeneration(time.Since(cur.genStart))
			close(cur.req.Tokens)
			delete(s.live, ev.RequestID)
		}
		s.mu.Unlock()
	}
}

func (s *Scheduler) failAll(err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for id, lr := range s.live {
		_ = s.rt.BatchCancel(id)
		lr.req.Err = err
		lr.req.State = queue.StateFailed
		s.metrics.IncFailed()
		close(lr.req.Tokens)
		delete(s.live, id)
	}
}

func (s *Scheduler) cancelAll() {
	for {
		select {
		case req, ok := <-s.q.Chan():
			if !ok {
				goto drained
			}
			req.Err = errors.New("server shutting down")
			close(req.Tokens)
		default:
			goto drained
		}
	}
drained:
	s.mu.Lock()
	defer s.mu.Unlock()
	for id, lr := range s.live {
		_ = s.rt.BatchCancel(id)
		lr.req.Err = errors.New("server shutting down")
		lr.req.State = queue.StateCancelled
		close(lr.req.Tokens)
		delete(s.live, id)
	}
}

func (s *Scheduler) Logf(format string, args ...any) {
	log.Printf(format, args...)
}

// PrepareChat uses the same loaded tokenizer as batch admission.
func (s *Scheduler) PrepareChat(messages []forge.Message) (forge.PreparedChat, error) {
	return s.rt.PrepareChat(messages)
}
