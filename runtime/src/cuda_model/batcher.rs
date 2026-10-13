//! GPU continuous batching with paged KV + chunked prefill.
//!
//! Contiguous `forward_tokens` / `generate_cuda` remain the reference path.
//! This batcher allocates KV from a shared block pool and interleaves decode
//! with bounded prefill chunks under a per-step token budget.

use super::paged_kv::{GpuKvBlockPool, PagedKvSession, PagedKvStats};
use super::transformer::{CudaBatchScratch, CudaBatchStats, CudaScratch, CudaTransformer};
use crate::compute::ComputeBackend;
use crate::model::{BatchError, BatchEvent, ForgeTokenizer, GenError, GenerateParams};
use crate::sampling::{greedy, sample_temperature, sample_top_k};
use crate::tensor::TensorError;
use std::env;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlotStatus {
    Free,
    /// Prompt partially processed; waiting for more prefill chunks.
    Prefilling,
    /// Prefill done; logits ready; waiting to sample/emit.
    Ready,
    /// Has a next token to decode on the following shared step.
    Decoding,
}

struct CudaSlot {
    id: u64,
    status: SlotStatus,
    session: PagedKvSession,
    /// Full prompt + generated tokens. During Prefilling, `prefill_pos` indexes into this.
    tokens: Vec<u32>,
    /// How many prompt tokens have been written into the KV cache.
    prefill_pos: usize,
    /// Prompt length at admit (generation starts only after prefill_pos == prompt_len).
    prompt_len: usize,
    emitted: String,
    params: GenerateParams,
    generated: usize,
    cancelled: bool,
    rng_state: u64,
    last_logits: Vec<f32>,
    /// GPU argmax result for greedy slots (avoids full logits D2H).
    cached_next: Option<u32>,
    pending_decode: Option<u32>,
}

/// Continuous GPU batcher: paged KV, chunked prefill, token-budget scheduling.
pub struct CudaBatcher {
    model: CudaTransformer,
    tok: Option<ForgeTokenizer>,
    max_seq: usize,
    max_batch: usize,
    slots: Vec<CudaSlot>,
    pool: GpuKvBlockPool,
    batch: CudaBatchScratch,
    prefill_scratch: CudaScratch,
    next_id: u64,
    stats: CudaBatchStats,
    prefill_chunk: usize,
    step_token_budget: usize,
    /// Round-robin cursor for Prefilling slots.
    planner: crate::scheduling::Planner,
    policy: crate::scheduling::Policy,
    /// Sum of wall time (ns) spent running prefill while decode slots waited.
    prefill_stall_ns: u64,
}

impl CudaBatcher {
    pub fn new(
        model: CudaTransformer,
        tok: ForgeTokenizer,
        max_seq: usize,
        max_batch: usize,
    ) -> Result<Self, BatchError> {
        Self::new_inner(model, Some(tok), max_seq, max_batch)
    }

    /// Tokenizer-free constructor for unit / parity tests (`admit_tokens` only).
    pub fn new_untokenized(
        model: CudaTransformer,
        max_seq: usize,
        max_batch: usize,
    ) -> Result<Self, BatchError> {
        Self::new_inner(model, None, max_seq, max_batch)
    }

    fn new_inner(
        model: CudaTransformer,
        tok: Option<ForgeTokenizer>,
        max_seq: usize,
        max_batch: usize,
    ) -> Result<Self, BatchError> {
        let max_batch = resolve_max_batch(max_batch);
        if max_batch == 0 {
            return Err(BatchError::Tensor(TensorError::ZeroDimension));
        }
        let backend = model.backend();
        let cfg = &model.cfg;
        let pool = GpuKvBlockPool::new(
            backend,
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim(),
            max_seq,
            max_batch,
        )
        .map_err(paged_err)?;

        let mut slots = Vec::with_capacity(max_batch);
        for _ in 0..max_batch {
            let session =
                PagedKvSession::new(backend, max_seq, pool.block_size).map_err(paged_err)?;
            slots.push(CudaSlot {
                id: 0,
                status: SlotStatus::Free,
                session,
                tokens: Vec::new(),
                prefill_pos: 0,
                prompt_len: 0,
                emitted: String::new(),
                params: GenerateParams::default(),
                generated: 0,
                cancelled: false,
                rng_state: 0xDEADBEEF,
                last_logits: Vec::new(),
                cached_next: None,
                pending_decode: None,
            });
        }
        let batch = CudaBatchScratch::new(backend, cfg, max_batch).map_err(tensor_err)?;
        // Prefill scratch sized to chunk (and at least 1); capped by max_seq.
        let prefill_chunk = resolve_prefill_chunk(max_seq);
        let scratch_seq = resolve_step_budget()
            .min(prefill_chunk.saturating_mul(max_batch))
            .max(prefill_chunk)
            .max(1);
        let prefill_scratch = CudaScratch::new(backend, cfg, scratch_seq).map_err(tensor_err)?;
        Ok(Self {
            model,
            tok,
            max_seq,
            max_batch,
            slots,
            pool,
            batch,
            prefill_scratch,
            next_id: 1,
            stats: CudaBatchStats::default(),
            prefill_chunk,
            step_token_budget: resolve_step_budget(),
            planner: crate::scheduling::Planner::default(),
            policy: crate::scheduling::Policy::from_name(
                &env::var("FORGE_SCHED_POLICY").unwrap_or_default(),
            ),
            prefill_stall_ns: 0,
        })
    }

    pub fn max_batch(&self) -> usize {
        self.max_batch
    }

    pub fn active_count(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| {
                matches!(
                    s.status,
                    SlotStatus::Ready | SlotStatus::Decoding | SlotStatus::Prefilling
                )
            })
            .count()
    }

    pub fn free_slots(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| s.status == SlotStatus::Free)
            .count()
    }

    pub fn active_decode(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| matches!(s.status, SlotStatus::Ready | SlotStatus::Decoding))
            .count()
    }

    pub fn active_prefill(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| s.status == SlotStatus::Prefilling)
            .count()
    }

    pub fn stats(&self) -> &CudaBatchStats {
        &self.stats
    }

    pub fn kv_stats(&self) -> PagedKvStats {
        self.pool.stats()
    }

    pub fn prefill_stall_ns(&self) -> u64 {
        self.prefill_stall_ns
    }

    pub fn synchronize(&self) -> Result<(), crate::compute::ComputeError> {
        self.model.backend().synchronize()
    }

    pub fn device_mem_info(&self) -> (Option<u64>, Option<u64>) {
        self.model.device_mem_info()
    }

    pub fn admit(&mut self, prompt: &str, params: GenerateParams) -> Result<u64, BatchError> {
        let tok = self.tok.as_ref().ok_or_else(|| {
            BatchError::Gen(GenError::Tok(crate::model::TokError::Encode(
                "tokenizer not configured".into(),
            )))
        })?;
        let started = std::time::Instant::now();
        let tokens = tok.encode(prompt, true)?;
        if env::var_os("FORGE_SERVING_PROFILE").is_some() {
            eprintln!(
                "[runtime tokenize] tokens={} host_ns={}",
                tokens.len(),
                started.elapsed().as_nanos()
            );
        }
        self.admit_tokens(tokens, params)
    }

    pub fn admit_tokens(
        &mut self,
        tokens: Vec<u32>,
        params: GenerateParams,
    ) -> Result<u64, BatchError> {
        let idx = self
            .slots
            .iter()
            .position(|s| s.status == SlotStatus::Free)
            .ok_or(BatchError::Full)?;

        if tokens.is_empty() || tokens.len() >= self.max_seq {
            return Err(BatchError::Gen(GenError::PromptTooLong));
        }

        // Capacity-aware admission: reserve enough blocks for the full prompt.
        self.pool
            .ensure_capacity_tokens(tokens.len())
            .map_err(paged_err)?;

        let id = self.next_id;
        self.next_id += 1;

        let slot = &mut self.slots[idx];
        slot.session.clear(&mut self.pool).map_err(paged_err)?;
        slot.tokens = tokens;
        slot.prompt_len = slot.tokens.len();
        slot.prefill_pos = 0;
        slot.params = params;
        slot.generated = 0;
        slot.emitted.clear();
        slot.cancelled = false;
        slot.rng_state = 0xDEADBEEF ^ id;
        slot.pending_decode = None;
        slot.last_logits.clear();
        slot.cached_next = None;
        slot.id = id;
        slot.status = SlotStatus::Prefilling;
        Ok(id)
    }

    pub fn cancel(&mut self, id: u64) -> Result<(), BatchError> {
        let idx = self
            .slots
            .iter()
            .position(|s| s.id == id && s.status != SlotStatus::Free)
            .ok_or(BatchError::UnknownId(id))?;
        self.free_slot(idx)?;
        Ok(())
    }

    /// One continuous-batching step: sample → decode + chunked prefill under token budget.
    pub fn step(&mut self) -> Result<Vec<BatchEvent>, BatchError> {
        if self.active_count() == 0 {
            return Err(BatchError::Empty);
        }

        let profile = env::var_os("FORGE_SERVING_PROFILE").is_some();
        let step_start = std::time::Instant::now();
        let mut events = Vec::new();
        let vocab = self.model.cfg.vocab_size;

        // --- Sample: sample & emit for Ready slots ---
        for si in 0..self.slots.len() {
            if self.slots[si].cancelled {
                self.free_slot(si)?;
                continue;
            }
            if self.slots[si].status != SlotStatus::Ready {
                continue;
            }

            let next = {
                let slot = &mut self.slots[si];
                Self::sample_slot(slot)
            };
            let slot = &mut self.slots[si];
            slot.tokens.push(next);
            slot.generated += 1;

            let done = slot.params.eos_token_id == Some(next)
                || slot.generated >= slot.params.max_tokens
                || slot.tokens.len() >= self.max_seq;
            let piece = match &self.tok {
                Some(tok) => {
                    tok.stream_piece(&slot.tokens[slot.prompt_len..], &mut slot.emitted, done)?
                }
                None => String::new(),
            };

            events.push(BatchEvent {
                request_id: slot.id,
                token_id: next,
                piece,
                done,
            });

            if done {
                self.free_slot(si)?;
            } else {
                slot.pending_decode = Some(next);
                slot.status = SlotStatus::Decoding;
            }
        }

        // Returning ready tokens avoids holding SSE events behind the next GPU round.
        if !events.is_empty() && env::var("FORGE_EARLY_EMIT").as_deref() == Ok("1") {
            self.stats.last_scheduled_decode = 0;
            self.stats.last_scheduled_prefill = 0;
            return Ok(events);
        }
        // --- Schedule: schedule under token budget ---
        let decode_idxs: Vec<usize> = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, s)| s.status == SlotStatus::Decoding && s.pending_decode.is_some())
            .map(|(i, _)| i)
            .collect();

        let prefill: Vec<_> = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, s)| s.status == SlotStatus::Prefilling)
            .map(|(i, s)| (i, s.prompt_len - s.prefill_pos))
            .collect();
        let plan = self.planner.plan(
            self.policy,
            self.step_token_budget,
            self.prefill_chunk,
            &decode_idxs,
            &prefill,
            self.slots.len(),
        );
        let decode_idxs = plan.decode;
        let decode_cost = decode_idxs.len();
        // Prefer decode: always run pending decode if any (even if budget is tight).
        let run_decode = !decode_idxs.is_empty();
        let prefill_plan = plan.prefill;
        let scheduling_ns = step_start.elapsed().as_nanos();

        self.stats.last_scheduled_decode = if run_decode { decode_cost } else { 0 };
        self.stats.last_scheduled_prefill = prefill_plan.iter().map(|(_, n)| *n).sum();

        // --- Execute: execute decode then prefill ---
        let decode_start = std::time::Instant::now();
        if run_decode {
            let timer = if profile {
                Some(crate::compute::DeviceTimer::start().map_err(compute_err)?)
            } else {
                None
            };
            self.forward_batch_decode(&decode_idxs)?;
            let device_ms = if let Some(timer) = timer {
                timer.elapsed().map_err(compute_err)?
            } else {
                0.0
            };
            if profile {
                self.synchronize().map_err(compute_err)?;
            }
            let compute_ns = decode_start.elapsed().as_nanos();
            let transfer_start = std::time::Instant::now();
            self.store_decode_results(&decode_idxs, vocab)?;
            if profile {
                eprintln!("[runtime decode] rows={} synchronized_compute_host_ns={} sample_transfer_host_ns={} stream_region_ms={}", decode_idxs.len(), compute_ns, transfer_start.elapsed().as_nanos(),device_ms);
            }
        }

        let prefill_start = std::time::Instant::now();
        let prefill_timer = if profile && !prefill_plan.is_empty() {
            Some(crate::compute::DeviceTimer::start().map_err(compute_err)?)
        } else {
            None
        };
        let had_decode_waiting = run_decode || self.active_decode() > 0;
        let packed =
            env::var("FORGE_PACKED_PREFILL").as_deref() == Ok("1") && !prefill_plan.is_empty();
        if packed {
            self.run_packed_prefill(&prefill_plan)?;
            if had_decode_waiting {
                self.prefill_stall_ns = self
                    .prefill_stall_ns
                    .saturating_add(prefill_start.elapsed().as_nanos() as u64);
            }
            self.stats.prefill_chunks = self
                .stats
                .prefill_chunks
                .saturating_add(prefill_plan.len() as u64);
        } else {
            for &(si, chunk_len) in &prefill_plan {
                let t0 = std::time::Instant::now();
                self.run_prefill_chunk(si, chunk_len)?;
                if had_decode_waiting && self.slots[si].status == SlotStatus::Prefilling {
                    // Still prefilling while others may wait next iteration.
                    self.prefill_stall_ns = self
                        .prefill_stall_ns
                        .saturating_add(t0.elapsed().as_nanos() as u64);
                } else if !decode_idxs.is_empty() {
                    self.prefill_stall_ns = self
                        .prefill_stall_ns
                        .saturating_add(t0.elapsed().as_nanos() as u64);
                }
                self.stats.prefill_chunks = self.stats.prefill_chunks.saturating_add(1);
            }
        }
        let prefill_device_ms = if let Some(timer) = prefill_timer {
            timer.elapsed().map_err(compute_err)?
        } else {
            0.0
        };
        if profile {
            self.synchronize().map_err(compute_err)?;
            eprintln!("[runtime step] scheduling_host_ns={} prefill_synchronized_host_ns={} prefill_rows={} prefill_requests={} decode_rows={} total_host_ns={} prefill_stream_region_ms={}", scheduling_ns, prefill_start.elapsed().as_nanos(), self.stats.last_scheduled_prefill, prefill_plan.len(), decode_idxs.len(), step_start.elapsed().as_nanos(),prefill_device_ms);
        }
        if env::var_os("FORGE_CUDA_BATCH_TRACE").is_some() {
            let s = &self.stats;
            let kv = self.pool.stats();
            eprintln!(
                "[forge cuda batch] B={} decode={} prefill={} q_proj=({},{},{}) kv_free={} kv_used={} steps={}",
                s.last_batch_size,
                s.last_scheduled_decode,
                s.last_scheduled_prefill,
                s.last_q_proj_m,
                s.last_q_proj_n,
                s.last_q_proj_k,
                kv.free_blocks,
                kv.allocated_blocks,
                s.decode_steps
            );
        }

        Ok(events)
    }

    fn run_packed_prefill(&mut self, plan: &[(usize, usize)]) -> Result<(), BatchError> {
        let idxs: Vec<usize> = plan.iter().map(|&(i, _)| i).collect();
        let chunks: Vec<Vec<u32>> = plan
            .iter()
            .map(|&(i, n)| {
                let s = &self.slots[i];
                s.tokens[s.prefill_pos..s.prefill_pos + n].to_vec()
            })
            .collect();
        let completed: Vec<bool> = plan
            .iter()
            .map(|&(i, n)| self.slots[i].prefill_pos + n == self.slots[i].prompt_len)
            .collect();
        let mut refs = slot_sessions_mut(&mut self.slots, &idxs)?;
        let finished = self
            .model
            .forward_prefill_paged(
                &chunks,
                &completed,
                &mut self.pool,
                &mut refs,
                &mut self.prefill_scratch,
                &mut self.batch,
            )
            .map_err(compute_err)?;
        for &(i, n) in plan {
            self.slots[i].prefill_pos += n;
        }
        let ready: Vec<usize> = finished.iter().map(|&row| idxs[row]).collect();
        if !ready.is_empty() {
            self.store_decode_results(&ready, self.model.cfg.vocab_size)?;
        }
        Ok(())
    }

    fn run_prefill_chunk(&mut self, si: usize, chunk_len: usize) -> Result<(), BatchError> {
        let pos = self.slots[si].prefill_pos;
        let prompt_len = self.slots[si].prompt_len;
        let end = pos + chunk_len;
        if end > prompt_len {
            return Err(BatchError::Tensor(TensorError::ShapeMismatch {
                expected: vec![prompt_len],
                got: vec![end],
            }));
        }
        let chunk: Vec<u32> = self.slots[si].tokens[pos..end].to_vec();

        // Disjoint field borrows: model / pool / slots / prefill_scratch.
        self.model
            .forward_tokens_paged(
                &chunk,
                pos,
                &mut self.pool,
                &mut self.slots[si].session,
                &mut self.prefill_scratch,
            )
            .map_err(compute_err)?;

        let vocab = self.model.cfg.vocab_size;
        self.slots[si].prefill_pos = end;
        if self.slots[si].prefill_pos == self.slots[si].prompt_len {
            if use_gpu_greedy(&self.slots[si].params) {
                let tok = self
                    .model
                    .greedy_from_scratch(&self.prefill_scratch, vocab)
                    .map_err(compute_err)?;
                self.slots[si].cached_next = Some(tok);
                self.slots[si].last_logits.clear();
            } else {
                self.slots[si].cached_next = None;
                self.slots[si].last_logits = self
                    .model
                    .read_logits(&self.prefill_scratch, vocab)
                    .map_err(compute_err)?;
            }
            self.slots[si].status = SlotStatus::Ready;
        }
        Ok(())
    }

    /// After decode: GPU argmax for greedy rows; D2H logits only for non-greedy.
    fn store_decode_results(
        &mut self,
        decode_idxs: &[usize],
        vocab: usize,
    ) -> Result<(), BatchError> {
        let b = decode_idxs.len();
        let greedy_mask: Vec<bool> = decode_idxs
            .iter()
            .map(|&si| use_gpu_greedy(&self.slots[si].params))
            .collect();
        let any_greedy = greedy_mask.iter().any(|&g| g);
        let any_sample = greedy_mask.iter().any(|&g| !g);

        let gpu_ids = if any_greedy {
            Some(
                self.model
                    .greedy_from_batch_scratch(&self.batch, b, vocab)
                    .map_err(compute_err)?,
            )
        } else {
            None
        };
        let host_logits = if any_sample {
            Some(
                self.model
                    .read_batch_logits(&self.batch, b, vocab)
                    .map_err(compute_err)?,
            )
        } else {
            None
        };

        for (row, &si) in decode_idxs.iter().enumerate() {
            let slot = &mut self.slots[si];
            if greedy_mask[row] {
                slot.cached_next = Some(gpu_ids.as_ref().unwrap()[row]);
                slot.last_logits.clear();
            } else {
                slot.cached_next = None;
                let off = row * vocab;
                slot.last_logits = host_logits.as_ref().unwrap()[off..off + vocab].to_vec();
            }
            slot.pending_decode = None;
            slot.status = SlotStatus::Ready;
        }
        Ok(())
    }

    fn free_slot(&mut self, si: usize) -> Result<(), BatchError> {
        let slot = &mut self.slots[si];
        slot.session.clear(&mut self.pool).map_err(paged_err)?;
        slot.status = SlotStatus::Free;
        slot.cancelled = false;
        slot.pending_decode = None;
        slot.tokens.clear();
        slot.last_logits.clear();
        slot.cached_next = None;
        slot.prefill_pos = 0;
        slot.prompt_len = 0;
        slot.generated = 0;
        slot.id = 0;
        Ok(())
    }

    fn sample_slot(slot: &mut CudaSlot) -> u32 {
        if let Some(tok) = slot.cached_next.take() {
            return tok;
        }
        let logits = &slot.last_logits;
        if slot.params.temperature <= 0.0 {
            greedy(logits)
        } else if slot.params.top_k > 0 {
            sample_top_k(logits, slot.params.top_k, slot.params.temperature, || {
                rng_f32(&mut slot.rng_state)
            })
        } else {
            sample_temperature(logits, slot.params.temperature, || {
                rng_f32(&mut slot.rng_state)
            })
        }
    }

    fn forward_batch_decode(&mut self, idxs: &[usize]) -> Result<(), BatchError> {
        let tokens: Vec<u32> = idxs
            .iter()
            .map(|&si| self.slots[si].pending_decode.ok_or(BatchError::Empty))
            .collect::<Result<Vec<_>, _>>()?;
        let positions: Vec<usize> = idxs
            .iter()
            .map(|&si| self.slots[si].tokens.len() - 1)
            .collect();

        let mut session_refs = slot_sessions_mut(&mut self.slots, idxs)?;

        self.model
            .forward_batch_decode_paged(
                &tokens,
                &positions,
                &mut self.pool,
                &mut session_refs,
                &mut self.batch,
                &mut self.stats,
            )
            .map_err(compute_err)?;
        Ok(())
    }
}

fn use_gpu_greedy(params: &GenerateParams) -> bool {
    params.temperature <= 0.0 && env::var_os("FORGE_CPU_GREEDY").is_none()
}

fn resolve_max_batch(requested: usize) -> usize {
    env::var("FORGE_CUDA_MAX_BATCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(requested)
        .max(1)
}

fn resolve_prefill_chunk(max_seq: usize) -> usize {
    env::var("FORGE_PREFILL_CHUNK")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(128)
        .min(max_seq)
        .max(1)
}

fn resolve_step_budget() -> usize {
    env::var("FORGE_STEP_TOKEN_BUDGET")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(256)
}

fn slot_sessions_mut<'a>(
    slots: &'a mut [CudaSlot],
    idxs: &[usize],
) -> Result<Vec<&'a mut PagedKvSession>, BatchError> {
    let n = slots.len();
    let mut seen = vec![false; n];
    for &i in idxs {
        if i >= n || seen[i] {
            return Err(BatchError::Tensor(TensorError::ShapeMismatch {
                expected: vec![n],
                got: vec![i],
            }));
        }
        seen[i] = true;
    }
    let base = slots.as_mut_ptr();
    let mut out = Vec::with_capacity(idxs.len());
    for &i in idxs {
        // SAFETY: idxs are unique and in-range; each yields a distinct CudaSlot.
        unsafe {
            out.push(&mut (*base.add(i)).session);
        }
    }
    Ok(out)
}

fn tensor_err(e: crate::compute::ComputeError) -> BatchError {
    BatchError::Tensor(TensorError::AllocFailed(e.to_string()))
}

fn compute_err(e: crate::compute::ComputeError) -> BatchError {
    match &e {
        crate::compute::ComputeError::OutOfMemory(s) => BatchError::KvExhausted(s.clone()),
        _ => BatchError::Tensor(TensorError::AllocFailed(e.to_string())),
    }
}

fn paged_err(e: super::paged_kv::PagedKvError) -> BatchError {
    match e {
        super::paged_kv::PagedKvError::Exhausted { need, free } => {
            BatchError::KvExhausted(format!("need {need} blocks, free {free}"))
        }
        other => BatchError::Tensor(TensorError::AllocFailed(other.to_string())),
    }
}

fn rng_f32(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    (*state >> 32) as f32 / u32::MAX as f32
}
