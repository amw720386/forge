//! Genuine continuous batching: shared decode matmuls across independent requests.
//!
//! Each request keeps its own KV caches and token state. Decode steps pack active
//! sequences into a batch dimension `B` and run linear/MLP/lm_head with `m=B`.
//! Attention remains per-sequence (different cache lengths) — linears dominate cost.

use super::generate::{GenError, GenerateParams};
use super::tokenizer::ForgeTokenizer;
use super::transformer::{ForwardScratch, LayerKvCaches, Transformer};
use crate::attention::AttnScratch;
use crate::sampling::{greedy, sample_temperature, sample_top_k};
use crate::tensor::rows::{add_rows, copy_rows, rmsnorm_rows};
use crate::tensor::{Tensor, TensorError};

/// Workspace for a decode step over up to `max_batch` sequences (one token each).
pub struct BatchScratch {
    pub x: Tensor,
    pub residual: Tensor,
    pub normed: Tensor,
    pub attn_out: Tensor,
    pub mlp_gate: Tensor,
    pub mlp_up: Tensor,
    pub mlp_out: Tensor,
    pub logits: Tensor,
    /// Batched Q/K/V/ctx before/after per-seq attention.
    pub q: Tensor,
    pub k: Tensor,
    pub v: Tensor,
    pub ctx: Tensor,
    /// Single-token attention scratch reused across sequences in a step.
    pub attn: AttnScratch,
}

impl BatchScratch {
    pub fn new(cfg: &super::config::ModelConfig, max_batch: usize) -> Result<Self, TensorError> {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let q_w = cfg.n_heads * cfg.head_dim();
        let kv_w = cfg.n_kv_heads * cfg.head_dim();
        Ok(Self {
            x: Tensor::zeros(&[max_batch, h])?,
            residual: Tensor::zeros(&[max_batch, h])?,
            normed: Tensor::zeros(&[max_batch, h])?,
            attn_out: Tensor::zeros(&[max_batch, h])?,
            mlp_gate: Tensor::zeros(&[max_batch, i])?,
            mlp_up: Tensor::zeros(&[max_batch, i])?,
            mlp_out: Tensor::zeros(&[max_batch, h])?,
            logits: Tensor::zeros(&[max_batch, cfg.vocab_size])?,
            q: Tensor::zeros(&[max_batch, q_w])?,
            k: Tensor::zeros(&[max_batch, kv_w])?,
            v: Tensor::zeros(&[max_batch, kv_w])?,
            ctx: Tensor::zeros(&[max_batch, q_w])?,
            attn: AttnScratch::new(&cfg.attention(), 1)?,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlotStatus {
    Free,
    /// Prefill done; logits ready; waiting to sample/emit.
    Ready,
    /// Has a next token to decode on the following shared step.
    Decoding,
}

struct Slot {
    id: u64,
    status: SlotStatus,
    caches: LayerKvCaches,
    tokens: Vec<u32>,
    prompt_len: usize,
    emitted: String,
    params: GenerateParams,
    generated: usize,
    rng_state: u64,
    /// Owned copy of last-token logits after prefill / shared decode.
    last_logits: Vec<f32>,
    /// Token queued for the next shared decode (set after sampling).
    pending_decode: Option<u32>,
}

/// Event emitted when a request produces a token during [`ContinuousBatcher::step`].
#[derive(Debug, Clone)]
pub struct BatchEvent {
    pub request_id: u64,
    pub token_id: u32,
    pub piece: String,
    pub done: bool,
}

#[derive(Debug)]
pub enum BatchError {
    Gen(GenError),
    Tensor(TensorError),
    Full,
    UnknownId(u64),
    Empty,
    /// GPU KV block pool cannot admit or grow this request.
    KvExhausted(String),
}

impl std::fmt::Display for BatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BatchError::Gen(e) => write!(f, "{e}"),
            BatchError::Tensor(e) => write!(f, "{e}"),
            BatchError::Full => write!(f, "batch full (backpressure)"),
            BatchError::UnknownId(id) => write!(f, "unknown request id {id}"),
            BatchError::Empty => write!(f, "no active requests"),
            BatchError::KvExhausted(s) => write!(f, "KV capacity exhausted: {s}"),
        }
    }
}

impl std::error::Error for BatchError {}
impl From<GenError> for BatchError {
    fn from(e: GenError) -> Self {
        BatchError::Gen(e)
    }
}
impl From<TensorError> for BatchError {
    fn from(e: TensorError) -> Self {
        BatchError::Tensor(e)
    }
}
impl From<super::tokenizer::TokError> for BatchError {
    fn from(e: super::tokenizer::TokError) -> Self {
        BatchError::Gen(GenError::Tok(e))
    }
}

/// Continuous batcher: admit/remove/cancel between shared decode steps.
pub struct ContinuousBatcher {
    model: Transformer,
    tok: Option<ForgeTokenizer>,
    max_seq: usize,
    max_batch: usize,
    slots: Vec<Slot>,
    /// Shared decode workspace (`m = B`).
    batch: BatchScratch,
    /// Shared prefill workspace — prefills are serialized; avoids per-slot `max_seq` scratch.
    prefill_scratch: ForwardScratch,
    next_id: u64,
}

impl ContinuousBatcher {
    pub fn new(
        model: Transformer,
        tok: ForgeTokenizer,
        max_seq: usize,
        max_batch: usize,
    ) -> Result<Self, BatchError> {
        Self::new_inner(model, Some(tok), max_seq, max_batch)
    }

    /// Tokenizer-free constructor for unit tests (`admit_tokens` only).
    pub fn new_untokenized(
        model: Transformer,
        max_seq: usize,
        max_batch: usize,
    ) -> Result<Self, BatchError> {
        Self::new_inner(model, None, max_seq, max_batch)
    }

    fn new_inner(
        model: Transformer,
        tok: Option<ForgeTokenizer>,
        max_seq: usize,
        max_batch: usize,
    ) -> Result<Self, BatchError> {
        if max_batch == 0 {
            return Err(BatchError::Tensor(TensorError::ZeroDimension));
        }
        let mut slots = Vec::with_capacity(max_batch);
        for _ in 0..max_batch {
            slots.push(Slot {
                id: 0,
                status: SlotStatus::Free,
                caches: LayerKvCaches::new(&model.cfg, max_seq)?,
                tokens: Vec::new(),
                prompt_len: 0,
                emitted: String::new(),
                params: GenerateParams::default(),
                generated: 0,
                rng_state: 0xDEADBEEF,
                last_logits: Vec::new(),
                pending_decode: None,
            });
        }
        let batch = BatchScratch::new(&model.cfg, max_batch)?;
        let prefill_scratch = ForwardScratch::new(&model.cfg, max_seq)?;
        Ok(Self {
            model,
            tok,
            max_seq,
            max_batch,
            slots,
            batch,
            prefill_scratch,
            next_id: 1,
        })
    }

    pub fn max_batch(&self) -> usize {
        self.max_batch
    }

    pub fn active_count(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| matches!(s.status, SlotStatus::Ready | SlotStatus::Decoding))
            .count()
    }

    pub fn free_slots(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| s.status == SlotStatus::Free)
            .count()
    }

    /// Admit a new request: tokenize + prefill on its private KV/scratch.
    /// Returns `BatchError::Full` when no free slot (backpressure).
    pub fn admit(&mut self, prompt: &str, params: GenerateParams) -> Result<u64, BatchError> {
        let tok = self.tok.as_ref().ok_or_else(|| {
            BatchError::Gen(GenError::Tok(super::tokenizer::TokError::Encode(
                "tokenizer not configured".into(),
            )))
        })?;
        let tokens = tok.encode(prompt, true)?;
        self.admit_tokens(tokens, params)
    }

    /// Admit with pre-tokenized ids (tests / FFI that already tokenized).
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

        let id = self.next_id;
        self.next_id += 1;

        let slot = &mut self.slots[idx];
        slot.caches.clear();
        slot.tokens = tokens;
        slot.prompt_len = slot.tokens.len();
        slot.params = params;
        slot.generated = 0;
        slot.emitted.clear();
        slot.rng_state = 0xDEADBEEF ^ id;
        slot.pending_decode = None;
        slot.id = id;

        let tokens = self.slots[idx].tokens.clone();
        self.model.forward_tokens(
            &tokens,
            0,
            &mut self.slots[idx].caches,
            &mut self.prefill_scratch,
        )?;
        let vocab = self.model.cfg.vocab_size;
        self.slots[idx].last_logits = self.prefill_scratch.logits.data()[..vocab].to_vec();
        self.slots[idx].status = SlotStatus::Ready;
        Ok(id)
    }

    /// Cancel without touching other requests' KV/state.
    pub fn cancel(&mut self, id: u64) -> Result<(), BatchError> {
        let idx = self
            .slots
            .iter()
            .position(|s| s.id == id && s.status != SlotStatus::Free)
            .ok_or(BatchError::UnknownId(id))?;
        self.free_slot(idx);
        Ok(())
    }

    /// One continuous-batching step:
    /// 1. Sample/emit for Ready slots (and Decoding slots that already have logits).
    /// 2. Queue pending decode tokens.
    /// 3. Run a **shared** batched decode for all pending sequences.
    ///
    /// Fairness: every active request participates in the same step (no starvation
    /// within the batch). Admission order is slot-scan order.
    pub fn step(&mut self) -> Result<Vec<BatchEvent>, BatchError> {
        if self.active_count() == 0 {
            return Err(BatchError::Empty);
        }

        let mut events = Vec::new();
        let vocab = self.model.cfg.vocab_size;

        // --- Sample: sample & emit for slots that have fresh logits ---
        for si in 0..self.slots.len() {
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
                self.free_slot(si);
            } else {
                slot.pending_decode = Some(next);
                slot.status = SlotStatus::Decoding;
            }
        }

        // --- Decode: shared batched decode for all Decoding slots ---
        let decode_idxs: Vec<usize> = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, s)| s.status == SlotStatus::Decoding && s.pending_decode.is_some())
            .map(|(i, _)| i)
            .collect();

        if decode_idxs.is_empty() {
            return Ok(events);
        }

        self.forward_batch_decode(&decode_idxs)?;

        for (row, &si) in decode_idxs.iter().enumerate() {
            let slot = &mut self.slots[si];
            let off = row * vocab;
            slot.last_logits = self.batch.logits.data()[off..off + vocab].to_vec();
            slot.pending_decode = None;
            slot.status = SlotStatus::Ready;
        }

        Ok(events)
    }

    fn free_slot(&mut self, si: usize) {
        let slot = &mut self.slots[si];
        slot.status = SlotStatus::Free;
        slot.pending_decode = None;
        slot.tokens.clear();
        slot.last_logits.clear();
        slot.caches.clear();
    }

    fn sample_slot(slot: &mut Slot) -> u32 {
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

    /// Shared decode: embed B tokens, run layers with batched linears, per-seq attention.
    fn forward_batch_decode(&mut self, idxs: &[usize]) -> Result<(), BatchError> {
        let b = idxs.len();
        let h = self.model.cfg.hidden_size;
        let i_sz = self.model.cfg.intermediate_size;
        let q_w = self.model.cfg.n_heads * self.model.cfg.head_dim();
        let kv_w = self.model.cfg.n_kv_heads * self.model.cfg.head_dim();
        let vocab = self.model.cfg.vocab_size;
        let eps = self.model.cfg.rms_norm_eps;

        // Embed pending tokens into batch.x[0..b)
        for (row, &si) in idxs.iter().enumerate() {
            let tok = self.slots[si].pending_decode.ok_or(BatchError::Empty)? as usize;
            if tok >= self.model.cfg.vocab_size {
                return Err(BatchError::Tensor(TensorError::ShapeMismatch {
                    expected: vec![self.model.cfg.vocab_size],
                    got: vec![tok],
                }));
            }
            let src = &self.model.embed.data()[tok * h..(tok + 1) * h];
            self.batch.x.data_mut()[row * h..(row + 1) * h].copy_from_slice(src);
        }

        let positions: Vec<usize> = idxs
            .iter()
            .map(|&si| self.slots[si].tokens.len() - 1)
            .collect();

        for layer_i in 0..self.model.layers.len() {
            // residual = x
            copy_rows(self.batch.x.data(), b, h, self.batch.residual.data_mut())?;

            let w = &self.model.layers[layer_i].weights;
            let attn_cfg = self.model.layers[layer_i].attn_cfg.clone();

            rmsnorm_rows(
                self.batch.x.data(),
                b,
                h,
                h,
                w.input_layernorm.data(),
                eps,
                self.batch.normed.data_mut(),
                h,
            )?;

            // Batched QKV (m=B)
            w.q_proj.apply_rows(
                self.batch.normed.data(),
                b,
                h,
                h,
                self.batch.q.data_mut(),
                q_w,
            )?;
            w.k_proj.apply_rows(
                self.batch.normed.data(),
                b,
                h,
                h,
                self.batch.k.data_mut(),
                kv_w,
            )?;
            w.v_proj.apply_rows(
                self.batch.normed.data(),
                b,
                h,
                h,
                self.batch.v.data_mut(),
                kv_w,
            )?;

            // Per-sequence attention (independent KV lengths / positions)
            for (row, &si) in idxs.iter().enumerate() {
                let q_row = &self.batch.q.data()[row * q_w..(row + 1) * q_w];
                let k_row = &self.batch.k.data()[row * kv_w..(row + 1) * kv_w];
                let v_row = &self.batch.v.data()[row * kv_w..(row + 1) * kv_w];
                self.batch.attn.q.data_mut()[..q_w].copy_from_slice(q_row);
                self.batch.attn.k.data_mut()[..kv_w].copy_from_slice(k_row);
                self.batch.attn.v.data_mut()[..kv_w].copy_from_slice(v_row);

                crate::attention::attend(
                    &attn_cfg,
                    &mut self.batch.attn,
                    1,
                    &mut self.slots[si].caches.layers[layer_i],
                    positions[row],
                )?;

                self.batch.ctx.data_mut()[row * q_w..(row + 1) * q_w]
                    .copy_from_slice(&self.batch.attn.ctx.data()[..q_w]);
            }

            w.o_proj.apply_rows(
                self.batch.ctx.data(),
                b,
                q_w,
                q_w,
                self.batch.attn_out.data_mut(),
                h,
            )?;
            add_rows(
                self.batch.residual.data(),
                self.batch.attn_out.data(),
                b,
                h,
                self.batch.x.data_mut(),
            )?;

            // MLP (batched)
            copy_rows(self.batch.x.data(), b, h, self.batch.residual.data_mut())?;
            rmsnorm_rows(
                self.batch.x.data(),
                b,
                h,
                h,
                w.post_attention_layernorm.data(),
                eps,
                self.batch.normed.data_mut(),
                h,
            )?;
            w.mlp.gate.apply_rows(
                self.batch.normed.data(),
                b,
                h,
                h,
                self.batch.mlp_gate.data_mut(),
                i_sz,
            )?;
            w.mlp.up.apply_rows(
                self.batch.normed.data(),
                b,
                h,
                h,
                self.batch.mlp_up.data_mut(),
                i_sz,
            )?;
            {
                let n = b * i_sz;
                let status = unsafe {
                    crate::ffi::kernels::forge_kernels_silu_f32(
                        n,
                        self.batch.mlp_gate.data().as_ptr(),
                        self.batch.mlp_gate.data_mut().as_mut_ptr(),
                    )
                };
                crate::ffi::kernels::check(status).map_err(TensorError::Kernel)?;
                let status = unsafe {
                    crate::ffi::kernels::forge_kernels_mul_f32(
                        n,
                        self.batch.mlp_gate.data().as_ptr(),
                        self.batch.mlp_up.data().as_ptr(),
                        self.batch.mlp_gate.data_mut().as_mut_ptr(),
                    )
                };
                crate::ffi::kernels::check(status).map_err(TensorError::Kernel)?;
            }
            w.mlp.down.apply_rows(
                self.batch.mlp_gate.data(),
                b,
                i_sz,
                i_sz,
                self.batch.mlp_out.data_mut(),
                h,
            )?;
            add_rows(
                self.batch.residual.data(),
                self.batch.mlp_out.data(),
                b,
                h,
                self.batch.x.data_mut(),
            )?;
        }

        // Final norm + batched lm_head
        rmsnorm_rows(
            self.batch.x.data(),
            b,
            h,
            h,
            self.model.final_norm.data(),
            eps,
            self.batch.normed.data_mut(),
            h,
        )?;
        self.model.lm_head.apply_rows(
            self.batch.normed.data(),
            b,
            h,
            h,
            self.batch.logits.data_mut(),
            vocab,
        )?;
        Ok(())
    }
}

fn rng_f32(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 32) as f32) / (u32::MAX as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelConfig;

    #[test]
    fn concurrent_greedy_matches_sequential() {
        let cfg = ModelConfig::tiny();
        let model = Transformer::synthetic(cfg.clone(), 7).unwrap();
        let max_seq = 64;
        let params = GenerateParams {
            max_tokens: 5,
            temperature: 0.0,
            top_k: 0,
            eos_token_id: None,
        };

        // Sequential baselines.
        let prompts: [Vec<u32>; 3] = [vec![1, 2, 3], vec![10, 11], vec![20, 21, 22, 23]];
        let mut seq_out: Vec<Vec<u32>> = Vec::new();
        for p in &prompts {
            let mut caches = LayerKvCaches::new(&cfg, max_seq).unwrap();
            let mut scratch = ForwardScratch::new(&cfg, max_seq).unwrap();
            let mut tokens = p.clone();
            model
                .forward_tokens(&tokens, 0, &mut caches, &mut scratch)
                .unwrap();
            let mut logits = scratch.logits.data()[..cfg.vocab_size].to_vec();
            let mut gen = Vec::new();
            for _ in 0..params.max_tokens {
                let next = greedy(&logits);
                gen.push(next);
                tokens.push(next);
                let pos = tokens.len() - 1;
                model
                    .forward_tokens(&[next], pos, &mut caches, &mut scratch)
                    .unwrap();
                logits = scratch.logits.data()[..cfg.vocab_size].to_vec();
            }
            seq_out.push(gen);
        }

        // Continuous batch: admit all, step until done.
        let mut batcher = ContinuousBatcher::new_untokenized(
            Transformer::synthetic(cfg.clone(), 7).unwrap(),
            max_seq,
            4,
        )
        .unwrap();
        let mut ids = Vec::new();
        for p in &prompts {
            ids.push(batcher.admit_tokens(p.clone(), params.clone()).unwrap());
        }
        assert_eq!(batcher.free_slots(), 1);
        assert!(batcher.admit_tokens(vec![1], params.clone()).is_ok());
        assert!(matches!(
            batcher.admit_tokens(vec![1], params.clone()),
            Err(BatchError::Full)
        ));
        // Free the filler slot so we only track the three prompts.
        let filler = batcher
            .slots
            .iter()
            .find(|s| s.status != SlotStatus::Free && !ids.contains(&s.id))
            .unwrap()
            .id;
        batcher.cancel(filler).unwrap();

        let mut batched: std::collections::HashMap<u64, Vec<u32>> =
            ids.iter().map(|&id| (id, Vec::new())).collect();
        while batcher.active_count() > 0 {
            let evs = batcher.step().unwrap();
            for ev in evs {
                batched.get_mut(&ev.request_id).unwrap().push(ev.token_id);
            }
        }
        for (i, id) in ids.iter().enumerate() {
            assert_eq!(&batched[id], &seq_out[i], "request {i} token mismatch");
        }
    }

    #[test]
    fn cancel_does_not_corrupt_sibling() {
        let cfg = ModelConfig::tiny();
        let max_seq = 64;
        let params = GenerateParams {
            max_tokens: 4,
            temperature: 0.0,
            top_k: 0,
            eos_token_id: None,
        };

        let mut batcher = ContinuousBatcher::new_untokenized(
            Transformer::synthetic(cfg.clone(), 99).unwrap(),
            max_seq,
            2,
        )
        .unwrap();
        let keep = batcher.admit_tokens(vec![5, 6], params.clone()).unwrap();
        let drop = batcher.admit_tokens(vec![7, 8, 9], params.clone()).unwrap();

        // Shared step, then cancel the sibling mid-flight.
        let first = batcher.step().unwrap();
        assert_eq!(first.len(), 2);
        batcher.cancel(drop).unwrap();
        assert_eq!(batcher.active_count(), 1);

        let mut got: Vec<u32> = first
            .into_iter()
            .filter(|e| e.request_id == keep)
            .map(|e| e.token_id)
            .collect();
        while batcher.active_count() > 0 {
            for ev in batcher.step().unwrap() {
                assert_eq!(ev.request_id, keep);
                got.push(ev.token_id);
            }
        }
        assert_eq!(got.len(), params.max_tokens);

        // Sequential greedy for the kept prompt must match.
        let model2 = Transformer::synthetic(cfg.clone(), 99).unwrap();
        let mut caches = LayerKvCaches::new(&cfg, max_seq).unwrap();
        let mut scratch = ForwardScratch::new(&cfg, max_seq).unwrap();
        let mut tokens = vec![5u32, 6];
        model2
            .forward_tokens(&tokens, 0, &mut caches, &mut scratch)
            .unwrap();
        let mut logits = scratch.logits.data()[..cfg.vocab_size].to_vec();
        let mut expect = Vec::new();
        for _ in 0..params.max_tokens {
            let next = greedy(&logits);
            expect.push(next);
            tokens.push(next);
            let pos = tokens.len() - 1;
            model2
                .forward_tokens(&[next], pos, &mut caches, &mut scratch)
                .unwrap();
            logits = scratch.logits.data()[..cfg.vocab_size].to_vec();
        }
        assert_eq!(got, expect);
    }
}
