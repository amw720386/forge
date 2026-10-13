use super::kv::{GpuKvCache, LayerGpuKvCaches};
use super::paged_kv::{GpuKvBlockPool, PagedKvSession};
use super::weights::{
    upload_model, upload_model_dtype, GpuDecoderWeights, GpuModelWeights, GpuWeightDType,
};
use crate::attention::AttentionConfig;
use crate::compute::{Buffer, ComputeBackend, ComputeError, CudaBackend};
use crate::model::{ModelConfig, Transformer};
use crate::tensor::TensorError;

#[derive(Debug)]
pub enum CudaLoadError {
    Compute(ComputeError),
    QuantizedWeights,
    Tensor(TensorError),
}

impl std::fmt::Display for CudaLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Compute(e) => write!(f, "{e}"),
            Self::QuantizedWeights => write!(f, "model uses quantized weights; load FP32 for CUDA"),
            Self::Tensor(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for CudaLoadError {}

pub struct CudaScratch {
    pub max_seq: usize,
    x: Buffer,
    residual: Buffer,
    normed: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    ctx: Buffer,
    attn_out: Buffer,
    mlp_gate: Buffer,
    mlp_up: Buffer,
    /// Temporary for SiLU(gate) before gate *= up (avoids overlapping borrows).
    mlp_act: Buffer,
    mlp_out: Buffer,
    logits: Buffer,
    token_ids: Buffer,
    /// Reused last-token hidden row for lm_head (avoids per-step alloc).
    last_hidden: Buffer,
    normed_bf16: Buffer,
}

impl CudaScratch {
    pub fn new(
        backend: &dyn ComputeBackend,
        cfg: &ModelConfig,
        max_seq: usize,
    ) -> Result<Self, ComputeError> {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let q_w = cfg.n_heads * cfg.head_dim();
        let kv_w = cfg.n_kv_heads * cfg.head_dim();
        let v = cfg.vocab_size;
        Ok(Self {
            max_seq,
            x: backend.alloc(max_seq * h * 4)?,
            residual: backend.alloc(max_seq * h * 4)?,
            normed: backend.alloc(max_seq * h * 4)?,
            q: backend.alloc(max_seq * q_w * 4)?,
            k: backend.alloc(max_seq * kv_w * 4)?,
            v: backend.alloc(max_seq * kv_w * 4)?,
            ctx: backend.alloc(max_seq * q_w * 4)?,
            attn_out: backend.alloc(max_seq * h * 4)?,
            mlp_gate: backend.alloc(max_seq * i * 4)?,
            mlp_up: backend.alloc(max_seq * i * 4)?,
            mlp_act: backend.alloc(max_seq * i * 4)?,
            mlp_out: backend.alloc(max_seq * h * 4)?,
            logits: backend.alloc(v * 4)?,
            token_ids: backend.alloc(max_seq * 4)?,
            last_hidden: backend.alloc(h * 4)?,
            normed_bf16: backend.alloc(max_seq * h * 2)?,
        })
    }
}

/// Workspace for a shared decode step over up to `max_batch` sequences (one token each).
/// Dense linears use `m=B`; attention uses per-row temps against each request's KV.
pub struct CudaBatchScratch {
    dimensions: [usize; 8],
    pub max_batch: usize,
    pub x: Buffer,
    pub residual: Buffer,
    pub normed: Buffer,
    pub q: Buffer,
    pub k: Buffer,
    pub v: Buffer,
    pub ctx: Buffer,
    pub attn_out: Buffer,
    pub mlp_gate: Buffer,
    pub mlp_up: Buffer,
    pub mlp_act: Buffer,
    pub mlp_out: Buffer,
    /// `[max_batch, vocab]` logits after batched lm_head.
    pub logits: Buffer,
    pub token_ids: Buffer,
    /// Single-row temps for per-request RoPE / KV append / GQA.
    pub q_row: Buffer,
    pub k_row: Buffer,
    pub v_row: Buffer,
    pub ctx_row: Buffer,
    pub metadata: Buffer,
    pub normed_bf16: Buffer,
    graphs: Vec<Option<crate::compute::DecodeGraph>>,
    graph_warmed: Vec<bool>,
    graph_disabled: Vec<bool>,
}

impl CudaBatchScratch {
    pub fn new(
        backend: &dyn ComputeBackend,
        cfg: &ModelConfig,
        max_batch: usize,
    ) -> Result<Self, ComputeError> {
        if max_batch == 0 {
            return Err(ComputeError::InvalidArgument(
                "max_batch must be >= 1".into(),
            ));
        }
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let q_w = cfg.n_heads * cfg.head_dim();
        let kv_w = cfg.n_kv_heads * cfg.head_dim();
        let v = cfg.vocab_size;
        Ok(Self {
            max_batch,
            dimensions: cfg.execution_shape(),
            x: backend.alloc(max_batch * h * 4)?,
            residual: backend.alloc(max_batch * h * 4)?,
            normed: backend.alloc(max_batch * h * 4)?,
            q: backend.alloc(max_batch * q_w * 4)?,
            k: backend.alloc(max_batch * kv_w * 4)?,
            v: backend.alloc(max_batch * kv_w * 4)?,
            ctx: backend.alloc(max_batch * q_w * 4)?,
            attn_out: backend.alloc(max_batch * h * 4)?,
            mlp_gate: backend.alloc(max_batch * i * 4)?,
            mlp_up: backend.alloc(max_batch * i * 4)?,
            mlp_act: backend.alloc(max_batch * i * 4)?,
            mlp_out: backend.alloc(max_batch * h * 4)?,
            logits: backend.alloc(max_batch * v * 4)?,
            token_ids: backend.alloc(max_batch * 4)?,
            q_row: backend.alloc(q_w * 4)?,
            k_row: backend.alloc(kv_w * 4)?,
            v_row: backend.alloc(kv_w * 4)?,
            ctx_row: backend.alloc(q_w * 4)?,
            metadata: backend.alloc(max_batch * 24)?,
            normed_bf16: backend.alloc(max_batch * h * 2)?,
            graphs: (0..=max_batch).map(|_| None).collect(),
            graph_warmed: vec![false; max_batch + 1],
            graph_disabled: vec![false; max_batch + 1],
        })
    }
}

/// Last shared-decode instrumentation (verification).
#[derive(Debug, Clone, Default)]
pub struct CudaBatchStats {
    pub last_batch_size: usize,
    pub last_q_proj_m: usize,
    pub last_q_proj_n: usize,
    pub last_q_proj_k: usize,
    pub last_lm_head_m: usize,
    pub last_lm_head_n: usize,
    pub last_lm_head_k: usize,
    pub decode_steps: u64,
    /// Tokens scheduled as decode in the last step.
    pub last_scheduled_decode: usize,
    /// Tokens scheduled as prefill chunks in the last step.
    pub last_scheduled_prefill: usize,
    pub prefill_chunks: u64,
}

pub struct CudaTransformer {
    pub cfg: ModelConfig,
    pub plan: crate::model::DecoderPlan,
    generation: u64,
    pub attn_cfg: AttentionConfig,
    backend: CudaBackend,
    weights: GpuModelWeights,
    pub weight_dtype: GpuWeightDType,
}

impl CudaTransformer {
    pub fn from_cpu(model: &Transformer, device: i32) -> Result<Self, CudaLoadError> {
        Self::from_cpu_dtype(model, device, GpuWeightDType::Fp32)
    }

    pub fn from_cpu_dtype(
        model: &Transformer,
        device: i32,
        dtype: GpuWeightDType,
    ) -> Result<Self, CudaLoadError> {
        if model.lm_head.weight.is_quantized()
            || model.layers.iter().any(|l| {
                l.weights.q_proj.weight.is_quantized() || l.weights.k_proj.weight.is_quantized()
            })
        {
            return Err(CudaLoadError::QuantizedWeights);
        }
        let plan = crate::model::DecoderPlan::new(&model.cfg).map_err(CudaLoadError::Tensor)?;
        let backend = CudaBackend::try_new(device).map_err(CudaLoadError::Compute)?;
        let weights = match dtype {
            GpuWeightDType::Fp32 => {
                upload_model(&backend, model).map_err(CudaLoadError::Compute)?
            }
            GpuWeightDType::Bf16 => {
                upload_model_dtype(&backend, model, dtype).map_err(CudaLoadError::Compute)?
            }
        };
        let attn_cfg = model.cfg.attention();
        Ok(Self {
            cfg: model.cfg.clone(),
            plan,
            generation: {
                static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
                NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            },
            attn_cfg,
            backend,
            weights,
            weight_dtype: dtype,
        })
    }

    pub fn backend(&self) -> &CudaBackend {
        &self.backend
    }

    pub fn weight_bytes(&self) -> usize {
        self.weights.weight_bytes
    }

    pub fn weight_dtype(&self) -> GpuWeightDType {
        self.weight_dtype
    }

    pub fn device_mem_info(&self) -> (Option<u64>, Option<u64>) {
        let info = self.backend.info();
        (info.free_bytes, info.total_bytes)
    }

    pub fn forward_tokens(
        &self,
        tokens: &[u32],
        position_offset: usize,
        caches: &mut LayerGpuKvCaches,
        scratch: &mut CudaScratch,
    ) -> Result<(), ComputeError> {
        let t = tokens.len();
        if t == 0 || t > scratch.max_seq {
            return Err(ComputeError::InvalidArgument("bad token count".into()));
        }
        let backend = &self.backend as &dyn ComputeBackend;
        let h = self.cfg.hidden_size;
        let vocab = self.cfg.vocab_size;

        for &tok in tokens {
            if tok as usize >= vocab {
                return Err(ComputeError::InvalidArgument(format!(
                    "token id {tok} >= vocab {vocab}"
                )));
            }
        }
        let ids_i32: Vec<i32> = tokens.iter().map(|&x| x as i32).collect();
        backend.copy_h2d(&mut scratch.token_ids, cast_u8(&ids_i32))?;
        self.embed_tokens(backend, &scratch.token_ids, t, &mut scratch.x)?;

        for (layer, cache) in self.weights.layers.iter().zip(caches.layers.iter_mut()) {
            self.decoder_forward(layer, cache, backend, t, position_offset, scratch)?;
        }

        backend.rmsnorm_f32(
            t,
            h,
            &scratch.x,
            &self.weights.final_norm,
            self.cfg.rms_norm_eps,
            &mut scratch.normed,
        )?;

        let src_off = (t - 1) * h * 4;
        backend.copy_d2d(&mut scratch.last_hidden, 0, &scratch.normed, src_off, h * 4)?;

        self.lm_head_apply(
            backend,
            &scratch.last_hidden,
            1,
            h,
            &mut scratch.logits,
            vocab,
        )?;
        Ok(())
    }

    pub fn read_logits(
        &self,
        scratch: &CudaScratch,
        vocab: usize,
    ) -> Result<Vec<f32>, ComputeError> {
        let mut out = vec![0u8; vocab * 4];
        self.backend.copy_d2h(&mut out, &scratch.logits)?;
        Ok(bytes_to_f32(&out))
    }

    /// Read `[rows, vocab]` logits from a batch scratch into a contiguous host vec.
    pub fn read_batch_logits(
        &self,
        scratch: &CudaBatchScratch,
        rows: usize,
        vocab: usize,
    ) -> Result<Vec<f32>, ComputeError> {
        if rows > scratch.max_batch {
            return Err(ComputeError::InvalidArgument("rows > max_batch".into()));
        }
        let mut out = vec![0u8; rows * vocab * 4];
        // copy_d2h copies dst.len() bytes from the start of the device buffer.
        self.backend.copy_d2h(&mut out, &scratch.logits)?;
        Ok(bytes_to_f32(&out))
    }

    /// Device argmax over `[rows, cols]` logits; D2H only the int32 token ids.
    pub fn argmax_logits(
        &self,
        logits: &Buffer,
        rows: usize,
        cols: usize,
    ) -> Result<Vec<u32>, ComputeError> {
        if rows == 0 || cols == 0 {
            return Err(ComputeError::InvalidArgument("argmax dims".into()));
        }
        let mut ids_buf = self.backend.alloc(rows * 4)?;
        self.backend
            .argmax_rows_f32(rows, cols, logits, &mut ids_buf)?;
        let mut host = vec![0u8; rows * 4];
        self.backend.copy_d2h(&mut host, &ids_buf)?;
        Ok(bytes_to_u32(&host))
    }

    /// Greedy next token from single-sequence scratch logits (no full logits D2H).
    pub fn greedy_from_scratch(
        &self,
        scratch: &CudaScratch,
        vocab: usize,
    ) -> Result<u32, ComputeError> {
        let ids = self.argmax_logits(&scratch.logits, 1, vocab)?;
        Ok(ids[0])
    }

    /// Greedy tokens for `rows` batch logits (no full logits D2H).
    pub fn greedy_from_batch_scratch(
        &self,
        scratch: &CudaBatchScratch,
        rows: usize,
        vocab: usize,
    ) -> Result<Vec<u32>, ComputeError> {
        if rows > scratch.max_batch {
            return Err(ComputeError::InvalidArgument("rows > max_batch".into()));
        }
        self.argmax_logits(&scratch.logits, rows, vocab)
    }

    /// Shared decode for `B` independent sequences (one new token each).
    ///
    /// Dense linears / norms / MLP / lm_head use `m=B`. Attention is per-request
    /// against `caches[i]` with RoPE position `positions[i]`.
    pub fn forward_batch_decode(
        &self,
        tokens: &[u32],
        positions: &[usize],
        caches: &mut [&mut LayerGpuKvCaches],
        scratch: &mut CudaBatchScratch,
        stats: &mut CudaBatchStats,
    ) -> Result<(), ComputeError> {
        let b = tokens.len();
        if b == 0 || b > scratch.max_batch {
            return Err(ComputeError::InvalidArgument("bad batch size".into()));
        }
        if positions.len() != b || caches.len() != b {
            return Err(ComputeError::InvalidArgument(
                "tokens/positions/caches length mismatch".into(),
            ));
        }

        let backend = &self.backend as &dyn ComputeBackend;
        let h = self.cfg.hidden_size;
        let vocab = self.cfg.vocab_size;
        let d = self.cfg.head_dim();
        let q_w = self.cfg.n_heads * d;
        let kv_w = self.cfg.n_kv_heads * d;
        let i_sz = self.cfg.intermediate_size;
        let scale = 1.0 / (d as f32).sqrt();

        for &tok in tokens {
            if tok as usize >= vocab {
                return Err(ComputeError::InvalidArgument(format!(
                    "token id {tok} >= vocab {vocab}"
                )));
            }
        }
        let ids_i32: Vec<i32> = tokens.iter().map(|&x| x as i32).collect();
        backend.copy_h2d(&mut scratch.token_ids, cast_u8(&ids_i32))?;
        self.embed_tokens(backend, &scratch.token_ids, b, &mut scratch.x)?;

        stats.last_batch_size = b;
        stats.last_q_proj_m = b;
        stats.last_q_proj_n = q_w;
        stats.last_q_proj_k = h;
        stats.decode_steps = stats.decode_steps.saturating_add(1);

        for layer_i in 0..self.weights.layers.len() {
            let w = &self.weights.layers[layer_i];
            backend.copy_rows_f32(b, h, &scratch.x, &mut scratch.residual)?;
            backend.rmsnorm_f32(
                b,
                h,
                &scratch.x,
                &w.input_layernorm,
                self.cfg.rms_norm_eps,
                &mut scratch.normed,
            )?;

            // Shared batched Q/K/V projections (m=B).
            w.q_proj
                .apply(backend, &scratch.normed, b, h, &mut scratch.q, q_w)?;
            w.k_proj
                .apply(backend, &scratch.normed, b, h, &mut scratch.k, kv_w)?;
            w.v_proj
                .apply(backend, &scratch.normed, b, h, &mut scratch.v, kv_w)?;

            // Per-request RoPE + KV append + GQA (independent cache lengths).
            for row in 0..b {
                let pos = positions[row];
                backend.copy_d2d(&mut scratch.q_row, 0, &scratch.q, row * q_w * 4, q_w * 4)?;
                backend.copy_d2d(&mut scratch.k_row, 0, &scratch.k, row * kv_w * 4, kv_w * 4)?;
                backend.copy_d2d(&mut scratch.v_row, 0, &scratch.v, row * kv_w * 4, kv_w * 4)?;

                backend.rope_f32(
                    1,
                    self.cfg.n_heads,
                    d,
                    &mut scratch.q_row,
                    pos,
                    self.cfg.rope_theta,
                )?;
                backend.rope_f32(
                    1,
                    self.cfg.n_kv_heads,
                    d,
                    &mut scratch.k_row,
                    pos,
                    self.cfg.rope_theta,
                )?;

                let cache = &mut caches[row].layers[layer_i];
                cache.append(backend, &scratch.k_row, &scratch.v_row, 1)?;
                let t_k = cache.len();
                backend.gqa_attn_f32(
                    1,
                    t_k,
                    self.cfg.n_heads,
                    self.cfg.n_kv_heads,
                    d,
                    scale,
                    &scratch.q_row,
                    q_w,
                    cache.k(),
                    kv_w,
                    cache.v(),
                    &mut scratch.ctx_row,
                    false,
                    0,
                )?;
                backend.copy_d2d(
                    &mut scratch.ctx,
                    row * q_w * 4,
                    &scratch.ctx_row,
                    0,
                    q_w * 4,
                )?;
            }

            w.o_proj
                .apply(backend, &scratch.ctx, b, q_w, &mut scratch.attn_out, h)?;
            backend.add_rows_f32(b, h, &scratch.residual, &scratch.attn_out, &mut scratch.x)?;

            // Batched MLP / SwiGLU.
            backend.copy_rows_f32(b, h, &scratch.x, &mut scratch.residual)?;
            backend.rmsnorm_f32(
                b,
                h,
                &scratch.x,
                &w.post_attention_layernorm,
                self.cfg.rms_norm_eps,
                &mut scratch.normed,
            )?;
            w.mlp
                .gate
                .apply(backend, &scratch.normed, b, h, &mut scratch.mlp_gate, i_sz)?;
            w.mlp
                .up
                .apply(backend, &scratch.normed, b, h, &mut scratch.mlp_up, i_sz)?;
            let n = b * i_sz;
            backend.silu_f32(n, &scratch.mlp_gate, &mut scratch.mlp_act)?;
            backend.mul_f32(n, &scratch.mlp_act, &scratch.mlp_up, &mut scratch.mlp_gate)?;
            w.mlp
                .down
                .apply(backend, &scratch.mlp_gate, b, i_sz, &mut scratch.mlp_out, h)?;
            backend.add_rows_f32(b, h, &scratch.residual, &scratch.mlp_out, &mut scratch.x)?;
        }

        backend.rmsnorm_f32(
            b,
            h,
            &scratch.x,
            &self.weights.final_norm,
            self.cfg.rms_norm_eps,
            &mut scratch.normed,
        )?;

        stats.last_lm_head_m = b;
        stats.last_lm_head_n = vocab;
        stats.last_lm_head_k = h;
        self.lm_head_apply(backend, &scratch.normed, b, h, &mut scratch.logits, vocab)?;
        Ok(())
    }

    fn embed_tokens(
        &self,
        backend: &dyn ComputeBackend,
        token_ids: &Buffer,
        t: usize,
        out: &mut Buffer,
    ) -> Result<(), ComputeError> {
        let vocab = self.cfg.vocab_size;
        let h = self.cfg.hidden_size;
        match self.weight_dtype {
            GpuWeightDType::Fp32 => {
                backend.embedding_f32(&self.weights.embed, vocab, h, token_ids, t, out)
            }
            GpuWeightDType::Bf16 => {
                backend.embedding_bf16(&self.weights.embed, vocab, h, token_ids, t, out)
            }
        }
    }

    fn lm_head_apply(
        &self,
        backend: &dyn ComputeBackend,
        x: &Buffer,
        rows: usize,
        x_cols: usize,
        y: &mut Buffer,
        y_cols: usize,
    ) -> Result<(), ComputeError> {
        if let Some(head) = &self.weights.lm_head {
            head.apply(backend, x, rows, x_cols, y, y_cols)
        } else {
            match self.weight_dtype {
                GpuWeightDType::Fp32 => backend.linear_f32(
                    rows,
                    self.cfg.vocab_size,
                    self.cfg.hidden_size,
                    x,
                    x_cols,
                    &self.weights.embed,
                    self.cfg.hidden_size,
                    None,
                    y,
                    y_cols,
                ),
                GpuWeightDType::Bf16 => backend.linear_bf16_w(
                    rows,
                    self.cfg.vocab_size,
                    self.cfg.hidden_size,
                    x,
                    x_cols,
                    &self.weights.embed,
                    self.cfg.hidden_size,
                    None,
                    y,
                    y_cols,
                ),
            }
        }
    }

    fn decoder_forward(
        &self,
        w: &GpuDecoderWeights,
        cache: &mut GpuKvCache,
        backend: &dyn ComputeBackend,
        t: usize,
        position_offset: usize,
        s: &mut CudaScratch,
    ) -> Result<(), ComputeError> {
        let h = self.cfg.hidden_size;
        let i_sz = self.cfg.intermediate_size;
        let d = self.cfg.head_dim();
        let q_w = self.cfg.n_heads * d;
        let kv_w = self.cfg.n_kv_heads * d;
        let scale = 1.0 / (d as f32).sqrt();

        backend.copy_rows_f32(t, h, &s.x, &mut s.residual)?;
        backend.rmsnorm_f32(
            t,
            h,
            &s.x,
            &w.input_layernorm,
            self.cfg.rms_norm_eps,
            &mut s.normed,
        )?;

        w.q_proj.apply(backend, &s.normed, t, h, &mut s.q, q_w)?;
        w.k_proj.apply(backend, &s.normed, t, h, &mut s.k, kv_w)?;
        w.v_proj.apply(backend, &s.normed, t, h, &mut s.v, kv_w)?;

        backend.rope_f32(
            t,
            self.cfg.n_heads,
            d,
            &mut s.q,
            position_offset,
            self.cfg.rope_theta,
        )?;
        backend.rope_f32(
            t,
            self.cfg.n_kv_heads,
            d,
            &mut s.k,
            position_offset,
            self.cfg.rope_theta,
        )?;

        cache.append(backend, &s.k, &s.v, t)?;
        let t_k = cache.len();
        let causal = t > 1;
        backend.gqa_attn_f32(
            t,
            t_k,
            self.cfg.n_heads,
            self.cfg.n_kv_heads,
            d,
            scale,
            &s.q,
            q_w,
            cache.k(),
            kv_w,
            cache.v(),
            &mut s.ctx,
            causal,
            if causal { position_offset } else { 0 },
        )?;

        w.o_proj
            .apply(backend, &s.ctx, t, q_w, &mut s.attn_out, h)?;
        backend.add_rows_f32(t, h, &s.residual, &s.attn_out, &mut s.x)?;

        backend.copy_rows_f32(t, h, &s.x, &mut s.residual)?;
        backend.rmsnorm_f32(
            t,
            h,
            &s.x,
            &w.post_attention_layernorm,
            self.cfg.rms_norm_eps,
            &mut s.normed,
        )?;

        w.mlp
            .gate
            .apply(backend, &s.normed, t, h, &mut s.mlp_gate, i_sz)?;
        w.mlp
            .up
            .apply(backend, &s.normed, t, h, &mut s.mlp_up, i_sz)?;
        let n = t * i_sz;
        // SwiGLU: silu(gate) * up  (distinct buffers â€” no overlapping borrows)
        backend.silu_f32(n, &s.mlp_gate, &mut s.mlp_act)?;
        backend.mul_f32(n, &s.mlp_act, &s.mlp_up, &mut s.mlp_gate)?;
        w.mlp
            .down
            .apply(backend, &s.mlp_gate, t, i_sz, &mut s.mlp_out, h)?;
        backend.add_rows_f32(t, h, &s.residual, &s.mlp_out, &mut s.x)?;
        Ok(())
    }

    /// Prefill or resume a prompt chunk into a paged KV session.
    ///
    /// Writes K/V for `tokens` at absolute positions starting at `position_offset`,
    /// attending to the full prefix already stored in `session` plus the new chunk.
    pub fn forward_tokens_paged(
        &self,
        tokens: &[u32],
        position_offset: usize,
        pool: &mut GpuKvBlockPool,
        session: &mut PagedKvSession,
        scratch: &mut CudaScratch,
    ) -> Result<(), ComputeError> {
        let t = tokens.len();
        if t == 0 || t > scratch.max_seq {
            return Err(ComputeError::InvalidArgument("bad token count".into()));
        }
        let backend = &self.backend as &dyn ComputeBackend;
        let h = self.cfg.hidden_size;
        let vocab = self.cfg.vocab_size;
        let d = self.cfg.head_dim();
        let q_w = self.cfg.n_heads * d;
        let kv_w = self.cfg.n_kv_heads * d;
        let scale = 1.0 / (d as f32).sqrt();

        for &tok in tokens {
            if tok as usize >= vocab {
                return Err(ComputeError::InvalidArgument(format!(
                    "token id {tok} >= vocab {vocab}"
                )));
            }
        }

        session
            .grow_for(pool, t)
            .map_err(|e| ComputeError::OutOfMemory(e.to_string()))?;
        session
            .sync_device_table(backend)
            .map_err(|e| ComputeError::Internal(e.to_string()))?;

        let ids_i32: Vec<i32> = tokens.iter().map(|&x| x as i32).collect();
        backend.copy_h2d(&mut scratch.token_ids, cast_u8(&ids_i32))?;
        self.embed_tokens(backend, &scratch.token_ids, t, &mut scratch.x)?;

        let dst_token_offset = session.len;
        for (layer_i, w) in self.weights.layers.iter().enumerate() {
            self.decoder_forward_paged(
                w,
                layer_i,
                pool,
                session,
                backend,
                t,
                position_offset,
                dst_token_offset,
                scratch,
                d,
                q_w,
                kv_w,
                scale,
            )?;
        }
        session.len = dst_token_offset + t;

        backend.rmsnorm_f32(
            t,
            h,
            &scratch.x,
            &self.weights.final_norm,
            self.cfg.rms_norm_eps,
            &mut scratch.normed,
        )?;

        let src_off = (t - 1) * h * 4;
        backend.copy_d2d(&mut scratch.last_hidden, 0, &scratch.normed, src_off, h * 4)?;
        self.lm_head_apply(
            backend,
            &scratch.last_hidden,
            1,
            h,
            &mut scratch.logits,
            vocab,
        )?;
        Ok(())
    }

    /// Ragged packed prefill: dense operations share all rows; attention retains
    /// each request's exact position, causal prefix and logical block table.
    /// Only completed prompts run LM head. Eager execution keeps decode graphs independent.
    pub fn forward_prefill_paged(
        &self,
        chunks: &[Vec<u32>],
        completed: &[bool],
        pool: &mut GpuKvBlockPool,
        sessions: &mut [&mut PagedKvSession],
        scratch: &mut CudaScratch,
        output: &mut CudaBatchScratch,
    ) -> Result<Vec<usize>, ComputeError> {
        let b = chunks.len();
        let t: usize = chunks.iter().map(Vec::len).sum();
        if b == 0
            || b != sessions.len()
            || b != completed.len()
            || b > output.max_batch
            || t > scratch.max_seq
            || chunks.iter().any(Vec::is_empty)
        {
            return Err(ComputeError::InvalidArgument(
                "invalid packed prefill shape".into(),
            ));
        }
        let backend = &self.backend as &dyn ComputeBackend;
        let h = self.cfg.hidden_size;
        let d = self.cfg.head_dim();
        let qw = self.cfg.n_heads * d;
        let kw = self.cfg.n_kv_heads * d;
        let isz = self.cfg.intermediate_size;
        let ids: Vec<i32> = chunks.iter().flatten().map(|&id| id as i32).collect();
        if chunks
            .iter()
            .flatten()
            .any(|&id| id as usize >= self.cfg.vocab_size)
        {
            return Err(ComputeError::InvalidArgument(
                "prefill token out of vocabulary".into(),
            ));
        }
        if sessions.iter().zip(chunks).any(|(s, c)| {
            s.len
                .checked_add(c.len())
                .filter(|&end| end <= s.max_seq)
                .is_none()
        }) {
            return Err(ComputeError::InvalidArgument(
                "prefill exceeds session capacity".into(),
            ));
        }
        // Preflight total free capacity before mutating any session.
        let profile_kv = std::env::var_os("FORGE_KERNEL_PROFILE").is_some();
        let allocation_start = std::time::Instant::now();
        let needed: usize = sessions
            .iter()
            .zip(chunks)
            .map(|(s, c)| {
                ((s.len + c.len() + pool.block_size - 1) / pool.block_size)
                    .saturating_sub(s.num_logical_blocks())
            })
            .sum();
        if needed > pool.stats().free_blocks {
            return Err(ComputeError::OutOfMemory(
                "packed prefill KV capacity".into(),
            ));
        }
        for (session, chunk) in sessions.iter_mut().zip(chunks) {
            session
                .grow_for(pool, chunk.len())
                .map_err(|e| ComputeError::OutOfMemory(e.to_string()))?;
        }
        if profile_kv {
            eprintln!(
                "[runtime kv] allocation_host_ns={} new_blocks={} requests={}",
                allocation_start.elapsed().as_nanos(),
                needed,
                b
            );
        }
        let table_start = std::time::Instant::now();
        for session in sessions.iter_mut() {
            session
                .sync_device_table(backend)
                .map_err(|e| ComputeError::Internal(e.to_string()))?;
        }
        if profile_kv {
            eprintln!(
                "[runtime kv] table_upload_host_ns={} requests={}",
                table_start.elapsed().as_nanos(),
                b
            );
        }
        backend.copy_h2d(&mut scratch.token_ids, cast_u8(&ids))?;
        self.embed_tokens(backend, &scratch.token_ids, t, &mut scratch.x)?;
        let reuse = self.weight_dtype == GpuWeightDType::Bf16
            && std::env::var("FORGE_REUSE_BF16").as_deref() == Ok("1");
        for (layer, w) in self.weights.layers.iter().enumerate() {
            backend.copy_rows_f32(t, h, &scratch.x, &mut scratch.residual)?;
            backend.rmsnorm_f32(
                t,
                h,
                &scratch.x,
                &w.input_layernorm,
                self.cfg.rms_norm_eps,
                &mut scratch.normed,
            )?;
            if reuse {
                backend.f32_to_bf16(t * h, &scratch.normed, &mut scratch.normed_bf16)?;
                w.q_proj.apply_precast(
                    &self.backend,
                    &scratch.normed_bf16,
                    t,
                    &mut scratch.q,
                    qw,
                )?;
                w.k_proj.apply_precast(
                    &self.backend,
                    &scratch.normed_bf16,
                    t,
                    &mut scratch.k,
                    kw,
                )?;
                w.v_proj.apply_precast(
                    &self.backend,
                    &scratch.normed_bf16,
                    t,
                    &mut scratch.v,
                    kw,
                )?;
            } else {
                w.q_proj
                    .apply(backend, &scratch.normed, t, h, &mut scratch.q, qw)?;
                w.k_proj
                    .apply(backend, &scratch.normed, t, h, &mut scratch.k, kw)?;
                w.v_proj
                    .apply(backend, &scratch.normed, t, h, &mut scratch.v, kw)?;
            }
            let mut row = 0;
            for (chunk, session) in chunks.iter().zip(sessions.iter()) {
                let n = chunk.len();
                let pos = session.len;
                let mut q = scratch.q.view_mut(row * qw * 4, n * qw * 4)?;
                let mut k = scratch.k.view_mut(row * kw * 4, n * kw * 4)?;
                let v = scratch.v.view_mut(row * kw * 4, n * kw * 4)?;
                let mut ctx = scratch.ctx.view_mut(row * qw * 4, n * qw * 4)?;
                backend.rope_f32(n, self.cfg.n_heads, d, &mut q, pos, self.cfg.rope_theta)?;
                backend.rope_f32(n, self.cfg.n_kv_heads, d, &mut k, pos, self.cfg.rope_theta)?;
                let bs = pool.block_size;
                let blocks = pool.num_blocks;
                let layers = pool.n_layers;
                let (kp, vp) = pool.pools_mut();
                backend.paged_kv_append_f32(
                    kw,
                    n,
                    pos,
                    bs,
                    blocks,
                    layers,
                    layer,
                    session.device_table(),
                    session.num_logical_blocks(),
                    &k,
                    &v,
                    kp,
                    vp,
                )?;
                backend.paged_gqa_attn_f32(
                    n,
                    pos + n,
                    self.cfg.n_heads,
                    self.cfg.n_kv_heads,
                    d,
                    1.0 / (d as f32).sqrt(),
                    &q,
                    qw,
                    pool.k_pool(),
                    pool.v_pool(),
                    session.device_table(),
                    session.num_logical_blocks(),
                    bs,
                    blocks,
                    layers,
                    layer,
                    kw,
                    &mut ctx,
                    true,
                    pos,
                )?;
                row += n;
            }
            w.o_proj
                .apply(backend, &scratch.ctx, t, qw, &mut scratch.attn_out, h)?;
            backend.add_rows_f32(t, h, &scratch.residual, &scratch.attn_out, &mut scratch.x)?;
            backend.copy_rows_f32(t, h, &scratch.x, &mut scratch.residual)?;
            backend.rmsnorm_f32(
                t,
                h,
                &scratch.x,
                &w.post_attention_layernorm,
                self.cfg.rms_norm_eps,
                &mut scratch.normed,
            )?;
            if reuse {
                backend.f32_to_bf16(t * h, &scratch.normed, &mut scratch.normed_bf16)?;
                w.mlp.gate.apply_precast(
                    &self.backend,
                    &scratch.normed_bf16,
                    t,
                    &mut scratch.mlp_gate,
                    isz,
                )?;
                w.mlp.up.apply_precast(
                    &self.backend,
                    &scratch.normed_bf16,
                    t,
                    &mut scratch.mlp_up,
                    isz,
                )?;
            } else {
                w.mlp
                    .gate
                    .apply(backend, &scratch.normed, t, h, &mut scratch.mlp_gate, isz)?;
                w.mlp
                    .up
                    .apply(backend, &scratch.normed, t, h, &mut scratch.mlp_up, isz)?;
            }
            backend.silu_f32(t * isz, &scratch.mlp_gate, &mut scratch.mlp_act)?;
            backend.mul_f32(
                t * isz,
                &scratch.mlp_act,
                &scratch.mlp_up,
                &mut scratch.mlp_gate,
            )?;
            w.mlp
                .down
                .apply(backend, &scratch.mlp_gate, t, isz, &mut scratch.mlp_out, h)?;
            backend.add_rows_f32(t, h, &scratch.residual, &scratch.mlp_out, &mut scratch.x)?;
        }
        let mut row = 0;
        let mut finished = Vec::new();
        for (i, (session, chunk)) in sessions.iter_mut().zip(chunks).enumerate() {
            row += chunk.len();
            session.len += chunk.len();
            if completed[i] {
                backend.copy_d2d(
                    &mut output.x,
                    finished.len() * h * 4,
                    &scratch.x,
                    (row - 1) * h * 4,
                    h * 4,
                )?;
                finished.push(i);
            }
        }
        if !finished.is_empty() {
            backend.rmsnorm_f32(
                finished.len(),
                h,
                &output.x,
                &self.weights.final_norm,
                self.cfg.rms_norm_eps,
                &mut output.normed,
            )?;
            self.lm_head_apply(
                backend,
                &output.normed,
                finished.len(),
                h,
                &mut output.logits,
                self.cfg.vocab_size,
            )?;
        }
        Ok(finished)
    }

    /// Shared decode for `B` sequences against a shared paged KV pool.
    pub fn forward_batch_decode_paged(
        &self,
        tokens: &[u32],
        positions: &[usize],
        pool: &mut GpuKvBlockPool,
        sessions: &mut [&mut PagedKvSession],
        scratch: &mut CudaBatchScratch,
        stats: &mut CudaBatchStats,
    ) -> Result<(), ComputeError> {
        if scratch.dimensions != self.cfg.execution_shape() {
            return Err(ComputeError::InvalidArgument(
                "batch scratch belongs to an incompatible model configuration".into(),
            ));
        }
        let b = tokens.len();
        if b == 0 || b > scratch.max_batch {
            return Err(ComputeError::InvalidArgument("bad batch size".into()));
        }
        if positions.len() != b || sessions.len() != b {
            return Err(ComputeError::InvalidArgument(
                "tokens/positions/sessions length mismatch".into(),
            ));
        }

        let backend = &self.backend as &dyn ComputeBackend;
        let h = self.cfg.hidden_size;
        let vocab = self.cfg.vocab_size;
        let d = self.cfg.head_dim();
        let q_w = self.cfg.n_heads * d;
        let kv_w = self.cfg.n_kv_heads * d;
        let i_sz = self.cfg.intermediate_size;
        let scale = 1.0 / (d as f32).sqrt();

        for &tok in tokens {
            if tok as usize >= vocab {
                return Err(ComputeError::InvalidArgument(format!(
                    "token id {tok} >= vocab {vocab}"
                )));
            }
        }
        let ids_i32: Vec<i32> = tokens.iter().map(|&x| x as i32).collect();
        backend.copy_h2d(&mut scratch.token_ids, cast_u8(&ids_i32))?;
        self.embed_tokens(backend, &scratch.token_ids, b, &mut scratch.x)?;

        stats.last_batch_size = b;
        stats.last_q_proj_m = b;
        stats.last_q_proj_n = q_w;
        stats.last_q_proj_k = h;
        stats.decode_steps = stats.decode_steps.saturating_add(1);

        // Grow all sessions for one new token before the layer loop.
        for session in sessions.iter_mut() {
            session
                .grow_for(pool, 1)
                .map_err(|e| ComputeError::OutOfMemory(e.to_string()))?;
            session
                .sync_device_table(backend)
                .map_err(|e| ComputeError::Internal(e.to_string()))?;
        }

        let batched = b <= 8
            && d <= 128
            && d % 2 == 0
            && std::env::var("FORGE_BATCHED_PAGED").as_deref() == Ok("1")
            && std::env::var_os("FORGE_PAGED_ATTN_NAIVE").is_none();
        if batched {
            let mut metadata = Vec::<u64>::with_capacity(b * 3);
            for row in 0..b {
                metadata.extend_from_slice(&[
                    sessions[row].device_table().ptr as u64,
                    positions[row] as u64,
                    sessions[row].len as u64,
                ]);
            }
            backend.copy_h2d(&mut scratch.metadata, cast_u8(&metadata))?;
        }

        let reuse_bf16 = self.weight_dtype == GpuWeightDType::Bf16
            && std::env::var("FORGE_REUSE_BF16").as_deref() == Ok("1");
        // Preserve validated BF16 greedy decisions and avoid launching idle warps
        // for long large batches or short contexts. Re-capture at dispatch changes.
        let precision = if self.weight_dtype == GpuWeightDType::Bf16 {
            crate::model::ExecutionPrecision::CudaBf16
        } else {
            crate::model::ExecutionPrecision::CudaFp32
        };
        let partitioned = std::env::var("FORGE_DECODE_ATTN_BLOCK").as_deref() != Ok("0")
            && precision.partitioned_attention_lengths(b, sessions.iter().map(|s| s.len + 1));
        let graph_enabled = batched
            && std::env::var("FORGE_CUDA_GRAPHS").as_deref() == Ok("1")
            && !scratch.graph_disabled[b];
        let key = crate::model::DecodeGraphKey {
            model: crate::model::GraphModelIdentity::new(&self.cfg, precision, self.generation),
            batch: b,
            kv_block_size: pool.block_size,
            kv_blocks: pool.num_blocks,
            allocations: [
                scratch.x.ptr as usize,
                scratch.metadata.ptr as usize,
                pool.k_pool().ptr as usize,
                pool.v_pool().ptr as usize,
                self.backend.bf16_scratch_address(),
            ],
            workspace: [
                scratch.residual.ptr as usize,
                scratch.normed.ptr as usize,
                scratch.q.ptr as usize,
                scratch.k.ptr as usize,
                scratch.v.ptr as usize,
                scratch.ctx.ptr as usize,
                scratch.attn_out.ptr as usize,
                scratch.mlp_gate.ptr as usize,
                scratch.mlp_up.ptr as usize,
                scratch.mlp_act.ptr as usize,
                scratch.mlp_out.ptr as usize,
                scratch.logits.ptr as usize,
                scratch.token_ids.ptr as usize,
                scratch.q_row.ptr as usize,
                scratch.k_row.ptr as usize,
                scratch.v_row.ptr as usize,
                scratch.ctx_row.ptr as usize,
                scratch.metadata.ptr as usize,
                scratch.normed_bf16.ptr as usize,
            ],
            reuse_bf16,
            gemm_lt: std::env::var("FORGE_GEMM_LT").as_deref() == Ok("1"),
            partitioned_attention: partitioned,
            paged_decode_fp32_kv: true,
        };
        if scratch.graphs[b]
            .as_ref()
            .map(|g| g.key != key)
            .unwrap_or(false)
        {
            scratch.graphs[b] = None;
            scratch.graph_warmed[b] = false;
        }
        let execute = |scratch: &mut CudaBatchScratch,
                       pool: &mut GpuKvBlockPool|
         -> Result<(), ComputeError> {
            for layer_i in 0..self.weights.layers.len() {
                let w = &self.weights.layers[layer_i];
                backend.copy_rows_f32(b, h, &scratch.x, &mut scratch.residual)?;
                backend.rmsnorm_f32(
                    b,
                    h,
                    &scratch.x,
                    &w.input_layernorm,
                    self.cfg.rms_norm_eps,
                    &mut scratch.normed,
                )?;

                if reuse_bf16 {
                    backend.f32_to_bf16(b * h, &scratch.normed, &mut scratch.normed_bf16)?;
                    w.q_proj.apply_precast(
                        &self.backend,
                        &scratch.normed_bf16,
                        b,
                        &mut scratch.q,
                        q_w,
                    )?;
                    w.k_proj.apply_precast(
                        &self.backend,
                        &scratch.normed_bf16,
                        b,
                        &mut scratch.k,
                        kv_w,
                    )?;
                    w.v_proj.apply_precast(
                        &self.backend,
                        &scratch.normed_bf16,
                        b,
                        &mut scratch.v,
                        kv_w,
                    )?;
                } else {
                    w.q_proj
                        .apply(backend, &scratch.normed, b, h, &mut scratch.q, q_w)?;
                    w.k_proj
                        .apply(backend, &scratch.normed, b, h, &mut scratch.k, kv_w)?;
                    w.v_proj
                        .apply(backend, &scratch.normed, b, h, &mut scratch.v, kv_w)?;
                }

                if batched {
                    let bs = pool.block_size;
                    let blocks = pool.num_blocks;
                    let (kp, vp) = pool.pools_mut();
                    self.backend.batch_paged_prepare(
                        b,
                        self.cfg.n_heads,
                        self.cfg.n_kv_heads,
                        d,
                        self.cfg.rope_theta,
                        &scratch.metadata,
                        &mut scratch.q,
                        &scratch.k,
                        &scratch.v,
                        kp,
                        vp,
                        bs,
                        blocks,
                        layer_i,
                    )?;
                    self.backend.batch_paged_attn(
                        b,
                        self.cfg.n_heads,
                        self.cfg.n_kv_heads,
                        d,
                        scale,
                        &scratch.metadata,
                        &scratch.q,
                        kp,
                        vp,
                        &mut scratch.ctx,
                        bs,
                        blocks,
                        layer_i,
                        partitioned,
                    )?;
                } else {
                    for row in 0..b {
                        let pos = positions[row];
                        backend.copy_d2d(
                            &mut scratch.q_row,
                            0,
                            &scratch.q,
                            row * q_w * 4,
                            q_w * 4,
                        )?;
                        backend.copy_d2d(
                            &mut scratch.k_row,
                            0,
                            &scratch.k,
                            row * kv_w * 4,
                            kv_w * 4,
                        )?;
                        backend.copy_d2d(
                            &mut scratch.v_row,
                            0,
                            &scratch.v,
                            row * kv_w * 4,
                            kv_w * 4,
                        )?;

                        backend.rope_f32(
                            1,
                            self.cfg.n_heads,
                            d,
                            &mut scratch.q_row,
                            pos,
                            self.cfg.rope_theta,
                        )?;
                        backend.rope_f32(
                            1,
                            self.cfg.n_kv_heads,
                            d,
                            &mut scratch.k_row,
                            pos,
                            self.cfg.rope_theta,
                        )?;

                        let dst = sessions[row].len;
                        let num_blocks_logical = sessions[row].num_logical_blocks();
                        let block_size = pool.block_size;
                        let num_blocks = pool.num_blocks;
                        let n_layers = pool.n_layers;
                        {
                            let table = sessions[row].device_table();
                            let (k_pool, v_pool) = pool.pools_mut();
                            backend.paged_kv_append_f32(
                                kv_w,
                                1,
                                dst,
                                block_size,
                                num_blocks,
                                n_layers,
                                layer_i,
                                table,
                                num_blocks_logical,
                                &scratch.k_row,
                                &scratch.v_row,
                                k_pool,
                                v_pool,
                            )?;
                        }
                        let t_k = dst + 1;
                        backend.paged_gqa_attn_f32(
                            1,
                            t_k,
                            self.cfg.n_heads,
                            self.cfg.n_kv_heads,
                            d,
                            scale,
                            &scratch.q_row,
                            q_w,
                            pool.k_pool(),
                            pool.v_pool(),
                            sessions[row].device_table(),
                            num_blocks_logical,
                            block_size,
                            num_blocks,
                            n_layers,
                            layer_i,
                            kv_w,
                            &mut scratch.ctx_row,
                            false,
                            0,
                        )?;
                        backend.copy_d2d(
                            &mut scratch.ctx,
                            row * q_w * 4,
                            &scratch.ctx_row,
                            0,
                            q_w * 4,
                        )?;
                    }
                }

                w.o_proj
                    .apply(backend, &scratch.ctx, b, q_w, &mut scratch.attn_out, h)?;
                backend.add_rows_f32(b, h, &scratch.residual, &scratch.attn_out, &mut scratch.x)?;

                backend.copy_rows_f32(b, h, &scratch.x, &mut scratch.residual)?;
                backend.rmsnorm_f32(
                    b,
                    h,
                    &scratch.x,
                    &w.post_attention_layernorm,
                    self.cfg.rms_norm_eps,
                    &mut scratch.normed,
                )?;
                if reuse_bf16 {
                    backend.f32_to_bf16(b * h, &scratch.normed, &mut scratch.normed_bf16)?;
                    w.mlp.gate.apply_precast(
                        &self.backend,
                        &scratch.normed_bf16,
                        b,
                        &mut scratch.mlp_gate,
                        i_sz,
                    )?;
                    w.mlp.up.apply_precast(
                        &self.backend,
                        &scratch.normed_bf16,
                        b,
                        &mut scratch.mlp_up,
                        i_sz,
                    )?;
                } else {
                    w.mlp.gate.apply(
                        backend,
                        &scratch.normed,
                        b,
                        h,
                        &mut scratch.mlp_gate,
                        i_sz,
                    )?;
                    w.mlp
                        .up
                        .apply(backend, &scratch.normed, b, h, &mut scratch.mlp_up, i_sz)?;
                }
                let n = b * i_sz;
                backend.silu_f32(n, &scratch.mlp_gate, &mut scratch.mlp_act)?;
                backend.mul_f32(n, &scratch.mlp_act, &scratch.mlp_up, &mut scratch.mlp_gate)?;
                w.mlp
                    .down
                    .apply(backend, &scratch.mlp_gate, b, i_sz, &mut scratch.mlp_out, h)?;
                backend.add_rows_f32(b, h, &scratch.residual, &scratch.mlp_out, &mut scratch.x)?;
            }
            backend.rmsnorm_f32(
                b,
                h,
                &scratch.x,
                &self.weights.final_norm,
                self.cfg.rms_norm_eps,
                &mut scratch.normed,
            )?;

            self.lm_head_apply(backend, &scratch.normed, b, h, &mut scratch.logits, vocab)?;
            Ok(())
        };
        if graph_enabled && scratch.graphs[b].is_some() {
            scratch.graphs[b].as_ref().unwrap().replay()?;
        } else if graph_enabled && scratch.graph_warmed[b] {
            let free_before_capture = self.backend.info().free_bytes;
            let capture_start = std::time::Instant::now();
            if crate::compute::DecodeGraph::begin().is_ok() {
                match execute(scratch, pool) {
                    Ok(()) => match crate::compute::DecodeGraph::end(key) {
                        Ok(graph) => {
                            let capture_ms = capture_start.elapsed().as_secs_f64() * 1000.0;
                            let free_after_capture = self.backend.info().free_bytes;
                            let overhead = free_before_capture
                                .zip(free_after_capture)
                                .map(|(before, after)| before.saturating_sub(after));
                            eprintln!("CUDA_GRAPH_CAPTURE_OK batch={b} ms={capture_ms:.3} device_memory_delta_bytes={overhead:?}");
                            graph.replay()?;
                            scratch.graphs[b] = Some(graph);
                        }
                        Err(e) => {
                            eprintln!("CUDA_GRAPH_FALLBACK batch={b}: {e}");
                            scratch.graph_disabled[b] = true;
                            execute(scratch, pool)?;
                        }
                    },
                    Err(e) => {
                        crate::compute::DecodeGraph::abort();
                        eprintln!("CUDA_GRAPH_FALLBACK batch={b}: {e}");
                        scratch.graph_disabled[b] = true;
                        execute(scratch, pool)?;
                    }
                }
            } else {
                scratch.graph_disabled[b] = true;
                execute(scratch, pool)?;
            }
        } else {
            execute(scratch, pool)?;
            scratch.graph_warmed[b] = true;
        }
        for session in sessions.iter_mut() {
            session.len += 1;
        }
        stats.last_lm_head_m = b;
        stats.last_lm_head_n = vocab;
        stats.last_lm_head_k = h;
        Ok(())
    }

    fn decoder_forward_paged(
        &self,
        w: &GpuDecoderWeights,
        layer_i: usize,
        pool: &mut GpuKvBlockPool,
        session: &mut PagedKvSession,
        backend: &dyn ComputeBackend,
        t: usize,
        position_offset: usize,
        dst_token_offset: usize,
        s: &mut CudaScratch,
        d: usize,
        q_w: usize,
        kv_w: usize,
        scale: f32,
    ) -> Result<(), ComputeError> {
        let h = self.cfg.hidden_size;
        let i_sz = self.cfg.intermediate_size;

        backend.copy_rows_f32(t, h, &s.x, &mut s.residual)?;
        backend.rmsnorm_f32(
            t,
            h,
            &s.x,
            &w.input_layernorm,
            self.cfg.rms_norm_eps,
            &mut s.normed,
        )?;

        w.q_proj.apply(backend, &s.normed, t, h, &mut s.q, q_w)?;
        w.k_proj.apply(backend, &s.normed, t, h, &mut s.k, kv_w)?;
        w.v_proj.apply(backend, &s.normed, t, h, &mut s.v, kv_w)?;

        backend.rope_f32(
            t,
            self.cfg.n_heads,
            d,
            &mut s.q,
            position_offset,
            self.cfg.rope_theta,
        )?;
        backend.rope_f32(
            t,
            self.cfg.n_kv_heads,
            d,
            &mut s.k,
            position_offset,
            self.cfg.rope_theta,
        )?;

        let num_logical = session.num_logical_blocks();
        let block_size = pool.block_size;
        let num_blocks = pool.num_blocks;
        let n_layers = pool.n_layers;
        {
            let table = session.device_table();
            let (k_pool, v_pool) = pool.pools_mut();
            backend.paged_kv_append_f32(
                kv_w,
                t,
                dst_token_offset,
                block_size,
                num_blocks,
                n_layers,
                layer_i,
                table,
                num_logical,
                &s.k,
                &s.v,
                k_pool,
                v_pool,
            )?;
        }

        let t_k = dst_token_offset + t;
        // Multi-token chunks need causal masking; single-token attends to full prefix.
        let causal = t > 1;
        let causal_off = if causal { position_offset } else { 0 };
        backend.paged_gqa_attn_f32(
            t,
            t_k,
            self.cfg.n_heads,
            self.cfg.n_kv_heads,
            d,
            scale,
            &s.q,
            q_w,
            pool.k_pool(),
            pool.v_pool(),
            session.device_table(),
            num_logical,
            block_size,
            num_blocks,
            n_layers,
            layer_i,
            kv_w,
            &mut s.ctx,
            causal,
            causal_off,
        )?;

        w.o_proj
            .apply(backend, &s.ctx, t, q_w, &mut s.attn_out, h)?;
        backend.add_rows_f32(t, h, &s.residual, &s.attn_out, &mut s.x)?;

        backend.copy_rows_f32(t, h, &s.x, &mut s.residual)?;
        backend.rmsnorm_f32(
            t,
            h,
            &s.x,
            &w.post_attention_layernorm,
            self.cfg.rms_norm_eps,
            &mut s.normed,
        )?;

        w.mlp
            .gate
            .apply(backend, &s.normed, t, h, &mut s.mlp_gate, i_sz)?;
        w.mlp
            .up
            .apply(backend, &s.normed, t, h, &mut s.mlp_up, i_sz)?;
        let n = t * i_sz;
        backend.silu_f32(n, &s.mlp_gate, &mut s.mlp_act)?;
        backend.mul_f32(n, &s.mlp_act, &s.mlp_up, &mut s.mlp_gate)?;
        w.mlp
            .down
            .apply(backend, &s.mlp_gate, t, i_sz, &mut s.mlp_out, h)?;
        backend.add_rows_f32(t, h, &s.residual, &s.mlp_out, &mut s.x)?;
        Ok(())
    }
}

fn cast_u8<T: Copy>(s: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

fn bytes_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn bytes_to_u32(b: &[u8]) -> Vec<u32> {
    b.chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u32)
        .collect()
}
