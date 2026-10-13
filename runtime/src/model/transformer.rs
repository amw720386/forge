use super::config::ModelConfig;
use super::decoder::{DecoderBlock, DecoderWeights};
use super::linear::Linear;
use super::mlp::MlpWeights;
use super::quant::QuantBits;
use crate::attention::AttnScratch;
use crate::cache::KvCache;
use crate::tensor::rows::rmsnorm_rows;
use crate::tensor::{Tensor, TensorError};

#[derive(Clone, Debug)]
pub struct TransformerWeights {
    pub embed: Tensor, // [vocab, H]
    pub layers: Vec<DecoderWeights>,
    pub final_norm: Tensor,      // [H]
    pub lm_head: Option<Tensor>, // [vocab, H] if not tied
}

pub struct LayerKvCaches {
    pub layers: Vec<KvCache>,
}

impl LayerKvCaches {
    pub fn new(cfg: &ModelConfig, max_seq: usize) -> Result<Self, TensorError> {
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for _ in 0..cfg.n_layers {
            layers.push(KvCache::new(cfg.n_kv_heads, cfg.head_dim(), max_seq)?);
        }
        Ok(Self { layers })
    }

    pub fn clear(&mut self) {
        for c in &mut self.layers {
            c.clear();
        }
    }
}

/// Reusable forward workspace — allocate once per request/session.
pub struct ForwardScratch {
    pub x: Tensor,
    pub residual: Tensor,
    pub normed: Tensor,
    pub attn_out: Tensor,
    pub mlp_gate: Tensor,
    pub mlp_up: Tensor,
    pub mlp_out: Tensor,
    pub logits: Tensor,
    pub attn: AttnScratch,
}

impl ForwardScratch {
    pub fn new(cfg: &ModelConfig, max_seq: usize) -> Result<Self, TensorError> {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        Ok(Self {
            x: Tensor::zeros(&[max_seq, h])?,
            residual: Tensor::zeros(&[max_seq, h])?,
            normed: Tensor::zeros(&[max_seq, h])?,
            attn_out: Tensor::zeros(&[max_seq, h])?,
            mlp_gate: Tensor::zeros(&[max_seq, i])?,
            mlp_up: Tensor::zeros(&[max_seq, i])?,
            mlp_out: Tensor::zeros(&[max_seq, h])?,
            // Only the last-token logits are required for autoregressive decode.
            logits: Tensor::zeros(&[1, cfg.vocab_size])?,
            attn: AttnScratch::new(&cfg.attention(), max_seq)?,
        })
    }

    /// Decode-only workspace (one token row) — used after prefill to cut per-request RAM.
    pub fn new_decode(cfg: &ModelConfig) -> Result<Self, TensorError> {
        Self::new(cfg, 1)
    }
}

pub struct Transformer {
    pub plan: super::DecoderPlan,
    pub cfg: ModelConfig,
    pub layers: Vec<DecoderBlock>,
    pub embed: Tensor,
    pub final_norm: Tensor,
    /// Always present; may be a clone of embed when tied. Embeddings stay FP32 for lookup;
    /// `lm_head` may be quantized independently.
    pub lm_head: Linear,
}

impl Transformer {
    pub fn from_weights(cfg: ModelConfig, w: TransformerWeights) -> Result<Self, TensorError> {
        cfg.validate()?;
        for (tensor, expected) in [
            (&w.embed, vec![cfg.vocab_size, cfg.hidden_size]),
            (&w.final_norm, vec![cfg.hidden_size]),
        ] {
            if tensor.shape() != expected {
                return Err(TensorError::ShapeMismatch {
                    expected,
                    got: tensor.shape().to_vec(),
                });
            }
        }
        if let Some(head) = &w.lm_head {
            if head.shape() != [cfg.vocab_size, cfg.hidden_size] {
                return Err(TensorError::ShapeMismatch {
                    expected: vec![cfg.vocab_size, cfg.hidden_size],
                    got: head.shape().to_vec(),
                });
            }
        }
        if w.layers.len() != cfg.n_layers {
            return Err(TensorError::ShapeMismatch {
                expected: vec![cfg.n_layers],
                got: vec![w.layers.len()],
            });
        }
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for lw in w.layers {
            layers.push(DecoderBlock::new(cfg.clone(), lw)?);
        }
        // Share storage when tied — avoids a second 519 MiB FP32 copy of embeddings.
        let lm_tensor = if cfg.tie_word_embeddings {
            w.embed.share()
        } else {
            w.lm_head.ok_or(TensorError::EmptyShape)?
        };
        let lm_head = Linear::new(lm_tensor, None)?;
        Ok(Self {
            plan: super::DecoderPlan::new(&cfg)?,
            cfg,
            layers,
            embed: w.embed,
            final_norm: w.final_norm,
            lm_head,
        })
    }

    /// Deterministic synthetic weights for tests.
    pub fn synthetic(cfg: ModelConfig, seed: u32) -> Result<Self, TensorError> {
        cfg.validate()?;
        let mut rng = SimpleRng::new(seed);
        let h = cfg.hidden_size;
        let embed = random_tensor(&mut rng, &[cfg.vocab_size, h], 0.02);
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for _ in 0..cfg.n_layers {
            layers.push(synthetic_layer(&cfg, &mut rng)?);
        }
        let final_norm = Tensor::filled(&[h], 1.0)?;
        Self::from_weights(
            cfg,
            TransformerWeights {
                embed,
                layers,
                final_norm,
                lm_head: None,
            },
        )
    }

    /// Weight-only quantize all linear projections + lm_head. Norms, biases, embeddings stay FP32.
    pub fn quantize_weights(
        &mut self,
        bits: QuantBits,
        group_size: usize,
    ) -> Result<(), TensorError> {
        for layer in &mut self.layers {
            let w = &mut layer.weights;
            w.q_proj.quantize(bits, group_size)?;
            w.k_proj.quantize(bits, group_size)?;
            w.v_proj.quantize(bits, group_size)?;
            w.o_proj.quantize(bits, group_size)?;
            w.mlp.gate.quantize(bits, group_size)?;
            w.mlp.up.quantize(bits, group_size)?;
            w.mlp.down.quantize(bits, group_size)?;
        }
        self.lm_head.quantize(bits, group_size)?;
        Ok(())
    }

    /// Approximate resident weight bytes (embeddings + norms + linears + biases).
    /// Tied FP32 lm_head sharing embed storage is counted once.
    pub fn weight_nbytes(&self) -> usize {
        let mut n = self.embed.data().len() * 4 + self.final_norm.data().len() * 4;
        let lm_shares_embed = match &self.lm_head.weight {
            super::linear::WeightStorage::Fp32(t) => {
                std::ptr::eq(t.data().as_ptr(), self.embed.data().as_ptr())
            }
            super::linear::WeightStorage::Quant(_) => false,
        };
        if !lm_shares_embed {
            n += self.lm_head.nbytes();
        }
        for layer in &self.layers {
            let w = &layer.weights;
            n += w.input_layernorm.data().len() * 4;
            n += w.post_attention_layernorm.data().len() * 4;
            n += w.q_proj.nbytes()
                + w.k_proj.nbytes()
                + w.v_proj.nbytes()
                + w.o_proj.nbytes()
                + w.mlp.gate.nbytes()
                + w.mlp.up.nbytes()
                + w.mlp.down.nbytes();
        }
        n
    }

    /// Prefill or decode: embed `tokens`, run layers, write logits for last position.
    pub fn forward_tokens(
        &self,
        tokens: &[u32],
        position_offset: usize,
        caches: &mut LayerKvCaches,
        scratch: &mut ForwardScratch,
    ) -> Result<(), TensorError> {
        let t = tokens.len();
        if t == 0 {
            return Err(TensorError::ZeroDimension);
        }
        let h = self.cfg.hidden_size;
        if t > scratch.x.shape()[0] {
            return Err(TensorError::ShapeMismatch {
                expected: vec![t],
                got: vec![scratch.x.shape()[0]],
            });
        }
        if position_offset
            .checked_add(t)
            .is_none_or(|end| end > self.cfg.max_position_embeddings)
        {
            return Err(TensorError::Configuration(
                "token positions exceed model context limit".into(),
            ));
        }
        if caches.layers.len() != self.plan.layers {
            return Err(TensorError::ShapeMismatch {
                expected: vec![self.plan.layers],
                got: vec![caches.layers.len()],
            });
        }

        for (i, &tok) in tokens.iter().enumerate() {
            let id = tok as usize;
            if id >= self.cfg.vocab_size {
                return Err(TensorError::ShapeMismatch {
                    expected: vec![self.cfg.vocab_size],
                    got: vec![id],
                });
            }
            let src = &self.embed.data()[id * h..(id + 1) * h];
            scratch.x.data_mut()[i * h..(i + 1) * h].copy_from_slice(src);
        }

        for (layer, cache) in self.layers.iter().zip(caches.layers.iter_mut()) {
            layer.forward(
                &mut scratch.x,
                t,
                position_offset,
                cache,
                &mut scratch.attn,
                &mut scratch.residual,
                &mut scratch.normed,
                &mut scratch.attn_out,
                &mut scratch.mlp_gate,
                &mut scratch.mlp_up,
                &mut scratch.mlp_out,
            )?;
        }

        {
            if scratch.normed.shape()[0] < t || scratch.normed.shape()[1] != h {
                scratch.normed = Tensor::zeros(&[t.max(1), h])?;
            }
            let x_cols = scratch.x.shape()[1];
            let normed_cols = scratch.normed.shape()[1];
            rmsnorm_rows(
                scratch.x.data(),
                t,
                h,
                x_cols,
                self.final_norm.data(),
                self.cfg.rms_norm_eps,
                scratch.normed.data_mut(),
                normed_cols,
            )?;

            let last_off = (t - 1) * h;
            let vocab = self.cfg.vocab_size;
            if scratch.logits.shape() != [1, vocab] {
                scratch.logits = Tensor::zeros(&[1, vocab])?;
            }
            self.lm_head.apply_rows(
                &scratch.normed.data()[last_off..last_off + h],
                1,
                h,
                h,
                scratch.logits.data_mut(),
                vocab,
            )?;
        }
        Ok(())
    }

    /// Prefill and return logits for every position `[t, vocab]` (test / parity use).
    pub fn forward_logits_all(
        &self,
        tokens: &[u32],
        position_offset: usize,
        caches: &mut LayerKvCaches,
        scratch: &mut ForwardScratch,
    ) -> Result<Tensor, TensorError> {
        let t = tokens.len();
        self.forward_tokens(tokens, position_offset, caches, scratch)?;
        let h = self.cfg.hidden_size;
        let mut logits = Tensor::zeros(&[t, self.cfg.vocab_size])?;
        self.lm_head.apply_rows(
            scratch.normed.data(),
            t,
            h,
            scratch.normed.shape()[1],
            logits.data_mut(),
            self.cfg.vocab_size,
        )?;
        Ok(logits)
    }

    /// Hidden states after layers+final norm for first `t` tokens (for parity tests).
    pub fn forward_hidden(
        &self,
        tokens: &[u32],
        position_offset: usize,
        caches: &mut LayerKvCaches,
        scratch: &mut ForwardScratch,
    ) -> Result<Tensor, TensorError> {
        self.forward_tokens(tokens, position_offset, caches, scratch)?;
        let t = tokens.len();
        let h = self.cfg.hidden_size;
        let mut out = Tensor::zeros(&[t, h])?;
        out.data_mut()[..t * h].copy_from_slice(&scratch.normed.data()[..t * h]);
        Ok(out)
    }
}

fn synthetic_layer(cfg: &ModelConfig, rng: &mut SimpleRng) -> Result<DecoderWeights, TensorError> {
    let h = cfg.hidden_size;
    let i = cfg.intermediate_size;
    let d = cfg.head_dim();
    let q_out = cfg.n_heads * d;
    let kv_out = cfg.n_kv_heads * d;
    Ok(DecoderWeights {
        input_layernorm: Tensor::filled(&[h], 1.0)?,
        post_attention_layernorm: Tensor::filled(&[h], 1.0)?,
        q_proj: Linear::new(
            random_tensor(rng, &[q_out, h], 0.02),
            Some(random_tensor(rng, &[q_out], 0.01)),
        )?,
        k_proj: Linear::new(
            random_tensor(rng, &[kv_out, h], 0.02),
            Some(random_tensor(rng, &[kv_out], 0.01)),
        )?,
        v_proj: Linear::new(
            random_tensor(rng, &[kv_out, h], 0.02),
            Some(random_tensor(rng, &[kv_out], 0.01)),
        )?,
        o_proj: Linear::new(random_tensor(rng, &[h, q_out], 0.02), None)?,
        mlp: MlpWeights {
            gate: Linear::new(random_tensor(rng, &[i, h], 0.02), None)?,
            up: Linear::new(random_tensor(rng, &[i, h], 0.02), None)?,
            down: Linear::new(random_tensor(rng, &[h, i], 0.02), None)?,
        },
    })
}

fn random_tensor(rng: &mut SimpleRng, shape: &[usize], scale: f32) -> Tensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|_| (rng.next_f32() * 2.0 - 1.0) * scale)
        .collect();
    Tensor::from_vec(shape, data).expect("shape")
}

struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u32) -> Self {
        Self {
            state: seed as u64 | 0x9E3779B97F4A7C15,
        }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.state >> 32) as u32
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32) / (u32::MAX as f32)
    }
}
