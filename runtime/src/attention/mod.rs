//! Causal self-attention with grouped-query attention (GQA).

use crate::cache::KvCache;
use crate::ffi::kernels;
use crate::tensor::ops::softmax_inplace_causal;
use crate::tensor::{Tensor, TensorError};

#[derive(Clone, Debug)]
pub struct AttentionConfig {
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f32,
}

impl AttentionConfig {
    pub fn validate(&self) -> Result<(), TensorError> {
        if self.n_heads == 0 || self.n_kv_heads == 0 || self.head_dim == 0 {
            return Err(TensorError::ZeroDimension);
        }
        if self.n_heads % self.n_kv_heads != 0 {
            return Err(TensorError::ShapeMismatch {
                expected: vec![self.n_kv_heads],
                got: vec![self.n_heads],
            });
        }
        if self.head_dim % 2 != 0 {
            return Err(TensorError::ShapeMismatch {
                expected: vec![2],
                got: vec![self.head_dim],
            });
        }
        Ok(())
    }

    pub fn heads_per_kv(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }

    pub fn kv_index(n_heads: usize, n_kv: usize, q_head: usize) -> usize {
        q_head / (n_heads / n_kv)
    }
}

#[derive(Debug)]
pub struct AttnScratch {
    pub q: Tensor,
    pub k: Tensor,
    pub v: Tensor,
    pub scores: Tensor,
    pub ctx: Tensor,
}

impl AttnScratch {
    pub fn new(cfg: &AttentionConfig, max_seq: usize) -> Result<Self, TensorError> {
        cfg.validate()?;
        let q_w = cfg.n_heads * cfg.head_dim;
        let kv_w = cfg.n_kv_heads * cfg.head_dim;
        Ok(Self {
            q: Tensor::zeros(&[max_seq, q_w])?,
            k: Tensor::zeros(&[max_seq, kv_w])?,
            v: Tensor::zeros(&[max_seq, kv_w])?,
            scores: Tensor::zeros(&[cfg.n_heads, max_seq, max_seq])?,
            ctx: Tensor::zeros(&[max_seq, q_w])?,
        })
    }

    fn resize_scores(&mut self, n_heads: usize, t_q: usize, t_k: usize) -> Result<(), TensorError> {
        if self.scores.shape() != [n_heads, t_q, t_k] {
            self.scores = Tensor::zeros(&[n_heads, t_q, t_k])?;
        }
        Ok(())
    }
}

/// Attend using projected Q/K/V in `scratch` for `t_q` rows.
pub fn attend(
    cfg: &AttentionConfig,
    scratch: &mut AttnScratch,
    t_q: usize,
    cache: &mut KvCache,
    position_offset: usize,
) -> Result<(), TensorError> {
    cfg.validate()?;
    if t_q == 0 {
        return Err(TensorError::ZeroDimension);
    }
    let d = cfg.head_dim;
    let scale = 1.0f32 / (d as f32).sqrt();
    let q_w = cfg.n_heads * d;
    let kv_w = cfg.n_kv_heads * d;

    rope_on_prefix(
        &mut scratch.q,
        t_q,
        cfg.n_heads,
        d,
        position_offset,
        cfg.rope_theta,
    )?;
    rope_on_prefix(
        &mut scratch.k,
        t_q,
        cfg.n_kv_heads,
        d,
        position_offset,
        cfg.rope_theta,
    )?;

    // Append without allocating intermediate tensors.
    cache.append_rows(scratch.k.data(), scratch.v.data(), t_q)?;

    let t_k = cache.len();
    scratch.resize_scores(cfg.n_heads, t_q, t_k)?;

    let q = scratch.q.data();
    let k = cache.keys_prefix();
    let scores = scratch.scores.data_mut();
    let reps = cfg.heads_per_kv();

    for h in 0..cfg.n_heads {
        let kv_h = h / reps;
        for qi in 0..t_q {
            let q_off = qi * q_w + h * d;
            for kj in 0..t_k {
                let k_off = kj * kv_w + kv_h * d;
                let mut dot = 0.0f32;
                for i in 0..d {
                    dot += q[q_off + i] * k[k_off + i];
                }
                scores[h * t_q * t_k + qi * t_k + kj] = dot * scale;
            }
        }
    }

    let use_causal = t_q > 1;
    softmax_inplace_causal(
        &mut scratch.scores,
        cfg.n_heads,
        use_causal,
        if use_causal { position_offset } else { 0 },
    )?;

    let probs = scratch.scores.data();
    let v = cache.values_prefix();
    let ctx = scratch.ctx.data_mut();
    for i in 0..t_q * q_w {
        ctx[i] = 0.0;
    }
    for h in 0..cfg.n_heads {
        let kv_h = h / reps;
        for qi in 0..t_q {
            let out_off = qi * q_w + h * d;
            for kj in 0..t_k {
                let p = probs[h * t_q * t_k + qi * t_k + kj];
                let v_off = kj * kv_w + kv_h * d;
                for i in 0..d {
                    ctx[out_off + i] += p * v[v_off + i];
                }
            }
        }
    }
    Ok(())
}

fn rope_on_prefix(
    buf: &mut Tensor,
    t: usize,
    n_heads: usize,
    head_dim: usize,
    position_offset: usize,
    rope_theta: f32,
) -> Result<(), TensorError> {
    let width = n_heads * head_dim;
    if buf.shape().len() != 2 || buf.shape()[1] != width || t > buf.shape()[0] {
        return Err(TensorError::ShapeMismatch {
            expected: vec![t, width],
            got: buf.shape().to_vec(),
        });
    }
    let status = unsafe {
        kernels::forge_kernels_rope_f32(
            t,
            n_heads,
            head_dim,
            buf.data_mut().as_mut_ptr(),
            std::ptr::null(),
            position_offset,
            rope_theta,
        )
    };
    kernels::check(status).map_err(TensorError::Kernel)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gqa_mapping() {
        assert_eq!(AttentionConfig::kv_index(14, 2, 0), 0);
        assert_eq!(AttentionConfig::kv_index(14, 2, 6), 0);
        assert_eq!(AttentionConfig::kv_index(14, 2, 7), 1);
        assert_eq!(AttentionConfig::kv_index(14, 2, 13), 1);
    }

    #[test]
    fn rejects_bad_gqa() {
        let cfg = AttentionConfig {
            n_heads: 5,
            n_kv_heads: 2,
            head_dim: 4,
            rope_theta: 10000.0,
        };
        assert!(cfg.validate().is_err());
    }
}
