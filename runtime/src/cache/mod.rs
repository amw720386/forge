//! Per-request KV cache.
//!
//! Layout: K and V each `[max_seq, n_kv_heads * head_dim]` contiguous.
//! Positions `[0, len)` are valid.

use crate::tensor::{Tensor, TensorError};

#[derive(Debug)]
pub struct KvCache {
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq: usize,
    len: usize,
    k: Tensor,
    v: Tensor,
}

impl KvCache {
    pub fn new(n_kv_heads: usize, head_dim: usize, max_seq: usize) -> Result<Self, TensorError> {
        if n_kv_heads == 0 || head_dim == 0 || max_seq == 0 {
            return Err(TensorError::ZeroDimension);
        }
        let width = n_kv_heads * head_dim;
        Ok(Self {
            n_kv_heads,
            head_dim,
            max_seq,
            len: 0,
            k: Tensor::zeros(&[max_seq, width])?,
            v: Tensor::zeros(&[max_seq, width])?,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }

    pub fn width(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }

    /// Append `t` rows from contiguous `[t, width]` prefixes (may be larger buffers).
    pub fn append_rows(
        &mut self,
        k_src: &[f32],
        v_src: &[f32],
        t: usize,
    ) -> Result<(), TensorError> {
        let width = self.width();
        if t == 0 || self.len + t > self.max_seq {
            return Err(TensorError::ShapeMismatch {
                expected: vec![t, width],
                got: vec![self.len, self.max_seq],
            });
        }
        if k_src.len() < t * width || v_src.len() < t * width {
            return Err(TensorError::ShapeMismatch {
                expected: vec![t * width],
                got: vec![k_src.len(), v_src.len()],
            });
        }
        let start = self.len * width;
        let end = start + t * width;
        self.k.data_mut()[start..end].copy_from_slice(&k_src[..t * width]);
        self.v.data_mut()[start..end].copy_from_slice(&v_src[..t * width]);
        self.len += t;
        Ok(())
    }

    /// Append `t` new key/value rows. `k_new`/`v_new` shape `[t, n_kv*D]`.
    pub fn append(&mut self, k_new: &Tensor, v_new: &Tensor) -> Result<(), TensorError> {
        if k_new.ndim() != 2 || v_new.ndim() != 2 {
            return Err(TensorError::ShapeMismatch {
                expected: vec![2],
                got: vec![k_new.ndim(), v_new.ndim()],
            });
        }
        let t = k_new.shape()[0];
        let width = self.width();
        if k_new.shape()[1] != width || v_new.shape() != k_new.shape() {
            return Err(TensorError::ShapeMismatch {
                expected: vec![t, width],
                got: k_new.shape().to_vec(),
            });
        }
        self.append_rows(k_new.data(), v_new.data(), t)
    }

    pub fn keys_prefix(&self) -> &[f32] {
        let n = self.len * self.width();
        &self.k.data()[..n]
    }

    pub fn values_prefix(&self) -> &[f32] {
        let n = self.len * self.width();
        &self.v.data()[..n]
    }
}
