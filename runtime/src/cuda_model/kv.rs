use crate::compute::{Buffer, ComputeBackend, ComputeError};

pub struct GpuKvCache {
    pub width: usize,
    pub max_seq: usize,
    len: usize,
    k: Buffer,
    v: Buffer,
}

impl GpuKvCache {
    pub fn new(
        backend: &dyn ComputeBackend,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> Result<Self, ComputeError> {
        let width = n_kv_heads * head_dim;
        let bytes = max_seq * width * 4;
        Ok(Self {
            width,
            max_seq,
            len: 0,
            k: backend.alloc(bytes)?,
            v: backend.alloc(bytes)?,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }

    pub fn append(
        &mut self,
        backend: &dyn ComputeBackend,
        k_src: &Buffer,
        v_src: &Buffer,
        t: usize,
    ) -> Result<(), ComputeError> {
        if self.len + t > self.max_seq {
            return Err(ComputeError::InvalidArgument("KV cache overflow".into()));
        }
        backend.kv_append_f32(self.width, t, self.len, k_src, v_src, &mut self.k, &mut self.v)?;
        self.len += t;
        Ok(())
    }

    pub fn k(&self) -> &Buffer {
        &self.k
    }

    pub fn v(&self) -> &Buffer {
        &self.v
    }
}

pub struct LayerGpuKvCaches {
    pub layers: Vec<GpuKvCache>,
}

impl LayerGpuKvCaches {
    pub fn new(
        backend: &dyn ComputeBackend,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> Result<Self, ComputeError> {
        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            layers.push(GpuKvCache::new(backend, n_kv_heads, head_dim, max_seq)?);
        }
        Ok(Self { layers })
    }

    pub fn clear(&mut self) {
        for c in &mut self.layers {
            c.clear();
        }
    }
}
