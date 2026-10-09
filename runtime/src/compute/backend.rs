use std::fmt;

/// Backend identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    Cpu,
    Cuda,
}

#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub kind: BackendKind,
    pub name: String,
    pub index: i32,
    /// Device free / total bytes when known (CUDA).
    pub free_bytes: Option<u64>,
    pub total_bytes: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComputeError {
    Unavailable(String),
    InvalidArgument(String),
    OutOfMemory(String),
    DimensionMismatch,
    Internal(String),
    NotBuilt,
}

impl fmt::Display for ComputeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable(s) => write!(f, "compute unavailable: {s}"),
            Self::InvalidArgument(s) => write!(f, "invalid argument: {s}"),
            Self::OutOfMemory(s) => write!(f, "out of memory: {s}"),
            Self::DimensionMismatch => write!(f, "dimension mismatch"),
            Self::Internal(s) => write!(f, "internal: {s}"),
            Self::NotBuilt => write!(f, "CUDA backend not built (enable --features cuda)"),
        }
    }
}

impl std::error::Error for ComputeError {}

/// Opaque device (or host) buffer owned by a backend.
pub struct Buffer {
    pub(crate) kind: BackendKind,
    pub(crate) ptr: *mut u8,
    pub(crate) bytes: usize,
    pub(crate) free_fn: Option<fn(*mut u8, usize)>,
}

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Drop for Buffer {
    fn drop(&mut self) {
        if let Some(f) = self.free_fn.take() {
            if !self.ptr.is_null() {
                f(self.ptr, self.bytes);
            }
        }
        self.ptr = std::ptr::null_mut();
        self.bytes = 0;
    }
}

impl Buffer {
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn kind(&self) -> BackendKind {
        self.kind
    }

    pub fn as_f32_ptr(&self) -> *const f32 {
        self.ptr as *const f32
    }

    pub fn as_f32_mut_ptr(&mut self) -> *mut f32 {
        self.ptr as *mut f32
    }
}

// A bounds-checked non-owning view whose lifetime keeps its allocation borrowed.
pub(crate) struct BufferView<'a> {
    buffer: Buffer,
    _owner: std::marker::PhantomData<&'a mut Buffer>,
}
impl std::ops::Deref for BufferView<'_> {
    type Target = Buffer;
    fn deref(&self) -> &Buffer {
        &self.buffer
    }
}
impl std::ops::DerefMut for BufferView<'_> {
    fn deref_mut(&mut self) -> &mut Buffer {
        &mut self.buffer
    }
}
impl Buffer {
    pub(crate) fn view_mut(
        &mut self,
        offset: usize,
        bytes: usize,
    ) -> Result<BufferView<'_>, ComputeError> {
        if offset
            .checked_add(bytes)
            .filter(|&end| end <= self.bytes)
            .is_none()
            || self.ptr.is_null()
        {
            return Err(ComputeError::InvalidArgument(
                "buffer view out of bounds".into(),
            ));
        }
        // SAFETY: range checked above; view cannot outlive or reallocate its owner.
        let ptr = unsafe { self.ptr.add(offset) };
        Ok(BufferView {
            buffer: Buffer {
                kind: self.kind,
                ptr,
                bytes,
                free_fn: None,
            },
            _owner: std::marker::PhantomData,
        })
    }
}

/// Minimal compute backend for Phase 8A primitives.
pub trait ComputeBackend: Send + Sync {
    fn kind(&self) -> BackendKind;
    fn info(&self) -> DeviceInfo;
    fn synchronize(&self) -> Result<(), ComputeError>;

    fn alloc(&self, bytes: usize) -> Result<Buffer, ComputeError>;
    fn copy_h2d(&self, dst: &mut Buffer, src: &[u8]) -> Result<(), ComputeError>;
    fn copy_d2h(&self, dst: &mut [u8], src: &Buffer) -> Result<(), ComputeError>;

    /// Device-to-device copy within buffers (CUDA only; CPU uses memcpy when both host).
    fn copy_d2d(
        &self,
        dst: &mut Buffer,
        dst_offset: usize,
        src: &Buffer,
        src_offset: usize,
        bytes: usize,
    ) -> Result<(), ComputeError> {
        let _ = (dst, dst_offset, src, src_offset, bytes);
        Err(ComputeError::Unavailable(
            "copy_d2d not implemented for this backend".into(),
        ))
    }

    fn matmul_f32(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: &Buffer,
        b: &Buffer,
        c: &mut Buffer,
    ) -> Result<(), ComputeError>;

    fn matvec_f32(
        &self,
        m: usize,
        n: usize,
        a: &Buffer,
        x: &Buffer,
        y: &mut Buffer,
    ) -> Result<(), ComputeError>;

    fn rmsnorm_f32(
        &self,
        rows: usize,
        dim: usize,
        x: &Buffer,
        weight: &Buffer,
        eps: f32,
        y: &mut Buffer,
    ) -> Result<(), ComputeError>;

    fn silu_f32(&self, n: usize, x: &Buffer, y: &mut Buffer) -> Result<(), ComputeError>;
    fn add_f32(&self, n: usize, a: &Buffer, b: &Buffer, y: &mut Buffer)
        -> Result<(), ComputeError>;
    fn mul_f32(&self, n: usize, a: &Buffer, b: &Buffer, y: &mut Buffer)
        -> Result<(), ComputeError>;

    fn linear_f32(
        &self,
        m: usize,
        n: usize,
        k: usize,
        x: &Buffer,
        x_cols: usize,
        w: &Buffer,
        w_cols: usize,
        bias: Option<&Buffer>,
        y: &mut Buffer,
        y_cols: usize,
    ) -> Result<(), ComputeError>;

    /// y = x @ W^T + bias with BF16 weights, FP32 activations (CUDA).
    fn linear_bf16_w(
        &self,
        m: usize,
        n: usize,
        k: usize,
        x: &Buffer,
        x_cols: usize,
        w: &Buffer,
        w_cols: usize,
        bias: Option<&Buffer>,
        y: &mut Buffer,
        y_cols: usize,
    ) -> Result<(), ComputeError>;

    /// Convert `n` FP32 values to BF16 on device.
    fn f32_to_bf16(&self, n: usize, src: &Buffer, dst: &mut Buffer) -> Result<(), ComputeError>;

    /// Row-wise argmax over `[rows, cols]` FP32 logits → int32 ids (lowest index on ties).
    fn argmax_rows_f32(
        &self,
        rows: usize,
        cols: usize,
        logits: &Buffer,
        out_ids: &mut Buffer,
    ) -> Result<(), ComputeError>;

    fn rope_f32(
        &self,
        seq_len: usize,
        n_heads: usize,
        head_dim: usize,
        x: &mut Buffer,
        position_offset: usize,
        rope_theta: f32,
    ) -> Result<(), ComputeError>;

    fn embedding_f32(
        &self,
        table: &Buffer,
        vocab: usize,
        hidden: usize,
        token_ids: &Buffer,
        t: usize,
        out: &mut Buffer,
    ) -> Result<(), ComputeError>;

    /// Embedding gather from BF16 table → FP32 out (CUDA).
    fn embedding_bf16(
        &self,
        table: &Buffer,
        vocab: usize,
        hidden: usize,
        token_ids: &Buffer,
        t: usize,
        out: &mut Buffer,
    ) -> Result<(), ComputeError>;

    fn gqa_attn_f32(
        &self,
        t_q: usize,
        t_k: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        scale: f32,
        q: &Buffer,
        q_cols: usize,
        k: &Buffer,
        kv_cols: usize,
        v: &Buffer,
        ctx: &mut Buffer,
        causal: bool,
        causal_q_offset: usize,
    ) -> Result<(), ComputeError>;

    fn copy_rows_f32(
        &self,
        rows: usize,
        cols: usize,
        src: &Buffer,
        dst: &mut Buffer,
    ) -> Result<(), ComputeError>;

    fn add_rows_f32(
        &self,
        rows: usize,
        cols: usize,
        a: &Buffer,
        b: &Buffer,
        y: &mut Buffer,
    ) -> Result<(), ComputeError>;

    fn kv_append_f32(
        &self,
        width: usize,
        t: usize,
        dst_row_offset: usize,
        k_src: &Buffer,
        v_src: &Buffer,
        k_dst: &mut Buffer,
        v_dst: &mut Buffer,
    ) -> Result<(), ComputeError>;

    /// Paged GQA: read K/V via block_table into pool layout
    /// `[n_layers][num_blocks][block_size][kv_width]`.
    fn paged_gqa_attn_f32(
        &self,
        t_q: usize,
        t_k: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        scale: f32,
        q: &Buffer,
        q_cols: usize,
        k_pool: &Buffer,
        v_pool: &Buffer,
        block_table: &Buffer,
        num_logical_blocks: usize,
        block_size: usize,
        num_blocks: usize,
        n_layers: usize,
        layer: usize,
        kv_cols: usize,
        ctx: &mut Buffer,
        causal: bool,
        causal_q_offset: usize,
    ) -> Result<(), ComputeError>;

    /// Naive / reference paged GQA (same signature as [`Self::paged_gqa_attn_f32`]).
    fn paged_gqa_attn_naive_f32(
        &self,
        t_q: usize,
        t_k: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        scale: f32,
        q: &Buffer,
        q_cols: usize,
        k_pool: &Buffer,
        v_pool: &Buffer,
        block_table: &Buffer,
        num_logical_blocks: usize,
        block_size: usize,
        num_blocks: usize,
        n_layers: usize,
        layer: usize,
        kv_cols: usize,
        ctx: &mut Buffer,
        causal: bool,
        causal_q_offset: usize,
    ) -> Result<(), ComputeError>;

    /// Append `t` rows into the paged K/V pool for one layer.
    fn paged_kv_append_f32(
        &self,
        width: usize,
        t: usize,
        dst_token_offset: usize,
        block_size: usize,
        num_blocks: usize,
        n_layers: usize,
        layer: usize,
        block_table: &Buffer,
        num_logical_blocks: usize,
        k_src: &Buffer,
        v_src: &Buffer,
        k_pool: &mut Buffer,
        v_pool: &mut Buffer,
    ) -> Result<(), ComputeError>;
}
