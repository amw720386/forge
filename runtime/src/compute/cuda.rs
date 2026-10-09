//! CUDA compute backend (compiled only with `--features cuda`).

use super::backend::{BackendKind, Buffer, ComputeBackend, ComputeError, DeviceInfo};
use std::env;
use std::ffi::CStr;
use std::os::raw::c_int;

#[repr(C)]
struct ForgeCudaBuffer {
    ptr: *mut u8,
    bytes: usize,
}

const FORGE_CUDA_OK: c_int = 0;
const FORGE_CUDA_ERR_INVALID_ARGUMENT: c_int = 1;
const FORGE_CUDA_ERR_DIMENSION_MISMATCH: c_int = 2;
const FORGE_CUDA_ERR_OUT_OF_MEMORY: c_int = 3;
const FORGE_CUDA_ERR_UNAVAILABLE: c_int = 5;
const FORGE_CUDA_ERR_NOT_BUILT: c_int = 6;

extern "C" {
    fn forge_cuda_timer_start(timer: *mut *mut u8) -> c_int;
    fn forge_cuda_timer_elapsed(timer: *mut u8, ms: *mut f32) -> c_int;
    fn forge_cuda_timer_destroy(timer: *mut u8) -> c_int;
    fn forge_cuda_bf16_scratch_address() -> u64;
    fn forge_cuda_graph_begin() -> c_int;
    fn forge_cuda_graph_end(executable: *mut *mut u8) -> c_int;
    fn forge_cuda_graph_replay(executable: *mut u8) -> c_int;
    fn forge_cuda_graph_destroy(executable: *mut u8) -> c_int;

    fn forge_cuda_linear_bf16_xw(
        m: usize,
        n: usize,
        k: usize,
        x: *const u8,
        ldx: usize,
        w: *const u8,
        ldw: usize,
        bias: *const f32,
        y: *mut f32,
        ldy: usize,
    ) -> c_int;

    fn forge_cuda_batch_paged_prepare_f32(
        batch: usize,
        heads: usize,
        kv_heads: usize,
        dim: usize,
        theta: f32,
        metadata: *const u64,
        q: *mut f32,
        k: *const f32,
        v: *const f32,
        kp: *mut f32,
        vp: *mut f32,
        bs: usize,
        blocks: usize,
        layer: usize,
    ) -> c_int;
    fn forge_cuda_batch_paged_attn_dispatch_f32(
        batch: usize,
        heads: usize,
        kv_heads: usize,
        dim: usize,
        scale: f32,
        metadata: *const u64,
        q: *const f32,
        kp: *const f32,
        vp: *const f32,
        ctx: *mut f32,
        bs: usize,
        blocks: usize,
        layer: usize,
        partitioned: c_int,
    ) -> c_int;

    fn forge_cuda_version() -> *const i8;
    fn forge_cuda_is_available() -> c_int;
    fn forge_cuda_set_device(device: c_int) -> c_int;
    fn forge_cuda_device_name(device: c_int, out: *mut i8, out_len: usize) -> c_int;
    fn forge_cuda_mem_info(free_bytes: *mut usize, total_bytes: *mut usize) -> c_int;
    fn forge_cuda_last_error() -> *const i8;
    fn forge_cuda_malloc(buf: *mut ForgeCudaBuffer, bytes: usize) -> c_int;
    fn forge_cuda_free(buf: *mut ForgeCudaBuffer) -> c_int;
    fn forge_cuda_memcpy_h2d(dst: *mut ForgeCudaBuffer, src: *const u8, bytes: usize) -> c_int;
    fn forge_cuda_memcpy_d2h(dst: *mut u8, src: *const ForgeCudaBuffer, bytes: usize) -> c_int;
    fn forge_cuda_memcpy_d2d(
        dst: *mut ForgeCudaBuffer,
        dst_off: usize,
        src: *const ForgeCudaBuffer,
        src_off: usize,
        bytes: usize,
    ) -> c_int;
    fn forge_cuda_synchronize() -> c_int;
    fn forge_cuda_matmul_f32(
        m: usize,
        n: usize,
        k: usize,
        a: *const f32,
        lda: usize,
        b: *const f32,
        ldb: usize,
        c: *mut f32,
        ldc: usize,
    ) -> c_int;
    fn forge_cuda_matvec_f32(
        m: usize,
        n: usize,
        a: *const f32,
        lda: usize,
        x: *const f32,
        y: *mut f32,
    ) -> c_int;
    fn forge_cuda_rmsnorm_f32(
        rows: usize,
        dim: usize,
        x: *const f32,
        ldx: usize,
        weight: *const f32,
        eps: f32,
        y: *mut f32,
        ldy: usize,
    ) -> c_int;
    fn forge_cuda_silu_f32(n: usize, x: *const f32, y: *mut f32) -> c_int;
    fn forge_cuda_add_f32(n: usize, a: *const f32, b: *const f32, y: *mut f32) -> c_int;
    fn forge_cuda_mul_f32(n: usize, a: *const f32, b: *const f32, y: *mut f32) -> c_int;
    fn forge_cuda_linear_f32(
        m: usize,
        n: usize,
        k: usize,
        x: *const f32,
        ldx: usize,
        w: *const f32,
        ldw: usize,
        bias: *const f32,
        y: *mut f32,
        ldy: usize,
    ) -> c_int;
    fn forge_cuda_f32_to_bf16(n: usize, src: *const f32, dst_bf16: *mut u8) -> c_int;
    fn forge_cuda_linear_bf16_w(
        m: usize,
        n: usize,
        k: usize,
        x: *const f32,
        ldx: usize,
        w_bf16: *const u8,
        ldw: usize,
        bias: *const f32,
        y: *mut f32,
        ldy: usize,
    ) -> c_int;
    fn forge_cuda_argmax_rows_f32(
        rows: usize,
        cols: usize,
        logits: *const f32,
        out_ids: *mut i32,
    ) -> c_int;
    fn forge_cuda_rope_f32(
        seq_len: usize,
        n_heads: usize,
        head_dim: usize,
        x: *mut f32,
        position_offset: usize,
        rope_theta: f32,
    ) -> c_int;
    fn forge_cuda_embedding_f32(
        embed: *const f32,
        vocab: usize,
        hidden: usize,
        token_ids: *const i32,
        t: usize,
        out: *mut f32,
    ) -> c_int;
    fn forge_cuda_embedding_bf16(
        embed_bf16: *const u8,
        vocab: usize,
        hidden: usize,
        token_ids: *const i32,
        t: usize,
        out: *mut f32,
    ) -> c_int;
    fn forge_cuda_gqa_attn_f32(
        t_q: usize,
        t_k: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        scale: f32,
        q: *const f32,
        q_w: usize,
        k: *const f32,
        kv_w: usize,
        v: *const f32,
        ctx: *mut f32,
        causal: c_int,
        causal_q_offset: usize,
    ) -> c_int;
    fn forge_cuda_copy_rows_f32(rows: usize, cols: usize, src: *const f32, dst: *mut f32) -> c_int;
    fn forge_cuda_add_rows_f32(
        rows: usize,
        cols: usize,
        a: *const f32,
        b: *const f32,
        y: *mut f32,
    ) -> c_int;
    fn forge_cuda_kv_append_f32(
        width: usize,
        t: usize,
        dst_row_offset: usize,
        k_src: *const f32,
        v_src: *const f32,
        k_dst: *mut f32,
        v_dst: *mut f32,
    ) -> c_int;
    fn forge_cuda_paged_gqa_attn_f32(
        t_q: usize,
        t_k: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        scale: f32,
        q: *const f32,
        q_w: usize,
        k_pool: *const f32,
        v_pool: *const f32,
        block_table: *const u32,
        num_logical_blocks: usize,
        block_size: usize,
        num_blocks: usize,
        n_layers: usize,
        layer: usize,
        kv_w: usize,
        ctx: *mut f32,
        causal: c_int,
        causal_q_offset: usize,
    ) -> c_int;
    fn forge_cuda_paged_gqa_attn_naive_f32(
        t_q: usize,
        t_k: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        scale: f32,
        q: *const f32,
        q_w: usize,
        k_pool: *const f32,
        v_pool: *const f32,
        block_table: *const u32,
        num_logical_blocks: usize,
        block_size: usize,
        num_blocks: usize,
        n_layers: usize,
        layer: usize,
        kv_w: usize,
        ctx: *mut f32,
        causal: c_int,
        causal_q_offset: usize,
    ) -> c_int;
    fn forge_cuda_paged_kv_append_f32(
        width: usize,
        t: usize,
        dst_token_offset: usize,
        block_size: usize,
        num_blocks: usize,
        n_layers: usize,
        layer: usize,
        block_table: *const u32,
        num_logical_blocks: usize,
        k_src: *const f32,
        v_src: *const f32,
        k_pool: *mut f32,
        v_pool: *mut f32,
    ) -> c_int;
}

fn last_err() -> String {
    unsafe {
        let p = forge_cuda_last_error();
        if p.is_null() {
            return String::new();
        }
        CStr::from_ptr(p).to_string_lossy().into_owned()
    }
}

fn map(st: c_int) -> Result<(), ComputeError> {
    match st {
        FORGE_CUDA_OK => Ok(()),
        FORGE_CUDA_ERR_INVALID_ARGUMENT => Err(ComputeError::InvalidArgument(last_err())),
        FORGE_CUDA_ERR_DIMENSION_MISMATCH => Err(ComputeError::DimensionMismatch),
        FORGE_CUDA_ERR_OUT_OF_MEMORY => Err(ComputeError::OutOfMemory(last_err())),
        FORGE_CUDA_ERR_UNAVAILABLE => Err(ComputeError::Unavailable(last_err())),
        FORGE_CUDA_ERR_NOT_BUILT => Err(ComputeError::NotBuilt),
        other => Err(ComputeError::Internal(format!(
            "cuda status {other}: {}",
            last_err()
        ))),
    }
}

fn use_naive_paged_attn() -> bool {
    env::var_os("FORGE_PAGED_ATTN_NAIVE").is_some()
}

fn cuda_free_buf(ptr: *mut u8, bytes: usize) {
    let mut b = ForgeCudaBuffer { ptr, bytes };
    unsafe {
        let _ = forge_cuda_free(&mut b);
    }
}

pub struct CudaBackend {
    device: i32,
    name: String,
}

impl CudaBackend {
    pub(crate) fn bf16_scratch_address(&self) -> usize {
        unsafe { forge_cuda_bf16_scratch_address() as usize }
    }
    pub(crate) fn linear_bf16_xw(
        &self,
        m: usize,
        n: usize,
        k: usize,
        x: &Buffer,
        w: &Buffer,
        bias: Option<&Buffer>,
        y: &mut Buffer,
        ldy: usize,
    ) -> Result<(), ComputeError> {
        map(unsafe {
            forge_cuda_linear_bf16_xw(
                m,
                n,
                k,
                x.ptr,
                k,
                w.ptr,
                k,
                bias.map(|b| b.as_f32_ptr()).unwrap_or(std::ptr::null()),
                y.as_f32_mut_ptr(),
                ldy,
            )
        })
    }

    pub(crate) fn batch_paged_prepare(
        &self,
        b: usize,
        heads: usize,
        kvheads: usize,
        d: usize,
        theta: f32,
        metadata: &Buffer,
        q: &mut Buffer,
        k: &Buffer,
        v: &Buffer,
        kp: &mut Buffer,
        vp: &mut Buffer,
        bs: usize,
        blocks: usize,
        layer: usize,
    ) -> Result<(), ComputeError> {
        map(unsafe {
            forge_cuda_batch_paged_prepare_f32(
                b,
                heads,
                kvheads,
                d,
                theta,
                metadata.ptr.cast(),
                q.as_f32_mut_ptr(),
                k.as_f32_ptr(),
                v.as_f32_ptr(),
                kp.as_f32_mut_ptr(),
                vp.as_f32_mut_ptr(),
                bs,
                blocks,
                layer,
            )
        })
    }
    pub(crate) fn batch_paged_attn(
        &self,
        b: usize,
        heads: usize,
        kvheads: usize,
        d: usize,
        scale: f32,
        metadata: &Buffer,
        q: &Buffer,
        kp: &Buffer,
        vp: &Buffer,
        ctx: &mut Buffer,
        bs: usize,
        blocks: usize,
        layer: usize,
        partitioned: bool,
    ) -> Result<(), ComputeError> {
        map(unsafe {
            forge_cuda_batch_paged_attn_dispatch_f32(
                b,
                heads,
                kvheads,
                d,
                scale,
                metadata.ptr.cast(),
                q.as_f32_ptr(),
                kp.as_f32_ptr(),
                vp.as_f32_ptr(),
                ctx.as_f32_mut_ptr(),
                bs,
                blocks,
                layer,
                partitioned as c_int,
            )
        })
    }

    pub fn try_new(device: i32) -> Result<Self, ComputeError> {
        unsafe {
            if forge_cuda_is_available() == 0 {
                return Err(ComputeError::Unavailable(
                    "no CUDA device (or driver)".into(),
                ));
            }
            map(forge_cuda_set_device(device))?;
            let mut name_buf = [0i8; 256];
            map(forge_cuda_device_name(
                device,
                name_buf.as_mut_ptr(),
                name_buf.len(),
            ))?;
            let name = CStr::from_ptr(name_buf.as_ptr())
                .to_string_lossy()
                .into_owned();
            let _ = forge_cuda_version();
            Ok(Self { device, name })
        }
    }
}

impl ComputeBackend for CudaBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Cuda
    }

    fn info(&self) -> DeviceInfo {
        let mut free_b = 0usize;
        let mut total_b = 0usize;
        let (free_bytes, total_bytes) = unsafe {
            if forge_cuda_mem_info(&mut free_b, &mut total_b) == FORGE_CUDA_OK {
                (Some(free_b as u64), Some(total_b as u64))
            } else {
                (None, None)
            }
        };
        DeviceInfo {
            kind: BackendKind::Cuda,
            name: self.name.clone(),
            index: self.device,
            free_bytes,
            total_bytes,
        }
    }

    fn synchronize(&self) -> Result<(), ComputeError> {
        map(unsafe { forge_cuda_synchronize() })
    }

    fn alloc(&self, bytes: usize) -> Result<Buffer, ComputeError> {
        let mut b = ForgeCudaBuffer {
            ptr: std::ptr::null_mut(),
            bytes: 0,
        };
        map(unsafe { forge_cuda_malloc(&mut b, bytes) })?;
        Ok(Buffer {
            kind: BackendKind::Cuda,
            ptr: b.ptr,
            bytes: b.bytes,
            free_fn: Some(cuda_free_buf),
        })
    }

    fn copy_h2d(&self, dst: &mut Buffer, src: &[u8]) -> Result<(), ComputeError> {
        if dst.kind != BackendKind::Cuda {
            return Err(ComputeError::InvalidArgument("cuda buffer required".into()));
        }
        let mut b = ForgeCudaBuffer {
            ptr: dst.ptr,
            bytes: dst.bytes,
        };
        map(unsafe { forge_cuda_memcpy_h2d(&mut b, src.as_ptr(), src.len()) })
    }

    fn copy_d2h(&self, dst: &mut [u8], src: &Buffer) -> Result<(), ComputeError> {
        if src.kind != BackendKind::Cuda {
            return Err(ComputeError::InvalidArgument("cuda buffer required".into()));
        }
        let b = ForgeCudaBuffer {
            ptr: src.ptr,
            bytes: src.bytes,
        };
        let profile = env::var_os("FORGE_SERVING_PROFILE").is_some();
        let started = std::time::Instant::now();
        let result = map(unsafe { forge_cuda_memcpy_d2h(dst.as_mut_ptr(), &b, dst.len()) });
        if profile {
            eprintln!(
                "[phase14 d2h] bytes={} host_ns={}",
                dst.len(),
                started.elapsed().as_nanos()
            );
        }
        result
    }

    fn copy_d2d(
        &self,
        dst: &mut Buffer,
        dst_offset: usize,
        src: &Buffer,
        src_offset: usize,
        bytes: usize,
    ) -> Result<(), ComputeError> {
        if dst.kind != BackendKind::Cuda || src.kind != BackendKind::Cuda {
            return Err(ComputeError::InvalidArgument(
                "cuda buffers required".into(),
            ));
        }
        let mut db = ForgeCudaBuffer {
            ptr: dst.ptr,
            bytes: dst.bytes,
        };
        let sb = ForgeCudaBuffer {
            ptr: src.ptr,
            bytes: src.bytes,
        };
        map(unsafe { forge_cuda_memcpy_d2d(&mut db, dst_offset, &sb, src_offset, bytes) })
    }

    fn matmul_f32(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: &Buffer,
        b: &Buffer,
        c: &mut Buffer,
    ) -> Result<(), ComputeError> {
        map(unsafe {
            forge_cuda_matmul_f32(
                m,
                n,
                k,
                a.as_f32_ptr(),
                k,
                b.as_f32_ptr(),
                n,
                c.as_f32_mut_ptr(),
                n,
            )
        })
    }

    fn matvec_f32(
        &self,
        m: usize,
        n: usize,
        a: &Buffer,
        x: &Buffer,
        y: &mut Buffer,
    ) -> Result<(), ComputeError> {
        map(unsafe {
            forge_cuda_matvec_f32(m, n, a.as_f32_ptr(), n, x.as_f32_ptr(), y.as_f32_mut_ptr())
        })
    }

    fn rmsnorm_f32(
        &self,
        rows: usize,
        dim: usize,
        x: &Buffer,
        weight: &Buffer,
        eps: f32,
        y: &mut Buffer,
    ) -> Result<(), ComputeError> {
        map(unsafe {
            forge_cuda_rmsnorm_f32(
                rows,
                dim,
                x.as_f32_ptr(),
                dim,
                weight.as_f32_ptr(),
                eps,
                y.as_f32_mut_ptr(),
                dim,
            )
        })
    }

    fn silu_f32(&self, n: usize, x: &Buffer, y: &mut Buffer) -> Result<(), ComputeError> {
        map(unsafe { forge_cuda_silu_f32(n, x.as_f32_ptr(), y.as_f32_mut_ptr()) })
    }

    fn add_f32(
        &self,
        n: usize,
        a: &Buffer,
        b: &Buffer,
        y: &mut Buffer,
    ) -> Result<(), ComputeError> {
        map(unsafe { forge_cuda_add_f32(n, a.as_f32_ptr(), b.as_f32_ptr(), y.as_f32_mut_ptr()) })
    }

    fn mul_f32(
        &self,
        n: usize,
        a: &Buffer,
        b: &Buffer,
        y: &mut Buffer,
    ) -> Result<(), ComputeError> {
        map(unsafe { forge_cuda_mul_f32(n, a.as_f32_ptr(), b.as_f32_ptr(), y.as_f32_mut_ptr()) })
    }

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
    ) -> Result<(), ComputeError> {
        let bias_ptr = bias.map(|b| b.as_f32_ptr()).unwrap_or(std::ptr::null());
        map(unsafe {
            forge_cuda_linear_f32(
                m,
                n,
                k,
                x.as_f32_ptr(),
                x_cols,
                w.as_f32_ptr(),
                w_cols,
                bias_ptr,
                y.as_f32_mut_ptr(),
                y_cols,
            )
        })
    }

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
    ) -> Result<(), ComputeError> {
        let bias_ptr = bias.map(|b| b.as_f32_ptr()).unwrap_or(std::ptr::null());
        map(unsafe {
            forge_cuda_linear_bf16_w(
                m,
                n,
                k,
                x.as_f32_ptr(),
                x_cols,
                w.ptr,
                w_cols,
                bias_ptr,
                y.as_f32_mut_ptr(),
                y_cols,
            )
        })
    }

    fn f32_to_bf16(&self, n: usize, src: &Buffer, dst: &mut Buffer) -> Result<(), ComputeError> {
        map(unsafe { forge_cuda_f32_to_bf16(n, src.as_f32_ptr(), dst.ptr) })
    }

    fn argmax_rows_f32(
        &self,
        rows: usize,
        cols: usize,
        logits: &Buffer,
        out_ids: &mut Buffer,
    ) -> Result<(), ComputeError> {
        map(unsafe {
            forge_cuda_argmax_rows_f32(rows, cols, logits.as_f32_ptr(), out_ids.ptr as *mut i32)
        })
    }

    fn rope_f32(
        &self,
        seq_len: usize,
        n_heads: usize,
        head_dim: usize,
        x: &mut Buffer,
        position_offset: usize,
        rope_theta: f32,
    ) -> Result<(), ComputeError> {
        map(unsafe {
            forge_cuda_rope_f32(
                seq_len,
                n_heads,
                head_dim,
                x.as_f32_mut_ptr(),
                position_offset,
                rope_theta,
            )
        })
    }

    fn embedding_f32(
        &self,
        table: &Buffer,
        vocab: usize,
        hidden: usize,
        token_ids: &Buffer,
        t: usize,
        out: &mut Buffer,
    ) -> Result<(), ComputeError> {
        map(unsafe {
            forge_cuda_embedding_f32(
                table.as_f32_ptr(),
                vocab,
                hidden,
                token_ids.ptr as *const i32,
                t,
                out.as_f32_mut_ptr(),
            )
        })
    }

    fn embedding_bf16(
        &self,
        table: &Buffer,
        vocab: usize,
        hidden: usize,
        token_ids: &Buffer,
        t: usize,
        out: &mut Buffer,
    ) -> Result<(), ComputeError> {
        map(unsafe {
            forge_cuda_embedding_bf16(
                table.ptr,
                vocab,
                hidden,
                token_ids.ptr as *const i32,
                t,
                out.as_f32_mut_ptr(),
            )
        })
    }

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
    ) -> Result<(), ComputeError> {
        map(unsafe {
            forge_cuda_gqa_attn_f32(
                t_q,
                t_k,
                n_heads,
                n_kv_heads,
                head_dim,
                scale,
                q.as_f32_ptr(),
                q_cols,
                k.as_f32_ptr(),
                kv_cols,
                v.as_f32_ptr(),
                ctx.as_f32_mut_ptr(),
                if causal { 1 } else { 0 },
                causal_q_offset,
            )
        })
    }

    fn copy_rows_f32(
        &self,
        rows: usize,
        cols: usize,
        src: &Buffer,
        dst: &mut Buffer,
    ) -> Result<(), ComputeError> {
        map(unsafe { forge_cuda_copy_rows_f32(rows, cols, src.as_f32_ptr(), dst.as_f32_mut_ptr()) })
    }

    fn add_rows_f32(
        &self,
        rows: usize,
        cols: usize,
        a: &Buffer,
        b: &Buffer,
        y: &mut Buffer,
    ) -> Result<(), ComputeError> {
        map(unsafe {
            forge_cuda_add_rows_f32(
                rows,
                cols,
                a.as_f32_ptr(),
                b.as_f32_ptr(),
                y.as_f32_mut_ptr(),
            )
        })
    }

    fn kv_append_f32(
        &self,
        width: usize,
        t: usize,
        dst_row_offset: usize,
        k_src: &Buffer,
        v_src: &Buffer,
        k_dst: &mut Buffer,
        v_dst: &mut Buffer,
    ) -> Result<(), ComputeError> {
        map(unsafe {
            forge_cuda_kv_append_f32(
                width,
                t,
                dst_row_offset,
                k_src.as_f32_ptr(),
                v_src.as_f32_ptr(),
                k_dst.as_f32_mut_ptr(),
                v_dst.as_f32_mut_ptr(),
            )
        })
    }

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
    ) -> Result<(), ComputeError> {
        if use_naive_paged_attn() {
            return self.paged_gqa_attn_naive_f32(
                t_q,
                t_k,
                n_heads,
                n_kv_heads,
                head_dim,
                scale,
                q,
                q_cols,
                k_pool,
                v_pool,
                block_table,
                num_logical_blocks,
                block_size,
                num_blocks,
                n_layers,
                layer,
                kv_cols,
                ctx,
                causal,
                causal_q_offset,
            );
        }
        map(unsafe {
            forge_cuda_paged_gqa_attn_f32(
                t_q,
                t_k,
                n_heads,
                n_kv_heads,
                head_dim,
                scale,
                q.as_f32_ptr(),
                q_cols,
                k_pool.as_f32_ptr(),
                v_pool.as_f32_ptr(),
                block_table.ptr as *const u32,
                num_logical_blocks,
                block_size,
                num_blocks,
                n_layers,
                layer,
                kv_cols,
                ctx.as_f32_mut_ptr(),
                if causal { 1 } else { 0 },
                causal_q_offset,
            )
        })
    }

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
    ) -> Result<(), ComputeError> {
        map(unsafe {
            forge_cuda_paged_gqa_attn_naive_f32(
                t_q,
                t_k,
                n_heads,
                n_kv_heads,
                head_dim,
                scale,
                q.as_f32_ptr(),
                q_cols,
                k_pool.as_f32_ptr(),
                v_pool.as_f32_ptr(),
                block_table.ptr as *const u32,
                num_logical_blocks,
                block_size,
                num_blocks,
                n_layers,
                layer,
                kv_cols,
                ctx.as_f32_mut_ptr(),
                if causal { 1 } else { 0 },
                causal_q_offset,
            )
        })
    }

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
    ) -> Result<(), ComputeError> {
        map(unsafe {
            forge_cuda_paged_kv_append_f32(
                width,
                t,
                dst_token_offset,
                block_size,
                num_blocks,
                n_layers,
                layer,
                block_table.ptr as *const u32,
                num_logical_blocks,
                k_src.as_f32_ptr(),
                v_src.as_f32_ptr(),
                k_pool.as_f32_mut_ptr(),
                v_pool.as_f32_mut_ptr(),
            )
        })
    }
}

/// Owns graph executable only. Captured memory is owned by batch scratch,
/// model and KV pool; replay checks their identities before use.
pub(crate) struct DecodeGraph {
    handle: usize,
    pub key: crate::model::DecodeGraphKey,
}
impl Drop for DecodeGraph {
    fn drop(&mut self) {
        unsafe {
            forge_cuda_graph_destroy(self.handle as *mut u8);
        }
    }
}
impl DecodeGraph {
    pub fn begin() -> Result<(), ComputeError> {
        map(unsafe { forge_cuda_graph_begin() })
    }
    pub fn end(key: crate::model::DecodeGraphKey) -> Result<Self, ComputeError> {
        let mut ptr = std::ptr::null_mut();
        map(unsafe { forge_cuda_graph_end(&mut ptr) })?;
        Ok(Self {
            handle: ptr as usize,
            key,
        })
    }
    pub fn abort() {
        unsafe {
            forge_cuda_graph_end(std::ptr::null_mut());
        }
    }
    pub fn replay(&self) -> Result<(), ComputeError> {
        map(unsafe { forge_cuda_graph_replay(self.handle as *mut u8) })
    }
}

pub(crate) struct DeviceTimer {
    handle: *mut u8,
}
impl DeviceTimer {
    pub fn start() -> Result<Self, ComputeError> {
        let mut handle = std::ptr::null_mut();
        map(unsafe { forge_cuda_timer_start(&mut handle) })?;
        Ok(Self { handle })
    }
    pub fn elapsed(self) -> Result<f32, ComputeError> {
        let mut ms = 0f32;
        map(unsafe { forge_cuda_timer_elapsed(self.handle, &mut ms) })?;
        Ok(ms)
    }
}
impl Drop for DeviceTimer {
    fn drop(&mut self) {
        unsafe {
            forge_cuda_timer_destroy(self.handle);
        }
    }
}
