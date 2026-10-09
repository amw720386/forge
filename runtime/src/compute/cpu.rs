//! CPU compute backend — wraps existing `forge_kernels_*` FFI.

use super::backend::{
    BackendKind, Buffer, ComputeBackend, ComputeError, DeviceInfo,
};
use crate::ffi::kernels;

pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

fn map_status(st: i32) -> Result<(), ComputeError> {
    match st {
        0 => Ok(()),
        1 => Err(ComputeError::InvalidArgument("kernel".into())),
        2 => Err(ComputeError::DimensionMismatch),
        3 => Err(ComputeError::OutOfMemory("kernel".into())),
        other => Err(ComputeError::Internal(format!("kernel status {other}"))),
    }
}

impl ComputeBackend for CpuBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Cpu
    }

    fn info(&self) -> DeviceInfo {
        DeviceInfo {
            kind: BackendKind::Cpu,
            name: format!("cpu ({})", kernels::version()),
            index: -1,
            free_bytes: None,
            total_bytes: None,
        }
    }

    fn synchronize(&self) -> Result<(), ComputeError> {
        Ok(())
    }

    fn alloc(&self, bytes: usize) -> Result<Buffer, ComputeError> {
        if bytes == 0 {
            return Err(ComputeError::InvalidArgument("zero alloc".into()));
        }
        // Soft cap mirrors CUDA backend (avoid accidental huge host allocs in tests).
        const MAX: usize = 16 * 1024 * 1024 * 1024;
        if bytes > MAX {
            return Err(ComputeError::InvalidArgument("alloc too large".into()));
        }
        let layout = std::alloc::Layout::from_size_align(bytes, 64)
            .map_err(|e| ComputeError::InvalidArgument(e.to_string()))?;
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err(ComputeError::OutOfMemory(format!("{bytes} bytes")));
        }
        Ok(Buffer {
            kind: BackendKind::Cpu,
            ptr,
            bytes,
            free_fn: Some(|p, n| {
                if !p.is_null() {
                    let layout = std::alloc::Layout::from_size_align(n, 64).unwrap();
                    unsafe { std::alloc::dealloc(p, layout) };
                }
            }),
        })
    }

    fn copy_h2d(&self, dst: &mut Buffer, src: &[u8]) -> Result<(), ComputeError> {
        if dst.kind != BackendKind::Cpu {
            return Err(ComputeError::InvalidArgument("cpu buffer required".into()));
        }
        if src.len() > dst.bytes {
            return Err(ComputeError::InvalidArgument("H2D overflow".into()));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst.ptr, src.len());
        }
        Ok(())
    }

    fn copy_d2h(&self, dst: &mut [u8], src: &Buffer) -> Result<(), ComputeError> {
        if src.kind != BackendKind::Cpu {
            return Err(ComputeError::InvalidArgument("cpu buffer required".into()));
        }
        if dst.len() > src.bytes {
            return Err(ComputeError::InvalidArgument("D2H overflow".into()));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(src.ptr, dst.as_mut_ptr(), dst.len());
        }
        Ok(())
    }

    fn copy_d2d(
        &self,
        dst: &mut Buffer,
        dst_offset: usize,
        src: &Buffer,
        src_offset: usize,
        bytes: usize,
    ) -> Result<(), ComputeError> {
        if dst.kind != BackendKind::Cpu || src.kind != BackendKind::Cpu {
            return Err(ComputeError::InvalidArgument("cpu buffers required".into()));
        }
        if dst_offset + bytes > dst.bytes || src_offset + bytes > src.bytes {
            return Err(ComputeError::InvalidArgument("D2D overflow".into()));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.ptr.add(src_offset),
                dst.ptr.add(dst_offset),
                bytes,
            );
        }
        Ok(())
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
        let need_a = m.checked_mul(k).and_then(|x| x.checked_mul(4));
        let need_b = k.checked_mul(n).and_then(|x| x.checked_mul(4));
        let need_c = m.checked_mul(n).and_then(|x| x.checked_mul(4));
        let (Some(na), Some(nb), Some(nc)) = (need_a, need_b, need_c) else {
            return Err(ComputeError::InvalidArgument("size overflow".into()));
        };
        if a.bytes < na || b.bytes < nb || c.bytes < nc {
            return Err(ComputeError::InvalidArgument("buffer too small".into()));
        }
        map_status(unsafe {
            kernels::forge_kernels_matmul_f32(
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
        map_status(unsafe {
            kernels::forge_kernels_matvec_f32(
                m,
                n,
                a.as_f32_ptr(),
                n,
                x.as_f32_ptr(),
                y.as_f32_mut_ptr(),
            )
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
        map_status(unsafe {
            kernels::forge_kernels_rmsnorm_f32(
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
        map_status(unsafe {
            kernels::forge_kernels_silu_f32(n, x.as_f32_ptr(), y.as_f32_mut_ptr())
        })
    }

    fn add_f32(
        &self,
        n: usize,
        a: &Buffer,
        b: &Buffer,
        y: &mut Buffer,
    ) -> Result<(), ComputeError> {
        map_status(unsafe {
            kernels::forge_kernels_add_f32(n, a.as_f32_ptr(), b.as_f32_ptr(), y.as_f32_mut_ptr())
        })
    }

    fn mul_f32(
        &self,
        n: usize,
        a: &Buffer,
        b: &Buffer,
        y: &mut Buffer,
    ) -> Result<(), ComputeError> {
        map_status(unsafe {
            kernels::forge_kernels_mul_f32(n, a.as_f32_ptr(), b.as_f32_ptr(), y.as_f32_mut_ptr())
        })
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
        let bias_ptr = bias
            .map(|b| b.as_f32_ptr())
            .unwrap_or(std::ptr::null());
        map_status(unsafe {
            kernels::forge_kernels_linear_f32(
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
        _m: usize,
        _n: usize,
        _k: usize,
        _x: &Buffer,
        _x_cols: usize,
        _w: &Buffer,
        _w_cols: usize,
        _bias: Option<&Buffer>,
        _y: &mut Buffer,
        _y_cols: usize,
    ) -> Result<(), ComputeError> {
        Err(ComputeError::NotBuilt)
    }

    fn f32_to_bf16(
        &self,
        _n: usize,
        _src: &Buffer,
        _dst: &mut Buffer,
    ) -> Result<(), ComputeError> {
        Err(ComputeError::NotBuilt)
    }

    fn argmax_rows_f32(
        &self,
        rows: usize,
        cols: usize,
        logits: &Buffer,
        out_ids: &mut Buffer,
    ) -> Result<(), ComputeError> {
        if rows == 0 || cols == 0 {
            return Err(ComputeError::InvalidArgument("argmax dims".into()));
        }
        let need_logits = rows
            .checked_mul(cols)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| ComputeError::InvalidArgument("size overflow".into()))?;
        let need_ids = rows
            .checked_mul(4)
            .ok_or_else(|| ComputeError::InvalidArgument("size overflow".into()))?;
        if logits.bytes < need_logits || out_ids.bytes < need_ids {
            return Err(ComputeError::InvalidArgument("buffer too small".into()));
        }
        let logits_s =
            unsafe { std::slice::from_raw_parts(logits.as_f32_ptr(), rows * cols) };
        let ids =
            unsafe { std::slice::from_raw_parts_mut(out_ids.ptr as *mut i32, rows) };
        for r in 0..rows {
            let row = &logits_s[r * cols..(r + 1) * cols];
            let mut best_i = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for (i, &v) in row.iter().enumerate() {
                // Strict > keeps lowest index on ties (matches sampling::greedy).
                if v > best_v {
                    best_v = v;
                    best_i = i;
                }
            }
            ids[r] = best_i as i32;
        }
        Ok(())
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
        map_status(unsafe {
            kernels::forge_kernels_rope_f32(
                seq_len,
                n_heads,
                head_dim,
                x.as_f32_mut_ptr(),
                std::ptr::null(),
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
        let ids: &[i32] = unsafe {
            std::slice::from_raw_parts(token_ids.ptr as *const i32, t)
        };
        let embed = unsafe {
            std::slice::from_raw_parts(table.as_f32_ptr(), vocab * hidden)
        };
        let y = unsafe { std::slice::from_raw_parts_mut(out.as_f32_mut_ptr(), t * hidden) };
        for row in 0..t {
            let id = ids[row] as usize;
            if id >= vocab {
                return Err(ComputeError::InvalidArgument("token id out of range".into()));
            }
            let src = &embed[id * hidden..(id + 1) * hidden];
            y[row * hidden..(row + 1) * hidden].copy_from_slice(src);
        }
        Ok(())
    }

    fn embedding_bf16(
        &self,
        _table: &Buffer,
        _vocab: usize,
        _hidden: usize,
        _token_ids: &Buffer,
        _t: usize,
        _out: &mut Buffer,
    ) -> Result<(), ComputeError> {
        Err(ComputeError::NotBuilt)
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
        let _ = scale;
        let q_w = n_heads * head_dim;
        let _kv_w = n_kv_heads * head_dim;
        let q_s = unsafe { std::slice::from_raw_parts(q.as_f32_ptr(), t_q * q_cols) };
        let k_s = unsafe { std::slice::from_raw_parts(k.as_f32_ptr(), t_k * kv_cols) };
        let v_s = unsafe { std::slice::from_raw_parts(v.as_f32_ptr(), t_k * kv_cols) };
        let ctx_s =
            unsafe { std::slice::from_raw_parts_mut(ctx.as_f32_mut_ptr(), t_q * q_cols) };
        ctx_s[..t_q * q_w].fill(0.0);
        let reps = n_heads / n_kv_heads;
        for h in 0..n_heads {
            let kv_h = h / reps;
            for qi in 0..t_q {
                let mut valid_k = t_k;
                if causal {
                    let qpos = causal_q_offset + qi;
                    valid_k = (qpos + 1).min(t_k);
                }
                let q_off = qi * q_cols + h * head_dim;
                let mut vmax = f32::NEG_INFINITY;
                let mut scores = vec![0.0f32; valid_k];
                for kj in 0..valid_k {
                    let k_off = kj * kv_cols + kv_h * head_dim;
                    let mut dot = 0.0f32;
                    for i in 0..head_dim {
                        dot += q_s[q_off + i] * k_s[k_off + i];
                    }
                    scores[kj] = dot * scale;
                    vmax = vmax.max(scores[kj]);
                }
                let mut sum = 0.0f32;
                for s in scores.iter_mut() {
                    *s = (*s - vmax).exp();
                    sum += *s;
                }
                let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                let out_off = qi * q_cols + h * head_dim;
                for i in 0..head_dim {
                    let mut acc = 0.0f32;
                    for (kj, &p) in scores.iter().enumerate().take(valid_k) {
                        let v_off = kj * kv_cols + kv_h * head_dim;
                        acc += p * inv * v_s[v_off + i];
                    }
                    ctx_s[out_off + i] = acc;
                }
            }
        }
        Ok(())
    }

    fn copy_rows_f32(
        &self,
        rows: usize,
        cols: usize,
        src: &Buffer,
        dst: &mut Buffer,
    ) -> Result<(), ComputeError> {
        let n = rows * cols * 4;
        if src.bytes < n || dst.bytes < n {
            return Err(ComputeError::InvalidArgument("buffer too small".into()));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(src.ptr, dst.ptr, n);
        }
        Ok(())
    }

    fn add_rows_f32(
        &self,
        rows: usize,
        cols: usize,
        a: &Buffer,
        b: &Buffer,
        y: &mut Buffer,
    ) -> Result<(), ComputeError> {
        let n = rows * cols;
        let aa = unsafe { std::slice::from_raw_parts(a.as_f32_ptr(), n) };
        let bb = unsafe { std::slice::from_raw_parts(b.as_f32_ptr(), n) };
        let yy = unsafe { std::slice::from_raw_parts_mut(y.as_f32_mut_ptr(), n) };
        for i in 0..n {
            yy[i] = aa[i] + bb[i];
        }
        Ok(())
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
        let n = t * width;
        let off = dst_row_offset * width;
        let ks = unsafe { std::slice::from_raw_parts(k_src.as_f32_ptr(), n) };
        let vs = unsafe { std::slice::from_raw_parts(v_src.as_f32_ptr(), n) };
        let kd = unsafe {
            std::slice::from_raw_parts_mut(k_dst.as_f32_mut_ptr(), off + n)
        };
        let vd = unsafe {
            std::slice::from_raw_parts_mut(v_dst.as_f32_mut_ptr(), off + n)
        };
        kd[off..off + n].copy_from_slice(ks);
        vd[off..off + n].copy_from_slice(vs);
        Ok(())
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
        if layer >= n_layers || block_size == 0 || num_blocks == 0 {
            return Err(ComputeError::InvalidArgument("paged attn dims".into()));
        }
        let need = (t_k + block_size - 1) / block_size;
        if need > num_logical_blocks {
            return Err(ComputeError::InvalidArgument("block table too short".into()));
        }
        let q_w = n_heads * head_dim;
        let q_s = unsafe { std::slice::from_raw_parts(q.as_f32_ptr(), t_q * q_cols) };
        let table = unsafe {
            std::slice::from_raw_parts(block_table.ptr as *const u32, num_logical_blocks)
        };
        let pool_elems = n_layers * num_blocks * block_size * kv_cols;
        let k_s = unsafe { std::slice::from_raw_parts(k_pool.as_f32_ptr(), pool_elems) };
        let v_s = unsafe { std::slice::from_raw_parts(v_pool.as_f32_ptr(), pool_elems) };
        let ctx_s =
            unsafe { std::slice::from_raw_parts_mut(ctx.as_f32_mut_ptr(), t_q * q_cols) };
        ctx_s[..t_q * q_w].fill(0.0);
        let reps = n_heads / n_kv_heads;
        let kv_base = |phys: u32, row: usize, kv_h: usize| {
            ((layer * num_blocks + phys as usize) * block_size + row) * kv_cols
                + kv_h * head_dim
        };
        for h in 0..n_heads {
            let kv_h = h / reps;
            for qi in 0..t_q {
                let mut valid_k = t_k;
                if causal {
                    let qpos = causal_q_offset + qi;
                    valid_k = (qpos + 1).min(t_k);
                }
                let q_off = qi * q_cols + h * head_dim;
                let mut vmax = f32::NEG_INFINITY;
                let mut scores = vec![0.0f32; valid_k];
                for kj in 0..valid_k {
                    let phys = table[kj / block_size];
                    let row = kj % block_size;
                    let k_off = kv_base(phys, row, kv_h);
                    let mut dot = 0.0f32;
                    for i in 0..head_dim {
                        dot += q_s[q_off + i] * k_s[k_off + i];
                    }
                    scores[kj] = dot * scale;
                    vmax = vmax.max(scores[kj]);
                }
                let mut sum = 0.0f32;
                for s in scores.iter_mut() {
                    *s = (*s - vmax).exp();
                    sum += *s;
                }
                let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                let out_off = qi * q_cols + h * head_dim;
                for i in 0..head_dim {
                    let mut acc = 0.0f32;
                    for (kj, &p) in scores.iter().enumerate().take(valid_k) {
                        let phys = table[kj / block_size];
                        let row = kj % block_size;
                        let v_off = kv_base(phys, row, kv_h);
                        acc += p * inv * v_s[v_off + i];
                    }
                    ctx_s[out_off + i] = acc;
                }
            }
        }
        Ok(())
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
        self.paged_gqa_attn_f32(
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
        )
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
        if layer >= n_layers || block_size == 0 {
            return Err(ComputeError::InvalidArgument("paged append dims".into()));
        }
        let end = dst_token_offset + t;
        let need = (end + block_size - 1) / block_size;
        if need > num_logical_blocks {
            return Err(ComputeError::InvalidArgument("block table too short".into()));
        }
        let table = unsafe {
            std::slice::from_raw_parts(block_table.ptr as *const u32, num_logical_blocks)
        };
        let n = t * width;
        let ks = unsafe { std::slice::from_raw_parts(k_src.as_f32_ptr(), n) };
        let vs = unsafe { std::slice::from_raw_parts(v_src.as_f32_ptr(), n) };
        let pool_elems = n_layers * num_blocks * block_size * width;
        let kd = unsafe { std::slice::from_raw_parts_mut(k_pool.as_f32_mut_ptr(), pool_elems) };
        let vd = unsafe { std::slice::from_raw_parts_mut(v_pool.as_f32_mut_ptr(), pool_elems) };
        for tok in 0..t {
            let logical = dst_token_offset + tok;
            let phys = table[logical / block_size] as usize;
            let row = logical % block_size;
            let dst = ((layer * num_blocks + phys) * block_size + row) * width;
            let src = tok * width;
            kd[dst..dst + width].copy_from_slice(&ks[src..src + width]);
            vd[dst..dst + width].copy_from_slice(&vs[src..src + width]);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::greedy;

    #[test]
    fn argmax_rows_matches_greedy_tie_break() {
        let cpu = CpuBackend::new();
        let rows = 2usize;
        let cols = 4usize;
        let logits_host: Vec<f32> = vec![
            1.0, 5.0, 5.0, 2.0, // row0 → idx 1
            3.0, 3.0, 1.0, 0.0, // row1 → idx 0
        ];
        let mut logits = cpu.alloc(rows * cols * 4).unwrap();
        let bytes: Vec<u8> = logits_host.iter().flat_map(|f| f.to_le_bytes()).collect();
        cpu.copy_h2d(&mut logits, &bytes).unwrap();
        let mut ids = cpu.alloc(rows * 4).unwrap();
        cpu.argmax_rows_f32(rows, cols, &logits, &mut ids).unwrap();
        let mut out = vec![0u8; rows * 4];
        cpu.copy_d2h(&mut out, &ids).unwrap();
        let got: Vec<u32> = out
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u32)
            .collect();
        assert_eq!(got[0], greedy(&logits_host[0..4]));
        assert_eq!(got[1], greedy(&logits_host[4..8]));
        assert_eq!(got, vec![1, 0]);
    }
}
