//! Unsafe bindings to the C++ kernel library compiled via `build.rs`.

use std::ffi::CStr;
use std::os::raw::{c_char, c_int};

pub const FORGE_OK: c_int = 0;

extern "C" {
    pub fn forge_kernels_matmul_f32(
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

    pub fn forge_kernels_matvec_f32(
        m: usize,
        n: usize,
        a: *const f32,
        lda: usize,
        x: *const f32,
        y: *mut f32,
    ) -> c_int;

    pub fn forge_kernels_linear_f32(
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

    pub fn forge_kernels_linear_i8_f32(
        m: usize,
        n: usize,
        k: usize,
        x: *const f32,
        ldx: usize,
        w_q: *const i8,
        scales: *const f32,
        group_size: usize,
        bias: *const f32,
        y: *mut f32,
        ldy: usize,
    ) -> c_int;

    pub fn forge_kernels_linear_i4_f32(
        m: usize,
        n: usize,
        k: usize,
        x: *const f32,
        ldx: usize,
        w_q: *const u8,
        scales: *const f32,
        group_size: usize,
        bias: *const f32,
        y: *mut f32,
        ldy: usize,
    ) -> c_int;

    pub fn forge_kernels_rmsnorm_f32(
        rows: usize,
        dim: usize,
        x: *const f32,
        ldx: usize,
        weight: *const f32,
        eps: f32,
        y: *mut f32,
        ldy: usize,
    ) -> c_int;

    pub fn forge_kernels_softmax_f32(
        batch: usize,
        rows: usize,
        cols: usize,
        scores: *mut f32,
        causal: c_int,
        causal_q_offset: usize,
    ) -> c_int;

    pub fn forge_kernels_rope_f32(
        seq_len: usize,
        n_heads: usize,
        head_dim: usize,
        x: *mut f32,
        positions: *const i32,
        position_offset: usize,
        rope_theta: f32,
    ) -> c_int;

    pub fn forge_kernels_silu_f32(n: usize, x: *const f32, y: *mut f32) -> c_int;

    pub fn forge_kernels_mul_f32(n: usize, a: *const f32, b: *const f32, y: *mut f32) -> c_int;

    pub fn forge_kernels_add_f32(n: usize, a: *const f32, b: *const f32, y: *mut f32) -> c_int;

    pub fn forge_kernels_version() -> *const c_char;

    pub fn forge_kernels_num_threads() -> u32;
}

pub fn version() -> String {
    unsafe {
        let ptr = forge_kernels_version();
        if ptr.is_null() {
            return String::new();
        }
        CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }
}

pub fn check(status: c_int) -> Result<(), i32> {
    if status == FORGE_OK {
        Ok(())
    } else {
        Err(status)
    }
}
