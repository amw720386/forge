//! Public C ABI exported for Go.

use crate::ffi::kernels;
use crate::mem::MemoryBudget;
use crate::model::{
    load_model_path, BatchEvent, ContinuousBatcher, ForgeTokenizer, GenerateParams,
};
use std::collections::VecDeque;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::ptr;
use std::sync::OnceLock;

static VERSION: OnceLock<CString> = OnceLock::new();

pub(crate) enum RuntimeHandleInner {
    Cpu(ContinuousBatcher),
    #[cfg(feature = "cuda")]
    Cuda(crate::cuda_model::CudaBatcher),
}

impl RuntimeHandleInner {
    fn max_batch(&self) -> usize {
        match self {
            Self::Cpu(b) => b.max_batch(),
            #[cfg(feature = "cuda")]
            Self::Cuda(b) => b.max_batch(),
        }
    }

    fn active_count(&self) -> usize {
        match self {
            Self::Cpu(b) => b.active_count(),
            #[cfg(feature = "cuda")]
            Self::Cuda(b) => b.active_count(),
        }
    }

    fn free_slots(&self) -> usize {
        match self {
            Self::Cpu(b) => b.free_slots(),
            #[cfg(feature = "cuda")]
            Self::Cuda(b) => b.free_slots(),
        }
    }

    fn sched_stats(&self) -> ForgeBatchSchedStats {
        match self {
            Self::Cpu(b) => ForgeBatchSchedStats {
                active_decode: b.active_count() as u64,
                active_prefill: 0,
                last_scheduled_decode: 0,
                last_scheduled_prefill: 0,
                kv_num_blocks: 0,
                kv_block_size: 0,
                kv_free_blocks: 0,
                kv_allocated_blocks: 0,
                prefill_stall_ns: 0,
            },
            #[cfg(feature = "cuda")]
            Self::Cuda(b) => {
                let kv = b.kv_stats();
                let st = b.stats();
                ForgeBatchSchedStats {
                    active_decode: b.active_decode() as u64,
                    active_prefill: b.active_prefill() as u64,
                    last_scheduled_decode: st.last_scheduled_decode as u64,
                    last_scheduled_prefill: st.last_scheduled_prefill as u64,
                    kv_num_blocks: kv.num_blocks as u64,
                    kv_block_size: kv.block_size as u64,
                    kv_free_blocks: kv.free_blocks as u64,
                    kv_allocated_blocks: kv.allocated_blocks as u64,
                    prefill_stall_ns: b.prefill_stall_ns(),
                }
            }
        }
    }

    pub(crate) fn admit(
        &mut self,
        prompt: &str,
        params: GenerateParams,
    ) -> Result<u64, crate::model::BatchError> {
        match self {
            Self::Cpu(b) => b.admit(prompt, params),
            #[cfg(feature = "cuda")]
            Self::Cuda(b) => b.admit(prompt, params),
        }
    }

    pub(crate) fn cancel(&mut self, id: u64) -> Result<(), crate::model::BatchError> {
        match self {
            Self::Cpu(b) => b.cancel(id),
            #[cfg(feature = "cuda")]
            Self::Cuda(b) => b.cancel(id),
        }
    }

    pub(crate) fn step(&mut self) -> Result<Vec<BatchEvent>, crate::model::BatchError> {
        match self {
            Self::Cpu(b) => b.step(),
            #[cfg(feature = "cuda")]
            Self::Cuda(b) => b.step(),
        }
    }
}

pub struct RuntimeHandle {
    pub(crate) inner: RuntimeHandleInner,
    pub(crate) tokenizer: ForgeTokenizer,
    pub(crate) tokenizer_dir: std::path::PathBuf,
    pub(crate) max_seq: usize,
    /// Compatibility path for single-flight generate_* API.
    pub(crate) eos_token_id: u32,
    single_id: Option<u64>,
    single_events: VecDeque<BatchEvent>,
    batch_events: Vec<BatchEvent>,
}

#[repr(C)]
pub struct ForgeBatchEvent {
    pub request_id: u64,
    pub token_id: u32,
    pub done: u8,
    pub _pad: [u8; 3],
    pub piece: [c_char; 64],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct ForgeBatchSchedStats {
    pub active_decode: u64,
    pub active_prefill: u64,
    pub last_scheduled_decode: u64,
    pub last_scheduled_prefill: u64,
    pub kv_num_blocks: u64,
    pub kv_block_size: u64,
    pub kv_free_blocks: u64,
    pub kv_allocated_blocks: u64,
    pub prefill_stall_ns: u64,
}

#[no_mangle]
pub unsafe extern "C" fn forge_version() -> *const c_char {
    VERSION
        .get_or_init(|| {
            CString::new(format!("forge-runtime/{}", env!("CARGO_PKG_VERSION"))).expect("version")
        })
        .as_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn forge_matmul_f32(
    m: usize,
    n: usize,
    k: usize,
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    c: *mut f32,
    ldc: usize,
) -> c_int {
    if a.is_null() || b.is_null() || c.is_null() {
        return 1;
    }
    kernels::forge_kernels_matmul_f32(m, n, k, a, lda, b, ldb, c, ldc)
}

#[no_mangle]
pub unsafe extern "C" fn forge_matvec_f32(
    m: usize,
    n: usize,
    a: *const f32,
    lda: usize,
    x: *const f32,
    y: *mut f32,
) -> c_int {
    if a.is_null() || x.is_null() || y.is_null() {
        return 1;
    }
    kernels::forge_kernels_matvec_f32(m, n, a, lda, x, y)
}

fn write_err(buf: *mut c_char, len: usize, msg: &str) {
    if buf.is_null() || len == 0 {
        return;
    }
    let bytes = msg.as_bytes();
    let n = bytes.len().min(len - 1);
    unsafe {
        ptr::copy_nonoverlapping(bytes.as_ptr(), buf as *mut u8, n);
        *buf.add(n) = 0;
    }
}

fn write_piece(dst: &mut [c_char], piece: &str) {
    let bytes = piece.as_bytes();
    let mut n = bytes.len().min(dst.len().saturating_sub(1));
    while !piece.is_char_boundary(n) {
        n -= 1;
    }
    for i in 0..n {
        dst[i] = bytes[i] as c_char;
    }
    if !dst.is_empty() {
        dst[n] = 0;
    }
}

/// device: 0 = CPU, 1 = CUDA FP32, 2 = CUDA BF16 (when built with `--features cuda`).
#[no_mangle]
pub unsafe extern "C" fn forge_runtime_load_ex(
    model_dir: *const c_char,
    max_seq: usize,
    device: c_int,
    err_buf: *mut c_char,
    err_len: usize,
) -> *mut RuntimeHandle {
    if model_dir.is_null() || max_seq == 0 {
        write_err(err_buf, err_len, "invalid argument");
        return ptr::null_mut();
    }
    let dir = match CStr::from_ptr(model_dir).to_str() {
        Ok(s) => s,
        Err(_) => {
            write_err(err_buf, err_len, "model_dir not utf8");
            return ptr::null_mut();
        }
    };
    let model = match load_model_path(dir) {
        Ok(m) => m,
        Err(e) => {
            write_err(err_buf, err_len, &e.to_string());
            return ptr::null_mut();
        }
    };
    let budget = MemoryBudget::from_env();
    let max_seq = max_seq.min(budget.max_seq);
    let max_batch = budget.max_batch;
    if let Err(e) = budget.check_allows(&model.cfg, model.weight_nbytes()) {
        write_err(err_buf, err_len, &e.to_string());
        return ptr::null_mut();
    }
    if max_seq > model.cfg.max_position_embeddings {
        write_err(
            err_buf,
            err_len,
            "requested max-seq exceeds model context limit",
        );
        return ptr::null_mut();
    }
    let tok_dir = match crate::model::tokenizer_dir(dir, &model.cfg) {
        Ok(p) => p,
        Err(e) => {
            write_err(err_buf, err_len, &e.to_string());
            return ptr::null_mut();
        }
    };
    let tok = match ForgeTokenizer::for_model(&tok_dir, &model.cfg) {
        Ok(t) => t,
        Err(e) => {
            write_err(err_buf, err_len, &e.to_string());
            return ptr::null_mut();
        }
    };
    let eos_token_id = model.cfg.eos_token_id;
    let tokenizer = tok.clone();
    let handle = if device == 1 || device == 2 {
        #[cfg(all(feature = "cuda-stub", not(feature = "cuda")))]
        {
            let _ = device;
            write_err(
                err_buf,
                err_len,
                "CUDA stub build cannot run inference (rebuild with --features cuda + libforge_cuda)",
            );
            return ptr::null_mut();
        }
        #[cfg(feature = "cuda")]
        {
            use crate::cuda_model::{CudaBatcher, CudaTransformer, GpuWeightDType};
            let dtype = if device == 2 {
                GpuWeightDType::Bf16
            } else {
                GpuWeightDType::Fp32
            };
            let cuda = match CudaTransformer::from_cpu_dtype(&model, 0, dtype) {
                Ok(c) => c,
                Err(e) => {
                    write_err(err_buf, err_len, &e.to_string());
                    return ptr::null_mut();
                }
            };
            match CudaBatcher::new(cuda, tok, max_seq, max_batch) {
                Ok(b) => RuntimeHandleInner::Cuda(b),
                Err(e) => {
                    write_err(err_buf, err_len, &e.to_string());
                    return ptr::null_mut();
                }
            }
        }
        #[cfg(not(any(feature = "cuda", feature = "cuda-stub")))]
        {
            let _ = device;
            write_err(err_buf, err_len, "CUDA not built (enable --features cuda)");
            return ptr::null_mut();
        }
    } else if device == 0 {
        match ContinuousBatcher::new(model, tok, max_seq, max_batch) {
            Ok(b) => RuntimeHandleInner::Cpu(b),
            Err(e) => {
                write_err(err_buf, err_len, &e.to_string());
                return ptr::null_mut();
            }
        }
    } else {
        write_err(
            err_buf,
            err_len,
            "unknown device (0=CPU, 1=CUDA FP32, 2=CUDA BF16)",
        );
        return ptr::null_mut();
    };
    Box::into_raw(Box::new(RuntimeHandle {
        inner: handle,
        tokenizer,
        tokenizer_dir: tok_dir,
        max_seq,
        eos_token_id,
        single_id: None,
        single_events: VecDeque::new(),
        batch_events: Vec::new(),
    }))
}

#[no_mangle]
pub unsafe extern "C" fn forge_runtime_load(
    model_dir: *const c_char,
    max_seq: usize,
    err_buf: *mut c_char,
    err_len: usize,
) -> *mut RuntimeHandle {
    forge_runtime_load_ex(model_dir, max_seq, 0, err_buf, err_len)
}

#[no_mangle]
pub unsafe extern "C" fn forge_runtime_free(rt: *mut RuntimeHandle) {
    if !rt.is_null() {
        drop(Box::from_raw(rt));
    }
}

#[no_mangle]
pub unsafe extern "C" fn forge_batch_capacity(rt: *mut RuntimeHandle) -> usize {
    if rt.is_null() {
        return 0;
    }
    (*rt).inner.max_batch()
}

#[no_mangle]
pub unsafe extern "C" fn forge_batch_active(rt: *mut RuntimeHandle) -> usize {
    if rt.is_null() {
        return 0;
    }
    (*rt).inner.active_count()
}

#[no_mangle]
pub unsafe extern "C" fn forge_batch_free_slots(rt: *mut RuntimeHandle) -> usize {
    if rt.is_null() {
        return 0;
    }
    (*rt).inner.free_slots()
}

#[no_mangle]
pub unsafe extern "C" fn forge_batch_sched_stats(
    rt: *mut RuntimeHandle,
    out: *mut ForgeBatchSchedStats,
) -> c_int {
    if rt.is_null() || out.is_null() {
        return 1; // INVALID_ARGUMENT
    }
    *out = (*rt).inner.sched_stats();
    0 // OK
}

/// Admit a request into the continuous batcher. Writes request_id on success.
#[no_mangle]
pub unsafe extern "C" fn forge_batch_admit(
    rt: *mut RuntimeHandle,
    prompt_utf8: *const c_char,
    max_tokens: i32,
    temperature_milli: i32,
    out_id: *mut u64,
    err_buf: *mut c_char,
    err_len: usize,
) -> c_int {
    if rt.is_null() || prompt_utf8.is_null() || out_id.is_null() || max_tokens <= 0 {
        return 1;
    }
    let handle = &mut *rt;
    let prompt = match CStr::from_ptr(prompt_utf8).to_str() {
        Ok(s) => s,
        Err(_) => {
            write_err(err_buf, err_len, "prompt not utf8");
            return 1;
        }
    };
    let params = GenerateParams {
        max_tokens: max_tokens as usize,
        temperature: (temperature_milli as f32) / 1000.0,
        top_k: 0,
        eos_token_id: Some(handle.eos_token_id),
    };
    match handle.inner.admit(prompt, params) {
        Ok(id) => {
            *out_id = id;
            0
        }
        Err(crate::model::BatchError::Full) => 5, // BUSY / backpressure
        Err(crate::model::BatchError::KvExhausted(s)) => {
            write_err(err_buf, err_len, &s);
            3 // OUT_OF_MEMORY
        }
        Err(e) => {
            write_err(err_buf, err_len, &e.to_string());
            4
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn forge_batch_cancel(rt: *mut RuntimeHandle, request_id: u64) -> c_int {
    if rt.is_null() {
        return 1;
    }
    match (*rt).inner.cancel(request_id) {
        Ok(()) => 0,
        Err(crate::model::BatchError::UnknownId(_)) => 6,
        Err(_) => 4,
    }
}

/// Run one shared decode step. Writes up to `cap` events; returns count (or negative status).
#[no_mangle]
pub unsafe extern "C" fn forge_batch_step(
    rt: *mut RuntimeHandle,
    out: *mut ForgeBatchEvent,
    cap: usize,
) -> c_int {
    if rt.is_null() || out.is_null() {
        return -1;
    }
    let handle = &mut *rt;
    if cap < handle.inner.max_batch() {
        return -1;
    }
    handle.batch_events.clear();
    if handle.inner.active_count() == 0 {
        return 0;
    }
    match handle.inner.step() {
        Ok(events) => {
            let n = events.len().min(cap);
            for (i, ev) in events.iter().take(n).enumerate() {
                let slot = &mut *out.add(i);
                slot.request_id = ev.request_id;
                slot.token_id = ev.token_id;
                slot.done = if ev.done { 1 } else { 0 };
                slot._pad = [0; 3];
                write_piece(&mut slot.piece, &ev.piece);
            }
            handle.batch_events = events;
            n as c_int
        }
        Err(crate::model::BatchError::Empty) => 0,
        Err(_) => -4,
    }
}

/// Full UTF-8 piece from the last batch step. Return required bytes including NUL.
/// A short buffer is untouched; retry with the returned size before the next step.
#[no_mangle]
pub unsafe extern "C" fn forge_batch_piece(
    rt: *mut RuntimeHandle,
    id: u64,
    buf: *mut c_char,
    len: usize,
) -> usize {
    if rt.is_null() {
        return 0;
    }
    let Some(event) = (*rt).batch_events.iter().find(|e| e.request_id == id) else {
        return 0;
    };
    let needed = event.piece.len() + 1;
    if !buf.is_null() && len >= needed {
        write_err(buf, len, &event.piece);
    }
    needed
}

#[cfg(test)]
mod piece_tests {
    use super::*;

    #[test]
    fn inline_piece_keeps_utf8_boundaries() {
        let mut buf = [0i8; 64];
        write_piece(&mut buf, &("a".repeat(62) + "é"));
        let text = unsafe { CStr::from_ptr(buf.as_ptr()) }.to_str().unwrap();
        assert_eq!(text, "a".repeat(62));
    }

    #[test]
    #[ignore = "requires local qualified model"]
    fn full_piece_is_not_truncated_and_short_event_buffer_does_not_step() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../models/SmolLM2-135M-Instruct");
        let mut runtime = RuntimeHandle::load(path.to_str().unwrap(), 64, 0).unwrap();
        runtime.batch_events.push(BatchEvent {
            request_id: 42,
            token_id: 0,
            piece: "é".repeat(128),
            done: true,
        });
        let ptr = &mut *runtime as *mut RuntimeHandle;
        let mut short = [7u8; 64];
        unsafe {
            assert_eq!(
                forge_batch_piece(ptr, 42, short.as_mut_ptr().cast(), short.len()),
                257
            );
            assert_eq!(short, [7u8; 64]);
            let mut full = vec![0u8; 257];
            assert_eq!(
                forge_batch_piece(ptr, 42, full.as_mut_ptr().cast(), full.len()),
                257
            );
            assert_eq!(std::str::from_utf8(&full[..256]).unwrap(), "é".repeat(128));
            let mut event: ForgeBatchEvent = std::mem::zeroed();
            assert_eq!(forge_batch_step(ptr, &mut event, 1), -1);
        }
    }
}

// ---- Single-flight compatibility (uses the same continuous batcher) --------

#[no_mangle]
pub unsafe extern "C" fn forge_generate_begin(
    rt: *mut RuntimeHandle,
    prompt_utf8: *const c_char,
    max_tokens: i32,
    temperature_milli: i32,
    err_buf: *mut c_char,
    err_len: usize,
) -> c_int {
    if rt.is_null() || prompt_utf8.is_null() || max_tokens <= 0 {
        return 1;
    }
    {
        let handle = &mut *rt;
        if handle.single_id.is_some() || handle.inner.active_count() > 0 {
            return 5; // BUSY
        }
    }
    let mut id: u64 = 0;
    let st = forge_batch_admit(
        rt,
        prompt_utf8,
        max_tokens,
        temperature_milli,
        &mut id,
        err_buf,
        err_len,
    );
    if st == 0 {
        let handle = &mut *rt;
        handle.single_id = Some(id);
        handle.single_events.clear();
    }
    st
}

#[no_mangle]
pub unsafe extern "C" fn forge_generate_next(
    rt: *mut RuntimeHandle,
    token_id: *mut u32,
    piece_buf: *mut c_char,
    piece_len: usize,
) -> c_int {
    if rt.is_null() || token_id.is_null() {
        return 1;
    }
    let handle = &mut *rt;
    let id = match handle.single_id {
        Some(id) => id,
        None => return 7,
    };

    if handle.single_events.is_empty() {
        if handle.inner.active_count() == 0 {
            handle.single_id = None;
            return 7;
        }
        match handle.inner.step() {
            Ok(events) => {
                for ev in events {
                    if ev.request_id == id {
                        handle.single_events.push_back(ev);
                    }
                }
            }
            Err(_) => {
                handle.single_id = None;
                return 4;
            }
        }
    }

    if let Some(ev) = handle.single_events.front() {
        if piece_buf.is_null() || piece_len <= ev.piece.len() {
            return 1;
        }
    }
    match handle.single_events.pop_front() {
        Some(ev) => {
            *token_id = ev.token_id;
            write_err(piece_buf, piece_len, &ev.piece);
            if ev.done {
                handle.single_id = None;
                handle.single_events.clear();
            }
            0
        }
        None => {
            handle.single_id = None;
            7
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn forge_generate_cancel(rt: *mut RuntimeHandle) {
    if rt.is_null() {
        return;
    }
    let handle = &mut *rt;
    if let Some(id) = handle.single_id.take() {
        let _ = handle.inner.cancel(id);
    }
    handle.single_events.clear();
}
