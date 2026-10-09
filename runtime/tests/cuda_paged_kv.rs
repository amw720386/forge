//! paged KV allocator + paged vs contiguous attention (CPU path).
//!
//! Runs under `--features cuda-stub` without a GPU. Full-model GPU parity lives in
//! `cuda_batch_parity`.

#![cfg(any(feature = "cuda", feature = "cuda-stub"))]

use forge_runtime::compute::{Buffer, ComputeBackend, CpuBackend};
use forge_runtime::cuda_model::paged_kv::{
    BlockAllocator, GpuKvBlockPool, PagedKvError, PagedKvSession, DEFAULT_BLOCK_SIZE,
};

fn fill_linear(buf: &mut Buffer, n: usize, start: f32) {
    let mut host = vec![0f32; n];
    for (i, v) in host.iter_mut().enumerate() {
        *v = start + i as f32 * 0.01;
    }
    let bytes: Vec<u8> = host.iter().flat_map(|f| f.to_le_bytes()).collect();
    // CpuBackend buffers are host-backed; use copy via backend.
    let cpu = CpuBackend;
    cpu.copy_h2d(buf, &bytes).unwrap();
}

#[test]
fn block_allocator_reuse_and_exhaustion() {
    let mut a = BlockAllocator::new(DEFAULT_BLOCK_SIZE, 3).unwrap();
    let b0 = a.alloc_one().unwrap();
    let b1 = a.alloc_one().unwrap();
    a.free_one(b0).unwrap();
    let b2 = a.alloc_one().unwrap();
    assert_eq!(b2, b0); // LIFO reuse
    a.alloc_one().unwrap();
    assert!(matches!(
        a.alloc_one(),
        Err(PagedKvError::Exhausted { .. })
    ));
    a.free_one(b1).unwrap();
    a.free_one(b2).unwrap();
}

#[test]
fn paged_vs_contiguous_attention_cpu() {
    let cpu = CpuBackend;
    let n_heads = 4usize;
    let n_kv = 2usize;
    let d = 8usize;
    let q_w = n_heads * d;
    let kv_w = n_kv * d;
    let t_k = 20usize; // crosses block boundary (block=16)
    let block_size = 16usize;
    let n_layers = 1usize;
    let num_blocks = 4usize;
    let scale = 1.0 / (d as f32).sqrt();

    // Contiguous K/V
    let mut k_c = cpu.alloc(t_k * kv_w * 4).unwrap();
    let mut v_c = cpu.alloc(t_k * kv_w * 4).unwrap();
    fill_linear(&mut k_c, t_k * kv_w, 0.1);
    fill_linear(&mut v_c, t_k * kv_w, 0.5);

    // Paged pool + table (identity mapping into first ceil(t_k/16) blocks)
    let pool_elems = n_layers * num_blocks * block_size * kv_w;
    let mut k_pool = cpu.alloc(pool_elems * 4).unwrap();
    let mut v_pool = cpu.alloc(pool_elems * 4).unwrap();
    // zero pools
    cpu.copy_h2d(&mut k_pool, &vec![0u8; pool_elems * 4]).unwrap();
    cpu.copy_h2d(&mut v_pool, &vec![0u8; pool_elems * 4]).unwrap();

    let need_blocks = (t_k + block_size - 1) / block_size;
    let table_host: Vec<u32> = (0..need_blocks as u32).collect();
    let mut table_bytes = Vec::new();
    for id in &table_host {
        table_bytes.extend_from_slice(&id.to_le_bytes());
    }
    let mut block_table = cpu.alloc(table_bytes.len().max(4)).unwrap();
    cpu.copy_h2d(&mut block_table, &table_bytes).unwrap();

    // Copy contiguous rows into paged layout via paged_kv_append
    let mut k_src = cpu.alloc(t_k * kv_w * 4).unwrap();
    let mut v_src = cpu.alloc(t_k * kv_w * 4).unwrap();
    // copy k_c/v_c into src (already filled same pattern — refill)
    fill_linear(&mut k_src, t_k * kv_w, 0.1);
    fill_linear(&mut v_src, t_k * kv_w, 0.5);
    cpu.paged_kv_append_f32(
        kv_w,
        t_k,
        0,
        block_size,
        num_blocks,
        n_layers,
        0,
        &block_table,
        need_blocks,
        &k_src,
        &v_src,
        &mut k_pool,
        &mut v_pool,
    )
    .unwrap();

    let t_q = 3usize;
    let mut q = cpu.alloc(t_q * q_w * 4).unwrap();
    fill_linear(&mut q, t_q * q_w, 0.01);
    let mut ctx_c = cpu.alloc(t_q * q_w * 4).unwrap();
    let mut ctx_p = cpu.alloc(t_q * q_w * 4).unwrap();

    cpu.gqa_attn_f32(
        t_q, t_k, n_heads, n_kv, d, scale, &q, q_w, &k_c, kv_w, &v_c, &mut ctx_c, true, 0,
    )
    .unwrap();
    cpu.paged_gqa_attn_f32(
        t_q,
        t_k,
        n_heads,
        n_kv,
        d,
        scale,
        &q,
        q_w,
        &k_pool,
        &v_pool,
        &block_table,
        need_blocks,
        block_size,
        num_blocks,
        n_layers,
        0,
        kv_w,
        &mut ctx_p,
        true,
        0,
    )
    .unwrap();

    let mut hc = vec![0u8; t_q * q_w * 4];
    let mut hp = vec![0u8; t_q * q_w * 4];
    cpu.copy_d2h(&mut hc, &ctx_c).unwrap();
    cpu.copy_d2h(&mut hp, &ctx_p).unwrap();
    assert_eq!(hc, hp, "paged attention must match contiguous on CPU");

    // Naive path must match the default CPU paged implementation.
    let mut ctx_n = cpu.alloc(t_q * q_w * 4).unwrap();
    cpu.paged_gqa_attn_naive_f32(
        t_q,
        t_k,
        n_heads,
        n_kv,
        d,
        scale,
        &q,
        q_w,
        &k_pool,
        &v_pool,
        &block_table,
        need_blocks,
        block_size,
        num_blocks,
        n_layers,
        0,
        kv_w,
        &mut ctx_n,
        true,
        0,
    )
    .unwrap();
    let mut hn = vec![0u8; t_q * q_w * 4];
    cpu.copy_d2h(&mut hn, &ctx_n).unwrap();
    assert_eq!(hp, hn, "naive paged attention must match default on CPU");
}

#[test]
fn session_grow_clear_returns_blocks() {
    let cpu = CpuBackend;
    let mut pool = GpuKvBlockPool::new(&cpu, 2, 2, 8, 64, 2).unwrap();
    let free0 = pool.free_count();
    let mut session = PagedKvSession::new(&cpu, 64, pool.block_size).unwrap();
    session.grow_for(&mut pool, 17).unwrap(); // 2 blocks
    assert_eq!(session.num_logical_blocks(), 2);
    assert_eq!(pool.free_count(), free0 - 2);
    session.clear(&mut pool).unwrap();
    assert_eq!(pool.free_count(), free0);
    assert_eq!(session.len, 0);
}
