//! CPU vs CUDA transformer parity.
//!
//! **Status: GPU execution unverified** until these tests pass on a real NVIDIA GPU
//! with `--features cuda` (not `cuda-stub`).
//!
//! - With `cuda-stub`: tests compile but skip (no device).
//! - With `cuda` + GPU: runs automatically.
//! - With `cuda` + no GPU: skips with a message.

#![cfg(any(feature = "cuda", feature = "cuda-stub"))]

use forge_runtime::compute::CudaBackend;
use forge_runtime::cuda_model::kv::LayerGpuKvCaches;
use forge_runtime::cuda_model::{CudaScratch, CudaTransformer};
use forge_runtime::model::{ForwardScratch, LayerKvCaches, ModelConfig, Transformer};
use forge_runtime::sampling::greedy;

fn approx_logits(a: &[f32], b: &[f32], atol: f32, rtol: f32) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let d = (x - y).abs();
        let tol = atol + rtol * x.abs().max(y.abs());
        assert!(d <= tol, "logit {i}: {x} vs {y} (tol {tol})");
    }
}

fn require_real_gpu() -> Option<()> {
    #[cfg(all(feature = "cuda-stub", not(feature = "cuda")))]
    {
        eprintln!(
            "SKIP: cuda-stub build — GPU parity unverified (needs --features cuda + NVIDIA GPU)"
        );
        return None;
    }
    match CudaBackend::try_new(0) {
        Ok(_) => Some(()),
        Err(e) => {
            eprintln!("SKIP: no CUDA device ({e}) — GPU parity unverified");
            None
        }
    }
}

#[test]
fn synthetic_prefill_parity() {
    if require_real_gpu().is_none() {
        return;
    }

    let cfg = ModelConfig::tiny();
    let model = Transformer::synthetic(cfg.clone(), 99).unwrap();
    let tokens: Vec<u32> = vec![3, 7, 11, 2];
    let max_seq = 16;

    let mut cpu_cache = LayerKvCaches::new(&cfg, max_seq).unwrap();
    let mut cpu_scratch = ForwardScratch::new(&cfg, max_seq).unwrap();
    model
        .forward_tokens(&tokens, 0, &mut cpu_cache, &mut cpu_scratch)
        .unwrap();
    let cpu_logits = cpu_scratch.logits.data().to_vec();

    let gpu = CudaTransformer::from_cpu(&model, 0).expect("cuda load");
    let backend = gpu.backend();
    let mut gpu_cache = LayerGpuKvCaches::new(
        backend,
        cfg.n_layers,
        cfg.n_kv_heads,
        cfg.head_dim(),
        max_seq,
    )
    .unwrap();
    let mut gpu_scratch = CudaScratch::new(backend, &cfg, max_seq).unwrap();
    gpu.forward_tokens(&tokens, 0, &mut gpu_cache, &mut gpu_scratch)
        .unwrap();
    let gpu_logits = gpu.read_logits(&gpu_scratch, cfg.vocab_size).unwrap();

    approx_logits(&cpu_logits, &gpu_logits, 1e-3, 1e-2);
    assert_eq!(greedy(&cpu_logits), greedy(&gpu_logits));
}

#[test]
fn synthetic_decode_parity() {
    if require_real_gpu().is_none() {
        return;
    }
    let cfg = ModelConfig::tiny();
    let model = Transformer::synthetic(cfg.clone(), 123).unwrap();
    let max_seq = 16;
    let prompt = vec![1u32, 2, 3];

    let mut cpu_cache = LayerKvCaches::new(&cfg, max_seq).unwrap();
    let mut cpu_scratch = ForwardScratch::new(&cfg, max_seq).unwrap();
    model
        .forward_tokens(&prompt, 0, &mut cpu_cache, &mut cpu_scratch)
        .unwrap();
    let t1_cpu = greedy(cpu_scratch.logits.data());
    model
        .forward_tokens(&[t1_cpu], prompt.len(), &mut cpu_cache, &mut cpu_scratch)
        .unwrap();
    let t2_cpu = greedy(cpu_scratch.logits.data());

    let gpu = CudaTransformer::from_cpu(&model, 0).unwrap();
    let backend = gpu.backend();
    let mut gpu_cache = LayerGpuKvCaches::new(
        backend,
        cfg.n_layers,
        cfg.n_kv_heads,
        cfg.head_dim(),
        max_seq,
    )
    .unwrap();
    let mut gpu_scratch = CudaScratch::new(backend, &cfg, max_seq).unwrap();
    gpu.forward_tokens(&prompt, 0, &mut gpu_cache, &mut gpu_scratch)
        .unwrap();
    let l1 = gpu.read_logits(&gpu_scratch, cfg.vocab_size).unwrap();
    let t1_gpu = greedy(&l1);
    assert_eq!(t1_cpu, t1_gpu);
    gpu.forward_tokens(&[t1_gpu], prompt.len(), &mut gpu_cache, &mut gpu_scratch)
        .unwrap();
    let l2 = gpu.read_logits(&gpu_scratch, cfg.vocab_size).unwrap();
    assert_eq!(t2_cpu, greedy(&l2));
}
