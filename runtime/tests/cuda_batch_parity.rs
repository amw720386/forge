//! CUDA continuous-batching correctness (paged KV + chunked prefill).
//!
//! **Status: GPU execution unverified** until these tests pass on a real NVIDIA GPU
//! with `--features cuda` (not `cuda-stub`).
//!
//! Covers:
//! - B=1 matches single-request `forward_tokens` / sequential generate
//! - Batched greedy matches sequential per-prompt CUDA
//! - Mixed prompt lengths, cancel isolation, slot reuse, unknown id, full batch
//! - Chunked prefill: admit returns Prefilling; tokens emit after prefill completes

#![cfg(any(feature = "cuda", feature = "cuda-stub"))]

use forge_runtime::cuda_model::{CudaBatcher, CudaTransformer};
use forge_runtime::model::{
    BatchError, ForwardScratch, GenerateParams, LayerKvCaches, ModelConfig, Transformer,
};
use forge_runtime::sampling::greedy;
use std::collections::HashMap;

fn require_real_gpu() -> Option<()> {
    #[cfg(feature = "cuda")]
    {
        use forge_runtime::compute::CudaBackend;
        return match CudaBackend::try_new(0) {
            Ok(_) => Some(()),
            Err(e) => {
                eprintln!("SKIP: no CUDA device ({e}) — CUDA batch parity unverified");
                None
            }
        };
    }
    #[cfg(not(feature = "cuda"))]
    {
        eprintln!(
            "SKIP: cuda-stub build — CUDA batch parity unverified (needs --features cuda + NVIDIA GPU)"
        );
        None
    }
}

fn sequential_cuda_greedy(
    model: &CudaTransformer,
    prompt: &[u32],
    max_tokens: usize,
    max_seq: usize,
) -> Vec<u32> {
    use forge_runtime::cuda_model::kv::LayerGpuKvCaches;
    use forge_runtime::cuda_model::CudaScratch;

    let backend = model.backend();
    let cfg = &model.cfg;
    let mut caches = LayerGpuKvCaches::new(
        backend,
        cfg.n_layers,
        cfg.n_kv_heads,
        cfg.head_dim(),
        max_seq,
    )
    .unwrap();
    let mut scratch = CudaScratch::new(backend, cfg, max_seq).unwrap();
    let mut tokens = prompt.to_vec();
    model
        .forward_tokens(&tokens, 0, &mut caches, &mut scratch)
        .unwrap();
    let vocab = cfg.vocab_size;
    let mut logits = model.read_logits(&scratch, vocab).unwrap();
    let mut gen = Vec::new();
    for _ in 0..max_tokens {
        let next = greedy(&logits);
        gen.push(next);
        tokens.push(next);
        let pos = tokens.len() - 1;
        model
            .forward_tokens(&[next], pos, &mut caches, &mut scratch)
            .unwrap();
        logits = model.read_logits(&scratch, vocab).unwrap();
    }
    gen
}

#[test]
fn batch_size_one_matches_sequential() {
    if require_real_gpu().is_none() {
        return;
    }
    let cfg = ModelConfig::tiny();
    let cpu = Transformer::synthetic(cfg.clone(), 42).unwrap();
    let cuda = CudaTransformer::from_cpu(&cpu, 0).unwrap();
    let max_seq = 64;
    let params = GenerateParams {
        max_tokens: 6,
        temperature: 0.0,
        top_k: 0,
        eos_token_id: None,
    };
    let prompt = vec![1u32, 2, 3, 4];

    let expect = sequential_cuda_greedy(&cuda, &prompt, params.max_tokens, max_seq);

    let mut batcher = CudaBatcher::new_untokenized(
        CudaTransformer::from_cpu(&Transformer::synthetic(cfg, 42).unwrap(), 0).unwrap(),
        max_seq,
        1,
    )
    .unwrap();
    assert_eq!(batcher.max_batch(), 1);
    let id = batcher.admit_tokens(prompt, params.clone()).unwrap();
    let mut got = Vec::new();
    while batcher.active_count() > 0 {
        for ev in batcher.step().unwrap() {
            assert_eq!(ev.request_id, id);
            got.push(ev.token_id);
        }
    }
    assert_eq!(got, expect, "B=1 must match single-request CUDA");
    assert_eq!(batcher.stats().last_batch_size, 1);
    assert_eq!(batcher.stats().last_q_proj_m, 1);
}

#[test]
fn batched_greedy_matches_sequential() {
    if require_real_gpu().is_none() {
        return;
    }
    let cfg = ModelConfig::tiny();
    let seed = 7u32;
    let max_seq = 64;
    let params = GenerateParams {
        max_tokens: 5,
        temperature: 0.0,
        top_k: 0,
        eos_token_id: None,
    };
    let prompts: [Vec<u32>; 3] = [vec![1, 2, 3], vec![10, 11], vec![20, 21, 22, 23]];

    let cuda_ref =
        CudaTransformer::from_cpu(&Transformer::synthetic(cfg.clone(), seed).unwrap(), 0).unwrap();
    let mut seq_out: Vec<Vec<u32>> = Vec::new();
    for p in &prompts {
        seq_out.push(sequential_cuda_greedy(
            &cuda_ref,
            p,
            params.max_tokens,
            max_seq,
        ));
    }

    let mut batcher = CudaBatcher::new_untokenized(
        CudaTransformer::from_cpu(&Transformer::synthetic(cfg, seed).unwrap(), 0).unwrap(),
        max_seq,
        4,
    )
    .unwrap();
    let mut ids = Vec::new();
    for p in &prompts {
        ids.push(batcher.admit_tokens(p.clone(), params.clone()).unwrap());
    }
    assert_eq!(batcher.active_count(), 3);
    assert_eq!(batcher.free_slots(), 1);

    // Identical prompts in separate slots
    let twin = batcher
        .admit_tokens(prompts[0].clone(), params.clone())
        .unwrap();
    assert!(matches!(
        batcher.admit_tokens(vec![1], params.clone()),
        Err(BatchError::Full)
    ));
    batcher.cancel(twin).unwrap();

    let mut batched: HashMap<u64, Vec<u32>> = ids.iter().map(|&id| (id, Vec::new())).collect();
    while batcher.active_count() > 0 {
        let evs = batcher.step().unwrap();
        // After first decode with 3 active, stats should show B>=1
        if batcher.stats().decode_steps > 0 {
            assert!(batcher.stats().last_batch_size >= 1);
            assert_eq!(
                batcher.stats().last_q_proj_m,
                batcher.stats().last_batch_size
            );
        }
        for ev in evs {
            batched.get_mut(&ev.request_id).unwrap().push(ev.token_id);
        }
    }
    for (i, id) in ids.iter().enumerate() {
        assert_eq!(&batched[id], &seq_out[i], "request {i} token mismatch");
    }
}

#[test]
fn cancel_does_not_corrupt_sibling() {
    if require_real_gpu().is_none() {
        return;
    }
    let cfg = ModelConfig::tiny();
    let max_seq = 64;
    let params = GenerateParams {
        max_tokens: 4,
        temperature: 0.0,
        top_k: 0,
        eos_token_id: None,
    };
    let mut batcher = CudaBatcher::new_untokenized(
        CudaTransformer::from_cpu(&Transformer::synthetic(cfg.clone(), 99).unwrap(), 0).unwrap(),
        max_seq,
        2,
    )
    .unwrap();
    let keep = batcher.admit_tokens(vec![5, 6], params.clone()).unwrap();
    let drop = batcher.admit_tokens(vec![7, 8, 9], params.clone()).unwrap();

    // Chunked prefill: drain until both requests have emitted their first token.
    let mut first = Vec::new();
    while first.len() < 2 && batcher.active_count() > 0 {
        first.extend(batcher.step().unwrap());
    }
    assert_eq!(first.len(), 2);
    batcher.cancel(drop).unwrap();
    assert_eq!(batcher.active_count(), 1);
    assert!(matches!(
        batcher.cancel(drop),
        Err(BatchError::UnknownId(_))
    ));

    let mut got: Vec<u32> = first
        .into_iter()
        .filter(|e| e.request_id == keep)
        .map(|e| e.token_id)
        .collect();
    while batcher.active_count() > 0 {
        for ev in batcher.step().unwrap() {
            assert_eq!(ev.request_id, keep);
            got.push(ev.token_id);
        }
    }
    assert_eq!(got.len(), params.max_tokens);

    let cuda = CudaTransformer::from_cpu(&Transformer::synthetic(cfg, 99).unwrap(), 0).unwrap();
    let expect = sequential_cuda_greedy(&cuda, &[5, 6], params.max_tokens, max_seq);
    assert_eq!(got, expect);
}

#[test]
fn slot_reuse_after_completion() {
    if require_real_gpu().is_none() {
        return;
    }
    let cfg = ModelConfig::tiny();
    let max_seq = 64;
    let params = GenerateParams {
        max_tokens: 2,
        temperature: 0.0,
        top_k: 0,
        eos_token_id: None,
    };
    let mut batcher = CudaBatcher::new_untokenized(
        CudaTransformer::from_cpu(&Transformer::synthetic(cfg, 3).unwrap(), 0).unwrap(),
        max_seq,
        2,
    )
    .unwrap();
    let a = batcher.admit_tokens(vec![1, 2], params.clone()).unwrap();
    while batcher.active_count() > 0 {
        let _ = batcher.step().unwrap();
    }
    assert_eq!(batcher.free_slots(), 2);
    let b = batcher.admit_tokens(vec![3, 4, 5], params).unwrap();
    assert_ne!(a, b);
    assert_eq!(batcher.active_count(), 1);
}

#[test]
fn cancel_before_step_and_invalid_id() {
    if require_real_gpu().is_none() {
        return;
    }
    let cfg = ModelConfig::tiny();
    let mut batcher = CudaBatcher::new_untokenized(
        CudaTransformer::from_cpu(&Transformer::synthetic(cfg, 1).unwrap(), 0).unwrap(),
        32,
        2,
    )
    .unwrap();
    let id = batcher
        .admit_tokens(
            vec![1, 2],
            GenerateParams {
                max_tokens: 3,
                temperature: 0.0,
                top_k: 0,
                eos_token_id: None,
            },
        )
        .unwrap();
    batcher.cancel(id).unwrap();
    assert_eq!(batcher.active_count(), 0);
    assert!(matches!(
        batcher.cancel(999_999),
        Err(BatchError::UnknownId(_))
    ));
    assert!(matches!(batcher.step(), Err(BatchError::Empty)));
}

#[test]
fn cuda_batch_types_link() {
    let _ = std::any::type_name::<forge_runtime::cuda_model::CudaBatchScratch>();
    let _ = std::any::type_name::<forge_runtime::cuda_model::CudaBatchStats>();
    let _ = std::any::type_name::<forge_runtime::cuda_model::CudaBatcher>();
}

/// Host-only reference: CPU continuous batcher still matches sequential (regression).
#[test]
fn cpu_batch_regression_still_passes() {
    let cfg = ModelConfig::tiny();
    let model = Transformer::synthetic(cfg.clone(), 7).unwrap();
    let max_seq = 64;
    let params = GenerateParams {
        max_tokens: 4,
        temperature: 0.0,
        top_k: 0,
        eos_token_id: None,
    };
    let prompt = vec![1u32, 2, 3];

    let mut caches = LayerKvCaches::new(&cfg, max_seq).unwrap();
    let mut scratch = ForwardScratch::new(&cfg, max_seq).unwrap();
    let mut tokens = prompt.clone();
    model
        .forward_tokens(&tokens, 0, &mut caches, &mut scratch)
        .unwrap();
    let mut logits = scratch.logits.data()[..cfg.vocab_size].to_vec();
    let mut expect = Vec::new();
    for _ in 0..params.max_tokens {
        let next = greedy(&logits);
        expect.push(next);
        tokens.push(next);
        let pos = tokens.len() - 1;
        model
            .forward_tokens(&[next], pos, &mut caches, &mut scratch)
            .unwrap();
        logits = scratch.logits.data()[..cfg.vocab_size].to_vec();
    }

    let mut batcher = forge_runtime::model::ContinuousBatcher::new_untokenized(
        Transformer::synthetic(cfg, 7).unwrap(),
        max_seq,
        2,
    )
    .unwrap();
    let id = batcher.admit_tokens(prompt, params).unwrap();
    let mut got = Vec::new();
    while batcher.active_count() > 0 {
        for ev in batcher.step().unwrap() {
            assert_eq!(ev.request_id, id);
            got.push(ev.token_id);
        }
    }
    assert_eq!(got, expect);
}
