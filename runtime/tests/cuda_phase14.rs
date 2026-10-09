//! GPU-required packed/chunked prefill parity. Run with --test-threads=1.
#![cfg(feature = "cuda")]
use forge_runtime::cuda_model::paged_kv::{GpuKvBlockPool, PagedKvSession};
use forge_runtime::cuda_model::{CudaBatchScratch, CudaScratch, CudaTransformer, GpuWeightDType};
use forge_runtime::model::{ModelConfig, Transformer};
#[test]
fn packed_logits_match_individual_chunks() {
    std::env::set_var("FORGE_KV_BLOCK_SIZE", "16");
    for dtype in [GpuWeightDType::Fp32, GpuWeightDType::Bf16] {
        let cfg = ModelConfig::tiny();
        let cpu = Transformer::synthetic(cfg.clone(), 37).unwrap();
        let model = CudaTransformer::from_cpu_dtype(&cpu, 0, dtype).unwrap();
        let backend = model.backend();
        for b in [1, 2, 4, 8] {
            let mut pool = GpuKvBlockPool::new(
                backend,
                cfg.n_layers,
                cfg.n_kv_heads,
                cfg.head_dim(),
                128,
                b,
            )
            .unwrap();
            let mut oracle_pool = GpuKvBlockPool::new(
                backend,
                cfg.n_layers,
                cfg.n_kv_heads,
                cfg.head_dim(),
                128,
                b,
            )
            .unwrap();
            let mut sessions: Vec<_> = (0..b)
                .map(|_| PagedKvSession::new(backend, 128, 16).unwrap())
                .collect();
            let mut oracle: Vec<_> = (0..b)
                .map(|_| PagedKvSession::new(backend, 128, 16).unwrap())
                .collect();
            let mut scratch = CudaScratch::new(backend, &cfg, 512).unwrap();
            let mut single = CudaScratch::new(backend, &cfg, 128).unwrap();
            let mut output = CudaBatchScratch::new(backend, &cfg, b).unwrap();
            let mut max_error = 0f32;
            // Different prefixes, physical block crossings and different chunk lengths.
            for round in 0..3 {
                let chunks: Vec<Vec<u32>> = (0..b)
                    .map(|r| {
                        (0..(if round == 0 { 15 + r } else { 3 + r }))
                            .map(|j| ((r + j + round * 7) % cfg.vocab_size) as u32)
                            .collect()
                    })
                    .collect();
                let mut expected = Vec::new();
                for r in 0..b {
                    let pos = oracle[r].len;
                    model
                        .forward_tokens_paged(
                            &chunks[r],
                            pos,
                            &mut oracle_pool,
                            &mut oracle[r],
                            &mut single,
                        )
                        .unwrap();
                    expected.extend(model.read_logits(&single, cfg.vocab_size).unwrap());
                }
                let mut refs: Vec<_> = sessions.iter_mut().collect();
                let rows = model
                    .forward_prefill_paged(
                        &chunks,
                        &vec![true; b],
                        &mut pool,
                        &mut refs,
                        &mut scratch,
                        &mut output,
                    )
                    .unwrap();
                assert_eq!(rows, (0..b).collect::<Vec<_>>());
                let got = model.read_batch_logits(&output, b, cfg.vocab_size).unwrap();
                for (x, y) in expected.iter().zip(got.iter()) {
                    max_error = max_error.max((x - y).abs());
                    assert!(
                        (x - y).abs() <= 1e-4 + 1e-3 * x.abs().max(y.abs()),
                        "dtype={dtype:?} B={b} round={round}: {x} {y}"
                    );
                }
                for r in 0..b {
                    assert_eq!(sessions[r].len, oracle[r].len);
                }
            }
            // Partial chunk omits logits, then slot reuse must not read previous ownership.
            sessions[0].clear(&mut pool).unwrap();
            oracle[0].clear(&mut oracle_pool).unwrap();
            let chunks = vec![vec![1, 2, 3]];
            let mut refs = vec![&mut sessions[0]];
            assert!(model
                .forward_prefill_paged(
                    &chunks,
                    &[false],
                    &mut pool,
                    &mut refs,
                    &mut scratch,
                    &mut output
                )
                .unwrap()
                .is_empty());
            println!("PHASE14_PREFILL_PARITY dtype={dtype:?} B={b} max_abs_error={max_error}");
        }
    }
    std::env::remove_var("FORGE_KV_BLOCK_SIZE");
}

#[test]
#[ignore = "requires the full Qwen checkpoint; explicit bounded GPU validation"]
fn real_model_packed_logits() {
    let path = std::env::var("FORGE_TEST_MODEL").expect("FORGE_TEST_MODEL required");
    let cpu = forge_runtime::model::load_qwen2_dir(path).unwrap();
    let cfg = cpu.cfg.clone();
    for dtype in [GpuWeightDType::Fp32, GpuWeightDType::Bf16] {
        let model = CudaTransformer::from_cpu_dtype(&cpu, 0, dtype).unwrap();
        let backend = model.backend();
        for b in [1, 2, 4, 8] {
            let mut pool = GpuKvBlockPool::new(
                backend,
                cfg.n_layers,
                cfg.n_kv_heads,
                cfg.head_dim(),
                512,
                b,
            )
            .unwrap();
            let mut oracle_pool = GpuKvBlockPool::new(
                backend,
                cfg.n_layers,
                cfg.n_kv_heads,
                cfg.head_dim(),
                512,
                b,
            )
            .unwrap();
            let mut sessions: Vec<_> = (0..b)
                .map(|_| PagedKvSession::new(backend, 512, pool.block_size).unwrap())
                .collect();
            let mut oracle: Vec<_> = (0..b)
                .map(|_| PagedKvSession::new(backend, 512, oracle_pool.block_size).unwrap())
                .collect();
            let mut scratch = CudaScratch::new(backend, &cfg, 256).unwrap();
            let mut single = CudaScratch::new(backend, &cfg, 128).unwrap();
            let mut output = CudaBatchScratch::new(backend, &cfg, b).unwrap();
            let mut maximum = 0f32;
            let mut disagreement = 0;
            for round in 0..8 {
                let chunks: Vec<Vec<u32>> = (0..b)
                    .map(|row| {
                        (0..32)
                            .map(|j| ((row * 101 + round * 32 + j + 1000) % cfg.vocab_size) as u32)
                            .collect()
                    })
                    .collect();
                std::env::set_var("FORGE_GEMM_LT", "0");
                std::env::set_var("FORGE_PREFILL_ATTN_BLOCK", "0");
                let mut expected = Vec::new();
                for row in 0..b {
                    let pos = oracle[row].len;
                    model
                        .forward_tokens_paged(
                            &chunks[row],
                            pos,
                            &mut oracle_pool,
                            &mut oracle[row],
                            &mut single,
                        )
                        .unwrap();
                    expected.extend(model.read_logits(&single, cfg.vocab_size).unwrap());
                }
                std::env::set_var("FORGE_GEMM_LT", "1");
                std::env::set_var("FORGE_PREFILL_ATTN_BLOCK", "1");
                let mut refs: Vec<_> = sessions.iter_mut().collect();
                model
                    .forward_prefill_paged(
                        &chunks,
                        &vec![true; b],
                        &mut pool,
                        &mut refs,
                        &mut scratch,
                        &mut output,
                    )
                    .unwrap();
                let got = model.read_batch_logits(&output, b, cfg.vocab_size).unwrap();
                for (x, y) in expected.iter().zip(&got) {
                    maximum = maximum.max((x - y).abs());
                }
                for row in 0..b {
                    let a = &expected[row * cfg.vocab_size..(row + 1) * cfg.vocab_size];
                    let z = &got[row * cfg.vocab_size..(row + 1) * cfg.vocab_size];
                    disagreement += usize::from(
                        forge_runtime::sampling::greedy(a) != forge_runtime::sampling::greedy(z),
                    );
                }
            }
            println!("PHASE14_REAL_PARITY dtype={dtype:?} B={b} max_abs_error={maximum} greedy_disagreements={disagreement}");
            // Report full pretrained-model errors before asserting greedy stability.
            assert_eq!(
                disagreement, 0,
                "investigate pretrained greedy disagreement; do not loosen tolerances"
            );
        }
    }
    std::env::remove_var("FORGE_GEMM_LT");
    std::env::remove_var("FORGE_PREFILL_ATTN_BLOCK");
}

#[test]
fn mixed_lifecycle_and_exhaustion() {
    use forge_runtime::cuda_model::CudaBatcher;
    use forge_runtime::model::{BatchError, GenerateParams};
    std::env::set_var("FORGE_KV_BLOCK_SIZE", "16");
    std::env::set_var("FORGE_PREFILL_CHUNK", "8");
    std::env::set_var("FORGE_STEP_TOKEN_BUDGET", "32");
    for dtype in [GpuWeightDType::Fp32, GpuWeightDType::Bf16] {
        let mut streams = Vec::new();
        for packed in [false, true] {
            std::env::set_var("FORGE_PACKED_PREFILL", if packed { "1" } else { "0" });
            std::env::set_var("FORGE_EARLY_EMIT", if packed { "1" } else { "0" });
            let cfg = ModelConfig::tiny();
            let model = CudaTransformer::from_cpu_dtype(
                &Transformer::synthetic(cfg.clone(), 37).unwrap(),
                0,
                dtype,
            )
            .unwrap();
            let mut batch = CudaBatcher::new_untokenized(model, 256, 8).unwrap();
            let mut ids = Vec::new();
            for (r, len) in [1, 3, 17, 33, 65, 121, 189, 205].into_iter().enumerate() {
                ids.push(
                    batch
                        .admit_tokens(
                            (0..len)
                                .map(|i| ((i + r) % cfg.vocab_size) as u32)
                                .collect(),
                            GenerateParams {
                                max_tokens: 3 + r % 5,
                                ..GenerateParams::default()
                            },
                        )
                        .unwrap(),
                );
            }
            // Cancel during partial prefill, then reuse the slot during ongoing sibling work.
            batch.step().unwrap();
            batch.cancel(ids[3]).unwrap();
            let replacement = batch
                .admit_tokens(
                    vec![2, 3, 4, 5, 6],
                    GenerateParams {
                        max_tokens: 4,
                        ..GenerateParams::default()
                    },
                )
                .unwrap();
            let mut outputs = std::collections::BTreeMap::<u64, Vec<u32>>::new();
            let mut iterations = 0;
            while batch.active_count() > 0 {
                iterations += 1;
                assert!(iterations < 1000, "prefill starvation");
                for event in batch.step().unwrap() {
                    assert_ne!(event.request_id, ids[3]);
                    outputs
                        .entry(event.request_id)
                        .or_default()
                        .push(event.token_id);
                }
            }
            assert_eq!(outputs[&replacement].len(), 4);
            assert_eq!(batch.kv_stats().allocated_blocks, 0);
            streams.push(outputs);
        }
        assert_eq!(
            streams[0], streams[1],
            "mixed arrival/cancellation greedy parity"
        );
    }
    std::env::set_var("FORGE_KV_NUM_BLOCKS", "1");
    let cfg = ModelConfig::tiny();
    let model = CudaTransformer::from_cpu(&Transformer::synthetic(cfg, 37).unwrap(), 0).unwrap();
    let mut batch = CudaBatcher::new_untokenized(model, 64, 2).unwrap();
    let id = batch
        .admit_tokens(vec![1; 16], GenerateParams::default())
        .unwrap();
    batch.step().unwrap();
    assert!(matches!(
        batch.admit_tokens(vec![2; 17], GenerateParams::default()),
        Err(BatchError::KvExhausted(_))
    ));
    batch.cancel(id).unwrap();
    assert_eq!(batch.kv_stats().allocated_blocks, 0);
    for key in [
        "FORGE_KV_NUM_BLOCKS",
        "FORGE_KV_BLOCK_SIZE",
        "FORGE_PREFILL_CHUNK",
        "FORGE_STEP_TOKEN_BUDGET",
        "FORGE_PACKED_PREFILL",
        "FORGE_EARLY_EMIT",
    ] {
        std::env::remove_var(key);
    }
}
