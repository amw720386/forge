//! Explicit L4 validation; CPU/stub runs must not be counted as GPU passes.
#![cfg(feature = "cuda")]
use forge_runtime::cuda_model::paged_kv::{GpuKvBlockPool, PagedKvSession};
use forge_runtime::cuda_model::{
    CudaBatchScratch, CudaBatchStats, CudaScratch, CudaTransformer, GpuWeightDType,
};
use forge_runtime::model::{ModelConfig, Transformer};
use forge_runtime::sampling::greedy;

fn flags(side: usize, tile: &str) {
    std::env::set_var("FORGE_PREFILL_ATTN_BLOCK", "1");
    std::env::set_var("FORGE_BATCHED_PAGED", "1");
    std::env::set_var("FORGE_REUSE_BF16", "1");
    std::env::set_var("FORGE_CUDA_GRAPHS", if side == 0 { "0" } else { "1" });
    std::env::set_var("FORGE_DECODE_ATTN_BLOCK", if side == 0 { "0" } else { "1" });
    std::env::set_var(
        "FORGE_ATTN_SOFTMAX_TILE",
        if side == 0 { "1" } else { tile },
    );
}
fn exercise(cpu: &Transformer, real: bool) {
    for dtype in [GpuWeightDType::Fp32, GpuWeightDType::Bf16] {
        let cfg = &cpu.cfg;
        let model = CudaTransformer::from_cpu_dtype(cpu, 0, dtype).unwrap();
        let backend = model.backend();
        for b in [1, 2, 4, 8] {
            for tile in if real { vec!["1"] } else { vec!["1"] } {
                let mut pools: Vec<_> = (0..2)
                    .map(|_| {
                        GpuKvBlockPool::new(
                            backend,
                            cfg.n_layers,
                            cfg.n_kv_heads,
                            cfg.head_dim(),
                            4608,
                            b,
                        )
                        .unwrap()
                    })
                    .collect();
                let mut sessions: Vec<Vec<_>> = (0..2)
                    .map(|_| {
                        (0..b)
                            .map(|_| {
                                PagedKvSession::new(backend, 4608, pools[0].block_size).unwrap()
                            })
                            .collect()
                    })
                    .collect();
                let mut prefill: Vec<_> = (0..2)
                    .map(|_| CudaScratch::new(backend, cfg, b * 128).unwrap())
                    .collect();
                let mut output: Vec<_> = (0..2)
                    .map(|_| CudaBatchScratch::new(backend, cfg, b).unwrap())
                    .collect();
                let mut maximum = 0f32;
                let mut decisions = 0;
                for target in if real {
                    vec![512, 4096]
                } else {
                    vec![128, 512, 1024, 2048, 4096]
                } {
                    while sessions[0][0].len < target {
                        let n = (target - sessions[0][0].len).min(128);
                        let chunks: Vec<Vec<u32>> = (0..b)
                            .map(|r| {
                                (0..n)
                                    .map(|j| {
                                        ((1000 + r * 101 + sessions[0][r].len + j) % cfg.vocab_size)
                                            as u32
                                    })
                                    .collect()
                            })
                            .collect();
                        for side in 0..2 {
                            flags(side, tile);
                            let mut refs: Vec<_> = sessions[side].iter_mut().collect();
                            model
                                .forward_prefill_paged(
                                    &chunks,
                                    &vec![true; b],
                                    &mut pools[side],
                                    &mut refs,
                                    &mut prefill[side],
                                    &mut output[side],
                                )
                                .unwrap();
                        }
                    }
                    for step in 0..4 {
                        let mut logits = Vec::new();
                        for side in 0..2 {
                            flags(side, tile);
                            logits.push(
                                model
                                    .read_batch_logits(&output[side], b, cfg.vocab_size)
                                    .unwrap(),
                            );
                        }
                        for (x, y) in logits[0].iter().zip(&logits[1]) {
                            assert!(x.is_finite() && y.is_finite());
                            maximum = maximum.max((x - y).abs());
                            // BF16 rounds intermediate activations; full-path checks follow the
                            // Phase 14 measured-logit/greedy contract. FP32 and standalone
                            // attention retain the original strict numerical bound.
                            if !real && dtype == GpuWeightDType::Fp32 {
                                assert!(
                                    (x - y).abs() <= 1e-4 + 1e-3 * x.abs().max(y.abs()),
                                    "{dtype:?} B={b} tile={tile} prefix={target}: {x} {y}"
                                );
                            }
                        }
                        let tokens:Vec<_>=(0..b).map(|r|{
      let a=greedy(&logits[0][r*cfg.vocab_size..(r+1)*cfg.vocab_size]);
      let z=greedy(&logits[1][r*cfg.vocab_size..(r+1)*cfg.vocab_size]);
      assert_eq!(a,z,"greedy {dtype:?} B={b} tile={tile} target={target} step={step}");decisions+=1;a
     }).collect();
                        for side in 0..2 {
                            flags(side, tile);
                            let positions: Vec<_> = sessions[side].iter().map(|s| s.len).collect();
                            let mut refs: Vec<_> = sessions[side].iter_mut().collect();
                            model
                                .forward_batch_decode_paged(
                                    &tokens,
                                    &positions,
                                    &mut pools[side],
                                    &mut refs,
                                    &mut output[side],
                                    &mut CudaBatchStats::default(),
                                )
                                .unwrap();
                        }
                    }
                }
                // Clear and replace a slot after captured block growth, then replay the same shape.
                for side in 0..2 {
                    flags(side, tile);
                    sessions[side][0].clear(&mut pools[side]).unwrap();
                    let mut refs = vec![&mut sessions[side][0]];
                    model
                        .forward_prefill_paged(
                            &[vec![7; 17]],
                            &[true],
                            &mut pools[side],
                            &mut refs,
                            &mut prefill[side],
                            &mut output[side],
                        )
                        .unwrap();
                    let positions: Vec<_> = sessions[side].iter().map(|s| s.len).collect();
                    let mut refs: Vec<_> = sessions[side].iter_mut().collect();
                    model
                        .forward_batch_decode_paged(
                            &vec![3; b],
                            &positions,
                            &mut pools[side],
                            &mut refs,
                            &mut output[side],
                            &mut CudaBatchStats::default(),
                        )
                        .unwrap();
                }
                let a = model
                    .read_batch_logits(&output[0], b, cfg.vocab_size)
                    .unwrap();
                let z = model
                    .read_batch_logits(&output[1], b, cfg.vocab_size)
                    .unwrap();
                for r in 0..b {
                    assert_eq!(
                        greedy(&a[r * cfg.vocab_size..(r + 1) * cfg.vocab_size]),
                        greedy(&z[r * cfg.vocab_size..(r + 1) * cfg.vocab_size])
                    );
                }
                println!("PHASE15_LONG_PARITY real={real} dtype={dtype:?} B={b} tile={tile} max_abs_error={maximum} greedy_decisions={decisions}");
            }
        }
    }
    for key in [
        "FORGE_DECODE_ATTN_BLOCK",
        "FORGE_ATTN_SOFTMAX_TILE",
        "FORGE_CUDA_GRAPHS",
        "FORGE_PREFILL_ATTN_BLOCK",
        "FORGE_BATCHED_PAGED",
        "FORGE_REUSE_BF16",
    ] {
        std::env::remove_var(key);
    }
}
#[test]
fn long_attention_graph_growth_and_replacement() {
    let mut cfg = ModelConfig::tiny();
    cfg.hidden_size = 896;
    cfg.n_heads = 14;
    cfg.n_kv_heads = 2;
    cfg.intermediate_size = 512;
    cfg.max_position_embeddings = 4608;
    exercise(&Transformer::synthetic(cfg, 37).unwrap(), false);
}
#[test]
#[ignore = "full checkpoint and bounded paid L4 session required"]
fn real_long_greedy_agreement() {
    let cpu = forge_runtime::model::load_qwen2_dir(
        std::env::var("FORGE_TEST_MODEL").expect("FORGE_TEST_MODEL"),
    )
    .unwrap();
    exercise(&cpu, true);
}

#[test]
#[ignore = "full checkpoint and paid L4 required"]
fn real_http_long_teacher_forced() {
    let dir = std::env::var("FORGE_TEST_MODEL").unwrap();
    let cpu = forge_runtime::model::load_qwen2_dir(&dir).unwrap();
    let tok = tokenizers::Tokenizer::from_file(format!("{dir}/tokenizer.json")).unwrap();
    let prompt = format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        "Explain how a computer stores and retrieves information. ".repeat(100)
    );
    let ids = tok.encode(prompt, false).unwrap().get_ids().to_vec();
    for dtype in [GpuWeightDType::Fp32, GpuWeightDType::Bf16] {
        let model = CudaTransformer::from_cpu_dtype(&cpu, 0, dtype).unwrap();
        let cfg = &cpu.cfg;
        for b in [1, 2, 4, 8] {
            let mut pools: Vec<_> = (0..2)
                .map(|_| {
                    GpuKvBlockPool::new(
                        model.backend(),
                        cfg.n_layers,
                        cfg.n_kv_heads,
                        cfg.head_dim(),
                        2048,
                        b,
                    )
                    .unwrap()
                })
                .collect();
            let mut sessions: Vec<Vec<_>> = (0..2)
                .map(|_| {
                    (0..b)
                        .map(|_| {
                            PagedKvSession::new(model.backend(), 2048, pools[0].block_size).unwrap()
                        })
                        .collect()
                })
                .collect();
            let mut prefill: Vec<_> = (0..2)
                .map(|_| CudaScratch::new(model.backend(), cfg, b * 128).unwrap())
                .collect();
            let mut output: Vec<_> = (0..2)
                .map(|_| CudaBatchScratch::new(model.backend(), cfg, b).unwrap())
                .collect();
            for chunk in ids.chunks(128) {
                for side in 0..2 {
                    flags(side, "1");
                    std::env::set_var("FORGE_CUDA_GRAPHS", "0");
                    let mut refs: Vec<_> = sessions[side].iter_mut().collect();
                    model
                        .forward_prefill_paged(
                            &vec![chunk.to_vec(); b],
                            &vec![true; b],
                            &mut pools[side],
                            &mut refs,
                            &mut prefill[side],
                            &mut output[side],
                        )
                        .unwrap();
                }
            }
            let mut disagreements = 0;
            let mut maximum = 0f32;
            for step in 0..64 {
                let a = model
                    .read_batch_logits(&output[0], b, cfg.vocab_size)
                    .unwrap();
                let z = model
                    .read_batch_logits(&output[1], b, cfg.vocab_size)
                    .unwrap();
                maximum = a
                    .iter()
                    .zip(&z)
                    .fold(maximum, |m, (x, y)| m.max((x - y).abs()));
                let tokens: Vec<_>=(0..b).map(|r| {let x=greedy(&a[r*cfg.vocab_size..(r+1)*cfg.vocab_size]);let y=greedy(&z[r*cfg.vocab_size..(r+1)*cfg.vocab_size]);if x!=y {disagreements+=1;println!("NATURAL_MISMATCH dtype={dtype:?} step={step} row={r} baseline={x} candidate={y} baseline_margin={} candidate_margin={}",a[r*cfg.vocab_size+x as usize]-a[r*cfg.vocab_size+y as usize],z[r*cfg.vocab_size+x as usize]-z[r*cfg.vocab_size+y as usize]);}x}).collect();
                for side in 0..2 {
                    flags(side, "1");
                    std::env::set_var("FORGE_CUDA_GRAPHS", "0");
                    let positions: Vec<_> = sessions[side].iter().map(|s| s.len).collect();
                    let mut refs: Vec<_> = sessions[side].iter_mut().collect();
                    model
                        .forward_batch_decode_paged(
                            &tokens,
                            &positions,
                            &mut pools[side],
                            &mut refs,
                            &mut output[side],
                            &mut CudaBatchStats::default(),
                        )
                        .unwrap();
                }
            }
            println!("NATURAL_LONG dtype={dtype:?} B={b} max_abs_error={maximum} disagreements={disagreements}/{}",b*64);
            assert_eq!(disagreements, 0);
        }
    }
}
#[test]
fn scheduling_progress_cancellation_and_reuse() {
    use forge_runtime::cuda_model::CudaBatcher;
    use forge_runtime::model::GenerateParams;
    for policy in ["fixed"] {
        for budget in [1, 2, 16] {
            std::env::set_var("FORGE_SCHED_POLICY", policy);
            std::env::set_var("FORGE_STEP_TOKEN_BUDGET", budget.to_string());
            std::env::set_var("FORGE_PREFILL_CHUNK", "16");
            std::env::set_var("FORGE_PACKED_PREFILL", "1");
            std::env::set_var("FORGE_EARLY_EMIT", "1");
            let cpu = Transformer::synthetic(ModelConfig::tiny(), 37).unwrap();
            let model = CudaTransformer::from_cpu(&cpu, 0).unwrap();
            let mut batch = CudaBatcher::new_untokenized(model, 512, 4).unwrap();
            let params = GenerateParams {
                max_tokens: 32,
                ..GenerateParams::default()
            };
            let a = batch.admit_tokens(vec![1], params.clone()).unwrap();
            let cancelled = batch.admit_tokens(vec![2; 255], params.clone()).unwrap();
            let long = batch.admit_tokens(vec![3; 257], params.clone()).unwrap();
            let mut seen = std::collections::BTreeSet::new();
            let mut cancelled_decode = false;
            for turn in 0..3000 {
                if turn == 2 {
                    batch.cancel(cancelled).unwrap();
                    batch.admit_tokens(vec![4; 17], params.clone()).unwrap();
                }
                if batch.active_count() == 0 {
                    break;
                }
                for e in batch.step().unwrap() {
                    assert_ne!(e.request_id, cancelled);
                    seen.insert(e.request_id);
                    if e.request_id == a && !e.done {
                        batch.cancel(a).unwrap();
                        cancelled_decode = true;
                    }
                }
                assert!(
                    batch.stats().last_scheduled_decode + batch.stats().last_scheduled_prefill
                        <= budget
                );
                assert!(turn < 2999, "policy={policy} budget={budget}: starvation");
            }
            assert!(cancelled_decode);
            assert!(seen.contains(&long));
            assert_eq!(batch.kv_stats().allocated_blocks, 0);
            println!("PHASE15_SCHED_LIFECYCLE policy={policy} budget={budget} passed");
        }
    }
    for key in [
        "FORGE_SCHED_POLICY",
        "FORGE_STEP_TOKEN_BUDGET",
        "FORGE_PREFILL_CHUNK",
        "FORGE_PACKED_PREFILL",
        "FORGE_EARLY_EMIT",
    ] {
        std::env::remove_var(key);
    }
}
