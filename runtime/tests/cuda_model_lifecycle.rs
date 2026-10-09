//! Explicit pretrained second-model validation; never runs under cuda-stub.
#![cfg(any(feature = "cuda", feature = "cuda-stub"))]
use forge_runtime::cuda_model::{CudaBatcher, CudaTransformer, GpuWeightDType};
use forge_runtime::model::{load_hf_dir, GenerateParams};
use std::collections::HashMap;

#[test]
#[ignore = "requires real CUDA; checks graph scratch reuse across model lifetimes"]
fn graph_reuse_across_architecture_and_weight_change() {
    use forge_runtime::cuda_model::paged_kv::{GpuKvBlockPool, PagedKvSession};
    use forge_runtime::cuda_model::{CudaBatchScratch, CudaBatchStats, CudaScratch};
    use forge_runtime::model::{
        Architecture, ForwardScratch, LayerKvCaches, ModelConfig, Transformer, TransformerWeights,
    };
    std::env::set_var("FORGE_CUDA_GRAPHS", "1");
    std::env::set_var("FORGE_BATCHED_PAGED", "1");
    let cfg = ModelConfig::tiny();
    let cpu = Transformer::synthetic(cfg.clone(), 42).unwrap();
    let mut gpu = Some(CudaTransformer::from_cpu(&cpu, 0).unwrap());
    let mut scratch = CudaBatchScratch::new(gpu.as_ref().unwrap().backend(), &cfg, 1).unwrap();
    let mut pool = GpuKvBlockPool::new(
        gpu.as_ref().unwrap().backend(),
        cfg.n_layers,
        cfg.n_kv_heads,
        cfg.head_dim(),
        32,
        1,
    )
    .unwrap();
    let mut session =
        PagedKvSession::new(gpu.as_ref().unwrap().backend(), 32, pool.block_size).unwrap();
    for family in [Architecture::Qwen2, Architecture::Llama] {
        let original = Transformer::synthetic(
            cfg.clone(),
            if family == Architecture::Qwen2 {
                42
            } else {
                99
            },
        )
        .unwrap();
        let mut changed = cfg.clone();
        changed.architecture = family;
        let mut layers: Vec<_> = original.layers.iter().map(|l| l.weights.clone()).collect();
        if family == Architecture::Llama {
            for layer in &mut layers {
                layer.q_proj.bias = None;
                layer.k_proj.bias = None;
                layer.v_proj.bias = None;
            }
        }
        let model = Transformer::from_weights(
            changed.clone(),
            TransformerWeights {
                embed: original.embed,
                layers,
                final_norm: original.final_norm,
                lm_head: None,
            },
        )
        .unwrap();
        if family == Architecture::Llama {
            drop(gpu.take());
            gpu = Some(CudaTransformer::from_cpu(&model, 0).unwrap());
        }
        let gpu = gpu.as_ref().unwrap();
        session.clear(&mut pool).unwrap();
        let mut prefill = CudaScratch::new(gpu.backend(), &changed, 32).unwrap();
        gpu.forward_tokens_paged(&[3, 4, 5], 0, &mut pool, &mut session, &mut prefill)
            .unwrap();
        let mut caches = LayerKvCaches::new(&changed, 32).unwrap();
        let mut reference = ForwardScratch::new(&changed, 32).unwrap();
        model
            .forward_tokens(&[3, 4, 5], 0, &mut caches, &mut reference)
            .unwrap();
        for step in 0..4 {
            gpu.forward_batch_decode_paged(
                &[6],
                &[3 + step],
                &mut pool,
                &mut [&mut session],
                &mut scratch,
                &mut CudaBatchStats::default(),
            )
            .unwrap();
            model
                .forward_tokens(&[6], 3 + step, &mut caches, &mut reference)
                .unwrap();
            let logits = gpu
                .read_batch_logits(&scratch, 1, changed.vocab_size)
                .unwrap();
            for (&a, &b) in logits.iter().zip(reference.logits.data()) {
                assert!((a - b).abs() <= 1e-3 + 1e-2 * a.abs().max(b.abs()));
            }
        }
        println!("MODEL_GRAPH_MODEL_REPLACEMENT family={family:?} pass");
    }
}

#[test]
#[ignore = "requires real CUDA and downloaded SmolLM2; run in bounded session"]
fn smollm2_paged_batch_graph_cancel_and_reuse() {
    let dir = std::env::var("FORGE_SMOLLM_MODEL").expect("FORGE_SMOLLM_MODEL");
    let refs = std::env::var("FORGE_SMOLLM_REFS").expect("FORGE_SMOLLM_REFS");
    let json: serde_json::Value = serde_json::from_slice(
        &std::fs::read(std::path::Path::new(&refs).join("cases.json")).unwrap(),
    )
    .unwrap();
    let cases = json["cases"].as_array().unwrap();
    let cpu = load_hf_dir(dir).unwrap();
    for dtype in [GpuWeightDType::Fp32, GpuWeightDType::Bf16] {
        for graphs in ["0", "1"] {
            std::env::set_var("FORGE_CUDA_GRAPHS", graphs);
            std::env::set_var("FORGE_BATCHED_PAGED", "1");
            std::env::set_var("FORGE_REUSE_BF16", "1");
            std::env::set_var("FORGE_PREFILL_ATTN_BLOCK", "1");
            std::env::set_var("FORGE_DECODE_ATTN_BLOCK", "1");
            std::env::set_var("FORGE_PREFILL_CHUNK", "16");
            let gpu = CudaTransformer::from_cpu_dtype(&cpu, 0, dtype).unwrap();
            let mut batch = CudaBatcher::new_untokenized(gpu, 256, 4).unwrap();
            let mut expected = HashMap::new();
            let params = GenerateParams {
                max_tokens: 12,
                temperature: 0.0,
                top_k: 0,
                eos_token_id: Some(cpu.cfg.eos_token_id),
            };
            for case in cases {
                let ids = case["tokens"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_u64().unwrap() as u32)
                    .collect();
                let id = batch.admit_tokens(ids, params.clone()).unwrap();
                let output: Vec<u32> = case["greedy"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_u64().unwrap() as u32)
                    .collect();
                expected.insert(id, output);
            }
            let mut got: HashMap<u64, Vec<u32>> = HashMap::new();
            let mut steps = 0;
            while batch.active_count() > 0 {
                steps += 1;
                assert!(steps < 256, "batch stalled");
                for event in batch.step().unwrap() {
                    got.entry(event.request_id)
                        .or_default()
                        .push(event.token_id);
                }
            }
            assert_eq!(got, expected, "{dtype:?} graphs={graphs}");
            // Re-admission into recycled slots, then cancel one sibling.
            let tokens: Vec<u32> = cases[0]["tokens"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as u32)
                .collect();
            let cancel = batch.admit_tokens(tokens.clone(), params.clone()).unwrap();
            let survivor = batch.admit_tokens(tokens, params).unwrap();
            batch.cancel(cancel).unwrap();
            let mut output = vec![];
            while batch.active_count() > 0 {
                for event in batch.step().unwrap() {
                    if event.request_id == survivor {
                        output.push(event.token_id);
                    }
                }
            }
            let first: Vec<u32> = cases[0]["greedy"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as u32)
                .collect();
            assert_eq!(output, first);
            println!(
                "MODEL_SMOLLM_BATCH dtype={dtype:?} graphs={graphs} cases=4 cancel_reuse=pass"
            );
        }
    }
}
