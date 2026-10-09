//! Compare Forge against sequentially exported HF references, one model at a time.
use forge_runtime::model::{load_model_path, ForwardScratch, LayerKvCaches};
use std::{path::Path, time::Instant};

fn read(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn compare(label: &str, got: &[f32], reference: &[f32], atol: f32, rtol: f32) {
    assert_eq!(got.len(), reference.len(), "{label}");
    let mut max = 0.0f32;
    let mut squared = 0.0f64;
    for (i, (&g, &r)) in got.iter().zip(reference).enumerate() {
        let error = (g - r).abs();
        max = max.max(error);
        squared += (error as f64).powi(2);
        assert!(
            g.is_finite() && r.is_finite() && error <= atol + rtol * r.abs(),
            "{label} [{i}] {g} vs {r}, error {error}"
        );
    }
    println!(
        "{label}: max_abs={max:.8} rms={:.8}",
        (squared / got.len() as f64).sqrt()
    );
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let start = Instant::now();
    let model = load_model_path(&args[1]).unwrap();
    let fixtures = Path::new(&args[2]);
    let cases: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixtures.join("cases.json")).unwrap()).unwrap();
    println!(
        "model={} load_s={:.3} weights_mib={:.2}",
        model.cfg.architecture.name(),
        start.elapsed().as_secs_f64(),
        model.weight_nbytes() as f64 / 1048576.0
    );
    for (index, case) in cases["cases"].as_array().unwrap().iter().enumerate() {
        let tokens: Vec<u32> = case["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect();
        let expected: Vec<u32> = case["greedy"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect();
        let max_seq = tokens.len() + expected.len() + 1;
        let mut caches = LayerKvCaches::new(&model.cfg, max_seq).unwrap();
        let mut scratch = ForwardScratch::new(&model.cfg, max_seq).unwrap();
        // Layer outputs before final RMSNorm, independently compared to HF hooks.
        let h = model.cfg.hidden_size;
        for (row, &token) in tokens.iter().enumerate() {
            scratch.x.data_mut()[row * h..(row + 1) * h]
                .copy_from_slice(&model.embed.data()[token as usize * h..(token as usize + 1) * h]);
        }
        for (layer_index, (layer, cache)) in model
            .layers
            .iter()
            .zip(caches.layers.iter_mut())
            .enumerate()
        {
            layer
                .forward(
                    &mut scratch.x,
                    tokens.len(),
                    0,
                    cache,
                    &mut scratch.attn,
                    &mut scratch.residual,
                    &mut scratch.normed,
                    &mut scratch.attn_out,
                    &mut scratch.mlp_gate,
                    &mut scratch.mlp_up,
                    &mut scratch.mlp_out,
                )
                .unwrap();
            compare(
                &format!("case{index}/layer{layer_index}"),
                &scratch.x.data()[..tokens.len() * h],
                &read(&fixtures.join(format!("{index}_layer_{layer_index}.bin"))),
                1e-3,
                1e-4,
            );
        }
        caches.clear();
        model
            .forward_tokens(&tokens, 0, &mut caches, &mut scratch)
            .unwrap();
        let reference = read(&fixtures.join(format!("{index}_logits.bin")));
        compare(
            &format!("case{index}/logits"),
            scratch.logits.data(),
            &reference,
            1e-4,
            1e-4,
        );
        let prefill = scratch.logits.data().to_vec();
        caches.clear();
        let mut decode = ForwardScratch::new_decode(&model.cfg).unwrap();
        for (pos, &token) in tokens.iter().enumerate() {
            model
                .forward_tokens(&[token], pos, &mut caches, &mut decode)
                .unwrap();
        }
        compare(
            &format!("case{index}/cached"),
            decode.logits.data(),
            &prefill,
            1e-4,
            1e-4,
        );
        let start = Instant::now();
        for (step, &token) in expected.iter().enumerate() {
            let next = forge_runtime::sampling::greedy(decode.logits.data());
            assert_eq!(next, token, "case{index} greedy step{step}");
            model
                .forward_tokens(&[next], tokens.len() + step, &mut caches, &mut decode)
                .unwrap();
        }
        println!(
            "case{index} greedy={} elapsed_s={:.3} tokens_per_s={:.2} text={}",
            expected.len(),
            start.elapsed().as_secs_f64(),
            expected.len() as f64 / start.elapsed().as_secs_f64(),
            case["text"]
        );
        #[cfg(any(feature = "cuda", feature = "cuda-stub"))]
        if args.get(3).is_some_and(|s| s == "cuda") {
            gpu(&model, &tokens, &expected, &reference);
        }
    }
    cpu_batch(model, cases["cases"].as_array().unwrap());
}
fn cpu_batch(model: forge_runtime::model::Transformer, cases: &[serde_json::Value]) {
    use forge_runtime::model::{ContinuousBatcher, GenerateParams};
    use std::collections::HashMap;
    let eos = model.cfg.eos_token_id;
    let mut batch = ContinuousBatcher::new_untokenized(model, 256, 4).unwrap();
    let mut expected = HashMap::new();
    for case in cases {
        let tokens = case["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let id = batch
            .admit_tokens(
                tokens,
                GenerateParams {
                    max_tokens: 12,
                    temperature: 0.0,
                    top_k: 0,
                    eos_token_id: Some(eos),
                },
            )
            .unwrap();
        let output: Vec<u32> = case["greedy"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        expected.insert(id, output);
    }
    let mut output: HashMap<u64, Vec<u32>> = HashMap::new();
    while batch.active_count() > 0 {
        for event in batch.step().unwrap() {
            output
                .entry(event.request_id)
                .or_default()
                .push(event.token_id);
        }
    }
    assert_eq!(expected, output, "CPU shared batch vs HF");
    println!("CPU_BATCH_HF_PARITY cases=4 pass");
}
#[cfg(any(feature = "cuda", feature = "cuda-stub"))]
fn gpu(
    model: &forge_runtime::model::Transformer,
    tokens: &[u32],
    expected: &[u32],
    reference: &[f32],
) {
    use forge_runtime::cuda_model::kv::LayerGpuKvCaches;
    use forge_runtime::cuda_model::{CudaScratch, CudaTransformer, GpuWeightDType};
    for dtype in [GpuWeightDType::Fp32, GpuWeightDType::Bf16] {
        let gpu = CudaTransformer::from_cpu_dtype(model, 0, dtype).unwrap();
        let mut scratch =
            CudaScratch::new(gpu.backend(), &model.cfg, tokens.len() + expected.len() + 1).unwrap();
        let mut caches = LayerGpuKvCaches::new(
            gpu.backend(),
            model.cfg.n_layers,
            model.cfg.n_kv_heads,
            model.cfg.head_dim(),
            tokens.len() + expected.len() + 1,
        )
        .unwrap();
        gpu.forward_tokens(tokens, 0, &mut caches, &mut scratch)
            .unwrap();
        let logits = gpu.read_logits(&scratch, model.cfg.vocab_size).unwrap();
        if dtype == GpuWeightDType::Fp32 {
            compare("cuda/fp32/logits", &logits, reference, 1e-4, 1e-4);
        } else {
            let max = logits
                .iter()
                .zip(reference)
                .map(|(g, r)| (g - r).abs())
                .fold(0.0, f32::max);
            let rms = (logits
                .iter()
                .zip(reference)
                .map(|(g, r)| ((*g as f64) - (*r as f64)).powi(2))
                .sum::<f64>()
                / logits.len() as f64)
                .sqrt();
            assert!(
                max.is_finite() && max <= 0.25,
                "BF16 logit error outside Phase 16 qualified envelope: max={max} rms={rms}"
            );
            println!("cuda/bf16 max_abs={max} rms={rms} (qualified BF16 envelope; FP32 tolerances unchanged)");
            bf16_projection_attribution(model, &gpu, tokens, reference);
        }
        let mut logits = logits;
        for (step, &token) in expected.iter().enumerate() {
            let next = forge_runtime::sampling::greedy(&logits);
            assert_eq!(next, token, "{dtype:?} step{step}");
            gpu.forward_tokens(&[next], tokens.len() + step, &mut caches, &mut scratch)
                .unwrap();
            logits = gpu.read_logits(&scratch, model.cfg.vocab_size).unwrap();
        }
    }
}
#[cfg(any(feature = "cuda", feature = "cuda-stub"))]
fn bf16_projection_attribution(
    model: &forge_runtime::model::Transformer,
    gpu: &forge_runtime::cuda_model::CudaTransformer,
    tokens: &[u32],
    reference: &[f32],
) {
    use forge_runtime::compute::ComputeBackend;
    use forge_runtime::model::WeightStorage;
    let mut cache = LayerKvCaches::new(&model.cfg, tokens.len()).unwrap();
    let mut scratch = ForwardScratch::new(&model.cfg, tokens.len()).unwrap();
    model
        .forward_tokens(tokens, 0, &mut cache, &mut scratch)
        .unwrap();
    let h = model.cfg.hidden_size;
    let vocab = model.cfg.vocab_size;
    let x = &scratch.normed.data()[(tokens.len() - 1) * h..tokens.len() * h];
    let round = |v: f32| {
        let bits = v.to_bits();
        f32::from_bits(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff0000)
    };
    let weights = match &model.lm_head.weight {
        WeightStorage::Fp32(w) => w.data(),
        _ => panic!("FP32 source required"),
    };
    let rounded: Vec<f32> = x.iter().copied().map(round).collect();
    let oracle: Vec<f32> = weights
        .chunks_exact(h)
        .map(|row| {
            row.iter()
                .zip(&rounded)
                .map(|(&w, &a)| round(w) as f64 * a as f64)
                .sum::<f64>() as f32
        })
        .collect();
    let backend = gpu.backend();
    let mut input = backend.alloc(h * 4).unwrap();
    let bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    backend.copy_h2d(&mut input, &bytes).unwrap();
    let mut output = backend.alloc(vocab * 4).unwrap();
    let bytes: Vec<u8> = weights
        .iter()
        .flat_map(|&w| ((round(w).to_bits() >> 16) as u16).to_le_bytes())
        .collect();
    let mut weight = backend.alloc(bytes.len()).unwrap();
    backend.copy_h2d(&mut weight, &bytes).unwrap();
    backend
        .linear_bf16_w(1, vocab, h, &input, h, &weight, h, None, &mut output, vocab)
        .unwrap();
    let mut bytes = vec![0u8; vocab * 4];
    backend.copy_d2h(&mut bytes, &output).unwrap();
    let actual: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    compare(
        "bf16/projection_vs_explicit_rounding",
        &actual,
        &oracle,
        1e-4,
        1e-4,
    );
    let max = actual
        .iter()
        .zip(reference)
        .map(|(g, r)| (g - r).abs())
        .fold(0.0, f32::max);
    let rms = (actual
        .iter()
        .zip(reference)
        .map(|(g, r)| (*g as f64 - *r as f64).powi(2))
        .sum::<f64>()
        / vocab as f64)
        .sqrt();
    println!("BF16_FINAL_PROJECTION_ONLY max_abs={max} rms={rms}");
}
