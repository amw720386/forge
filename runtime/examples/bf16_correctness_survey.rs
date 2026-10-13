//! BF16 greedy correctness survey vs CUDA FP32 reference.
//!
//! Measures token agreement, first divergence, and logit errors on the shared
//! prefix at divergence (CPU-greedy path for full logits D2H).
//!
//! ```bash
//! cargo run --release --features cuda --example bf16_correctness_survey -- \
//!   ../models/Qwen2.5-0.5B-Instruct 32
//! ```

#![recursion_limit = "256"]

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("bf16_correctness_survey requires --features cuda");
    std::process::exit(1);
}

#[cfg(feature = "cuda")]
fn main() {
    use forge_runtime::cuda_model::{generate_cuda, CudaBatcher, CudaTransformer, GpuWeightDType};
    use forge_runtime::model::{load_qwen2_dir, ForgeTokenizer, GenerateParams};
    use forge_runtime::ComputeBackend;
    use serde_json::json;
    use std::env;
    use std::fs;
    use std::path::PathBuf;
    use std::time::Instant;

    // Force full logits D2H for numerical comparison path.
    env::set_var("FORGE_CPU_GREEDY", "1");

    let args: Vec<String> = env::args().collect();
    let dir = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "../models/Qwen2.5-0.5B-Instruct".into());
    let max_new: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(32);
    let json_out = args
        .iter()
        .position(|a| a == "--json-out")
        .and_then(|i| args.get(i + 1).map(PathBuf::from));

    let pad = "alpha ".repeat(180);
    let owned_prompts: Vec<(String, String)> = vec![
        (
            "hello".into(),
            "<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n".into(),
        ),
        (
            "short".into(),
            "<|im_start|>user\nShort.<|im_end|>\n<|im_start|>assistant\n".into(),
        ),
        (
            "primes".into(),
            "<|im_start|>user\nList three primes.<|im_end|>\n<|im_start|>assistant\n".into(),
        ),
        (
            "reason".into(),
            "<|im_start|>user\nIf 2+2=4, what is 3+5? Reply with only the number.<|im_end|>\n<|im_start|>assistant\n".into(),
        ),
        (
            "repeat".into(),
            "<|im_start|>user\na a a a a a a a a a a a a a a a<|im_end|>\n<|im_start|>assistant\n".into(),
        ),
        (
            "numeric".into(),
            "<|im_start|>user\nCompute 17*19. Show only the product.<|im_end|>\n<|im_start|>assistant\n".into(),
        ),
        (
            "medium".into(),
            format!(
                "<|im_start|>user\nSummarize:\n{pad}\n<|im_end|>\n<|im_start|>assistant\n"
            ),
        ),
    ];

    eprintln!("=== bf16_correctness_survey ===");
    eprintln!("model={dir} max_new={max_new}");

    let model = load_qwen2_dir(&dir).expect("load");
    let tok = ForgeTokenizer::from_dir(&dir, 151643, 151645).expect("tok");
    let params = GenerateParams {
        max_tokens: max_new,
        temperature: 0.0,
        top_k: 0,
        eos_token_id: Some(tok.eos_token_id),
    };

    eprintln!("upload FP32…");
    let t0 = Instant::now();
    let cuda_fp32 = CudaTransformer::from_cpu_dtype(&model, 0, GpuWeightDType::Fp32).expect("fp32");
    cuda_fp32.backend().synchronize().ok();
    eprintln!(
        "  {:.2}s weights={:.1} MiB",
        t0.elapsed().as_secs_f64(),
        cuda_fp32.weight_bytes() as f64 / (1024.0 * 1024.0)
    );

    eprintln!("upload BF16…");
    let t1 = Instant::now();
    let cuda_bf16 = CudaTransformer::from_cpu_dtype(&model, 0, GpuWeightDType::Bf16).expect("bf16");
    cuda_bf16.backend().synchronize().ok();
    eprintln!(
        "  {:.2}s weights={:.1} MiB",
        t1.elapsed().as_secs_f64(),
        cuda_bf16.weight_bytes() as f64 / (1024.0 * 1024.0)
    );

    let mut cases = Vec::new();
    let mut seq_agree = 0usize;
    let mut seq_total = 0usize;
    let mut tok_agree = 0usize;
    let mut tok_total = 0usize;

    for (id, prompt) in &owned_prompts {
        let ids = tok.encode(prompt, true).expect("encode");
        let max_seq = (ids.len() + max_new + 64).max(512).min(2048);
        eprintln!(
            "\n--- prompt={id} prompt_tokens={} max_seq={max_seq} ---",
            ids.len()
        );

        let out_fp = generate_cuda(&cuda_fp32, &tok, prompt, &params, max_seq).expect("fp32 gen");
        let out_bf = generate_cuda(&cuda_bf16, &tok, prompt, &params, max_seq).expect("bf16 gen");

        let (first_div, agree_n) = first_divergence(&out_fp.tokens, &out_bf.tokens);
        let seq_ok = first_div.is_none() && out_fp.tokens.len() == out_bf.tokens.len();
        seq_total += 1;
        if seq_ok {
            seq_agree += 1;
        }
        tok_agree += agree_n;
        tok_total += out_fp.tokens.len().max(out_bf.tokens.len());

        let mut logit_metrics = json!(null);
        if let Some(div) = first_div {
            // Compare logits on shared prefix (prompt + tokens[..div]).
            let mut prefix = ids.clone();
            prefix.extend_from_slice(&out_fp.tokens[..div]);
            logit_metrics = compare_logits_at_prefix(&cuda_fp32, &cuda_bf16, &prefix, max_seq);
            eprintln!(
                "  DIVERGE at pos={div} fp32_tok={} bf16_tok={} logits={}",
                out_fp.tokens.get(div).copied().unwrap_or(u32::MAX),
                out_bf.tokens.get(div).copied().unwrap_or(u32::MAX),
                logit_metrics
            );
        } else {
            eprintln!(
                "  AGREE n={} text_fp32={:?} text_bf16={:?}",
                out_fp.tokens.len(),
                truncate(&out_fp.text, 60),
                truncate(&out_bf.text, 60)
            );
            // Still sample logit error on final step prefix for transparency.
            let mut prefix = ids.clone();
            if !out_fp.tokens.is_empty() {
                prefix.extend_from_slice(&out_fp.tokens[..out_fp.tokens.len().saturating_sub(1)]);
            }
            if prefix.len() < max_seq {
                logit_metrics = compare_logits_at_prefix(&cuda_fp32, &cuda_bf16, &prefix, max_seq);
            }
        }

        cases.push(json!({
            "id": id,
            "prompt_tokens": ids.len(),
            "fp32_tokens": out_fp.tokens,
            "bf16_tokens": out_bf.tokens,
            "fp32_text": out_fp.text,
            "bf16_text": out_bf.text,
            "sequence_agree": seq_ok,
            "first_divergence": first_div,
            "prefix_token_agree": agree_n,
            "logit_metrics": logit_metrics,
        }));
    }

    // Batch B=1,2,4,8 smoke: identical prompts, compare concatenated token streams.
    eprintln!("\n=== batch agreement B=1,2,4,8 (hello, max_new={max_new}) ===");
    let mut batch_cases = Vec::new();
    let hello = owned_prompts
        .iter()
        .find(|(id, _)| id == "hello")
        .map(|(_, p)| p.clone())
        .unwrap();
    for &b in &[1usize, 2, 4, 8] {
        let tok_fp = ForgeTokenizer::from_dir(&dir, 151643, 151645).unwrap();
        let tok_bf = ForgeTokenizer::from_dir(&dir, 151643, 151645).unwrap();
        // Need fresh transformers? Reuse existing — batcher takes ownership.
        // Re-upload lightweight: from same CPU model.
        let fp = CudaTransformer::from_cpu_dtype(&model, 0, GpuWeightDType::Fp32).unwrap();
        let bf = CudaTransformer::from_cpu_dtype(&model, 0, GpuWeightDType::Bf16).unwrap();
        let mut batch_fp = CudaBatcher::new(fp, tok_fp, 512, b).unwrap();
        let mut batch_bf = CudaBatcher::new(bf, tok_bf, 512, b).unwrap();
        let p = params.clone();
        let mut ids_fp = Vec::new();
        let mut ids_bf = Vec::new();
        for _ in 0..b {
            ids_fp.push(batch_fp.admit(&hello, p.clone()).unwrap());
            ids_bf.push(batch_bf.admit(&hello, p.clone()).unwrap());
        }
        let mut map_fp: std::collections::HashMap<u64, Vec<u32>> =
            ids_fp.iter().map(|&id| (id, Vec::new())).collect();
        let mut map_bf: std::collections::HashMap<u64, Vec<u32>> =
            ids_bf.iter().map(|&id| (id, Vec::new())).collect();
        while batch_fp.active_count() > 0 {
            for ev in batch_fp.step().unwrap() {
                map_fp.get_mut(&ev.request_id).unwrap().push(ev.token_id);
            }
        }
        while batch_bf.active_count() > 0 {
            for ev in batch_bf.step().unwrap() {
                map_bf.get_mut(&ev.request_id).unwrap().push(ev.token_id);
            }
        }
        let mut all_ok = true;
        let mut divergences = Vec::new();
        for i in 0..b {
            let a = map_fp.get(&ids_fp[i]).cloned().unwrap_or_default();
            let c = map_bf.get(&ids_bf[i]).cloned().unwrap_or_default();
            let (div, _) = first_divergence(&a, &c);
            if div.is_some() || a != c {
                all_ok = false;
            }
            divergences.push(json!({"slot": i, "first_divergence": div, "fp32": a, "bf16": c}));
        }
        eprintln!("  B={b} sequence_agree_all={all_ok}");
        batch_cases.push(json!({
            "batch": b,
            "sequence_agree_all": all_ok,
            "slots": divergences,
        }));
    }

    // Chunked prefill stress
    eprintln!("\n=== chunked prefill (FORGE_PREFILL_CHUNK=32) ===");
    env::set_var("FORGE_PREFILL_CHUNK", "32");
    let (id, prompt) = &owned_prompts[0];
    let fp = CudaTransformer::from_cpu_dtype(&model, 0, GpuWeightDType::Fp32).unwrap();
    let bf = CudaTransformer::from_cpu_dtype(&model, 0, GpuWeightDType::Bf16).unwrap();
    let out_fp = generate_cuda(&fp, &tok, prompt, &params, 512).unwrap();
    let out_bf = generate_cuda(&bf, &tok, prompt, &params, 512).unwrap();
    let (div, _) = first_divergence(&out_fp.tokens, &out_bf.tokens);
    let chunk_ok = div.is_none() && out_fp.tokens == out_bf.tokens;
    eprintln!("  prompt={id} chunked_agree={chunk_ok} first_div={div:?}");
    env::remove_var("FORGE_PREFILL_CHUNK");

    let summary = json!({
        "harness": "bf16_correctness_survey",
        "reference": "cuda_fp32",
        "candidate": "cuda_bf16",
        "max_new": max_new,
        "sequence_agreement_rate": if seq_total > 0 { seq_agree as f64 / seq_total as f64 } else { 0.0 },
        "token_agreement_rate": if tok_total > 0 { tok_agree as f64 / tok_total as f64 } else { 0.0 },
        "sequences_agree": seq_agree,
        "sequences_total": seq_total,
        "note": "BF16 may legitimately diverge when competing logits have small margins; divergence ≠ bug by itself.",
        "single_seq_cases": cases,
        "batch_cases": batch_cases,
        "chunked_prefill_agree": chunk_ok,
        "chunked_first_divergence": div,
    });

    let text = serde_json::to_string_pretty(&summary).unwrap();
    if let Some(path) = json_out {
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        fs::write(&path, &text).expect("write");
        eprintln!("Wrote {}", path.display());
    }
    println!("{text}");
    eprintln!(
        "\nSUMMARY sequence_agree={}/{} ({:.1}%) token_agree={}/{} ({:.1}%)",
        seq_agree,
        seq_total,
        100.0 * seq_agree as f64 / seq_total.max(1) as f64,
        tok_agree,
        tok_total,
        100.0 * tok_agree as f64 / tok_total.max(1) as f64
    );
}

#[cfg(feature = "cuda")]
fn truncate(s: &str, n: usize) -> String {
    let mut t: String = s.chars().take(n).collect();
    if s.chars().count() > n {
        t.push('…');
    }
    t
}

#[cfg(feature = "cuda")]
fn first_divergence(a: &[u32], b: &[u32]) -> (Option<usize>, usize) {
    let n = a.len().min(b.len());
    for i in 0..n {
        if a[i] != b[i] {
            return (Some(i), i);
        }
    }
    if a.len() != b.len() {
        return (Some(n), n);
    }
    (None, n)
}

#[cfg(feature = "cuda")]
fn compare_logits_at_prefix(
    fp32: &forge_runtime::cuda_model::CudaTransformer,
    bf16: &forge_runtime::cuda_model::CudaTransformer,
    prefix: &[u32],
    max_seq: usize,
) -> serde_json::Value {
    use forge_runtime::cuda_model::kv::LayerGpuKvCaches;
    use forge_runtime::cuda_model::CudaScratch;
    use forge_runtime::sampling::greedy;
    use serde_json::json;

    if prefix.is_empty() || prefix.len() >= max_seq {
        return json!({"error": "bad prefix"});
    }

    let run = |model: &forge_runtime::cuda_model::CudaTransformer| -> Result<Vec<f32>, String> {
        let backend = model.backend();
        let mut caches = LayerGpuKvCaches::new(
            backend,
            model.cfg.n_layers,
            model.cfg.n_kv_heads,
            model.cfg.head_dim(),
            max_seq,
        )
        .map_err(|e| e.to_string())?;
        let mut scratch =
            CudaScratch::new(backend, &model.cfg, max_seq).map_err(|e| e.to_string())?;
        model
            .forward_tokens(prefix, 0, &mut caches, &mut scratch)
            .map_err(|e| e.to_string())?;
        model
            .read_logits(&scratch, model.cfg.vocab_size)
            .map_err(|e| e.to_string())
    };

    let a = match run(fp32) {
        Ok(v) => v,
        Err(e) => return json!({"error": e}),
    };
    let b = match run(bf16) {
        Ok(v) => v,
        Err(e) => return json!({"error": e}),
    };
    let vocab = a.len().min(b.len());
    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    let mut sum_abs = 0.0f64;
    for i in 0..vocab {
        let da = a[i] as f64;
        let db = b[i] as f64;
        let abs = (da - db).abs();
        max_abs = max_abs.max(abs);
        sum_abs += abs;
        let denom = da.abs().max(1e-6);
        max_rel = max_rel.max(abs / denom);
    }
    let top1_a = greedy(&a);
    let top1_b = greedy(&b);
    let (top2_a, margin_a) = top2_margin(&a);
    let (top2_b, margin_b) = top2_margin(&b);

    json!({
        "vocab_compared": vocab,
        "max_abs_logit_err": max_abs,
        "mean_abs_logit_err": sum_abs / vocab.max(1) as f64,
        "max_rel_logit_err": max_rel,
        "top1_fp32": top1_a,
        "top1_bf16": top1_b,
        "top1_agree": top1_a == top1_b,
        "top2_fp32": top2_a,
        "top2_bf16": top2_b,
        "top2_margin_fp32": margin_a,
        "top2_margin_bf16": margin_b,
    })
}

#[cfg(feature = "cuda")]
fn top2_margin(logits: &[f32]) -> ((u32, u32), f64) {
    let mut best = (0u32, f32::NEG_INFINITY);
    let mut second = (0u32, f32::NEG_INFINITY);
    for (i, &v) in logits.iter().enumerate() {
        if v > best.1 {
            second = best;
            best = (i as u32, v);
        } else if v > second.1 {
            second = (i as u32, v);
        }
    }
    ((best.0, second.0), (best.1 - second.1) as f64)
}
