//! Reproducible Phase 6 benchmark: FP32 / INT8 / INT4 + batch throughput.
//!
//! ```bash
//! cargo run --release --example bench_phase6 -- ../models/Qwen2.5-0.5B-Instruct
//! ```
//!
//! Env: FORGE_NUM_THREADS, optional FORGE_BENCH_PREFILL (default 64), FORGE_BENCH_DECODE (default 32).

use forge_runtime::ffi::kernels;
use forge_runtime::model::{
    load_qwen2_dir, ContinuousBatcher, ForgeTokenizer, ForwardScratch, GenerateParams,
    LayerKvCaches, QuantBits, Transformer,
};
use forge_runtime::sampling::greedy;
use std::env;
use std::time::Instant;

fn ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn measure_single(
    label: &str,
    model: &Transformer,
    tok: &ForgeTokenizer,
    prompt: &str,
    prefill_len: usize,
    decode_n: usize,
) {
    let max_seq = 2048usize;
    let mut ids = tok.encode(prompt, true).expect("encode");
    if ids.len() > prefill_len {
        ids.truncate(prefill_len);
    }
    let mut caches = LayerKvCaches::new(&model.cfg, max_seq).unwrap();
    let mut scratch = ForwardScratch::new(&model.cfg, max_seq).unwrap();

    // Cold prefill (first call after load — includes any lazy costs)
    let t_cold = Instant::now();
    model
        .forward_tokens(&ids, 0, &mut caches, &mut scratch)
        .unwrap();
    let cold_ms = ms(t_cold.elapsed());

    // Warm-up: a few decode steps then reset
    let mut logits = scratch.logits.data()[..model.cfg.vocab_size].to_vec();
    for _ in 0..2 {
        let next = greedy(&logits);
        ids.push(next);
        let pos = ids.len() - 1;
        model
            .forward_tokens(&[next], pos, &mut caches, &mut scratch)
            .unwrap();
        logits = scratch.logits.data()[..model.cfg.vocab_size].to_vec();
    }

    // Steady-state: fresh caches + warm prefill + timed decode
    let mut ids = tok.encode(prompt, true).unwrap();
    if ids.len() > prefill_len {
        ids.truncate(prefill_len);
    }
    let prompt_tok = ids.len();
    caches.clear();
    let t_pre = Instant::now();
    model
        .forward_tokens(&ids, 0, &mut caches, &mut scratch)
        .unwrap();
    let prefill_ms = ms(t_pre.elapsed());
    let ttft_ms = prefill_ms; // TTFT ≈ warm prefill (+ negligible sample)
    logits = scratch.logits.data()[..model.cfg.vocab_size].to_vec();

    let mut decode_ms = Vec::with_capacity(decode_n);
    let t_dec = Instant::now();
    for _ in 0..decode_n {
        let next = greedy(&logits);
        ids.push(next);
        let pos = ids.len() - 1;
        let t0 = Instant::now();
        model
            .forward_tokens(&[next], pos, &mut caches, &mut scratch)
            .unwrap();
        decode_ms.push(ms(t0.elapsed()));
        logits = scratch.logits.data()[..model.cfg.vocab_size].to_vec();
    }
    let dec_total = ms(t_dec.elapsed());
    decode_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());

    println!("=== {label} ===");
    println!(
        "  weights: {:.2} MiB",
        model.weight_nbytes() as f64 / (1024.0 * 1024.0)
    );
    println!(
        "  cold_prefill: {:.1} ms  warm_prefill({prompt_tok} tok): {:.1} ms ({:.2} tok/s)  TTFT≈{:.1} ms",
        cold_ms,
        prefill_ms,
        prompt_tok as f64 / (prefill_ms / 1e3),
        ttft_ms
    );
    println!(
        "  decode {} tok: mean={:.1} ms/tok ({:.2} tok/s)  p50={:.1}  p95={:.1}  total={:.1} ms",
        decode_n,
        dec_total / decode_n as f64,
        decode_n as f64 / (dec_total / 1e3),
        percentile(&decode_ms, 0.50),
        percentile(&decode_ms, 0.95),
        dec_total
    );
}

fn measure_batch(
    label: &str,
    model: Transformer,
    tok: ForgeTokenizer,
    prompts: &[&str],
    max_tokens: usize,
) {
    let max_seq = 512usize;
    let b = prompts.len();
    let mut batcher = ContinuousBatcher::new(model, tok, max_seq, b.max(1)).unwrap();
    let params = GenerateParams {
        max_tokens,
        temperature: 0.0,
        top_k: 0,
        eos_token_id: Some(151645),
    };
    let t_admit = Instant::now();
    let mut ids = Vec::new();
    for p in prompts {
        ids.push(batcher.admit(p, params.clone()).unwrap());
    }
    let admit_ms = ms(t_admit.elapsed());

    let t0 = Instant::now();
    let mut total_tokens = 0u64;
    let mut step_ms = Vec::new();
    while batcher.active_count() > 0 {
        let ts = Instant::now();
        let evs = batcher.step().unwrap();
        step_ms.push(ms(ts.elapsed()));
        total_tokens += evs.len() as u64;
    }
    let wall = ms(t0.elapsed());
    step_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("=== batch B={b} {label} ===");
    println!(
        "  admit(prefill all): {:.1} ms  decode_wall: {:.1} ms  tokens={}  aggregate={:.2} tok/s",
        admit_ms,
        wall,
        total_tokens,
        total_tokens as f64 / (wall / 1e3)
    );
    println!(
        "  step latency ms: p50={:.1}  p95={:.1}",
        percentile(&step_ms, 0.50),
        percentile(&step_ms, 0.95)
    );
}

fn main() {
    let dir = env::args()
        .nth(1)
        .unwrap_or_else(|| "../models/Qwen2.5-0.5B-Instruct".into());
    let prefill_len: usize = env::var("FORGE_BENCH_PREFILL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let decode_n: usize = env::var("FORGE_BENCH_DECODE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);

    println!("=== Forge Phase 6 reproducible bench ===");
    println!("kernels: {}", kernels::version());
    println!(
        "threads: {} (FORGE_NUM_THREADS)",
        unsafe { kernels::forge_kernels_num_threads() }
    );
    println!("model_dir: {dir}");
    println!("prefill_len={prefill_len} decode_n={decode_n}");
    println!("build: cargo --release");
    println!();

    let prompt = "<|im_start|>user\nSay hello in one short sentence.<|im_end|>\n<|im_start|>assistant\n";
    let prompts_mixed = [
        prompt,
        "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nName a primary color.<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWrite a tiny poem about rain.<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nList three mammals.<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nCapital of France?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nDefine latency briefly.<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nOne word for happiness.<|im_end|>\n<|im_start|>assistant\n",
    ];

    // Memory safeguard: this bench loads **one** backend at a time and drops it.
    // Override with FORGE_BENCH_MEM_CAP_MIB / FORGE_BENCH_RAM_FRACTION if needed.
    if let Err(e) = forge_runtime::mem::allow_multi_model_bench(3200) {
        eprintln!("bench refused: {e}");
        eprintln!("hint: use forge-bench --backend fp32|int8|forge for single-process runs");
        std::process::exit(3);
    }

    let tok = ForgeTokenizer::from_dir(&dir, 151643, 151645).expect("tokenizer");

    {
        let model_fp = load_qwen2_dir(&dir).expect("load");
        measure_single("FP32", &model_fp, &tok, prompt, prefill_len, decode_n);
    }

    {
        let mut model_i8 = load_qwen2_dir(&dir).expect("load");
        let tq = Instant::now();
        model_i8.quantize_weights(QuantBits::Int8, 64).unwrap();
        println!(
            "(INT8 group=64 quantize wall: {:.1} ms)",
            ms(tq.elapsed())
        );
        measure_single("INT8-g64", &model_i8, &tok, prompt, prefill_len, decode_n);
    }

    for &(label, gs) in &[("INT4-g32", 32usize), ("INT4-g64", 64), ("INT4-g128", 128)] {
        let mut m = load_qwen2_dir(&dir).expect("load");
        m.quantize_weights(QuantBits::Int4, gs).unwrap();
        measure_single(label, &m, &tok, prompt, prefill_len, decode_n);
    }

    println!();
    println!("--- Continuous batch throughput (INT8-g64) ---");
    for &b in &[1usize, 2, 4, 8] {
        let mut m = load_qwen2_dir(&dir).expect("load");
        m.quantize_weights(QuantBits::Int8, 64).unwrap();
        let t = ForgeTokenizer::from_dir(&dir, 151643, 151645).unwrap();
        measure_batch("INT8-g64", m, t, &prompts_mixed[..b], 16);
    }

    println!();
    println!("--- Next-token agreement (greedy; INT8 only — FP32 dropped) ---");
    {
        let mut iq = load_qwen2_dir(&dir).unwrap();
        iq.quantize_weights(QuantBits::Int8, 64).unwrap();
        for (i, p) in prompts_mixed.iter().take(4).enumerate() {
            let ids = tok.encode(p, true).unwrap();
            let trunc: Vec<u32> = ids.iter().copied().take(48).collect();
            let mut c2 = LayerKvCaches::new(&iq.cfg, 256).unwrap();
            let mut s2 = ForwardScratch::new(&iq.cfg, 256).unwrap();
            iq.forward_tokens(&trunc, 0, &mut c2, &mut s2).unwrap();
            let t_q = greedy(&s2.logits.data()[..iq.cfg.vocab_size]);
            println!("  prompt{i}: int8_tok={t_q}");
        }
    }
}
