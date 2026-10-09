//! Profile Qwen2.5-0.5B: load / warm-up / prefill / decode + micro-kernels.
//!
//! ```bash
//! cargo run --release --example profile_qwen -- ../models/Qwen2.5-0.5B-Instruct
//! ```

use forge_runtime::ffi::kernels;
use forge_runtime::model::{
    load_qwen2_dir, ForgeTokenizer, ForwardScratch, LayerKvCaches,
};
use forge_runtime::tensor::ops::linear_into;
use forge_runtime::tensor::Tensor;
use std::env;
use std::time::Instant;

fn ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn bench_linear(label: &str, m: usize, n: usize, k: usize, iters: usize) {
    let x = Tensor::filled(&[m, k], 0.01).unwrap();
    let w = Tensor::filled(&[n, k], 0.001).unwrap();
    let mut y = Tensor::zeros(&[m, n]).unwrap();
    for _ in 0..3 {
        linear_into(&x, &w, None, &mut y).unwrap();
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        linear_into(&x, &w, None, &mut y).unwrap();
    }
    let elapsed = t0.elapsed();
    let per = ms(elapsed) / iters as f64;
    let gflops = (2.0 * m as f64 * n as f64 * k as f64) / (per * 1e6);
    println!(
        "  linear {label:>18}  m={m:<4} n={n:<6} k={k:<4}  {per:8.3} ms  {gflops:6.2} GFLOP/s  (threads={})",
        unsafe { kernels::forge_kernels_num_threads() }
    );
}

fn main() {
    let dir = env::args()
        .nth(1)
        .unwrap_or_else(|| "../models/Qwen2.5-0.5B-Instruct".into());
    let prefill_len: usize = env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let decode_tokens: usize = env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);

    println!("=== Forge Phase 5 profiler ===");
    println!("kernels: {}", kernels::version());
    println!(
        "threads: {}",
        unsafe { kernels::forge_kernels_num_threads() }
    );
    println!("model:   {dir}");
    println!("prefill_len={prefill_len}  decode_tokens={decode_tokens}");
    println!();

    println!("--- Microbench: Qwen decode linear shapes (m=1) ---");
    let h = 896usize;
    let i = 4864usize;
    let kv = 128usize;
    let vocab = 151_936usize;
    bench_linear("q_proj", 1, h, h, 50);
    bench_linear("k_proj", 1, kv, h, 100);
    bench_linear("gate_proj", 1, i, h, 20);
    bench_linear("down_proj", 1, h, i, 20);
    bench_linear("lm_head", 1, vocab, h, 5);
    println!();

    println!("--- Microbench: prefill-ish linear (m={prefill_len}) ---");
    bench_linear("q_proj", prefill_len, h, h, 5);
    bench_linear("gate_proj", prefill_len, i, h, 3);
    bench_linear("down_proj", prefill_len, h, i, 3);
    println!();

    let t_load0 = Instant::now();
    let model = load_qwen2_dir(&dir).expect("load");
    let load_ms = ms(t_load0.elapsed());
    let tok = ForgeTokenizer::from_dir(&dir, 151643, 151645).expect("tok");
    println!("--- Model load: {load_ms:.1} ms ---");

    let prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nExplain what a KV cache is in one paragraph.<|im_end|>\n<|im_start|>assistant\n";
    let mut prompt_ids = tok.encode(prompt, true).expect("encode");
    if prompt_ids.len() > prefill_len {
        prompt_ids.truncate(prefill_len);
    } else {
        while prompt_ids.len() < prefill_len {
            prompt_ids.push(prompt_ids[prompt_ids.len() % prompt_ids.len().max(1)]);
        }
    }
    println!("prompt tokens used: {}", prompt_ids.len());

    let max_seq = prefill_len + decode_tokens + 8;
    let mut caches = LayerKvCaches::new(&model.cfg, max_seq).unwrap();
    let mut scratch = ForwardScratch::new(&model.cfg, max_seq).unwrap();

    {
        let warm = &prompt_ids[..prompt_ids.len().min(8)];
        caches.clear();
        let t0 = Instant::now();
        model
            .forward_tokens(warm, 0, &mut caches, &mut scratch)
            .unwrap();
        let _ = model.forward_tokens(&[1u32], warm.len(), &mut caches, &mut scratch);
        println!("--- Warm-up: {:.1} ms ---", ms(t0.elapsed()));
    }

    caches.clear();
    let t_pre0 = Instant::now();
    model
        .forward_tokens(&prompt_ids, 0, &mut caches, &mut scratch)
        .unwrap();
    let prefill_ms = ms(t_pre0.elapsed());
    let prefill_tps = prompt_ids.len() as f64 / (prefill_ms / 1e3);
    println!(
        "--- Prefill: {prefill_ms:.1} ms  ({prefill_tps:.2} tok/s, {} tokens) ---",
        prompt_ids.len()
    );

    let mut next = {
        let logits = scratch.logits.data();
        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best = i;
            }
        }
        best as u32
    };
    let mut decode_times = Vec::with_capacity(decode_tokens);
    let t_dec0 = Instant::now();
    for step in 0..decode_tokens {
        let pos = prompt_ids.len() + step;
        let t0 = Instant::now();
        model
            .forward_tokens(&[next], pos, &mut caches, &mut scratch)
            .unwrap();
        decode_times.push(ms(t0.elapsed()));
        let logits = scratch.logits.data();
        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best = i;
            }
        }
        next = best as u32;
        if next == 151645 {
            break;
        }
    }
    let decode_total = ms(t_dec0.elapsed());
    let n = decode_times.len().max(1);
    let mean = decode_times.iter().sum::<f64>() / n as f64;
    let mut sorted = decode_times.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = sorted[n / 2];
    let p95 = sorted[((n as f64) * 0.95) as usize].min(sorted[n - 1]);
    println!(
        "--- Decode: {decode_total:.1} ms total, {} tokens ---",
        decode_times.len()
    );
    println!(
        "    mean={mean:.1} ms/tok  ({:.2} tok/s)  p50={p50:.1}  p95={p95:.1}",
        1000.0 / mean
    );
    println!("\n=== done ===");
}
