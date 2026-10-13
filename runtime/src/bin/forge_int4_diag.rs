//! Safe single-backend INT4 diagnosis: memory/paging + kernel microbench + decode.
//!
//! One process = one backend. Orchestrate with `scripts/bench_int4_compare.ps1`.
//! Build explicitly with Cargo's `--features diagnostics`.
//!
//! ```text
//! forge-int4-diag --model ../models/Qwen2.5-0.5B-Instruct --backend int4 --group 64 --reps 3
//! ```
//!
//! Does **not** modify kernels. Aborts if available RAM is too low.

use forge_runtime::checkpoint::{load_forge_path, LoadOptions};
use forge_runtime::ffi::kernels;
use forge_runtime::mem::{allow_multi_model_bench, require_available_ram_mib, ProcessMemory};
use forge_runtime::model::{
    load_hf_dir, ForgeTokenizer, ForwardScratch, LayerKvCaches, QuantBits, QuantMatrix, Transformer,
};
use forge_runtime::sampling::greedy;
use forge_runtime::tensor::Tensor;
use std::env;
use std::path::PathBuf;
use std::process::ExitCode;
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

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

struct Args {
    model: PathBuf,
    backend: String,
    group: usize,
    prefill: usize,
    decode_n: usize,
    reps: usize,
    max_seq: usize,
    min_avail_mib: u64,
    skip_micro: bool,
    skip_model: bool,
}

fn parse_args() -> Args {
    let args: Vec<String> = env::args().collect();
    let mut a = Args {
        model: PathBuf::from("../models/Qwen2.5-0.5B-Instruct"),
        backend: "int4".into(),
        group: 64,
        prefill: 64,
        decode_n: 32,
        reps: 3,
        max_seq: 512,
        min_avail_mib: 2048,
        skip_micro: false,
        skip_model: false,
    };
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                a.model = PathBuf::from(args.get(i + 1).expect("--model"));
                i += 2;
            }
            "--backend" => {
                a.backend = args.get(i + 1).cloned().unwrap_or(a.backend);
                i += 2;
            }
            "--group" => {
                a.group = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(64);
                i += 2;
            }
            "--prefill" => {
                a.prefill = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(64);
                i += 2;
            }
            "--decode" => {
                a.decode_n = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(32);
                i += 2;
            }
            "--reps" => {
                a.reps = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(3);
                i += 2;
            }
            "--max-seq" => {
                a.max_seq = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(512);
                i += 2;
            }
            "--min-avail-mib" => {
                a.min_avail_mib = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(4096);
                i += 2;
            }
            "--skip-micro" => {
                a.skip_micro = true;
                i += 1;
            }
            "--skip-model" => {
                a.skip_model = true;
                i += 1;
            }
            _ => i += 1,
        }
    }
    a
}

/// Synthetic Qwen decode shapes — isolates kernel cost from paging / model load.
fn microbench_quant_linears(group: usize) {
    println!("--- kernel microbench (synthetic Qwen shapes, group={group}) ---");
    println!("kernels: {}  threads={}", kernels::version(), unsafe {
        kernels::forge_kernels_num_threads()
    });

    let shapes: &[(&str, usize, usize, usize)] = &[
        ("q_proj", 1, 896, 896),
        ("gate_proj", 1, 4864, 896),
        ("down_proj", 1, 896, 4864),
        ("lm_head", 1, 151_936, 896),
    ];

    for &(label, m, n, k) in shapes {
        assert_eq!(k % group, 0);
        // Cap synthetic rows for lm_head to keep microbench RAM modest (~96 MiB FP32).
        let n_bench = n.min(24_576);
        let mut x = Tensor::zeros(&[m, k]).unwrap();
        {
            let d = x.data_mut();
            for (i, v) in d.iter_mut().enumerate() {
                *v = ((i % 17) as f32) * 0.01 - 0.08;
            }
        }
        let mut w = Tensor::zeros(&[n_bench, k]).unwrap();
        {
            let d = w.data_mut();
            for (i, v) in d.iter_mut().enumerate() {
                *v = ((i % 31) as f32) * 0.001 - 0.015;
            }
        }

        let q8 = QuantMatrix::from_fp32(&w, QuantBits::Int8, group).unwrap();
        let q4 = QuantMatrix::from_fp32(&w, QuantBits::Int4, group).unwrap();
        drop(w);
        let mut y8 = Tensor::zeros(&[m, n_bench]).unwrap();
        let mut y4 = Tensor::zeros(&[m, n_bench]).unwrap();

        let iters = if n_bench > 8_000 { 8 } else { 20 };

        unsafe {
            kernels::check(kernels::forge_kernels_linear_i8_f32(
                m,
                n_bench,
                k,
                x.data().as_ptr(),
                k,
                q8.qweight.as_ptr() as *const i8,
                q8.scales.as_ptr(),
                group,
                std::ptr::null(),
                y8.data_mut().as_mut_ptr(),
                n_bench,
            ))
            .unwrap();
            kernels::check(kernels::forge_kernels_linear_i4_f32(
                m,
                n_bench,
                k,
                x.data().as_ptr(),
                k,
                q4.qweight.as_ptr(),
                q4.scales.as_ptr(),
                group,
                std::ptr::null(),
                y4.data_mut().as_mut_ptr(),
                n_bench,
            ))
            .unwrap();
        }

        let mem_before = ProcessMemory::sample();
        let t8 = Instant::now();
        for _ in 0..iters {
            unsafe {
                kernels::forge_kernels_linear_i8_f32(
                    m,
                    n_bench,
                    k,
                    x.data().as_ptr(),
                    k,
                    q8.qweight.as_ptr() as *const i8,
                    q8.scales.as_ptr(),
                    group,
                    std::ptr::null(),
                    y8.data_mut().as_mut_ptr(),
                    n_bench,
                );
            }
        }
        let i8_ms = ms(t8.elapsed()) / iters as f64;

        let t4 = Instant::now();
        for _ in 0..iters {
            unsafe {
                kernels::forge_kernels_linear_i4_f32(
                    m,
                    n_bench,
                    k,
                    x.data().as_ptr(),
                    k,
                    q4.qweight.as_ptr(),
                    q4.scales.as_ptr(),
                    group,
                    std::ptr::null(),
                    y4.data_mut().as_mut_ptr(),
                    n_bench,
                );
            }
        }
        let i4_ms = ms(t4.elapsed()) / iters as f64;
        let mem_after = ProcessMemory::sample();

        let mut mad = 0.0f64;
        let check_n = n_bench.min(256);
        for j in 0..check_n {
            mad += (y8.data()[j] - y4.data()[j]).abs() as f64;
        }
        mad /= check_n as f64;

        let note = if n_bench != n {
            format!(" (n capped {n}->{n_bench})")
        } else {
            String::new()
        };
        println!(
            "  {label:10} m={m} n={n_bench} k={k}{note}: INT8={i8_ms:.3} ms  INT4={i4_ms:.3} ms  \
             INT4/INT8={:.2}×  mad(first {check_n})={mad:.5}  Δfaults={}",
            i4_ms / i8_ms.max(1e-9),
            mem_after.page_faults.saturating_sub(mem_before.page_faults)
        );
    }
    println!();
}

/// Returns steady-state decode tok/s for this repetition.
fn run_model_rep(
    label: &str,
    model: &Transformer,
    tok: &ForgeTokenizer,
    prompt: &str,
    prefill_len: usize,
    decode_n: usize,
    max_seq: usize,
    rep: usize,
) -> f64 {
    let mut ids = tok.encode(prompt, true).expect("encode");
    if ids.len() > prefill_len {
        ids.truncate(prefill_len);
    }
    let prompt_tok = ids.len();

    let mem0 = ProcessMemory::sample();
    let faults0 = mem0.page_faults;

    let mut caches = LayerKvCaches::new(&model.cfg, max_seq).unwrap();
    let mut scratch = ForwardScratch::new(&model.cfg, max_seq).unwrap();

    let t_cold = Instant::now();
    model
        .forward_tokens(&ids, 0, &mut caches, &mut scratch)
        .unwrap();
    let cold_ms = ms(t_cold.elapsed());
    let mem_after_cold = ProcessMemory::sample();

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

    let mut ids = tok.encode(prompt, true).unwrap();
    if ids.len() > prefill_len {
        ids.truncate(prefill_len);
    }
    caches.clear();
    let faults_pre = ProcessMemory::sample().page_faults;

    let t_pre = Instant::now();
    model
        .forward_tokens(&ids, 0, &mut caches, &mut scratch)
        .unwrap();
    let prefill_ms = ms(t_pre.elapsed());
    logits = scratch.logits.data()[..model.cfg.vocab_size].to_vec();
    let faults_after_pre = ProcessMemory::sample().page_faults;

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
    let mem_end = ProcessMemory::sample();
    decode_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let tok_s = decode_n as f64 / (dec_total / 1e3);

    println!("=== {label} rep={rep} ===");
    println!(
        "  weights={:.1} MiB  prompt_tok={prompt_tok}",
        model.weight_nbytes() as f64 / (1024.0 * 1024.0)
    );
    println!(
        "  cold_prefill: {:.1} ms  warm_prefill: {:.1} ms ({:.2} tok/s)  TTFT≈{:.1} ms",
        cold_ms,
        prefill_ms,
        prompt_tok as f64 / (prefill_ms / 1e3),
        prefill_ms
    );
    println!(
        "  decode {decode_n} tok: mean={:.1} ms/tok ({tok_s:.2} tok/s)  p50={:.1}  p95={:.1}",
        dec_total / decode_n as f64,
        percentile(&decode_ms, 0.50),
        percentile(&decode_ms, 0.95)
    );
    println!("  mem_start: {}", mem0.format_short());
    println!(
        "  mem_after_cold: {}  Δfaults_cold={}",
        mem_after_cold.format_short(),
        mem_after_cold.page_faults.saturating_sub(faults0)
    );
    println!(
        "  Δfaults_warm_prefill={}  Δfaults_decode={}  major={}",
        faults_after_pre.saturating_sub(faults_pre),
        mem_end.page_faults.saturating_sub(faults_after_pre),
        mem_end.major_faults
    );
    println!("  mem_end: {}", mem_end.format_short());
    tok_s
}

fn main() -> ExitCode {
    let a = parse_args();

    println!("=== forge-int4-diag (single backend, conservative) ===");
    println!(
        "model={:?} backend={} group={}",
        a.model, a.backend, a.group
    );
    println!(
        "prefill={} decode={} reps={} max_seq={} min_avail_mib={}",
        a.prefill, a.decode_n, a.reps, a.max_seq, a.min_avail_mib
    );
    println!("mem: {}", ProcessMemory::sample().format_short());

    let est = match a.backend.as_str() {
        "fp32" => 2800u64,
        "int8" | "int4" => 1600,
        _ => 2000,
    };
    if let Err(e) = allow_multi_model_bench(est) {
        eprintln!("safety: {e}");
        return ExitCode::from(3);
    }
    if let Err(e) = require_available_ram_mib(a.min_avail_mib) {
        eprintln!("safety: {e}");
        return ExitCode::from(3);
    }

    if !a.skip_micro {
        microbench_quant_linears(a.group);
        if let Err(e) = require_available_ram_mib(a.min_avail_mib) {
            eprintln!("safety after micro: {e}");
            return ExitCode::from(3);
        }
    }

    if a.skip_model {
        return ExitCode::SUCCESS;
    }

    let prompt =
        "<|im_start|>user\nSay hello in one short sentence.<|im_end|>\n<|im_start|>assistant\n";

    let t_load = Instant::now();
    let mem_before_load = ProcessMemory::sample();
    // Prefer `.forge` when the path ends in .forge — avoids FP32+quant peak.
    let is_forge = a
        .model
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("forge"))
        .unwrap_or(false);

    let (model, tok_dir): (Transformer, PathBuf) = if is_forge {
        let loaded =
            load_forge_path(&a.model, LoadOptions { memory_map: true }).expect("forge load");
        let parent = forge_runtime::model::tokenizer_dir(&a.model, &loaded.model.cfg)
            .expect("tokenizer directory");
        (loaded.model, parent)
    } else {
        let mut model = load_hf_dir(&a.model).expect("load model");
        match a.backend.as_str() {
            "fp32" => {}
            "int8" => {
                model
                    .quantize_weights(QuantBits::Int8, a.group)
                    .expect("int8 quant");
            }
            "int4" => {
                model
                    .quantize_weights(QuantBits::Int4, a.group)
                    .expect("int4 quant");
            }
            other => {
                eprintln!("unknown --backend {other} (fp32|int8|int4)");
                return ExitCode::from(2);
            }
        }
        (model, a.model.clone())
    };
    let load_ms = ms(t_load.elapsed());
    let mem_after_load = ProcessMemory::sample();
    println!(
        "load{}: {:.1} ms  weights={:.1} MiB",
        if is_forge { "(.forge)" } else { "+quant" },
        load_ms,
        model.weight_nbytes() as f64 / (1024.0 * 1024.0)
    );
    println!("  before: {}", mem_before_load.format_short());
    println!(
        "  after:  {}  Δfaults={}",
        mem_after_load.format_short(),
        mem_after_load
            .page_faults
            .saturating_sub(mem_before_load.page_faults)
    );

    let commit_mib = mem_after_load.committed_bytes / (1024 * 1024);
    let cap = env::var("FORGE_BENCH_MEM_CAP_MIB")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(6144u64);
    if commit_mib > cap {
        eprintln!("safety: commit {commit_mib} MiB > FORGE_BENCH_MEM_CAP_MIB={cap}; aborting");
        return ExitCode::from(3);
    }

    let tok = ForgeTokenizer::for_model(&tok_dir, &model.cfg).expect("tok");
    let label = format!("{}-g{}", a.backend.to_uppercase(), a.group);

    let mut decode_tps = Vec::new();
    for rep in 1..=a.reps {
        if let Err(e) = require_available_ram_mib(a.min_avail_mib.saturating_sub(512)) {
            eprintln!("safety mid-run: {e}");
            return ExitCode::from(3);
        }
        decode_tps.push(run_model_rep(
            &label, &model, &tok, prompt, a.prefill, a.decode_n, a.max_seq, rep,
        ));
    }

    decode_tps.sort_by(|x, y| x.partial_cmp(y).unwrap());
    println!();
    println!(
        "SUMMARY {label}: decode tok/s over {} reps: mean={:.2}  p50={:.2}  min={:.2}  max={:.2}",
        decode_tps.len(),
        mean(&decode_tps),
        percentile(&decode_tps, 0.50),
        decode_tps.first().copied().unwrap_or(0.0),
        decode_tps.last().copied().unwrap_or(0.0)
    );
    println!("mem_final: {}", ProcessMemory::sample().format_short());
    ExitCode::SUCCESS
}
