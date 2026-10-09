//! Phase 12 audited CUDA continuous-batching benchmark harness.
//!
//! Methodology: docs/BENCHMARK_METHODOLOGY.md
//!
//! ```bash
//! cargo run --release --features cuda --example bench_phase12 -- \
//!   --model ../models/Qwen2.5-0.5B-Instruct --dtype bf16 --batch 1,2,4,8 \
//!   --max-new 16 --trials 3 --warmup 1 --token-policy fixed_len \
//!   --json-out ../benchmarks/results/forge_short.json
//! ```

#![recursion_limit = "256"]

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("bench_phase12 requires --features cuda");
    std::process::exit(1);
}

#[cfg(feature = "cuda")]
use forge_runtime::cuda_model::{CudaBatcher, CudaTransformer, GpuWeightDType};
#[cfg(feature = "cuda")]
use forge_runtime::model::{load_qwen2_dir, ForgeTokenizer, GenerateParams};
#[cfg(feature = "cuda")]
use forge_runtime::ComputeBackend;
#[cfg(feature = "cuda")]
use serde_json::{json, Value};
#[cfg(feature = "cuda")]
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::env;
#[cfg(feature = "cuda")]
use std::fs;
#[cfg(feature = "cuda")]
use std::path::PathBuf;
#[cfg(feature = "cuda")]
use std::time::Instant;

#[cfg(feature = "cuda")]
fn main() {

    let args: Vec<String> = env::args().collect();
    let cfg = Cli::parse(&args);

    eprintln!("=== bench_phase12 (audited) ===");
    eprintln!(
        "model={} dtype={:?} batches={:?} max_new={} max_seq={} trials={} warmup={} policy={}",
        cfg.model.display(),
        cfg.dtype,
        cfg.batches,
        cfg.max_new,
        cfg.max_seq,
        cfg.trials,
        cfg.warmup,
        cfg.token_policy
    );

    let t_load = Instant::now();
    let model = load_qwen2_dir(&cfg.model).expect("load model");
    let load_s = t_load.elapsed().as_secs_f64();
    let tok = ForgeTokenizer::from_dir(&cfg.model, 151643, 151645).expect("tok");

    let prompts = build_prompts(&cfg, &tok);
    for (i, p) in prompts.iter().enumerate() {
        let n = tok.encode(p, true).map(|t| t.len()).unwrap_or(0);
        eprintln!("  prompt[{i}] tokens={n}");
    }

    let t_up = Instant::now();
    let cuda = CudaTransformer::from_cpu_dtype(&model, 0, cfg.dtype).expect("cuda upload");
    cuda.backend().synchronize().ok();
    let upload_s = t_up.elapsed().as_secs_f64();
    let weight_mib = cuda.weight_bytes() as f64 / (1024.0 * 1024.0);
    let (free0, total0) = cuda.device_mem_info();
    eprintln!(
        "load={load_s:.2}s upload={upload_s:.2}s weights={weight_mib:.1} MiB VRAM free/total={:?}/{:?}",
        free0.map(|b| b as f64 / (1024.0 * 1024.0)),
        total0.map(|b| b as f64 / (1024.0 * 1024.0))
    );

    // One batcher sized to max B in this run (steady-state weights; no re-upload per B).
    let max_b = *cfg.batches.iter().max().unwrap_or(&1);
    let tok_b = ForgeTokenizer::from_dir(&cfg.model, 151643, 151645).expect("tok");
    let mut batcher = CudaBatcher::new(cuda, tok_b, cfg.max_seq, max_b).expect("batcher");
    assert_eq!(batcher.max_batch(), max_b);

    let eos = match cfg.token_policy.as_str() {
        "with_eos" => Some(tok.eos_token_id),
        _ => None, // fixed_len
    };
    let params = GenerateParams {
        max_tokens: cfg.max_new,
        temperature: 0.0,
        top_k: 0,
        eos_token_id: eos,
    };

    let mut all_results: Vec<Value> = Vec::new();

    for &b in &cfg.batches {
        eprintln!("\n=== configured_admit_B={b} (max_batch={max_b}) ===");
        let prompt_lens: Vec<usize> = prompts
            .iter()
            .map(|p| tok.encode(p, true).map(|t| t.len()).unwrap_or(0))
            .collect();

        // Warm-up (excluded)
        for w in 0..cfg.warmup {
            let _ = run_trial(&mut batcher, &prompts, &prompt_lens, b, &params, false);
            eprintln!("  warmup {w} done");
        }

        let mut trials: Vec<TrialMetrics> = Vec::new();
        for t in 0..cfg.trials {
            let m = run_trial(&mut batcher, &prompts, &prompt_lens, b, &params, true).expect("trial");
            eprintln!(
                "  trial {t}: tokens={} wall={:.3}s agg={:.2} tok/s ttft_p50={:.4}s e2e_p50={:.4}s occ={:.2}",
                m.total_tokens,
                m.e2e_wall_s,
                m.aggregate_tok_s,
                m.ttft_p50_s,
                m.e2e_p50_s,
                m.avg_occupancy
            );
            trials.push(m);
        }

        let summary = summarize_trials(&trials, b, max_b, &cfg, weight_mib, load_s, upload_s);
        eprintln!(
            "  MEDIAN agg={:.2} tok/s  decode≈{:.2}  prefill≈{:.2}  ttft_p50={:.4}s  (n={})",
            summary["aggregate_tok_s_median"].as_f64().unwrap_or(0.0),
            summary["steady_decode_tok_s_median"].as_f64().unwrap_or(0.0),
            summary["prefill_tok_s_median"].as_f64().unwrap_or(0.0),
            summary["ttft_p50_median_s"].as_f64().unwrap_or(0.0),
            cfg.trials
        );
        all_results.push(summary);
    }

    let out = json!({
        "harness": "bench_phase12",
        "methodology": "docs/BENCHMARK_METHODOLOGY.md",
        "model_dir": cfg.model.display().to_string(),
        "dtype": format!("{:?}", cfg.dtype),
        "token_policy": cfg.token_policy,
        "max_new": cfg.max_new,
        "max_seq": cfg.max_seq,
        "warmup": cfg.warmup,
        "trials": cfg.trials,
        "workload": cfg.workload,
        "label": cfg.label,
        "results": all_results,
    });

    let text = serde_json::to_string_pretty(&out).expect("json");
    if let Some(path) = &cfg.json_out {
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        fs::write(path, &text).expect("write json");
        eprintln!("\nWrote {}", path.display());
    }
    println!("{text}");

    // Silence unused in stub-free path
    let _ = model;
}

#[cfg(feature = "cuda")]
#[derive(Clone)]
struct Cli {
    model: PathBuf,
    dtype: forge_runtime::cuda_model::GpuWeightDType,
    batches: Vec<usize>,
    max_new: usize,
    max_seq: usize,
    trials: usize,
    warmup: usize,
    token_policy: String,
    workload: String,
    label: String,
    json_out: Option<PathBuf>,
    prompt: String,
    mixed: bool,
}

#[cfg(feature = "cuda")]
impl Cli {
    fn parse(args: &[String]) -> Self {
        use forge_runtime::cuda_model::GpuWeightDType;
        use std::path::PathBuf;

        let mut model = PathBuf::from("../models/Qwen2.5-0.5B-Instruct");
        let mut dtype = GpuWeightDType::Fp32;
        let mut batches = vec![1usize, 2, 4, 8];
        let mut max_new = 16usize;
        let mut max_seq = 512usize;
        let mut trials = 3usize;
        let mut warmup = 1usize;
        let mut token_policy = "fixed_len".to_string();
        let mut workload = "short_hello".to_string();
        let mut label = String::new();
        let mut json_out = None;
        let mut prompt = env_or(
            "FORGE_PROMPT",
            "<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n",
        );
        let mut mixed = env::var("FORGE_MIXED_PROMPTS").unwrap_or_default() == "1";

        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--model" => {
                    model = PathBuf::from(args.get(i + 1).expect("--model"));
                    i += 2;
                }
                "--dtype" => {
                    let s = args.get(i + 1).expect("--dtype").to_ascii_lowercase();
                    dtype = if s == "bf16" || s == "bfloat16" {
                        GpuWeightDType::Bf16
                    } else {
                        GpuWeightDType::Fp32
                    };
                    i += 2;
                }
                "--batch" => {
                    batches = args
                        .get(i + 1)
                        .expect("--batch")
                        .split(',')
                        .filter_map(|s| s.parse().ok())
                        .collect();
                    i += 2;
                }
                "--max-new" => {
                    max_new = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(16);
                    i += 2;
                }
                "--max-seq" => {
                    max_seq = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(512);
                    i += 2;
                }
                "--trials" => {
                    trials = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(3);
                    i += 2;
                }
                "--warmup" => {
                    warmup = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(1);
                    i += 2;
                }
                "--token-policy" => {
                    token_policy = args.get(i + 1).cloned().unwrap_or(token_policy);
                    i += 2;
                }
                "--workload" => {
                    workload = args.get(i + 1).cloned().unwrap_or(workload);
                    i += 2;
                }
                "--label" => {
                    label = args.get(i + 1).cloned().unwrap_or_default();
                    i += 2;
                }
                "--json-out" => {
                    json_out = Some(PathBuf::from(args.get(i + 1).expect("--json-out")));
                    i += 2;
                }
                "--prompt" => {
                    prompt = args.get(i + 1).cloned().unwrap_or(prompt);
                    i += 2;
                }
                "--mixed" => {
                    mixed = true;
                    i += 1;
                }
                "--medium" => {
                    // ~256-token-ish padded prompt
                    let pad = "alpha ".repeat(220);
                    prompt = format!(
                        "<|im_start|>user\nSummarize the following:\n{pad}\n<|im_end|>\n<|im_start|>assistant\n"
                    );
                    workload = "medium_256".into();
                    if max_seq < 1024 {
                        max_seq = 1024;
                    }
                    i += 1;
                }
                "--long" => {
                    let pad = "beta ".repeat(980);
                    prompt = format!(
                        "<|im_start|>user\nRead carefully then reply with one word: OK.\n{pad}\n<|im_end|>\n<|im_start|>assistant\n"
                    );
                    workload = "long_1024".into();
                    if max_seq < 2048 {
                        max_seq = 2048;
                    }
                    i += 1;
                }
                other if other.starts_with('-') => {
                    eprintln!("unknown flag {other}");
                    i += 1;
                }
                // positional: model [max_new]
                _ => {
                    if i == 1 {
                        model = PathBuf::from(&args[i]);
                    } else if i == 2 {
                        max_new = args[i].parse().unwrap_or(max_new);
                    }
                    i += 1;
                }
            }
        }

        // Env dtype override (matches Phase 11 scripts)
        if let Ok(d) = env::var("FORGE_CUDA_DTYPE") {
            let s = d.to_ascii_lowercase();
            if s == "bf16" || s == "bfloat16" {
                dtype = GpuWeightDType::Bf16;
            }
        }

        if batches.is_empty() {
            batches = vec![1];
        }

        Self {
            model,
            dtype,
            batches,
            max_new,
            max_seq,
            trials,
            warmup,
            token_policy,
            workload,
            label,
            json_out,
            prompt,
            mixed,
        }
    }
}

#[cfg(feature = "cuda")]
fn env_or(key: &str, default: &str) -> String {
    env::var(key).unwrap_or_else(|_| default.to_string())
}

#[cfg(feature = "cuda")]
fn build_prompts(cfg: &Cli, _tok: &forge_runtime::model::ForgeTokenizer) -> Vec<String> {
    if cfg.mixed || cfg.workload == "mixed_stagger" {
        return vec![
            cfg.prompt.clone(),
            "<|im_start|>user\nShort.<|im_end|>\n<|im_start|>assistant\n".into(),
            "<|im_start|>user\nExplain continuous batching in one sentence.<|im_end|>\n<|im_start|>assistant\n".into(),
            "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n".into(),
            cfg.prompt.clone(),
            "<|im_start|>user\nList three primes.<|im_end|>\n<|im_start|>assistant\n".into(),
            cfg.prompt.clone(),
            "<|im_start|>user\nCompute 17*19.<|im_end|>\n<|im_start|>assistant\n".into(),
        ];
    }
    vec![cfg.prompt.clone(); 8]
}

#[cfg(feature = "cuda")]
#[derive(Clone, Debug)]
struct TrialMetrics {
    n_req: usize,
    total_tokens: usize,
    prompt_tokens: usize,
    e2e_wall_s: f64,
    aggregate_tok_s: f64,
    mean_per_req_tok_s: f64,
    request_throughput: f64,
    prefill_tok_s: f64,
    steady_decode_tok_s: f64,
    ttft_p50_s: f64,
    ttft_p95_s: f64,
    e2e_p50_s: f64,
    e2e_p95_s: f64,
    avg_occupancy: f64,
    kv_util_peak: f64,
    sched_prefill: usize,
    sched_decode: usize,
}

#[cfg(feature = "cuda")]
fn run_trial(
    batcher: &mut forge_runtime::cuda_model::CudaBatcher,
    prompts: &[String],
    prompt_lens: &[usize],
    n_req: usize,
    params: &forge_runtime::model::GenerateParams,
    measure: bool,
) -> Result<TrialMetrics, String> {
    batcher.synchronize().map_err(|e| e.to_string())?;

    let mut prompt_tokens = 0usize;
    let t_admit = Instant::now();
    let mut ids = Vec::with_capacity(n_req);
    // Staggered admit for mixed: admit one, step once, then continue admitting.
    let stagger = env::var("FORGE_STAGGER_ADMIT").unwrap_or_default() == "1";
    if stagger {
        for i in 0..n_req {
            let id = batcher
                .admit(&prompts[i % prompts.len()], params.clone())
                .map_err(|e| e.to_string())?;
            ids.push(id);
            prompt_tokens += prompt_lens[i % prompt_lens.len()];
            if batcher.active_count() > 0 {
                let _ = batcher.step().map_err(|e| e.to_string())?;
            }
        }
    } else {
        for i in 0..n_req {
            let id = batcher
                .admit(&prompts[i % prompts.len()], params.clone())
                .map_err(|e| e.to_string())?;
            ids.push(id);
            prompt_tokens += prompt_lens[i % prompt_lens.len()];
        }
    }

    let mut per_tok: HashMap<u64, usize> = ids.iter().map(|&id| (id, 0)).collect();
    let mut first_s: HashMap<u64, f64> = HashMap::new();
    let mut done_s: HashMap<u64, f64> = HashMap::new();
    let mut occupancy_sum = 0usize;
    let mut occupancy_n = 0usize;
    let mut sched_prefill = 0usize;
    let mut sched_decode = 0usize;
    let mut kv_util_peak = 0.0f64;

    // Clock for e2e includes admit→done (methodology: TTFT from admit).
    let t0 = t_admit;
    while batcher.active_count() > 0 {
        let evs = batcher.step().map_err(|e| e.to_string())?;
        sched_prefill += batcher.stats().last_scheduled_prefill;
        sched_decode += batcher.stats().last_scheduled_decode;
        if batcher.stats().last_batch_size > 0 {
            occupancy_sum += batcher.stats().last_batch_size;
            occupancy_n += 1;
        }
        let kv = batcher.kv_stats();
        kv_util_peak = kv_util_peak.max(kv.utilization);
        let now = t0.elapsed().as_secs_f64();
        for ev in evs {
            *per_tok.get_mut(&ev.request_id).unwrap() += 1;
            first_s.entry(ev.request_id).or_insert(now);
            if ev.done {
                done_s.insert(ev.request_id, now);
            }
        }
    }
    batcher.synchronize().map_err(|e| e.to_string())?;
    let wall = t0.elapsed().as_secs_f64();

    let total: usize = per_tok.values().sum();
    let avg_occ = if occupancy_n > 0 {
        occupancy_sum as f64 / occupancy_n as f64
    } else {
        0.0
    };

    let mut ttfts: Vec<f64> = first_s.values().copied().collect();
    ttfts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut e2es: Vec<f64> = done_s.values().copied().collect();
    e2es.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut per_req_rates = Vec::new();
    for &id in &ids {
        let tok_n = *per_tok.get(&id).unwrap_or(&0);
        let e2e = *done_s.get(&id).unwrap_or(&wall);
        if e2e > 0.0 && tok_n > 0 {
            per_req_rates.push(tok_n as f64 / e2e);
        }
    }
    let mean_per_req = if per_req_rates.is_empty() {
        0.0
    } else {
        per_req_rates.iter().sum::<f64>() / per_req_rates.len() as f64
    };

    let t_last_first = ttfts.last().copied().unwrap_or(wall);
    let decode_tokens = total.saturating_sub(n_req); // after first token each
    let prefill_tok_s = if t_last_first > 0.0 && prompt_tokens > 0 {
        prompt_tokens as f64 / t_last_first
    } else {
        0.0
    };
    let steady_decode = if wall > t_last_first && decode_tokens > 0 {
        decode_tokens as f64 / (wall - t_last_first)
    } else if wall > 0.0 {
        total as f64 / wall
    } else {
        0.0
    };

    if !measure {
        return Ok(TrialMetrics {
            n_req,
            total_tokens: total,
            prompt_tokens,
            e2e_wall_s: wall,
            aggregate_tok_s: 0.0,
            mean_per_req_tok_s: 0.0,
            request_throughput: 0.0,
            prefill_tok_s: 0.0,
            steady_decode_tok_s: 0.0,
            ttft_p50_s: 0.0,
            ttft_p95_s: 0.0,
            e2e_p50_s: 0.0,
            e2e_p95_s: 0.0,
            avg_occupancy: avg_occ,
            kv_util_peak,
            sched_prefill,
            sched_decode,
        });
    }

    Ok(TrialMetrics {
        n_req,
        total_tokens: total,
        prompt_tokens,
        e2e_wall_s: wall,
        aggregate_tok_s: if wall > 0.0 {
            total as f64 / wall
        } else {
            0.0
        },
        mean_per_req_tok_s: mean_per_req,
        request_throughput: if wall > 0.0 {
            n_req as f64 / wall
        } else {
            0.0
        },
        prefill_tok_s,
        steady_decode_tok_s: steady_decode,
        ttft_p50_s: percentile(&ttfts, 0.50),
        ttft_p95_s: percentile(&ttfts, 0.95),
        e2e_p50_s: percentile(&e2es, 0.50),
        e2e_p95_s: percentile(&e2es, 0.95),
        avg_occupancy: avg_occ,
        kv_util_peak,
        sched_prefill,
        sched_decode,
    })
}

#[cfg(feature = "cuda")]
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

#[cfg(feature = "cuda")]
fn median(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    percentile(&v, 0.50)
}

#[cfg(feature = "cuda")]
fn summarize_trials(
    trials: &[TrialMetrics],
    admit_b: usize,
    max_batch: usize,
    cfg: &Cli,
    weight_mib: f64,
    load_s: f64,
    upload_s: f64,
) -> serde_json::Value {
    use serde_json::json;
    let agg: Vec<f64> = trials.iter().map(|t| t.aggregate_tok_s).collect();
    let dec: Vec<f64> = trials.iter().map(|t| t.steady_decode_tok_s).collect();
    let pref: Vec<f64> = trials.iter().map(|t| t.prefill_tok_s).collect();
    let ttft: Vec<f64> = trials.iter().map(|t| t.ttft_p50_s).collect();
    let e2e: Vec<f64> = trials.iter().map(|t| t.e2e_p50_s).collect();
    let walls: Vec<f64> = trials.iter().map(|t| t.e2e_wall_s).collect();
    let toks: Vec<f64> = trials.iter().map(|t| t.total_tokens as f64).collect();

    json!({
        "admit_batch": admit_b,
        "max_batch": max_batch,
        "dtype": format!("{:?}", cfg.dtype),
        "token_policy": cfg.token_policy,
        "max_new": cfg.max_new,
        "max_seq": cfg.max_seq,
        "workload": cfg.workload,
        "label": cfg.label,
        "weight_mib": weight_mib,
        "load_s": load_s,
        "upload_s": upload_s,
        "n_trials": trials.len(),
        "total_tokens_median": median(&toks),
        "e2e_wall_s_median": median(&walls),
        "e2e_wall_s_min": walls.iter().cloned().fold(f64::INFINITY, f64::min),
        "e2e_wall_s_max": walls.iter().cloned().fold(0.0, f64::max),
        "aggregate_tok_s_median": median(&agg),
        "aggregate_tok_s_min": agg.iter().cloned().fold(f64::INFINITY, f64::min),
        "aggregate_tok_s_max": agg.iter().cloned().fold(0.0, f64::max),
        "steady_decode_tok_s_median": median(&dec),
        "prefill_tok_s_median": median(&pref),
        "mean_per_req_tok_s_median": median(&trials.iter().map(|t| t.mean_per_req_tok_s).collect::<Vec<_>>()),
        "request_throughput_median": median(&trials.iter().map(|t| t.request_throughput).collect::<Vec<_>>()),
        "ttft_p50_median_s": median(&ttft),
        "ttft_p95_median_s": median(&trials.iter().map(|t| t.ttft_p95_s).collect::<Vec<_>>()),
        "e2e_p50_median_s": median(&e2e),
        "e2e_p95_median_s": median(&trials.iter().map(|t| t.e2e_p95_s).collect::<Vec<_>>()),
        "avg_occupancy_median": median(&trials.iter().map(|t| t.avg_occupancy).collect::<Vec<_>>()),
        "kv_util_peak_median": median(&trials.iter().map(|t| t.kv_util_peak).collect::<Vec<_>>()),
        "trials": trial_json_list(trials),
    })
}

#[cfg(feature = "cuda")]
fn trial_json_list(trials: &[TrialMetrics]) -> Vec<Value> {
    trials
        .iter()
        .map(|t| {
            json!({
                "total_tokens": t.total_tokens,
                "e2e_wall_s": t.e2e_wall_s,
                "aggregate_tok_s": t.aggregate_tok_s,
                "steady_decode_tok_s": t.steady_decode_tok_s,
                "prefill_tok_s": t.prefill_tok_s,
                "mean_per_req_tok_s": t.mean_per_req_tok_s,
                "ttft_p50_s": t.ttft_p50_s,
                "ttft_p95_s": t.ttft_p95_s,
                "e2e_p50_s": t.e2e_p50_s,
                "e2e_p95_s": t.e2e_p95_s,
                "avg_occupancy": t.avg_occupancy,
                "kv_util_peak": t.kv_util_peak,
                "sched_prefill": t.sched_prefill,
                "sched_decode": t.sched_decode,
            })
        })
        .collect()
}
