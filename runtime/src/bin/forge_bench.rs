//! Conservative bench: one backend per process, memory safeguards.
//!
//! ```text
//! forge-bench --model models/Qwen2.5-0.5B-Instruct --backend fp32
//! forge-bench --model models/Qwen2.5-0.5B-Instruct --backend int4 --group 64
//! forge-bench --model models/qwen2.5-int8.forge --backend forge
//! ```

use forge_runtime::checkpoint::{load_forge_path, LoadOptions};
use forge_runtime::mem::{allow_multi_model_bench, ProcessMemory};
use forge_runtime::model::{
    load_hf_dir, ForgeTokenizer, ForwardScratch, LayerKvCaches, QuantBits, Transformer,
};
use forge_runtime::sampling::greedy;
use std::env;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

fn ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let mut model_path = PathBuf::from("../models/Qwen2.5-0.5B-Instruct");
    let mut backend = "fp32".to_string();
    let mut group = 64usize;
    let mut prefill = 64usize;
    let mut decode_n = 16usize;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                model_path = PathBuf::from(args.get(i + 1).expect("--model path"));
                i += 2;
            }
            "--backend" => {
                backend = args.get(i + 1).cloned().unwrap_or(backend);
                i += 2;
            }
            "--group" => {
                group = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(64);
                i += 2;
            }
            "--prefill" => {
                prefill = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(64);
                i += 2;
            }
            "--decode" => {
                decode_n = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(16);
                i += 2;
            }
            _ => i += 1,
        }
    }

    let est = match backend.as_str() {
        "fp32" => 2800u64,
        _ => 1600,
    };
    if let Err(e) = allow_multi_model_bench(est) {
        eprintln!("safety: {e}");
        return ExitCode::from(3);
    }

    eprintln!("=== forge-bench (single backend) ===");
    eprintln!("model: {model_path:?}  backend={backend} group={group}");
    eprintln!("mem: {}", ProcessMemory::sample().format_short());

    let t_load = Instant::now();
    let (model, tok_dir): (Transformer, PathBuf) = match backend.as_str() {
        "fp32" => {
            let m = load_hf_dir(&model_path).expect("load");
            (m, model_path.clone())
        }
        "int8" => {
            let mut m = load_hf_dir(&model_path).expect("load");
            m.quantize_weights(QuantBits::Int8, group).expect("quant");
            (m, model_path.clone())
        }
        "int4" => {
            let mut m = load_hf_dir(&model_path).expect("load");
            m.quantize_weights(QuantBits::Int4, group).expect("quant");
            (m, model_path.clone())
        }
        "forge" => {
            let loaded =
                load_forge_path(&model_path, LoadOptions { memory_map: true }).expect("forge");
            let parent = forge_runtime::model::tokenizer_dir(&model_path, &loaded.model.cfg).expect("tokenizer directory");
            (loaded.model, parent)
        }
        other => {
            eprintln!("unknown --backend {other} (fp32|int8|int4|forge)");
            return ExitCode::from(2);
        }
    };
    let load_ms = ms(t_load.elapsed());
    eprintln!(
        "loaded in {:.1} ms  weights={:.1} MiB  {}",
        load_ms,
        model.weight_nbytes() as f64 / (1024.0 * 1024.0),
        ProcessMemory::sample().format_short()
    );

    let tok = ForgeTokenizer::for_model(&tok_dir, &model.cfg).expect("tok");
    let prompt = "<|im_start|>user\nSay hello briefly.<|im_end|>\n<|im_start|>assistant\n";
    let mut ids = tok.encode(prompt, true).unwrap();
    if ids.len() > prefill {
        ids.truncate(prefill);
    }
    let max_seq = 512usize;
    let mut caches = LayerKvCaches::new(&model.cfg, max_seq).unwrap();
    let mut scratch = ForwardScratch::new(&model.cfg, max_seq).unwrap();

    let t_cold = Instant::now();
    model
        .forward_tokens(&ids, 0, &mut caches, &mut scratch)
        .unwrap();
    let cold = ms(t_cold.elapsed());

    caches.clear();
    let t_warm = Instant::now();
    model
        .forward_tokens(&ids, 0, &mut caches, &mut scratch)
        .unwrap();
    let warm_pre = ms(t_warm.elapsed());
    let mut logits = scratch.logits.data()[..model.cfg.vocab_size].to_vec();

    let t_dec = Instant::now();
    for _ in 0..decode_n {
        let next = greedy(&logits);
        ids.push(next);
        let pos = ids.len() - 1;
        model
            .forward_tokens(&[next], pos, &mut caches, &mut scratch)
            .unwrap();
        logits = scratch.logits.data()[..model.cfg.vocab_size].to_vec();
    }
    let dec = ms(t_dec.elapsed());

    eprintln!("cold_prefill: {cold:.1} ms");
    eprintln!(
        "warm_prefill ({} tok): {warm_pre:.1} ms ({:.2} tok/s)  TTFT≈{warm_pre:.1} ms",
        ids.len() - decode_n,
        (ids.len() - decode_n) as f64 / (warm_pre / 1e3)
    );
    eprintln!(
        "decode {decode_n} tok: mean={:.1} ms/tok ({:.2} tok/s)",
        dec / decode_n as f64,
        decode_n as f64 / (dec / 1e3)
    );
    eprintln!("mem@end: {}", ProcessMemory::sample().format_short());
    ExitCode::SUCCESS
}
