//! Offline weight-only quantization → `.forge` checkpoint.
//!
//! ```text
//! forge-quantize --model models/Qwen2.5-0.5B-Instruct --bits 8 --group-size 64 \
//!   --output models/qwen2.5-int8.forge
//! ```

use forge_runtime::checkpoint::{save_forge, SaveOptions};
use forge_runtime::mem::ProcessMemory;
use forge_runtime::model::{load_hf_dir, QuantBits};
use std::env;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

fn usage() -> ! {
    eprintln!(
        "Usage: forge-quantize --model <HF_DIR> --bits <4|8> --group-size <N> --output <file.forge>"
    );
    std::process::exit(2);
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let mut model = None;
    let mut bits = None;
    let mut group = 64usize;
    let mut output = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                model = args.get(i + 1).cloned();
                i += 2;
            }
            "--bits" => {
                bits = args.get(i + 1).and_then(|s| s.parse().ok());
                i += 2;
            }
            "--group-size" | "--group" => {
                group = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(64);
                i += 2;
            }
            "--output" | "-o" => {
                output = args.get(i + 1).cloned();
                i += 2;
            }
            "-h" | "--help" => usage(),
            _ => i += 1,
        }
    }
    let model_dir = model.unwrap_or_else(|| usage());
    let bits: u8 = bits.unwrap_or_else(|| usage());
    let out: PathBuf = output.unwrap_or_else(|| usage()).into();
    let qb = match QuantBits::from_u8(bits) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("--bits must be 4 or 8");
            return ExitCode::from(2);
        }
    };

    if out.extension().and_then(|e| e.to_str()) != Some("forge") {
        eprintln!("warning: output path should end in .forge");
    }

    eprintln!("=== forge-quantize ===");
    eprintln!("model:  {model_dir}");
    eprintln!("bits:   {bits}  group: {group}");
    eprintln!("output: {}", out.display());
    eprintln!("mem@start: {}", ProcessMemory::sample().format_short());

    let t0 = Instant::now();
    let mut m = match load_hf_dir(&model_dir) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("load failed: {e}");
            return ExitCode::FAILURE;
        }
    };
    eprintln!(
        "loaded FP32 in {:.2}s  weights={:.1} MiB  {}",
        t0.elapsed().as_secs_f32(),
        m.weight_nbytes() as f64 / (1024.0 * 1024.0),
        ProcessMemory::sample().format_short()
    );

    let tq = Instant::now();
    if let Err(e) = m.quantize_weights(qb, group) {
        eprintln!("quantize failed: {e}");
        return ExitCode::FAILURE;
    }
    eprintln!(
        "quantized in {:.2}s  weights={:.1} MiB  {}",
        tq.elapsed().as_secs_f32(),
        m.weight_nbytes() as f64 / (1024.0 * 1024.0),
        ProcessMemory::sample().format_short()
    );

    let ts = Instant::now();
    let opts = SaveOptions {
        name: out
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model")
            .into(),
        bos_token_id: m.cfg.bos_token_id,
        eos_token_id: m.cfg.eos_token_id,
    };
    if let Err(e) = save_forge(&m, &out, opts) {
        eprintln!("save failed: {e}");
        return ExitCode::FAILURE;
    }
    let sz = std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0);
    eprintln!(
        "wrote {} ({:.1} MiB) in {:.2}s",
        out.display(),
        sz as f64 / (1024.0 * 1024.0),
        ts.elapsed().as_secs_f32()
    );
    eprintln!(
        "total {:.2}s  peak-ish {}",
        t0.elapsed().as_secs_f32(),
        ProcessMemory::sample().format_short()
    );
    ExitCode::SUCCESS
}
