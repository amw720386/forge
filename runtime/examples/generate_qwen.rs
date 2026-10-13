//! Load Qwen2.5-0.5B-Instruct and generate greedily.
//!
//! ```bash
//! cargo run --release --example generate_qwen -- ../models/Qwen2.5-0.5B-Instruct
//! cargo run --release --example generate_qwen -- ../models/Qwen2.5-0.5B-Instruct --bits 8 --group 64
//! ```

use forge_runtime::model::{generate, load_hf_dir, ForgeTokenizer, GenerateParams, QuantBits};
use std::env;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();
    let dir = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "../models/Qwen2.5-0.5B-Instruct".into());

    let mut bits: Option<u8> = None;
    let mut group: usize = 64;
    let mut max_tokens: usize = 32;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--bits" => {
                bits = args.get(i + 1).and_then(|s| s.parse().ok());
                i += 2;
            }
            "--group" => {
                group = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(64);
                i += 2;
            }
            "--max-tokens" => {
                max_tokens = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(32);
                i += 2;
            }
            other if other.parse::<usize>().is_ok() => {
                max_tokens = other.parse().unwrap();
                i += 1;
            }
            _ => i += 1,
        }
    }

    let prompt = env::var("FORGE_PROMPT")
        .unwrap_or_else(|_| "<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n".into());

    eprintln!("loading model from {dir} ...");
    let t0 = Instant::now();
    let mut model = load_hf_dir(&dir).expect("load model");
    if let Some(b) = bits {
        let qb = QuantBits::from_u8(b).expect("--bits must be 4 or 8");
        model.quantize_weights(qb, group).expect("quantize");
        eprintln!(
            "quantized to INT{b} group={group}  weights={:.2} MiB",
            model.weight_nbytes() as f64 / (1024.0 * 1024.0)
        );
    } else {
        eprintln!(
            "FP32 weights={:.2} MiB",
            model.weight_nbytes() as f64 / (1024.0 * 1024.0)
        );
    }
    eprintln!("loaded in {:.2}s", t0.elapsed().as_secs_f32());

    let tok = ForgeTokenizer::from_dir(&dir, model.cfg.bos_token_id, model.cfg.eos_token_id)
        .expect("tokenizer");
    let params = GenerateParams {
        max_tokens,
        temperature: 0.0,
        top_k: 0,
        eos_token_id: Some(tok.eos_token_id),
    };

    let t1 = Instant::now();
    let out = generate(&model, &tok, &prompt, &params, 512).expect("generate");
    let dt = t1.elapsed().as_secs_f32();
    eprintln!(
        "generated {} tokens in {:.2}s ({:.2} tok/s)",
        out.tokens.len(),
        dt,
        out.tokens.len() as f32 / dt.max(1e-6)
    );
    eprintln!("token ids: {:?}", out.tokens);
    println!("{}", out.text);
}
