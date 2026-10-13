//! Generate with CUDA backend (requires `--features cuda` + GPU).
//!
//! ```bash
//! cargo run --release --features cuda --example generate_cuda_qwen -- ../models/Qwen2.5-0.5B-Instruct
//! ```

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("generate_cuda_qwen requires --features cuda");
    std::process::exit(1);
}

#[cfg(feature = "cuda")]
fn main() {
    use forge_runtime::cuda_model::{generate_cuda, CudaTransformer, GpuWeightDType};
    use forge_runtime::model::{load_qwen2_dir, ForgeTokenizer, GenerateParams};
    use std::env;
    use std::time::Instant;

    let args: Vec<String> = env::args().collect();
    let dir = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "../models/Qwen2.5-0.5B-Instruct".into());
    let max_tokens: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(32);
    let prompt = env::var("FORGE_PROMPT")
        .unwrap_or_else(|_| "<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n".into());
    let dtype = match env::var("FORGE_CUDA_DTYPE")
        .unwrap_or_else(|_| "fp32".into())
        .to_ascii_lowercase()
        .as_str()
    {
        "bf16" | "bfloat16" => GpuWeightDType::Bf16,
        _ => GpuWeightDType::Fp32,
    };

    eprintln!("loading CPU FP32 from {dir} ...");
    let t0 = Instant::now();
    let model = load_qwen2_dir(&dir).expect("load");
    eprintln!("CPU load {:.2}s", t0.elapsed().as_secs_f32());

    eprintln!("uploading to CUDA (dtype={dtype:?}) ...");
    let t1 = Instant::now();
    let cuda = CudaTransformer::from_cpu_dtype(&model, 0, dtype).expect("cuda");
    eprintln!(
        "upload {:.2}s  GPU weights={:.2} MiB  VRAM free/total={:?}/{:?}",
        t1.elapsed().as_secs_f32(),
        cuda.weight_bytes() as f64 / (1024.0 * 1024.0),
        cuda.device_mem_info().0,
        cuda.device_mem_info().1,
    );

    let tok = ForgeTokenizer::from_dir(&dir, 151643, 151645).expect("tok");
    let params = GenerateParams {
        max_tokens,
        temperature: 0.0,
        top_k: 0,
        eos_token_id: Some(tok.eos_token_id),
    };
    let t2 = Instant::now();
    let out = generate_cuda(&cuda, &tok, &prompt, &params, 512).expect("gen");
    eprintln!("generate {:.2}s", t2.elapsed().as_secs_f32());
    eprintln!("tokens={:?}", out.tokens);
    println!("{}", out.text);
}
