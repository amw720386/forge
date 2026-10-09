//! Full-model CUDA benchmarks (requires `--features cuda` + NVIDIA GPU).
//!
//! ```bash
//! cargo run --release --features cuda --example bench_cuda_qwen -- ../models/Qwen2.5-0.5B-Instruct
//! ```
//!
//! Timing method: wall-clock around `forward_tokens` / generate loop, with
//! `backend.synchronize()` after measured regions. Not kernel-event microbenches.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("bench_cuda_qwen requires --features cuda");
    std::process::exit(1);
}

#[cfg(feature = "cuda")]
fn main() {
    use forge_runtime::cuda_model::{generate_cuda, CudaTransformer};
    use forge_runtime::model::{
        generate, load_qwen2_dir, ForgeTokenizer, ForwardScratch, GenerateParams, LayerKvCaches,
    };
    use forge_runtime::sampling::greedy;
    use forge_runtime::ComputeBackend;
    use std::env;
    use std::time::Instant;

    let args: Vec<String> = env::args().collect();
    let dir = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "../models/Qwen2.5-0.5B-Instruct".into());
    let max_new: usize = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    let max_seq = 512;

    let prompt = env::var("FORGE_PROMPT").unwrap_or_else(|_| {
        "<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n".into()
    });

    eprintln!("=== CPU FP32 baseline load ===");
    let t0 = Instant::now();
    let model = load_qwen2_dir(&dir).expect("load");
    eprintln!(
        "CPU load {:.2}s  weights={:.2} MiB",
        t0.elapsed().as_secs_f32(),
        model.weight_nbytes() as f64 / (1024.0 * 1024.0)
    );

    let tok = ForgeTokenizer::from_dir(&dir, 151643, 151645).expect("tok");
    let params = GenerateParams {
        max_tokens: max_new,
        temperature: 0.0,
        top_k: 0,
        eos_token_id: Some(tok.eos_token_id),
    };

    eprintln!("=== CUDA upload ===");
    let t_up = Instant::now();
    let cuda = CudaTransformer::from_cpu(&model, 0).expect("cuda upload");
    cuda.backend().synchronize().ok();
    let upload_s = t_up.elapsed().as_secs_f32();
    let (free_b, total_b) = cuda.device_mem_info();
    eprintln!(
        "GPU upload {:.2}s  weight_bytes={:.2} MiB  VRAM free/total={:?}/{:?} MiB",
        upload_s,
        cuda.weight_bytes() as f64 / (1024.0 * 1024.0),
        free_b.map(|b| b as f64 / (1024.0 * 1024.0)),
        total_b.map(|b| b as f64 / (1024.0 * 1024.0)),
    );

    let ids = tok.encode(&prompt, true).expect("encode");
    eprintln!("prompt_tokens={}", ids.len());

    // Prefill timing
    {
        use forge_runtime::cuda_model::kv::LayerGpuKvCaches;
        use forge_runtime::cuda_model::CudaScratch;
        let backend = cuda.backend();
        let mut caches = LayerGpuKvCaches::new(
            backend,
            cuda.cfg.n_layers,
            cuda.cfg.n_kv_heads,
            cuda.cfg.head_dim(),
            max_seq,
        )
        .unwrap();
        let mut scratch = CudaScratch::new(backend, &cuda.cfg, max_seq).unwrap();
        let t_pf = Instant::now();
        cuda.forward_tokens(&ids, 0, &mut caches, &mut scratch)
            .unwrap();
        backend.synchronize().unwrap();
        let pf = t_pf.elapsed().as_secs_f32();
        eprintln!(
            "GPU prefill {:.3}s  ({:.1} tok/s)",
            pf,
            ids.len() as f32 / pf
        );
    }

    eprintln!("=== GPU greedy generate (max_new={max_new}) ===");
    let t_gen = Instant::now();
    let out = generate_cuda(&cuda, &tok, &prompt, &params, max_seq).expect("gpu gen");
    cuda.backend().synchronize().ok();
    let gen_s = t_gen.elapsed().as_secs_f32();
    let n = out.tokens.len().max(1);
    eprintln!(
        "GPU generate {:.3}s  new_tokens={}  ~{:.1} decode tok/s (includes prefill+sampling)",
        gen_s,
        out.tokens.len(),
        n as f32 / gen_s
    );
    eprintln!("GPU text: {:?}", out.text);

    eprintln!("=== CPU greedy generate (same prompt) ===");
    let t_cpu = Instant::now();
    let cpu_out = generate(&model, &tok, &prompt, &params, max_seq).expect("cpu gen");
    let cpu_s = t_cpu.elapsed().as_secs_f32();
    eprintln!(
        "CPU generate {:.3}s  new_tokens={}  ~{:.1} tok/s",
        cpu_s,
        cpu_out.tokens.len(),
        cpu_out.tokens.len().max(1) as f32 / cpu_s
    );
    eprintln!("CPU text: {:?}", cpu_out.text);

    if out.tokens == cpu_out.tokens {
        eprintln!("TOKEN MATCH: greedy CUDA == CPU");
    } else {
        eprintln!(
            "TOKEN DIFF: cuda={:?} cpu={:?} (first mismatch check logits separately)",
            out.tokens, cpu_out.tokens
        );
        // Logit max abs error on prefill last token
        let mut cc = LayerKvCaches::new(&model.cfg, max_seq).unwrap();
        let mut cs = ForwardScratch::new(&model.cfg, max_seq).unwrap();
        model.forward_tokens(&ids, 0, &mut cc, &mut cs).unwrap();
        let cpu_logits = cs.logits.data().to_vec();
        let gpu_logits = {
            use forge_runtime::cuda_model::kv::LayerGpuKvCaches;
            use forge_runtime::cuda_model::CudaScratch;
            let backend = cuda.backend();
            let mut caches = LayerGpuKvCaches::new(
                backend,
                cuda.cfg.n_layers,
                cuda.cfg.n_kv_heads,
                cuda.cfg.head_dim(),
                max_seq,
            )
            .unwrap();
            let mut scratch = CudaScratch::new(backend, &cuda.cfg, max_seq).unwrap();
            cuda.forward_tokens(&ids, 0, &mut caches, &mut scratch)
                .unwrap();
            backend.synchronize().unwrap();
            cuda.read_logits(&scratch, model.cfg.vocab_size).unwrap()
        };
        let mut max_abs = 0.0f32;
        for (a, b) in cpu_logits.iter().zip(gpu_logits.iter()) {
            max_abs = max_abs.max((a - b).abs());
        }
        eprintln!(
            "prefill last-token logits max_abs_err={max_abs:.6e}  greedy_cpu={} greedy_gpu={}",
            greedy(&cpu_logits),
            greedy(&gpu_logits)
        );
    }
}
