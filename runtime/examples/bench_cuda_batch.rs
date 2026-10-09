//! Phase 9/10: CUDA continuous-batching throughput benches (B=1,2,4,8).
//!
//! Phase 10 uses paged KV + chunked prefill. TTFT is time from admit until the
//! first emitted token (includes prefill chunks).
//!
//! ```bash
//! cargo run --release --features cuda --example bench_cuda_batch -- \
//!   ../models/Qwen2.5-0.5B-Instruct 16
//! ```
//!
//! Records aggregate tok/s, per-request tok/s, TTFT, KV block stats, occupancy,
//! and GEMM shapes from `CudaBatchStats`. Wall-clock with `synchronize()`.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("bench_cuda_batch requires --features cuda");
    std::process::exit(1);
}

#[cfg(feature = "cuda")]
fn main() {
    use forge_runtime::cuda_model::{CudaBatcher, CudaTransformer, GpuWeightDType};
    use forge_runtime::model::{load_qwen2_dir, ForgeTokenizer, GenerateParams};
    use forge_runtime::ComputeBackend;
    use std::collections::HashMap;
    use std::env;
    use std::time::Instant;

    let args: Vec<String> = env::args().collect();
    let dir = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "../models/Qwen2.5-0.5B-Instruct".into());
    let max_new: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(16);
    let max_seq = 512;

    let prompt = env::var("FORGE_PROMPT").unwrap_or_else(|_| {
        "<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n".into()
    });
    let mix = env::var("FORGE_MIXED_PROMPTS").unwrap_or_default() == "1";
    let dtype = match env::var("FORGE_CUDA_DTYPE")
        .unwrap_or_else(|_| "fp32".into())
        .to_ascii_lowercase()
        .as_str()
    {
        "bf16" | "bfloat16" => GpuWeightDType::Bf16,
        _ => GpuWeightDType::Fp32,
    };

    eprintln!("=== load CPU FP32 + CUDA upload (dtype={dtype:?}) ===");
    let model = load_qwen2_dir(&dir).expect("load");
    let cuda0 = CudaTransformer::from_cpu_dtype(&model, 0, dtype).expect("cuda");
    cuda0.backend().synchronize().ok();
    let (free_b, total_b) = cuda0.device_mem_info();
    eprintln!(
        "weights={:.1} MiB  VRAM free/total={:?}/{:?}",
        cuda0.weight_bytes() as f64 / (1024.0 * 1024.0),
        free_b.map(|b| b as f64 / (1024.0 * 1024.0)),
        total_b.map(|b| b as f64 / (1024.0 * 1024.0)),
    );
    drop(cuda0);

    let tok0 = ForgeTokenizer::from_dir(&dir, 151643, 151645).expect("tok");
    let params = GenerateParams {
        max_tokens: max_new,
        temperature: 0.0,
        top_k: 0,
        eos_token_id: Some(tok0.eos_token_id),
    };

    let prompts: Vec<String> = if mix {
        vec![
            prompt.clone(),
            format!(
                "<|im_start|>user\nShort.<|im_end|>\n<|im_start|>assistant\n"
            ),
            format!(
                "<|im_start|>user\nExplain continuous batching in one sentence.<|im_end|>\n<|im_start|>assistant\n"
            ),
            prompt.clone(),
            prompt.clone(),
            format!("<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"),
            prompt.clone(),
            format!(
                "<|im_start|>user\nList three primes.<|im_end|>\n<|im_start|>assistant\n"
            ),
        ]
    } else {
        vec![prompt; 8]
    };

    // Sequential GPU baseline (one request at a time via batcher max_batch=1).
    eprintln!("\n=== sequential baseline (configured B=1, N=4 requests) ===");
    {
        let tok_s = ForgeTokenizer::from_dir(&dir, 151643, 151645).unwrap();
        let mut batcher =
            CudaBatcher::new(
                CudaTransformer::from_cpu_dtype(&model, 0, dtype).unwrap(),
                tok_s,
                max_seq,
                1,
            )
                .unwrap();
        let t0 = Instant::now();
        let mut total_tok = 0usize;
        for i in 0..4 {
            let p = &prompts[i % prompts.len()];
            let id = batcher.admit(p, params.clone()).unwrap();
            while batcher.active_count() > 0 {
                for ev in batcher.step().unwrap() {
                    assert_eq!(ev.request_id, id);
                    total_tok += 1;
                }
            }
        }
        batcher.synchronize().ok();
        let s = t0.elapsed().as_secs_f64();
        eprintln!(
            "sequential: {} tokens in {:.3}s → {:.2} aggregate tok/s  ({:.2} per-req tok/s avg)",
            total_tok,
            s,
            total_tok as f64 / s,
            (total_tok as f64 / 4.0) / s
        );
    }

    for &b in &[1usize, 2, 4, 8] {
        eprintln!("\n=== continuous batch configured_B={b} ===");
        let tok_b = ForgeTokenizer::from_dir(&dir, 151643, 151645).unwrap();
        let cuda_b = CudaTransformer::from_cpu_dtype(&model, 0, dtype).unwrap();
        let mut batcher = CudaBatcher::new(cuda_b, tok_b, max_seq, b).unwrap();
        assert_eq!(batcher.max_batch(), b);

        let n_req = b;
        let mut ids = Vec::new();
        let kv0 = batcher.kv_stats();
        eprintln!(
            "  kv_pool: blocks={} size={} free={} used={} util={:.1}%",
            kv0.num_blocks,
            kv0.block_size,
            kv0.free_blocks,
            kv0.allocated_blocks,
            kv0.utilization * 100.0
        );
        let t_admit = Instant::now();
        for i in 0..n_req {
            let id = batcher
                .admit(&prompts[i % prompts.len()], params.clone())
                .expect("admit");
            ids.push(id);
        }
        let admit_s = t_admit.elapsed().as_secs_f64();
        eprintln!(
            "  admit {} reqs (chunked prefill queued) {:.3}s  active={} prefill={} decode={} free_slots={}",
            n_req,
            admit_s,
            batcher.active_count(),
            batcher.active_prefill(),
            batcher.active_decode(),
            batcher.free_slots()
        );

        let mut per_req: HashMap<u64, usize> = ids.iter().map(|&id| (id, 0)).collect();
        let mut occupancy_sum = 0usize;
        let mut occupancy_n = 0usize;
        let mut first_token_s: Option<f64> = None;
        let mut sched_prefill = 0usize;
        let mut sched_decode = 0usize;
        let t0 = Instant::now();
        while batcher.active_count() > 0 {
            let evs = batcher.step().unwrap();
            sched_prefill += batcher.stats().last_scheduled_prefill;
            sched_decode += batcher.stats().last_scheduled_decode;
            if first_token_s.is_none() && !evs.is_empty() {
                first_token_s = Some(t0.elapsed().as_secs_f64());
            }
            if batcher.stats().last_batch_size > 0 {
                occupancy_sum += batcher.stats().last_batch_size;
                occupancy_n += 1;
            }
            for ev in evs {
                *per_req.get_mut(&ev.request_id).unwrap() += 1;
            }
        }
        batcher.synchronize().ok();
        let wall = t0.elapsed().as_secs_f64();
        let total: usize = per_req.values().sum();
        let avg_occ = if occupancy_n > 0 {
            occupancy_sum as f64 / occupancy_n as f64
        } else {
            0.0
        };
        let st = batcher.stats();
        let kv1 = batcher.kv_stats();
        eprintln!(
            "  tokens={} wall={:.3}s aggregate={:.2} tok/s  per-req_avg={:.2} tok/s",
            total,
            wall,
            total as f64 / wall,
            (total as f64 / n_req as f64) / wall
        );
        eprintln!(
            "  TTFT(first_token)={:?}s  avg_active_B={:.2}  scheduled_prefill={} scheduled_decode={}",
            first_token_s, avg_occ, sched_prefill, sched_decode
        );
        eprintln!(
            "  last_q_proj=({},{},{}) last_lm=({},{},{}) prefill_chunks={} stall_ms={:.2}",
            st.last_q_proj_m,
            st.last_q_proj_n,
            st.last_q_proj_k,
            st.last_lm_head_m,
            st.last_lm_head_n,
            st.last_lm_head_k,
            st.prefill_chunks,
            batcher.prefill_stall_ns() as f64 / 1e6
        );
        eprintln!(
            "  kv after: free={} used={} util={:.1}%",
            kv1.free_blocks,
            kv1.allocated_blocks,
            kv1.utilization * 100.0
        );
        let (free2, _) = batcher.device_mem_info();
        eprintln!(
            "  VRAM free after run: {:?} MiB",
            free2.map(|x| x as f64 / (1024.0 * 1024.0))
        );
    }
}
