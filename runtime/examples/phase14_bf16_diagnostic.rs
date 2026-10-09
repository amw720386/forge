#[cfg(feature = "cuda")]
use forge_runtime::cuda_model::paged_kv::{GpuKvBlockPool, PagedKvSession};
#[cfg(feature = "cuda")]
use forge_runtime::cuda_model::{CudaBatchScratch, CudaScratch, CudaTransformer, GpuWeightDType};
#[cfg(feature = "cuda")]
use forge_runtime::model::load_qwen2_dir;
#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("This diagnostic requires --features cuda and a real GPU.");
    std::process::exit(1);
}
#[cfg(feature = "cuda")]
fn main() {
    let path = std::env::args().nth(1).expect("model path");
    let cpu = load_qwen2_dir(path).unwrap();
    let cfg = cpu.cfg.clone();
    let model = CudaTransformer::from_cpu_dtype(&cpu, 0, GpuWeightDType::Bf16).unwrap();
    let backend = model.backend();
    std::env::set_var("FORGE_REUSE_BF16", "1");
    for b in [1, 8] {
        let mut pools: Vec<_> = (0..5)
            .map(|_| {
                GpuKvBlockPool::new(
                    backend,
                    cfg.n_layers,
                    cfg.n_kv_heads,
                    cfg.head_dim(),
                    512,
                    b,
                )
                .unwrap()
            })
            .collect();
        let mut sessions: Vec<Vec<_>> = (0..5)
            .map(|_| {
                (0..b)
                    .map(|_| PagedKvSession::new(backend, 512, pools[0].block_size).unwrap())
                    .collect()
            })
            .collect();
        let mut scratch = CudaScratch::new(backend, &cfg, 256).unwrap();
        let mut out = CudaBatchScratch::new(backend, &cfg, b).unwrap();
        let mut errors = [0f32; 4];
        let mut disagreements = [0usize; 4];
        for round in 0..8 {
            let chunks: Vec<Vec<u32>> = (0..b)
                .map(|row| {
                    (0..32)
                        .map(|j| ((row * 101 + round * 32 + j + 1000) % cfg.vocab_size) as u32)
                        .collect()
                })
                .collect();
            std::env::set_var("FORGE_GEMM_LT", "0");
            std::env::set_var("FORGE_PREFILL_ATTN_BLOCK", "0");
            let mut reference = Vec::new();
            for row in 0..b {
                let pos = sessions[0][row].len;
                model
                    .forward_tokens_paged(
                        &chunks[row],
                        pos,
                        &mut pools[0],
                        &mut sessions[0][row],
                        &mut scratch,
                    )
                    .unwrap();
                reference.extend(model.read_logits(&scratch, cfg.vocab_size).unwrap());
            }
            for mode in 0..4 {
                std::env::set_var("FORGE_GEMM_LT", if mode >= 2 { "1" } else { "0" });
                std::env::set_var(
                    "FORGE_PREFILL_ATTN_BLOCK",
                    if mode % 2 == 1 { "1" } else { "0" },
                );
                let mut refs: Vec<_> = sessions[mode + 1].iter_mut().collect();
                model
                    .forward_prefill_paged(
                        &chunks,
                        &vec![true; b],
                        &mut pools[mode + 1],
                        &mut refs,
                        &mut scratch,
                        &mut out,
                    )
                    .unwrap();
                let got = model.read_batch_logits(&out, b, cfg.vocab_size).unwrap();
                for (x, y) in reference.iter().zip(&got) {
                    errors[mode] = errors[mode].max((x - y).abs());
                }
                for row in 0..b {
                    let range = row * cfg.vocab_size..(row + 1) * cfg.vocab_size;
                    disagreements[mode] += usize::from(
                        forge_runtime::sampling::greedy(&reference[range.clone()])
                            != forge_runtime::sampling::greedy(&got[range]),
                    );
                }
            }
        }
        println!("{{\"batch\":{},\"modes\":[\"packed\",\"packed_attention\",\"packed_lt\",\"combined\"],\"max_abs_errors\":{:?},\"greedy_disagreements\":{:?}}}",b,errors,disagreements);
    }
}
