//! Logit parity across block boundaries and request replacement. Run serialized:
//! cargo test --release --features cuda --test cuda_phase13 -- --test-threads=1
#![cfg(feature = "cuda")]
use forge_runtime::cuda_model::{CudaTransformer,CudaScratch,CudaBatchScratch,CudaBatchStats,GpuWeightDType};
use forge_runtime::cuda_model::paged_kv::{GpuKvBlockPool,PagedKvSession};
use forge_runtime::model::{ModelConfig,Transformer};
use forge_runtime::ComputeBackend;

#[test]
fn logits_eager_optimized_graph_and_replacement() {
    // This is an explicitly GPU-required suite; absence of a device fails.
    for dtype in [GpuWeightDType::Fp32,GpuWeightDType::Bf16] {
        let cfg=ModelConfig::tiny();
        let model=CudaTransformer::from_cpu_dtype(&Transformer::synthetic(cfg.clone(),37).unwrap(),0,dtype).unwrap();
        let backend=model.backend();
        for b in [1,2,4,8,9] {
            let mut pools:Vec<_>=(0..2).map(|_|GpuKvBlockPool::new(backend,cfg.n_layers,cfg.n_kv_heads,cfg.head_dim(),128,b).unwrap()).collect();
            let mut sessions:Vec<Vec<_>>=(0..2).map(|_|(0..b).map(|_|PagedKvSession::new(backend,128,pools[0].block_size).unwrap()).collect()).collect();
            let mut scratches:Vec<_>=(0..2).map(|_|CudaBatchScratch::new(backend,&cfg,b).unwrap()).collect();
            let mut prefill=CudaScratch::new(backend,&cfg,128).unwrap();
            for side in 0..2 {
                for row in 0..b {
                    let prompt:Vec<u32>=(0..15+row*3).map(|i|((i+row)%cfg.vocab_size) as u32).collect();
                    model.forward_tokens_paged(&prompt,0,&mut pools[side],&mut sessions[side][row],&mut prefill).unwrap();
                }
            }
            let mut max_error=0f32;
            for step in 0..24 {
                if step==9 || step==15 {
                    // Replace a live request without changing pool or table addresses.
                    for side in 0..2 {
                        sessions[side][0].clear(&mut pools[side]).unwrap();
                        let replacement = if step==9 {vec![8,7,6]} else {vec![8;75]};
                        model.forward_tokens_paged(&replacement,0,&mut pools[side],&mut sessions[side][0],&mut prefill).unwrap();
                    }
                }
                let tokens:Vec<_>=(0..b).map(|row|((step+row+1)%cfg.vocab_size) as u32).collect();
                let mut results=Vec::new();
                for side in 0..2 {
                    std::env::set_var("FORGE_BATCHED_PAGED",if side==0 {"0"} else {"1"});
                    std::env::set_var("FORGE_REUSE_BF16",if side==0 {"0"} else {"1"});
                    std::env::set_var("FORGE_CUDA_GRAPHS",if side==0 {"0"} else {"1"});
                    let positions:Vec<_>=sessions[side].iter().map(|s|s.len).collect();
                    let mut refs:Vec<_>=sessions[side].iter_mut().collect();
                    model.forward_batch_decode_paged(&tokens,&positions,&mut pools[side],&mut refs,&mut scratches[side],&mut CudaBatchStats::default()).unwrap();
                    backend.synchronize().unwrap();
                    results.push(model.read_batch_logits(&scratches[side],b,cfg.vocab_size).unwrap());
                }
                for (x,y) in results[0].iter().zip(&results[1]) {
                    max_error=max_error.max((x-y).abs());
                    assert!((x-y).abs() <= 1e-4 + 1e-3*x.abs().max(y.abs()),"dtype={dtype:?} B={b} step={step}: {x} vs {y}");
                }
            }
            println!("PHASE13_LOGIT_PARITY dtype={dtype:?} batch={b} max_abs_error={max_error}");
        }
    }
    for key in ["FORGE_BATCHED_PAGED","FORGE_REUSE_BF16","FORGE_CUDA_GRAPHS"] {std::env::remove_var(key);}
}
