//! Greedy (and temperature) generation on CUDA.

use super::kv::LayerGpuKvCaches;
use super::transformer::{CudaScratch, CudaTransformer};
use crate::compute::ComputeError;
use crate::model::{ForgeTokenizer, GenError, GenerateParams};
use crate::sampling::{greedy, sample_temperature, sample_top_k};
use std::env;

#[derive(Debug)]
pub struct GenerateCudaOutput {
    pub tokens: Vec<u32>,
    pub text: String,
    pub weight_bytes: usize,
    pub free_bytes_after_load: Option<u64>,
    pub total_bytes: Option<u64>,
}

/// End-to-end GPU generation: tokenize on CPU, prefill+decode on CUDA.
/// Greedy (temperature <= 0) uses GPU argmax unless `FORGE_CPU_GREEDY=1`.
pub fn generate_cuda(
    model: &CudaTransformer,
    tok: &ForgeTokenizer,
    prompt: &str,
    params: &GenerateParams,
    max_seq: usize,
) -> Result<GenerateCudaOutput, GenError> {
    let mut tokens = tok.encode(prompt, true).map_err(GenError::Tok)?;
    if tokens.is_empty() || tokens.len() >= max_seq {
        return Err(GenError::PromptTooLong);
    }

    let backend = model.backend();
    let (free_bytes_after_load, total_bytes) = model.device_mem_info();
    let mut caches = LayerGpuKvCaches::new(
        backend,
        model.cfg.n_layers,
        model.cfg.n_kv_heads,
        model.cfg.head_dim(),
        max_seq,
    )
    .map_err(map_compute)?;
    let mut scratch = CudaScratch::new(backend, &model.cfg, max_seq).map_err(map_compute)?;

    model
        .forward_tokens(&tokens, 0, &mut caches, &mut scratch)
        .map_err(map_compute)?;
    let vocab = model.cfg.vocab_size;
    let use_gpu_greedy = params.temperature <= 0.0 && env::var_os("FORGE_CPU_GREEDY").is_none();
    let mut last_logits = if use_gpu_greedy {
        Vec::new()
    } else {
        model.read_logits(&scratch, vocab).map_err(map_compute)?
    };

    let mut generated = Vec::new();
    let mut rng_state = 0xDEADBEEFu64;

    for _ in 0..params.max_tokens {
        let next = if use_gpu_greedy {
            model
                .greedy_from_scratch(&scratch, vocab)
                .map_err(map_compute)?
        } else if params.temperature <= 0.0 {
            greedy(&last_logits)
        } else if params.top_k > 0 {
            sample_top_k(&last_logits, params.top_k, params.temperature, || {
                rng_f32(&mut rng_state)
            })
        } else {
            sample_temperature(&last_logits, params.temperature, || rng_f32(&mut rng_state))
        };
        generated.push(next);
        tokens.push(next);
        if params.eos_token_id == Some(next) || tokens.len() >= max_seq {
            break;
        }
        let pos = tokens.len() - 1;
        model
            .forward_tokens(&[next], pos, &mut caches, &mut scratch)
            .map_err(map_compute)?;
        if !use_gpu_greedy {
            last_logits = model.read_logits(&scratch, vocab).map_err(map_compute)?;
        }
    }

    let text = tok.decode(&generated, true).map_err(GenError::Tok)?;
    Ok(GenerateCudaOutput {
        tokens: generated,
        text,
        weight_bytes: model.weight_bytes(),
        free_bytes_after_load,
        total_bytes,
    })
}

fn map_compute(e: ComputeError) -> GenError {
    GenError::Tensor(crate::tensor::TensorError::AllocFailed(e.to_string()))
}

fn rng_f32(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    (*state >> 32) as f32 / u32::MAX as f32
}
