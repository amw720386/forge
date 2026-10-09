//! Autoregressive generation.

use super::tokenizer::ForgeTokenizer;
use super::transformer::{ForwardScratch, LayerKvCaches, Transformer};
use crate::sampling::{greedy, sample_temperature, sample_top_k};
use crate::tensor::TensorError;

#[derive(Clone, Debug)]
pub struct GenerateParams {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    /// Stop when this id is produced (typically EOS).
    pub eos_token_id: Option<u32>,
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self {
            max_tokens: 64,
            temperature: 0.0,
            top_k: 0,
            eos_token_id: None,
        }
    }
}

#[derive(Debug)]
pub struct GenerateOutput {
    pub tokens: Vec<u32>,
    pub text: String,
}

pub fn generate(
    model: &Transformer,
    tok: &ForgeTokenizer,
    prompt: &str,
    params: &GenerateParams,
    max_seq: usize,
) -> Result<GenerateOutput, GenError> {
    // Chat-style prompts: caller supplies fully formatted text.
    let mut tokens = tok.encode(prompt, true)?;
    if tokens.len() >= max_seq {
        return Err(GenError::PromptTooLong);
    }

    let mut caches = LayerKvCaches::new(&model.cfg, max_seq)?;
    let mut scratch = ForwardScratch::new(&model.cfg, max_seq)?;

    // Prefill
    model.forward_tokens(&tokens, 0, &mut caches, &mut scratch)?;
    let vocab = model.cfg.vocab_size;
    let mut last_logits = scratch.logits.data()[..vocab].to_vec();

    let mut generated = Vec::new();
    let mut rng_state = 0xDEADBEEFu64;

    for _ in 0..params.max_tokens {
        let next = if params.temperature <= 0.0 {
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
        if params.eos_token_id == Some(next) {
            break;
        }
        if tokens.len() >= max_seq {
            break;
        }

        let pos = tokens.len() - 1;
        model.forward_tokens(&[next], pos, &mut caches, &mut scratch)?;
        last_logits = scratch.logits.data()[..vocab].to_vec();
    }

    let text = tok.decode(&generated, true)?;
    Ok(GenerateOutput {
        tokens: generated,
        text,
    })
}

#[derive(Debug)]
pub enum GenError {
    Tok(super::tokenizer::TokError),
    Tensor(TensorError),
    PromptTooLong,
}

impl std::fmt::Display for GenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GenError::Tok(e) => write!(f, "{e}"),
            GenError::Tensor(e) => write!(f, "{e}"),
            GenError::PromptTooLong => write!(f, "prompt exceeds max_seq"),
        }
    }
}

impl std::error::Error for GenError {}
impl From<super::tokenizer::TokError> for GenError {
    fn from(e: super::tokenizer::TokError) -> Self {
        GenError::Tok(e)
    }
}
impl From<TensorError> for GenError {
    fn from(e: TensorError) -> Self {
        GenError::Tensor(e)
    }
}

fn rng_f32(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 32) as f32) / (u32::MAX as f32)
}
