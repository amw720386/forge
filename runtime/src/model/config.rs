use crate::attention::AttentionConfig;
use crate::tensor::TensorError;

/// Decoder-only transformer configuration (Qwen2-compatible fields).
#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub architecture: super::Architecture,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub pad_token_id: Option<u32>,
    pub tokenizer_sha256: Option<String>,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub tie_word_embeddings: bool,
}

impl ModelConfig {
    /// Tiny synthetic config for Phase 2 unit tests.
    pub fn tiny() -> Self {
        Self {
            architecture: super::Architecture::Qwen2,
            bos_token_id: 1,
            eos_token_id: 2,
            pad_token_id: None,
            tokenizer_sha256: None,
            hidden_size: 32,
            intermediate_size: 64,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            vocab_size: 128,
            max_position_embeddings: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            tie_word_embeddings: true,
        }
    }

    /// Qwen2.5-0.5B-Instruct.
    pub fn qwen25_0_5b() -> Self {
        Self {
            architecture: super::Architecture::Qwen2,
            bos_token_id: 151643,
            eos_token_id: 151645,
            pad_token_id: Some(151643),
            tokenizer_sha256: None,
            hidden_size: 896,
            intermediate_size: 4864,
            n_layers: 24,
            n_heads: 14,
            n_kv_heads: 2,
            vocab_size: 151_936,
            max_position_embeddings: 32_768,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            tie_word_embeddings: true,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.n_heads
    }
    pub fn execution_shape(&self) -> [usize; 8] {
        [self.hidden_size,self.intermediate_size,self.n_layers,self.n_heads,self.n_kv_heads,self.head_dim(),self.vocab_size,self.max_position_embeddings]
    }

    pub fn attention(&self) -> AttentionConfig {
        AttentionConfig {
            n_heads: self.n_heads,
            n_kv_heads: self.n_kv_heads,
            head_dim: self.head_dim(),
            rope_theta: self.rope_theta,
        }
    }

    pub fn validate(&self) -> Result<(), TensorError> {
        if self.hidden_size == 0
            || self.intermediate_size == 0
            || self.n_layers == 0
            || self.vocab_size == 0
            || self.n_heads == 0
            || self.n_kv_heads == 0
            || self.max_position_embeddings == 0
        {
            return Err(TensorError::ZeroDimension);
        }
        if !self.rms_norm_eps.is_finite() || self.rms_norm_eps <= 0.0
            || !self.rope_theta.is_finite() || self.rope_theta <= 0.0
            || self.bos_token_id as usize >= self.vocab_size
            || self.eos_token_id as usize >= self.vocab_size
            || self.pad_token_id.is_some_and(|id| id as usize >= self.vocab_size)
        {
            return Err(TensorError::Configuration("invalid norm, RoPE or special token configuration".into()));
        }
        if self.hidden_size % self.n_heads != 0 {
            return Err(TensorError::ShapeMismatch {
                expected: vec![self.n_heads],
                got: vec![self.hidden_size],
            });
        }
        for (a,b) in [(self.hidden_size,self.vocab_size),(self.hidden_size,self.intermediate_size),(self.max_position_embeddings,self.hidden_size)] {
            if a.checked_mul(b).and_then(|n|n.checked_mul(4)).is_none() { return Err(TensorError::Configuration("model dimensions overflow address space".into())); }
        }
        self.attention().validate()
    }
}
