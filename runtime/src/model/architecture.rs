//! Family metadata and a static, validated plan; no per-kernel trait dispatch.
use super::{LoadError, ModelConfig};
use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Architecture {
    Qwen2,
    Llama,
}

impl Architecture {
    pub fn name(self) -> &'static str {
        match self {
            Self::Qwen2 => "qwen2",
            Self::Llama => "llama",
        }
    }
    pub fn parse(name: &str) -> Result<Self, LoadError> {
        match name {
            "qwen2" => Ok(Self::Qwen2),
            "llama" => Ok(Self::Llama),
            _ => Err(LoadError::Unsupported(format!(
                "architecture {name:?}; supported: qwen2, llama"
            ))),
        }
    }
    pub fn qkv_bias(self) -> bool {
        self == Self::Qwen2
    }
    /// Both verified families use HF [out,in] projections and split-half RoPE.
    pub fn layer_tensor(self, layer: usize, suffix: &str) -> String {
        format!("model.layers.{layer}.{suffix}")
    }
    pub fn tensor_shapes(self, cfg: &ModelConfig) -> Vec<(String, Vec<usize>)> {
        let h = cfg.hidden_size;
        let kv = cfg.n_kv_heads * cfg.head_dim();
        let mid = cfg.intermediate_size;
        let mut shapes = vec![
            ("model.embed_tokens.weight".into(), vec![cfg.vocab_size, h]),
            ("model.norm.weight".into(), vec![h]),
        ];
        for layer in 0..cfg.n_layers {
            for (suffix, shape) in [
                ("input_layernorm.weight", vec![h]),
                ("post_attention_layernorm.weight", vec![h]),
                ("self_attn.q_proj.weight", vec![h, h]),
                ("self_attn.k_proj.weight", vec![kv, h]),
                ("self_attn.v_proj.weight", vec![kv, h]),
                ("self_attn.o_proj.weight", vec![h, h]),
                ("mlp.gate_proj.weight", vec![mid, h]),
                ("mlp.up_proj.weight", vec![mid, h]),
                ("mlp.down_proj.weight", vec![h, mid]),
            ] {
                shapes.push((self.layer_tensor(layer, suffix), shape));
            }
            if self.qkv_bias() {
                for (name, size) in [("q", h), ("k", kv), ("v", kv)] {
                    shapes.push((
                        self.layer_tensor(layer, &format!("self_attn.{name}_proj.bias")),
                        vec![size],
                    ));
                }
            }
        }
        if !cfg.tie_word_embeddings {
            shapes.push(("lm_head.weight".into(), vec![cfg.vocab_size, h]));
        }
        shapes
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DecoderOp {
    AttentionNorm,
    Qkv,
    SplitHalfRope,
    CausalGqa,
    AttentionOutput,
    Residual,
    MlpNorm,
    SwiGlu,
    MlpOutput,
}
pub const LAYER_OPS: &[DecoderOp] = &[
    DecoderOp::AttentionNorm,
    DecoderOp::Qkv,
    DecoderOp::SplitHalfRope,
    DecoderOp::CausalGqa,
    DecoderOp::AttentionOutput,
    DecoderOp::Residual,
    DecoderOp::MlpNorm,
    DecoderOp::SwiGlu,
    DecoderOp::MlpOutput,
    DecoderOp::Residual,
];

#[derive(Clone, Debug)]
pub struct DecoderPlan {
    pub architecture: Architecture,
    pub layers: usize,
    pub operations: &'static [DecoderOp],
    pub attention_scale: f32,
    pub causal: bool,
    pub full_rope: bool,
    pub fp32_kv: bool,
    pub final_rms_norm: bool,
    pub tied_output: bool,
    pub supported_precisions: &'static [super::ExecutionPrecision],
}
impl DecoderPlan {
    pub fn new(cfg: &ModelConfig) -> Result<Self, crate::tensor::TensorError> {
        cfg.validate()?;
        Ok(Self {
            architecture: cfg.architecture,
            layers: cfg.n_layers,
            operations: LAYER_OPS,
            attention_scale: 1.0 / (cfg.head_dim() as f32).sqrt(),
            causal: true,
            full_rope: true,
            fp32_kv: true,
            final_rms_norm: true,
            tied_output: cfg.tie_word_embeddings,
            supported_precisions: &[
                super::ExecutionPrecision::CpuFp32,
                super::ExecutionPrecision::CpuInt8,
                super::ExecutionPrecision::CpuInt4,
                super::ExecutionPrecision::CudaFp32,
                super::ExecutionPrecision::CudaBf16,
            ],
        })
    }
}

pub fn parse_hf_config(text: &str) -> Result<ModelConfig, LoadError> {
    let v: Value = serde_json::from_str(text).map_err(|e| LoadError::Json(e.to_string()))?;
    let architecture = Architecture::parse(
        v.get("model_type")
            .and_then(Value::as_str)
            .ok_or_else(|| LoadError::Json("missing model_type".into()))?,
    )?;
    let unsupported = |s: &str| LoadError::Unsupported(s.to_string());
    if let Some(names) = v.get("architectures") {
        let expected = if architecture == Architecture::Qwen2 {
            "Qwen2ForCausalLM"
        } else {
            "LlamaForCausalLM"
        };
        if names
            .as_array()
            .is_none_or(|a| a.len() != 1 || a[0].as_str() != Some(expected))
        {
            return Err(unsupported("architectures incompatible with model_type"));
        }
    }
    if v.get("hidden_act").and_then(Value::as_str) != Some("silu") {
        return Err(unsupported("only SiLU/SwiGLU is supported"));
    }
    for key in [
        "rope_scaling",
        "sliding_window",
        "attention_chunk_size",
        "quantization_config",
    ] {
        if v.get(key).is_some_and(|x| !x.is_null())
            && !(key == "sliding_window"
                && v.get("use_sliding_window") == Some(&Value::Bool(false)))
        {
            return Err(unsupported(&format!("{key} is not supported")));
        }
    }
    for key in ["use_sliding_window", "rope_interleaved", "mlp_bias"] {
        if v.get(key)
            .is_some_and(|x| *x != Value::Bool(false) && !x.is_null())
        {
            return Err(unsupported(key));
        }
    }
    if v.get("attention_bias")
        .is_some_and(|x| x.as_bool() != Some(architecture.qkv_bias()))
    {
        return Err(unsupported("attention_bias incompatible with family"));
    }
    if v.get("pretraining_tp")
        .is_some_and(|x| x.as_u64() != Some(1))
    {
        return Err(unsupported("pretraining_tp must be 1"));
    }
    let integer = |key: &str| -> Result<usize, LoadError> {
        v.get(key)
            .and_then(Value::as_u64)
            .and_then(|x| usize::try_from(x).ok())
            .ok_or_else(|| LoadError::Json(format!("missing/invalid {key}")))
    };
    let float = |key: &str| -> Result<f32, LoadError> {
        v.get(key)
            .and_then(Value::as_f64)
            .map(|x| x as f32)
            .ok_or_else(|| LoadError::Json(format!("missing/invalid {key}")))
    };
    let token = |key: &str| -> Result<u32, LoadError> {
        u32::try_from(integer(key)?).map_err(|_| LoadError::Json(format!("invalid {key}")))
    };
    let cfg = ModelConfig {
        architecture,
        bos_token_id: token("bos_token_id")?,
        eos_token_id: token("eos_token_id")?,
        pad_token_id: if v.get("pad_token_id").is_some_and(|x| !x.is_null()) {
            Some(token("pad_token_id")?)
        } else {
            None
        },
        tokenizer_sha256: None,
        hidden_size: integer("hidden_size")?,
        intermediate_size: integer("intermediate_size")?,
        n_layers: integer("num_hidden_layers")?,
        n_heads: integer("num_attention_heads")?,
        n_kv_heads: integer("num_key_value_heads")?,
        vocab_size: integer("vocab_size")?,
        max_position_embeddings: integer("max_position_embeddings")?,
        rms_norm_eps: float("rms_norm_eps")?,
        rope_theta: float("rope_theta")?,
        tie_word_embeddings: v
            .get("tie_word_embeddings")
            .and_then(Value::as_bool)
            .ok_or_else(|| LoadError::Json("missing tie_word_embeddings".into()))?,
    };
    if v.get("head_dim").is_some_and(|x| {
        x.as_u64() != Some(cfg.hidden_size.checked_div(cfg.n_heads).unwrap_or(0) as u64)
    }) {
        return Err(unsupported(
            "head_dim must equal hidden_size / num_attention_heads",
        ));
    }
    if v.get("partial_rotary_factor")
        .is_some_and(|x| x.as_f64() != Some(1.0))
    {
        return Err(unsupported("partial RoPE is not supported"));
    }
    cfg.validate()?;
    Ok(cfg)
}
