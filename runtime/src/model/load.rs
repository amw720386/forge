//! Load Qwen2-style models from Hugging Face `config.json` + safetensors.

use super::decoder::DecoderWeights;
use super::linear::Linear;
use super::mlp::MlpWeights;
use super::parse_hf_config;
use super::transformer::{Transformer, TransformerWeights};
use crate::tensor::{Tensor, TensorError};
use safetensors::tensor::{SafeTensors, TensorView};
use safetensors::Dtype;
use std::fs;
use std::path::Path;

#[derive(Debug)]
pub enum LoadError {
    Io(std::io::Error),
    Json(String),
    Safetensors(String),
    Tensor(TensorError),
    Missing(String),
    Unsupported(String),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Io(e) => write!(f, "io: {e}"),
            LoadError::Json(e) => write!(f, "json: {e}"),
            LoadError::Safetensors(e) => write!(f, "safetensors: {e}"),
            LoadError::Tensor(e) => write!(f, "tensor: {e}"),
            LoadError::Missing(e) => write!(f, "missing tensor: {e}"),
            LoadError::Unsupported(e) => write!(f, "unsupported: {e}"),
        }
    }
}

impl std::error::Error for LoadError {}

impl From<TensorError> for LoadError {
    fn from(e: TensorError) -> Self {
        LoadError::Tensor(e)
    }
}

/// Load either a Hugging Face directory or a `.forge` checkpoint.
pub fn load_model_path(path: impl AsRef<Path>) -> Result<Transformer, LoadError> {
    let path = path.as_ref();
    if path.is_file()
        && path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("forge"))
            .unwrap_or(false)
    {
        let loaded =
            crate::checkpoint::load_forge_path(path, crate::checkpoint::LoadOptions::default())
                .map_err(|e| LoadError::Unsupported(e.to_string()))?;
        return Ok(loaded.model);
    }
    load_hf_dir(path)
}

/// Load a Qwen2 HF directory (`config.json` + `model.safetensors`).
pub fn load_qwen2_dir(dir: impl AsRef<Path>) -> Result<Transformer, LoadError> {
    let cfg = parse_hf_config(
        &fs::read_to_string(dir.as_ref().join("config.json")).map_err(LoadError::Io)?,
    )?;
    if cfg.architecture != super::Architecture::Qwen2 {
        return Err(LoadError::Unsupported("expected qwen2".into()));
    }
    load_hf_dir(dir)
}

/// Architecture-detected HF loading. Both adapters use [out,in] without permutation.
pub fn load_hf_dir(dir: impl AsRef<Path>) -> Result<Transformer, LoadError> {
    let dir = dir.as_ref();
    let cfg_text = fs::read_to_string(dir.join("config.json")).map_err(LoadError::Io)?;
    let mut cfg = parse_hf_config(&cfg_text)?;
    if dir.join("tokenizer.json").is_file() {
        use sha2::{Digest, Sha256};
        cfg.tokenizer_sha256 = Some(format!(
            "{:x}",
            Sha256::digest(fs::read(dir.join("tokenizer.json")).map_err(LoadError::Io)?)
        ));
    }

    let st_path = dir.join("model.safetensors");
    let file = fs::File::open(&st_path).map_err(LoadError::Io)?;
    // Read-only mapping avoids a second complete checkpoint byte allocation.
    let bytes = unsafe { memmap2::Mmap::map(&file) }.map_err(LoadError::Io)?;
    let tensors =
        SafeTensors::deserialize(&bytes).map_err(|e| LoadError::Safetensors(e.to_string()))?;
    for (name, expected) in cfg.architecture.tensor_shapes(&cfg) {
        let view = tensors
            .tensor(&name)
            .map_err(|_| LoadError::Missing(name.clone()))?;
        if view.shape() != expected {
            return Err(LoadError::Unsupported(format!(
                "{name}: expected {expected:?}, got {:?}",
                view.shape()
            )));
        }
        if !matches!(view.dtype(), Dtype::F32 | Dtype::F16 | Dtype::BF16) {
            return Err(LoadError::Unsupported(format!(
                "{name}: dtype {:?}",
                view.dtype()
            )));
        }
    }
    for name in tensors.names() {
        if name.ends_with(".bias")
            && (!cfg.architecture.qkv_bias()
                || !["q_proj.bias", "k_proj.bias", "v_proj.bias"]
                    .iter()
                    .any(|suffix| name.ends_with(suffix)))
        {
            return Err(LoadError::Unsupported(format!("unexpected bias {name}")));
        }
    }

    let embed = load_tensor(&tensors, "model.embed_tokens.weight")?;
    let final_norm = load_tensor(&tensors, "model.norm.weight")?;

    let mut layers = Vec::with_capacity(cfg.n_layers);
    for i in 0..cfg.n_layers {
        let p = cfg.architecture.layer_tensor(i, "");
        let p = p.trim_end_matches('.');
        let q_w = load_tensor(&tensors, &format!("{p}.self_attn.q_proj.weight"))?;
        let q_b = if cfg.architecture.qkv_bias() {
            Some(load_tensor(
                &tensors,
                &format!("{p}.self_attn.q_proj.bias"),
            )?)
        } else {
            if tensors
                .tensor(&format!("{p}.self_attn.q_proj.bias"))
                .is_ok()
            {
                return Err(LoadError::Unsupported(
                    "unexpected Llama attention bias".into(),
                ));
            }
            None
        };
        let k_w = load_tensor(&tensors, &format!("{p}.self_attn.k_proj.weight"))?;
        let k_b = if cfg.architecture.qkv_bias() {
            Some(load_tensor(
                &tensors,
                &format!("{p}.self_attn.k_proj.bias"),
            )?)
        } else {
            if tensors
                .tensor(&format!("{p}.self_attn.k_proj.bias"))
                .is_ok()
            {
                return Err(LoadError::Unsupported(
                    "unexpected Llama attention bias".into(),
                ));
            }
            None
        };
        let v_w = load_tensor(&tensors, &format!("{p}.self_attn.v_proj.weight"))?;
        let v_b = if cfg.architecture.qkv_bias() {
            Some(load_tensor(
                &tensors,
                &format!("{p}.self_attn.v_proj.bias"),
            )?)
        } else {
            if tensors
                .tensor(&format!("{p}.self_attn.v_proj.bias"))
                .is_ok()
            {
                return Err(LoadError::Unsupported(
                    "unexpected Llama attention bias".into(),
                ));
            }
            None
        };
        let o_w = load_tensor(&tensors, &format!("{p}.self_attn.o_proj.weight"))?;
        layers.push(DecoderWeights {
            input_layernorm: load_tensor(&tensors, &format!("{p}.input_layernorm.weight"))?,
            post_attention_layernorm: load_tensor(
                &tensors,
                &format!("{p}.post_attention_layernorm.weight"),
            )?,
            q_proj: Linear::new(q_w, q_b)?,
            k_proj: Linear::new(k_w, k_b)?,
            v_proj: Linear::new(v_w, v_b)?,
            o_proj: Linear::new(o_w, None)?,
            mlp: MlpWeights {
                gate: Linear::new(
                    load_tensor(&tensors, &format!("{p}.mlp.gate_proj.weight"))?,
                    None,
                )?,
                up: Linear::new(
                    load_tensor(&tensors, &format!("{p}.mlp.up_proj.weight"))?,
                    None,
                )?,
                down: Linear::new(
                    load_tensor(&tensors, &format!("{p}.mlp.down_proj.weight"))?,
                    None,
                )?,
            },
        });
    }

    let lm_head = if tensors.names().iter().any(|n| *n == "lm_head.weight") {
        Some(load_tensor(&tensors, "lm_head.weight")?)
    } else {
        None
    };

    Transformer::from_weights(
        cfg,
        TransformerWeights {
            embed,
            layers,
            final_norm,
            lm_head,
        },
    )
    .map_err(LoadError::Tensor)
}

fn load_tensor(st: &SafeTensors<'_>, name: &str) -> Result<Tensor, LoadError> {
    let view = st
        .tensor(name)
        .map_err(|_| LoadError::Missing(name.to_string()))?;
    tensor_from_view(view)
}

fn tensor_from_view(view: TensorView<'_>) -> Result<Tensor, LoadError> {
    let shape: Vec<usize> = view.shape().to_vec();
    let data = match view.dtype() {
        Dtype::F32 => {
            let bytes = view.data();
            let mut out = vec![0.0f32; bytes.len() / 4];
            for (i, chunk) in bytes.chunks_exact(4).enumerate() {
                out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
            out
        }
        Dtype::BF16 => {
            let bytes = view.data();
            let mut out = Vec::with_capacity(bytes.len() / 2);
            for chunk in bytes.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(bf16_to_f32(bits));
            }
            out
        }
        Dtype::F16 => {
            let bytes = view.data();
            let mut out = Vec::with_capacity(bytes.len() / 2);
            for chunk in bytes.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(f16_to_f32(bits));
            }
            out
        }
        other => {
            return Err(LoadError::Unsupported(format!("dtype {other:?}")));
        }
    };
    Tensor::from_vec(&shape, data).map_err(LoadError::Tensor)
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn f16_to_f32(bits: u16) -> f32 {
    // IEEE half → float32
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;
    if exp == 0 {
        if frac == 0 {
            0.0
        } else {
            let mut e = -14i32;
            let mut m = frac;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            f32::from_bits((sign << 31) | (((e + 127) as u32) << 23) | (m << 13))
        }
    } else if exp == 31 {
        f32::from_bits((sign << 31) | 0x7f800000 | (frac << 13))
    } else {
        f32::from_bits((sign << 31) | ((exp + 112) << 23) | (frac << 13))
    }
}
