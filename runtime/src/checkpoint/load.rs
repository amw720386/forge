//! Load `.forge` checkpoints into a `Transformer` (copy or memory-map backed buffers).

use super::format::{DTypeCode, ModelMetaJson, QuantScheme, HEADER_BYTES};
use super::save::parse_tensor_table;
use super::validate::validate_checkpoint_bytes;
use super::CheckpointError;
use crate::model::{
    DecoderBlock, DecoderWeights, Linear, MlpWeights, ModelConfig, QuantBits, QuantMatrix,
    Transformer, TransformerWeights, WeightStorage,
};
use crate::tensor::Tensor;
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct LoadOptions {
    /// When true, memory-map the file and copy tensor bytes into owned buffers
    /// (kernels still receive stable owned pointers; mmap provides I/O efficiency).
    /// Full zero-copy kernel views are reserved for a future ABI that can pin lifetimes.
    pub memory_map: bool,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            memory_map: std::env::var("FORGE_MMAP")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(true),
        }
    }
}

pub struct LoadedCheckpoint {
    pub model: Transformer,
    pub meta: ModelMetaJson,
    /// Keeps mmap alive for the process lifetime of this load when used.
    _mmap: Option<Arc<Mmap>>,
}

fn f32_from_le(bytes: &[u8]) -> Result<Vec<f32>, CheckpointError> {
    if bytes.len() % 4 != 0 {
        return Err(CheckpointError::Format(
            "f32 bytes not multiple of 4".into(),
        ));
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok(out)
}

fn shape_usize(shape: &[u64]) -> Result<Vec<usize>, CheckpointError> {
    shape
        .iter()
        .map(|&d| usize::try_from(d).map_err(|_| CheckpointError::Format("dim too large".into())))
        .collect()
}

struct TensorStore<'a> {
    map: HashMap<String, (super::format::TensorEntry, &'a [u8])>,
}

impl<'a> TensorStore<'a> {
    fn get(&self, name: &str) -> Result<(&super::format::TensorEntry, &[u8]), CheckpointError> {
        self.map
            .get(name)
            .map(|(e, b)| (e, *b))
            .ok_or_else(|| CheckpointError::Format(format!("missing tensor {name}")))
    }

    fn tensor_f32(&self, name: &str) -> Result<Tensor, CheckpointError> {
        let (e, bytes) = self.get(name)?;
        if e.dtype != DTypeCode::F32 {
            return Err(CheckpointError::Format(format!("{name} not F32")));
        }
        let shape = shape_usize(&e.shape)?;
        let data = f32_from_le(bytes)?;
        Ok(Tensor::from_vec(&shape, data)?)
    }

    fn linear(&self, prefix: &str) -> Result<Linear, CheckpointError> {
        let wname = format!("{prefix}.weight");
        let (e, bytes) = self.get(&wname)?;
        let bias = match self.get(&format!("{prefix}.bias")) {
            Ok((_, bbytes)) => {
                let be = &self.map.get(&format!("{prefix}.bias")).unwrap().0;
                let shape = shape_usize(&be.shape)?;
                Some(Tensor::from_vec(&shape, f32_from_le(bbytes)?)?)
            }
            Err(_) => None,
        };
        match e.dtype {
            DTypeCode::F32 => {
                let shape = shape_usize(&e.shape)?;
                Linear::new(Tensor::from_vec(&shape, f32_from_le(bytes)?)?, bias)
                    .map_err(Into::into)
            }
            DTypeCode::I8 | DTypeCode::I4 => {
                if e.quant != QuantScheme::WeightOnlySymmetric {
                    return Err(CheckpointError::Format(format!(
                        "{wname} missing quant scheme"
                    )));
                }
                if e.shape.len() != 2 {
                    return Err(CheckpointError::Format(format!("{wname} bad rank")));
                }
                let out = e.shape[0] as usize;
                let inn = e.shape[1] as usize;
                let group = e.group_size as usize;
                if group == 0 || inn % group != 0 {
                    return Err(CheckpointError::Format(format!("{wname} bad group_size")));
                }
                let bits = if e.dtype == DTypeCode::I8 {
                    QuantBits::Int8
                } else {
                    QuantBits::Int4
                };
                let sname = format!("{prefix}.scales");
                let (se, sbytes) = self.get(&sname)?;
                if se.dtype != DTypeCode::F32Scale {
                    return Err(CheckpointError::Format(format!("{sname} not scales")));
                }
                let scales = f32_from_le(sbytes)?;
                let n_groups = inn / group;
                if scales.len() != out * n_groups {
                    return Err(CheckpointError::Format(format!("{sname} length mismatch")));
                }
                let expected_q = match bits {
                    QuantBits::Int8 => out * inn,
                    QuantBits::Int4 => out * ((inn + 1) / 2),
                };
                if bytes.len() != expected_q {
                    return Err(CheckpointError::Format(format!(
                        "{wname} packed length mismatch"
                    )));
                }
                let q = QuantMatrix {
                    bits,
                    out_features: out,
                    in_features: inn,
                    group_size: group,
                    qweight: bytes.to_vec(),
                    scales,
                };
                Ok(Linear {
                    weight: WeightStorage::Quant(q),
                    bias,
                })
            }
            DTypeCode::F32Scale => Err(CheckpointError::Format(format!(
                "{wname} unexpected scale dtype"
            ))),
        }
    }
}

fn build_model(
    store: &TensorStore<'_>,
    meta: &ModelMetaJson,
) -> Result<Transformer, CheckpointError> {
    let cfg = ModelConfig {
        architecture: crate::model::Architecture::parse(&meta.architecture)
            .map_err(|e| CheckpointError::Unsupported(e.to_string()))?,
        bos_token_id: meta.bos_token_id,
        eos_token_id: meta.eos_token_id,
        pad_token_id: meta.pad_token_id,
        tokenizer_sha256: meta.tokenizer_sha256.clone(),
        hidden_size: meta.hidden_size,
        intermediate_size: meta.intermediate_size,
        n_layers: meta.n_layers,
        n_heads: meta.n_heads,
        n_kv_heads: meta.n_kv_heads,
        vocab_size: meta.vocab_size,
        max_position_embeddings: meta.max_position_embeddings,
        rms_norm_eps: meta.rms_norm_eps,
        rope_theta: meta.rope_theta,
        tie_word_embeddings: meta.tie_word_embeddings,
    };
    cfg.validate()?;

    let embed = store.tensor_f32("model.embed_tokens.weight")?;
    let final_norm = store.tensor_f32("model.norm.weight")?;
    let mut layers = Vec::with_capacity(cfg.n_layers);
    for i in 0..cfg.n_layers {
        let p = format!("model.layers.{i}");
        layers.push(DecoderWeights {
            input_layernorm: store.tensor_f32(&format!("{p}.input_layernorm.weight"))?,
            post_attention_layernorm: store
                .tensor_f32(&format!("{p}.post_attention_layernorm.weight"))?,
            q_proj: store.linear(&format!("{p}.self_attn.q_proj"))?,
            k_proj: store.linear(&format!("{p}.self_attn.k_proj"))?,
            v_proj: store.linear(&format!("{p}.self_attn.v_proj"))?,
            o_proj: store.linear(&format!("{p}.self_attn.o_proj"))?,
            mlp: MlpWeights {
                gate: store.linear(&format!("{p}.mlp.gate_proj"))?,
                up: store.linear(&format!("{p}.mlp.up_proj"))?,
                down: store.linear(&format!("{p}.mlp.down_proj"))?,
            },
        });
    }

    match store.linear("lm_head") {
        Ok(l) => match &l.weight {
            WeightStorage::Quant(_) => {
                build_model_with_quant_lm_head(cfg, embed, final_norm, layers, l)
            }
            WeightStorage::Fp32(t) => Ok(Transformer::from_weights(
                cfg,
                TransformerWeights {
                    embed,
                    layers,
                    final_norm,
                    lm_head: Some(t.share()),
                },
            )?),
        },
        Err(_) if meta.tie_word_embeddings && !store.map.contains_key("lm_head.weight") => {
            Ok(Transformer::from_weights(
                cfg,
                TransformerWeights {
                    embed,
                    layers,
                    final_norm,
                    lm_head: None,
                },
            )?)
        }
        Err(e) => Err(e),
    }
}

fn build_model_with_quant_lm_head(
    cfg: ModelConfig,
    embed: Tensor,
    final_norm: Tensor,
    layers: Vec<DecoderWeights>,
    lm_head: Linear,
) -> Result<Transformer, CheckpointError> {
    if embed.shape() != [cfg.vocab_size, cfg.hidden_size]
        || final_norm.shape() != [cfg.hidden_size]
        || lm_head.weight.out_features() != cfg.vocab_size
        || lm_head.weight.in_features() != cfg.hidden_size
    {
        return Err(CheckpointError::Format(
            "embedding, norm or quantized output shape mismatch".into(),
        ));
    }
    let mut blocks = Vec::with_capacity(layers.len());
    for lw in layers {
        blocks.push(DecoderBlock::new(cfg.clone(), lw)?);
    }
    Ok(Transformer {
        plan: crate::model::DecoderPlan::new(&cfg)?,
        cfg,
        layers: blocks,
        embed,
        final_norm,
        lm_head,
    })
}

fn parse_store<'a>(bytes: &'a [u8]) -> Result<(ModelMetaJson, TensorStore<'a>), CheckpointError> {
    let header = validate_checkpoint_bytes(bytes)?;
    let meta_off = header.meta_offset as usize;
    let meta_len = header.meta_length as usize;
    let meta = &bytes[meta_off..meta_off + meta_len];
    let json_len = u32::from_le_bytes(meta[0..4].try_into().unwrap()) as usize;
    let json = &meta[4..4 + json_len];
    let model_meta: ModelMetaJson =
        serde_json::from_slice(json).map_err(|e| CheckpointError::Format(e.to_string()))?;
    let expected_arch = match model_meta.architecture.as_str() {
        "qwen2" => 1,
        "llama" => 2,
        _ => 0,
    };
    if expected_arch != header.arch_id {
        return Err(CheckpointError::Format(
            "header/metadata architecture mismatch".into(),
        ));
    }
    let table_len_off = 4 + json_len;
    let table_len =
        u32::from_le_bytes(meta[table_len_off..table_len_off + 4].try_into().unwrap()) as usize;
    let table = &meta[table_len_off + 4..table_len_off + 4 + table_len];
    let entries = parse_tensor_table(table, header.tensor_count as usize)?;
    let data_off = header.data_offset as usize;
    let mut map = HashMap::new();
    for e in entries {
        let abs = data_off
            .checked_add(e.data_rel_offset as usize)
            .ok_or_else(|| CheckpointError::Format("offset overflow".into()))?;
        let end = abs
            .checked_add(e.data_nbytes as usize)
            .ok_or_else(|| CheckpointError::Format("length overflow".into()))?;
        if end > bytes.len() {
            return Err(CheckpointError::Format(format!("tensor {} OOB", e.name)));
        }
        map.insert(e.name.clone(), (e, &bytes[abs..end]));
    }
    Ok((model_meta, TensorStore { map }))
}

pub fn load_forge_path(
    path: impl AsRef<Path>,
    opts: LoadOptions,
) -> Result<LoadedCheckpoint, CheckpointError> {
    let path = path.as_ref();
    if opts.memory_map {
        let file = File::open(path)?;
        // SAFETY: file is kept via Mmap owner; we only read.
        let mmap = unsafe { Mmap::map(&file)? };
        let mmap = Arc::new(mmap);
        let (meta, store) = parse_store(&mmap)?;
        let model = build_model(&store, &meta)?;
        // Copies into owned Tensor/QuantMatrix buffers above so dropping mmap is safe
        // after build — but we retain mmap optionally for future zero-copy.
        Ok(LoadedCheckpoint {
            model,
            meta,
            _mmap: Some(mmap),
        })
    } else {
        let bytes = std::fs::read(path)?;
        if bytes.len() < HEADER_BYTES {
            return Err(CheckpointError::Format("file too small".into()));
        }
        let (meta, store) = parse_store(&bytes)?;
        let model = build_model(&store, &meta)?;
        Ok(LoadedCheckpoint {
            model,
            meta,
            _mmap: None,
        })
    }
}
