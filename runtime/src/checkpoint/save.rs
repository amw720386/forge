//! Serialize a `Transformer` into a `.forge` file (atomic temp → rename).

use super::format::{
    ArchId, DTypeCode, ForgeHeader, ModelMetaJson, QuantScheme, TensorEntry, DATA_ALIGN,
    HEADER_BYTES,
};
use super::CheckpointError;
use crate::model::{Linear, ModelConfig, QuantBits, Transformer, WeightStorage};
use crate::tensor::Tensor;
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Default)]
pub struct SaveOptions {
    pub name: String,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
}

struct Payload {
    entry: TensorEntry,
    bytes: Vec<u8>,
}

fn align_up(x: u64, a: u64) -> u64 {
    (x + a - 1) / a * a
}

fn f32_bytes(t: &Tensor) -> Vec<u8> {
    let mut out = Vec::with_capacity(t.numel() * 4);
    for &v in t.data() {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn push_fp32(payloads: &mut Vec<Payload>, name: &str, t: &Tensor) {
    let shape: Vec<u64> = t.shape().iter().map(|&d| d as u64).collect();
    let bytes = f32_bytes(t);
    payloads.push(Payload {
        entry: TensorEntry {
            name: name.into(),
            shape,
            dtype: DTypeCode::F32,
            quant: QuantScheme::None,
            group_size: 0,
            data_rel_offset: 0,
            data_nbytes: bytes.len() as u64,
        },
        bytes,
    });
}

fn push_linear(payloads: &mut Vec<Payload>, name: &str, lin: &Linear) {
    match &lin.weight {
        WeightStorage::Fp32(t) => push_fp32(payloads, &format!("{name}.weight"), t),
        WeightStorage::Quant(q) => {
            let shape = vec![q.out_features as u64, q.in_features as u64];
            let dtype = match q.bits {
                QuantBits::Int8 => DTypeCode::I8,
                QuantBits::Int4 => DTypeCode::I4,
            };
            payloads.push(Payload {
                entry: TensorEntry {
                    name: format!("{name}.weight"),
                    shape,
                    dtype,
                    quant: QuantScheme::WeightOnlySymmetric,
                    group_size: q.group_size as u32,
                    data_rel_offset: 0,
                    data_nbytes: q.qweight.len() as u64,
                },
                bytes: q.qweight.clone(),
            });
            let mut scale_bytes = Vec::with_capacity(q.scales.len() * 4);
            for &s in &q.scales {
                scale_bytes.extend_from_slice(&s.to_le_bytes());
            }
            payloads.push(Payload {
                entry: TensorEntry {
                    name: format!("{name}.scales"),
                    shape: vec![q.out_features as u64, q.n_groups() as u64],
                    dtype: DTypeCode::F32Scale,
                    quant: QuantScheme::WeightOnlySymmetric,
                    group_size: q.group_size as u32,
                    data_rel_offset: 0,
                    data_nbytes: scale_bytes.len() as u64,
                },
                bytes: scale_bytes,
            });
        }
    }
    if let Some(b) = &lin.bias {
        push_fp32(payloads, &format!("{name}.bias"), b);
    }
}

fn detect_quant(model: &Transformer) -> (u8, u32) {
    for layer in &model.layers {
        if let WeightStorage::Quant(q) = &layer.weights.mlp.gate.weight {
            return (q.bits.bits(), q.group_size as u32);
        }
    }
    if let WeightStorage::Quant(q) = &model.lm_head.weight {
        return (q.bits.bits(), q.group_size as u32);
    }
    (0, 0)
}

fn collect_payloads(model: &Transformer) -> Vec<Payload> {
    let mut payloads = Vec::new();
    push_fp32(&mut payloads, "model.embed_tokens.weight", &model.embed);
    push_fp32(&mut payloads, "model.norm.weight", &model.final_norm);
    for (i, layer) in model.layers.iter().enumerate() {
        let p = format!("model.layers.{i}");
        let w = &layer.weights;
        push_fp32(
            &mut payloads,
            &format!("{p}.input_layernorm.weight"),
            &w.input_layernorm,
        );
        push_fp32(
            &mut payloads,
            &format!("{p}.post_attention_layernorm.weight"),
            &w.post_attention_layernorm,
        );
        push_linear(&mut payloads, &format!("{p}.self_attn.q_proj"), &w.q_proj);
        push_linear(&mut payloads, &format!("{p}.self_attn.k_proj"), &w.k_proj);
        push_linear(&mut payloads, &format!("{p}.self_attn.v_proj"), &w.v_proj);
        push_linear(&mut payloads, &format!("{p}.self_attn.o_proj"), &w.o_proj);
        push_linear(&mut payloads, &format!("{p}.mlp.gate_proj"), &w.mlp.gate);
        push_linear(&mut payloads, &format!("{p}.mlp.up_proj"), &w.mlp.up);
        push_linear(&mut payloads, &format!("{p}.mlp.down_proj"), &w.mlp.down);
    }
    // Always store lm_head separately (even if tied at load, we persist weights once).
    push_linear(&mut payloads, "lm_head", &model.lm_head);
    payloads
}

fn write_tensor_table(entries: &[TensorEntry]) -> Result<Vec<u8>, CheckpointError> {
    let mut buf = Vec::new();
    for e in entries {
        let name = e.name.as_bytes();
        if name.len() > u16::MAX as usize {
            return Err(CheckpointError::Format("tensor name too long".into()));
        }
        if e.shape.len() > 8 {
            return Err(CheckpointError::Format("ndim > 8".into()));
        }
        buf.extend_from_slice(&(name.len() as u16).to_le_bytes());
        buf.extend_from_slice(name);
        buf.push(e.shape.len() as u8);
        for _ in 0..(8 - e.shape.len()) {
            // pad dims to 8 × u64 for fixed scanning? Write only ndim dims.
        }
        for &d in &e.shape {
            buf.extend_from_slice(&d.to_le_bytes());
        }
        buf.push(e.dtype as u8);
        buf.push(e.quant as u8);
        buf.extend_from_slice(&e.group_size.to_le_bytes());
        buf.extend_from_slice(&e.data_rel_offset.to_le_bytes());
        buf.extend_from_slice(&e.data_nbytes.to_le_bytes());
    }
    Ok(buf)
}

fn read_u16(buf: &[u8], off: &mut usize) -> Result<u16, CheckpointError> {
    if *off + 2 > buf.len() {
        return Err(CheckpointError::Format("truncated meta".into()));
    }
    let v = u16::from_le_bytes(buf[*off..*off + 2].try_into().unwrap());
    *off += 2;
    Ok(v)
}

pub(crate) fn parse_tensor_table(
    buf: &[u8],
    count: usize,
) -> Result<Vec<TensorEntry>, CheckpointError> {
    let mut off = 0usize;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let nlen = read_u16(buf, &mut off)? as usize;
        if off + nlen > buf.len() {
            return Err(CheckpointError::Format("name OOB".into()));
        }
        let name = String::from_utf8(buf[off..off + nlen].to_vec())
            .map_err(|e| CheckpointError::Format(e.to_string()))?;
        off += nlen;
        if off >= buf.len() {
            return Err(CheckpointError::Format("ndim OOB".into()));
        }
        let ndim = buf[off] as usize;
        off += 1;
        if ndim == 0 || ndim > 8 || off + ndim * 8 > buf.len() {
            return Err(CheckpointError::Format("bad shape".into()));
        }
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            let d = u64::from_le_bytes(buf[off..off + 8].try_into().unwrap());
            off += 8;
            if d == 0 {
                return Err(CheckpointError::Format("zero dim".into()));
            }
            shape.push(d);
        }
        if off + 1 + 1 + 4 + 8 + 8 > buf.len() {
            return Err(CheckpointError::Format("entry truncated".into()));
        }
        let dtype = DTypeCode::from_u8(buf[off])
            .ok_or_else(|| CheckpointError::Format(format!("bad dtype {}", buf[off])))?;
        off += 1;
        let quant = QuantScheme::from_u8(buf[off])
            .ok_or_else(|| CheckpointError::Format(format!("bad quant {}", buf[off])))?;
        off += 1;
        let group_size = u32::from_le_bytes(buf[off..off + 4].try_into().unwrap());
        off += 4;
        let data_rel_offset = u64::from_le_bytes(buf[off..off + 8].try_into().unwrap());
        off += 8;
        let data_nbytes = u64::from_le_bytes(buf[off..off + 8].try_into().unwrap());
        off += 8;
        out.push(TensorEntry {
            name,
            shape,
            dtype,
            quant,
            group_size,
            data_rel_offset,
            data_nbytes,
        });
    }
    Ok(out)
}

/// Write `model` to `path` via tempfile + atomic rename.
pub fn save_forge(
    model: &Transformer,
    path: impl AsRef<Path>,
    opts: SaveOptions,
) -> Result<(), CheckpointError> {
    let path = path.as_ref();
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let mut payloads = collect_payloads(model);
    let (qbits, qgroup) = detect_quant(model);

    // Assign relative offsets with alignment.
    let mut cursor = 0u64;
    for p in &mut payloads {
        cursor = align_up(cursor, DATA_ALIGN);
        p.entry.data_rel_offset = cursor;
        p.entry.data_nbytes = p.bytes.len() as u64;
        cursor = cursor
            .checked_add(p.bytes.len() as u64)
            .ok_or_else(|| CheckpointError::Format("data overflow".into()))?;
    }
    let data_length = cursor;

    let entries: Vec<TensorEntry> = payloads.iter().map(|p| p.entry.clone()).collect();
    let table = write_tensor_table(&entries)?;

    let cfg: &ModelConfig = &model.cfg;
    let meta = ModelMetaJson {
        name: if opts.name.is_empty() {
            cfg.architecture.name().into()
        } else {
            opts.name.clone()
        },
        architecture: cfg.architecture.name().into(),
        pad_token_id: cfg.pad_token_id,
        tokenizer_sha256: cfg.tokenizer_sha256.clone(),
        hidden_size: cfg.hidden_size,
        intermediate_size: cfg.intermediate_size,
        n_layers: cfg.n_layers,
        n_heads: cfg.n_heads,
        n_kv_heads: cfg.n_kv_heads,
        vocab_size: cfg.vocab_size,
        max_position_embeddings: cfg.max_position_embeddings,
        rms_norm_eps: cfg.rms_norm_eps,
        rope_theta: cfg.rope_theta,
        tie_word_embeddings: cfg.tie_word_embeddings,
        quant_bits: qbits,
        quant_group_size: qgroup,
        bos_token_id: if opts.bos_token_id == 0 {
            cfg.bos_token_id
        } else {
            opts.bos_token_id
        },
        eos_token_id: if opts.eos_token_id == 0 {
            cfg.eos_token_id
        } else {
            opts.eos_token_id
        },
    };
    let meta_json =
        serde_json::to_vec(&meta).map_err(|e| CheckpointError::Format(e.to_string()))?;

    // Metadata section: u32 json_len | json | u32 table_len | table
    let mut meta_blob = Vec::new();
    meta_blob.extend_from_slice(&(meta_json.len() as u32).to_le_bytes());
    meta_blob.extend_from_slice(&meta_json);
    meta_blob.extend_from_slice(&(table.len() as u32).to_le_bytes());
    meta_blob.extend_from_slice(&table);

    let meta_offset = HEADER_BYTES as u64;
    let meta_length = meta_blob.len() as u64;
    let data_offset = align_up(meta_offset + meta_length, DATA_ALIGN);

    let meta_sha = Sha256::digest(&meta_blob);
    let mut data_hasher = Sha256::new();
    for p in &payloads {
        // Hash including padding zeros between tensors for determinism of file contents
        // for the payload bytes themselves (padding hashed as zeros when writing).
        data_hasher.update(&p.bytes);
    }
    let data_sha = data_hasher.finalize();

    let mut header = ForgeHeader::new_v1(
        match cfg.architecture {
            crate::model::Architecture::Qwen2 => ArchId::Qwen2,
            crate::model::Architecture::Llama => ArchId::Llama,
        },
        entries.len() as u32,
    );
    header.meta_offset = meta_offset;
    header.meta_length = meta_length;
    header.data_offset = data_offset;
    header.data_length = data_length;
    header.meta_sha256.copy_from_slice(&meta_sha);
    header.data_sha256.copy_from_slice(&data_sha);

    let tmp: PathBuf = path.with_extension("forge.tmp");
    {
        let mut f = File::create(&tmp)?;
        let mut hdr = [0u8; HEADER_BYTES];
        header.write_bytes(&mut hdr);
        f.write_all(&hdr)?;
        f.write_all(&meta_blob)?;
        let pad = (data_offset - (meta_offset + meta_length)) as usize;
        if pad > 0 {
            f.write_all(&vec![0u8; pad])?;
        }
        let mut at = 0u64;
        for p in &payloads {
            let gap = p.entry.data_rel_offset.saturating_sub(at);
            if gap > 0 {
                f.write_all(&vec![0u8; gap as usize])?;
            }
            f.write_all(&p.bytes)?;
            at = p.entry.data_rel_offset + p.entry.data_nbytes;
        }
        f.sync_all()?;
    }
    fs::rename(&tmp, path)?;
    Ok(())
}
