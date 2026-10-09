//! Structural validation of `.forge` byte ranges without building a model.

use super::format::{ArchId, ForgeHeader, HEADER_BYTES};
use super::save::parse_tensor_table;
use super::CheckpointError;
use sha2::{Digest, Sha256};

pub fn validate_checkpoint_bytes(bytes: &[u8]) -> Result<ForgeHeader, CheckpointError> {
    if bytes.len() < HEADER_BYTES {
        return Err(CheckpointError::Format("file too small".into()));
    }
    let header = ForgeHeader::read_bytes(&bytes[..HEADER_BYTES])
        .map_err(CheckpointError::Format)?;
    if header.header_size as usize != HEADER_BYTES {
        return Err(CheckpointError::Format("unexpected header_size".into()));
    }
    ArchId::from_u32(header.arch_id).ok_or_else(|| {
        CheckpointError::Unsupported(format!("arch_id {}", header.arch_id))
    })?;

    let meta_off = header.meta_offset as usize;
    let meta_len = header.meta_length as usize;
    let data_off = header.data_offset as usize;
    let data_len = header.data_length as usize;

    let end = bytes.len();
    let meta_end = meta_off
        .checked_add(meta_len)
        .ok_or_else(|| CheckpointError::Format("meta overflow".into()))?;
    let data_end = data_off
        .checked_add(data_len)
        .ok_or_else(|| CheckpointError::Format("data overflow".into()))?;
    if meta_off < HEADER_BYTES || meta_end > end {
        return Err(CheckpointError::Format("meta out of bounds".into()));
    }
    if data_off < meta_end || data_end > end {
        return Err(CheckpointError::Format("data out of bounds".into()));
    }
    if data_off < meta_off {
        return Err(CheckpointError::Format("data before meta".into()));
    }

    let meta = &bytes[meta_off..meta_end];
    if (header.flags & 1) != 0 {
        let dig = Sha256::digest(meta);
        if dig.as_slice() != header.meta_sha256 {
            return Err(CheckpointError::Integrity("meta sha256 mismatch".into()));
        }
    }

    if meta.len() < 4 {
        return Err(CheckpointError::Format("meta truncated".into()));
    }
    let json_len = u32::from_le_bytes(meta[0..4].try_into().unwrap()) as usize;
    let json_end = 4usize
        .checked_add(json_len)
        .ok_or_else(|| CheckpointError::Format("json len overflow".into()))?;
    if json_end + 4 > meta.len() {
        return Err(CheckpointError::Format("json OOB".into()));
    }
    let table_len =
        u32::from_le_bytes(meta[json_end..json_end + 4].try_into().unwrap()) as usize;
    let table_off = json_end + 4;
    let table_end = table_off
        .checked_add(table_len)
        .ok_or_else(|| CheckpointError::Format("table overflow".into()))?;
    if table_end > meta.len() {
        return Err(CheckpointError::Format("table OOB".into()));
    }
    let table = &meta[table_off..table_end];
    let entries = parse_tensor_table(table, header.tensor_count as usize)?;

    // Validate tensor ranges: no overflow, within data section, no overlaps.
    let mut spans: Vec<(u64, u64, String)> = Vec::new();
    for e in &entries {
        let start = e.data_rel_offset;
        let len = e.data_nbytes;
        let end = start
            .checked_add(len)
            .ok_or_else(|| CheckpointError::Format(format!("tensor {} size overflow", e.name)))?;
        if end > header.data_length {
            return Err(CheckpointError::Format(format!(
                "tensor {} exceeds data section",
                e.name
            )));
        }
        // Shape product sanity for element counts where applicable.
        let mut prod = 1u64;
        for &d in &e.shape {
            prod = prod
                .checked_mul(d)
                .ok_or_else(|| CheckpointError::Format(format!("{} shape overflow", e.name)))?;
        }
        spans.push((start, end, e.name.clone()));
    }
    spans.sort_by_key(|s| s.0);
    for w in spans.windows(2) {
        if w[0].1 > w[1].0 {
            return Err(CheckpointError::Format(format!(
                "overlapping tensors {} and {}",
                w[0].2, w[1].2
            )));
        }
    }

    if (header.flags & 1) != 0 {
        let mut hasher = Sha256::new();
        for e in &entries {
            let abs = data_off + e.data_rel_offset as usize;
            let len = e.data_nbytes as usize;
            hasher.update(&bytes[abs..abs + len]);
        }
        let dig = hasher.finalize();
        if dig.as_slice() != header.data_sha256 {
            return Err(CheckpointError::Integrity("data sha256 mismatch".into()));
        }
    }

    Ok(header)
}
