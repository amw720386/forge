//! On-disk constants and POD structures for `.forge` v1 (little-endian).

pub const MAGIC: [u8; 4] = *b"FRG1";
pub const FORMAT_VERSION: u32 = 1;
pub const HEADER_BYTES: usize = 128;
pub const DATA_ALIGN: u64 = 64;

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArchId {
    Qwen2 = 1,
    Llama = 2,
}

impl ArchId {
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            1 => Some(Self::Qwen2),
            2 => Some(Self::Llama),
            _ => None,
        }
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DTypeCode {
    F32 = 1,
    I8 = 2,
    I4 = 3,
    /// Separate scale tensor: f32 vector (or matrix flattened).
    F32Scale = 4,
}

impl DTypeCode {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(Self::F32),
            2 => Some(Self::I8),
            3 => Some(Self::I4),
            4 => Some(Self::F32Scale),
            _ => None,
        }
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantScheme {
    None = 0,
    WeightOnlySymmetric = 1,
}

impl QuantScheme {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::None),
            1 => Some(Self::WeightOnlySymmetric),
            _ => None,
        }
    }
}

/// Fixed 128-byte header.
#[derive(Clone, Debug)]
pub struct ForgeHeader {
    pub magic: [u8; 4],
    pub version: u32,
    pub header_size: u32,
    pub arch_id: u32,
    pub tensor_count: u32,
    pub flags: u32,
    pub meta_offset: u64,
    pub meta_length: u64,
    pub data_offset: u64,
    pub data_length: u64,
    pub meta_sha256: [u8; 32],
    pub data_sha256: [u8; 32],
}

impl ForgeHeader {
    pub fn new_v1(arch: ArchId, tensor_count: u32) -> Self {
        Self {
            magic: MAGIC,
            version: FORMAT_VERSION,
            header_size: HEADER_BYTES as u32,
            arch_id: arch as u32,
            tensor_count,
            flags: 1, // sha256 present
            meta_offset: 0,
            meta_length: 0,
            data_offset: 0,
            data_length: 0,
            meta_sha256: [0; 32],
            data_sha256: [0; 32],
        }
    }

    pub fn write_bytes(&self, out: &mut [u8; HEADER_BYTES]) {
        out.fill(0);
        out[0..4].copy_from_slice(&self.magic);
        out[4..8].copy_from_slice(&self.version.to_le_bytes());
        out[8..12].copy_from_slice(&self.header_size.to_le_bytes());
        out[12..16].copy_from_slice(&self.arch_id.to_le_bytes());
        out[16..20].copy_from_slice(&self.tensor_count.to_le_bytes());
        out[20..24].copy_from_slice(&self.flags.to_le_bytes());
        out[24..32].copy_from_slice(&self.meta_offset.to_le_bytes());
        out[32..40].copy_from_slice(&self.meta_length.to_le_bytes());
        out[40..48].copy_from_slice(&self.data_offset.to_le_bytes());
        out[48..56].copy_from_slice(&self.data_length.to_le_bytes());
        out[56..88].copy_from_slice(&self.meta_sha256);
        out[88..120].copy_from_slice(&self.data_sha256);
        // 120..128 reserved zero
    }

    pub fn read_bytes(buf: &[u8]) -> Result<Self, String> {
        if buf.len() < HEADER_BYTES {
            return Err("header truncated".into());
        }
        let mut magic = [0u8; 4];
        magic.copy_from_slice(&buf[0..4]);
        if magic != MAGIC {
            return Err(format!("bad magic {magic:?}"));
        }
        let version = u32::from_le_bytes(buf[4..8].try_into().unwrap());
        if version != FORMAT_VERSION {
            return Err(format!("unsupported version {version}"));
        }
        let mut meta_sha256 = [0u8; 32];
        let mut data_sha256 = [0u8; 32];
        meta_sha256.copy_from_slice(&buf[56..88]);
        data_sha256.copy_from_slice(&buf[88..120]);
        Ok(Self {
            magic,
            version,
            header_size: u32::from_le_bytes(buf[8..12].try_into().unwrap()),
            arch_id: u32::from_le_bytes(buf[12..16].try_into().unwrap()),
            tensor_count: u32::from_le_bytes(buf[16..20].try_into().unwrap()),
            flags: u32::from_le_bytes(buf[20..24].try_into().unwrap()),
            meta_offset: u64::from_le_bytes(buf[24..32].try_into().unwrap()),
            meta_length: u64::from_le_bytes(buf[32..40].try_into().unwrap()),
            data_offset: u64::from_le_bytes(buf[40..48].try_into().unwrap()),
            data_length: u64::from_le_bytes(buf[48..56].try_into().unwrap()),
            meta_sha256,
            data_sha256,
        })
    }
}

#[derive(Clone, Debug)]
pub struct TensorEntry {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: DTypeCode,
    pub quant: QuantScheme,
    pub group_size: u32,
    pub data_rel_offset: u64,
    pub data_nbytes: u64,
}

/// Model config blob stored in the metadata section (json for inspectability + versioned fields).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ModelMetaJson {
    pub name: String,
    #[serde(default)]
    pub pad_token_id: Option<u32>,
    #[serde(default)]
    pub tokenizer_sha256: Option<String>,
    pub architecture: String,
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
    pub quant_bits: u8,
    pub quant_group_size: u32,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
}
