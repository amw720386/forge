//! Hugging Face `tokenizer.json` wrapper.

use std::path::Path;
use tokenizers::Tokenizer;

/// Reuse the legacy sibling layout while permitting an explicit tokenizer directory.
pub fn tokenizer_dir(path: impl AsRef<Path>, cfg: &super::ModelConfig) -> Result<std::path::PathBuf, TokError> {
    if let Some(dir)=std::env::var_os("FORGE_TOKENIZER_DIR") { return Ok(dir.into()); }
    let path=path.as_ref();
    if path.is_dir() { return Ok(path.to_path_buf()); }
    let parent=path.parent().ok_or_else(||TokError::Load("checkpoint has no tokenizer directory".into()))?;
    let family=match cfg.architecture { super::Architecture::Qwen2=>"Qwen2.5-0.5B-Instruct",super::Architecture::Llama=>"SmolLM2-135M-Instruct" };
    for candidate in [path.with_extension(""),parent.join(family),parent.to_path_buf()] {
        if candidate.join("tokenizer.json").is_file() {return Ok(candidate);}
    }
    Err(TokError::Load("tokenizer.json not found; set FORGE_TOKENIZER_DIR to the checkpoint's tokenizer directory".into()))
}

#[derive(Debug)]
pub struct ForgeTokenizer {
    inner: Tokenizer,
    pub eos_token_id: u32,
    pub bos_token_id: u32,
}

#[derive(Debug)]
pub enum TokError {
    Load(String),
    Encode(String),
    Decode(String),
}

impl std::fmt::Display for TokError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokError::Load(e) | TokError::Encode(e) | TokError::Decode(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for TokError {}

impl ForgeTokenizer {
    pub fn for_model(dir: impl AsRef<Path>, cfg: &super::ModelConfig) -> Result<Self, TokError> {
        if let Some(expected) = &cfg.tokenizer_sha256 {
            use sha2::{Digest, Sha256};
            let bytes = std::fs::read(dir.as_ref().join("tokenizer.json")).map_err(|e| TokError::Load(e.to_string()))?;
            if format!("{:x}", Sha256::digest(bytes)) != *expected { return Err(TokError::Load("tokenizer identity differs from checkpoint".into())); }
        }
        let tok = Self::from_dir(dir, cfg.bos_token_id, cfg.eos_token_id)?;
        if tok.inner.get_vocab_size(true) > cfg.vocab_size
            || tok.inner.id_to_token(cfg.bos_token_id).is_none() || tok.inner.id_to_token(cfg.eos_token_id).is_none()
        { return Err(TokError::Load("tokenizer vocabulary/special tokens incompatible with model".into())); }
        Ok(tok)
    }
    pub fn from_dir(dir: impl AsRef<Path>, bos: u32, eos: u32) -> Result<Self, TokError> {
        let path = dir.as_ref().join("tokenizer.json");
        let inner = Tokenizer::from_file(&path).map_err(|e| TokError::Load(e.to_string()))?;
        Ok(Self {
            inner,
            eos_token_id: eos,
            bos_token_id: bos,
        })
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, TokError> {
        let enc = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| TokError::Encode(e.to_string()))?;
        Ok(enc.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, TokError> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| TokError::Decode(e.to_string()))
    }
}
