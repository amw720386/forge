//! Native `.forge` checkpoint format (version 1).
//!
//! See `docs/CHECKPOINT_FORMAT.md` for the on-disk layout.

mod format;
mod load;
mod save;
mod validate;

pub use format::ModelMetaJson;
pub use format::{ArchId, DTypeCode, ForgeHeader, QuantScheme, TensorEntry, FORMAT_VERSION, MAGIC};
pub use load::{load_forge_path, LoadOptions, LoadedCheckpoint};
pub use save::{save_forge, SaveOptions};
pub use validate::validate_checkpoint_bytes;

use crate::tensor::TensorError;

#[derive(Debug)]
pub enum CheckpointError {
    Io(std::io::Error),
    Format(String),
    Unsupported(String),
    Tensor(TensorError),
    Integrity(String),
}

impl std::fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CheckpointError::Io(e) => write!(f, "io: {e}"),
            CheckpointError::Format(e) => write!(f, "format: {e}"),
            CheckpointError::Unsupported(e) => write!(f, "unsupported: {e}"),
            CheckpointError::Tensor(e) => write!(f, "tensor: {e}"),
            CheckpointError::Integrity(e) => write!(f, "integrity: {e}"),
        }
    }
}

impl std::error::Error for CheckpointError {}
impl From<std::io::Error> for CheckpointError {
    fn from(e: std::io::Error) -> Self {
        CheckpointError::Io(e)
    }
}
impl From<TensorError> for CheckpointError {
    fn from(e: TensorError) -> Self {
        CheckpointError::Tensor(e)
    }
}
