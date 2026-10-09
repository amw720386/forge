//! Forge inference runtime.

pub mod attention;
pub mod cache;
pub mod chat;
pub mod checkpoint;
pub mod compute;
pub mod ffi;
pub mod mem;
pub mod model;
pub mod sampling;
pub mod scheduling;
pub mod tensor;

#[cfg(any(feature = "cuda", feature = "cuda-stub"))]
pub mod cuda_model;

// Re-export FFI for examples/profilers.
pub use ffi::kernels as kernel_ffi;

pub use checkpoint::{load_forge_path, save_forge, LoadOptions, SaveOptions};
pub use compute::{default_backend, BackendKind, ComputeBackend, CpuBackend};
pub use mem::{MemoryBudget, ProcessMemory};
pub use model::{
    ForwardScratch, LayerKvCaches, ModelConfig, QuantBits, Transformer, TransformerWeights,
};
pub use tensor::{Tensor, TensorError};
