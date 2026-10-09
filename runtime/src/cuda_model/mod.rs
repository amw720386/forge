//! Full-model CUDA inference (Phase 8B–10). Compiled with `--features cuda`.

mod batcher;
pub mod kv;
pub mod paged_kv;
mod generate;
mod transformer;
mod weights;

pub use batcher::CudaBatcher;
pub use generate::{generate_cuda, GenerateCudaOutput};
pub use paged_kv::{
    BlockAllocator, GpuKvBlockPool, PagedKvError, PagedKvSession, PagedKvStats, DEFAULT_BLOCK_SIZE,
};
pub use transformer::{
    CudaBatchScratch, CudaBatchStats, CudaLoadError, CudaScratch, CudaTransformer,
};
pub use weights::GpuWeightDType;
