//! Full-model CUDA inference. Compiled with `--features cuda`.

mod batcher;
mod generate;
pub mod kv;
pub mod paged_kv;
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
