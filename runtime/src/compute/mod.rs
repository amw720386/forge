//! Compute backend abstraction (CPU always; CUDA optional via `--features cuda`
//! or compile-checked via `--features cuda-stub`).

mod backend;
mod cpu;

#[cfg(any(feature = "cuda", feature = "cuda-stub"))]
mod cuda;

pub use backend::{BackendKind, Buffer, ComputeBackend, ComputeError, DeviceInfo};
pub use cpu::CpuBackend;

#[cfg(any(feature = "cuda", feature = "cuda-stub"))]
pub use cuda::CudaBackend;

/// Select the preferred backend. Real CUDA only when the `cuda` feature is
/// enabled **and** a device is visible; stub builds always fall back to CPU.
pub fn default_backend() -> Box<dyn ComputeBackend> {
    #[cfg(feature = "cuda")]
    {
        if let Ok(cuda) = CudaBackend::try_new(0) {
            return Box::new(cuda);
        }
    }
    Box::new(CpuBackend::new())
}

#[cfg(any(feature = "cuda", feature = "cuda-stub"))]
pub(crate) use cuda::{DecodeGraph, DeviceTimer};
