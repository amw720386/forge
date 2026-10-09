//! Compile/link verification for the CUDA backend surface.
//!
//! - `--features cuda-stub`: always runs (no GPU); proves Rust CUDA modules + FFI link.
//! - `--features cuda`: runs when a device is present; otherwise skips GPU assertions.
//!
//! GPU transformer parity / Qwen inference remain **unverified** until hardware runs
//! `cuda_transformer_parity` and `bench_cuda_qwen` successfully.

#![cfg(any(feature = "cuda", feature = "cuda-stub"))]

use forge_runtime::compute::{BackendKind, ComputeBackend, CudaBackend};

#[test]
fn cuda_backend_type_and_symbols_link() {
    // Forces linkage of forge_cuda_* (real or stub).
    let err = CudaBackend::try_new(0);
    match err {
        Ok(b) => {
            assert_eq!(b.kind(), BackendKind::Cuda);
            let info = b.info();
            assert!(!info.name.is_empty());
            eprintln!(
                "CUDA device available: {} (GPU execution still requires parity tests)",
                info.name
            );
        }
        Err(e) => {
            #[cfg(feature = "cuda-stub")]
            {
                let msg = e.to_string();
                assert!(
                    msg.contains("unavailable")
                        || msg.contains("no CUDA")
                        || msg.contains("stub")
                        || msg.contains("device")
                        || msg.contains("NOT")
                        || msg.contains("built")
                        || msg.to_lowercase().contains("cuda"),
                    "stub should fail device init gracefully: {e}"
                );
                eprintln!("OK: cuda-stub linked; device init unavailable as expected: {e}");
            }
            #[cfg(all(feature = "cuda", not(feature = "cuda-stub")))]
            {
                eprintln!("SKIP GPU assert: --features cuda but no device ({e})");
            }
        }
    }
}

#[test]
fn cuda_model_module_compiles() {
    // Type-level check: CudaTransformer is part of the crate when CUDA features are on.
    let _ = std::any::type_name::<forge_runtime::cuda_model::CudaTransformer>();
    let _ = std::any::type_name::<forge_runtime::cuda_model::CudaScratch>();
    let _ = std::any::type_name::<forge_runtime::cuda_model::CudaBatchScratch>();
    let _ = std::any::type_name::<forge_runtime::cuda_model::CudaBatchStats>();
    let _ = std::any::type_name::<forge_runtime::cuda_model::CudaBatcher>();
}
