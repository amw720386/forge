//! CPU compute backend tests (always). CUDA parity is gated and skipped without a GPU.

use forge_runtime::compute::{default_backend, BackendKind, ComputeBackend, CpuBackend};

fn upload_f32(backend: &dyn ComputeBackend, data: &[f32]) -> forge_runtime::compute::Buffer {
    let mut buf = backend.alloc(data.len() * 4).unwrap();
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    backend.copy_h2d(&mut buf, &bytes).unwrap();
    buf
}

fn download_f32(
    backend: &dyn ComputeBackend,
    buf: &forge_runtime::compute::Buffer,
    n: usize,
) -> Vec<f32> {
    let mut bytes = vec![0u8; n * 4];
    backend.copy_d2h(&mut bytes, buf).unwrap();
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn cpu_backend_matmul_hand_checked() {
    let b = CpuBackend::new();
    assert_eq!(b.kind(), BackendKind::Cpu);
    let a = upload_f32(&b, &[1., 2., 3., 4.]);
    let w = upload_f32(&b, &[5., 6., 7., 8.]);
    let mut c = b.alloc(4 * 4).unwrap();
    b.matmul_f32(2, 2, 2, &a, &w, &mut c).unwrap();
    let out = download_f32(&b, &c, 4);
    assert!((out[0] - 19.).abs() < 1e-5);
    assert!((out[1] - 22.).abs() < 1e-5);
    assert!((out[2] - 43.).abs() < 1e-5);
    assert!((out[3] - 50.).abs() < 1e-5);
}

#[test]
fn cpu_backend_silu_and_alloc_reject() {
    let b = CpuBackend::new();
    assert!(b.alloc(0).is_err());
    let x = upload_f32(&b, &[0., 2.]);
    let mut y = b.alloc(8).unwrap();
    b.silu_f32(2, &x, &mut y).unwrap();
    let out = download_f32(&b, &y, 2);
    assert!((out[0] - 0.).abs() < 1e-6);
    let ref1 = 2. / (1. + (-2f32).exp());
    assert!((out[1] - ref1).abs() < 1e-5);
}

#[test]
fn default_backend_is_cpu_without_cuda_feature() {
    let b = default_backend();
    // Without --features cuda this must be CPU. With cuda but no device, also CPU.
    #[cfg(not(feature = "cuda"))]
    assert_eq!(b.kind(), BackendKind::Cpu);
    let _ = b.info();
}

#[test]
fn cpu_copy_rejects_overflowing_ranges_without_modifying_destination() {
    let backend = CpuBackend;
    let mut src = backend.alloc(8).unwrap();
    let mut dst = backend.alloc(8).unwrap();
    backend
        .copy_h2d(&mut src, &[1, 2, 3, 4, 5, 6, 7, 8])
        .unwrap();
    backend.copy_h2d(&mut dst, &[9; 8]).unwrap();

    for (dst_offset, src_offset, bytes) in [
        (usize::MAX, 0, 2),
        (0, usize::MAX, 2),
        (1, 0, usize::MAX),
        (0, 1, usize::MAX),
        (8, 0, 1),
        (0, 8, 1),
        (9, 0, 0),
        (0, 9, 0),
    ] {
        assert!(matches!(
            backend.copy_d2d(&mut dst, dst_offset, &src, src_offset, bytes),
            Err(forge_runtime::compute::ComputeError::InvalidArgument(_))
        ));
        let mut actual = [0; 8];
        backend.copy_d2h(&mut actual, &dst).unwrap();
        assert_eq!(actual, [9; 8]);
    }

    backend.copy_d2d(&mut dst, 2, &src, 3, 3).unwrap();
    backend.copy_d2d(&mut dst, 8, &src, 8, 0).unwrap();
    let mut actual = [0; 8];
    backend.copy_d2h(&mut actual, &dst).unwrap();
    assert_eq!(actual, [9, 9, 4, 5, 6, 9, 9, 9]);
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_parity_when_available() {
    use forge_runtime::compute::CudaBackend;
    let Ok(cuda) = CudaBackend::try_new(0) else {
        eprintln!("SKIP: CUDA feature on but no device");
        return;
    };
    let cpu = CpuBackend::new();
    let m = 7usize;
    let n = 5usize;
    let k = 9usize;
    let mut ha = vec![0f32; m * k];
    let mut hb = vec![0f32; k * n];
    for (i, v) in ha.iter_mut().enumerate() {
        *v = ((i % 5) as f32) * 0.1 - 0.2;
    }
    for (i, v) in hb.iter_mut().enumerate() {
        *v = ((i % 3) as f32) * 0.05 - 0.05;
    }
    let a_c = upload_f32(&cpu, &ha);
    let b_c = upload_f32(&cpu, &hb);
    let mut c_c = cpu.alloc(m * n * 4).unwrap();
    cpu.matmul_f32(m, n, k, &a_c, &b_c, &mut c_c).unwrap();
    let ref_out = download_f32(&cpu, &c_c, m * n);

    let a_g = upload_f32(&cuda, &ha);
    let b_g = upload_f32(&cuda, &hb);
    let mut c_g = cuda.alloc(m * n * 4).unwrap();
    cuda.matmul_f32(m, n, k, &a_g, &b_g, &mut c_g).unwrap();
    cuda.synchronize().unwrap();
    let got = download_f32(&cuda, &c_g, m * n);
    for (i, (g, r)) in got.iter().zip(ref_out.iter()).enumerate() {
        let tol = 1e-4 + 1e-3 * r.abs();
        assert!((g - r).abs() < tol, "idx {i}: {g} vs {r}");
    }
}
