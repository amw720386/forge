//! Compare C++/Rust kernels against PyTorch-generated fixtures.

use forge_runtime::attention::{attend, AttnScratch, AttentionConfig};
use forge_runtime::cache::KvCache;
use forge_runtime::ffi::kernels;
use forge_runtime::tensor::ops::{rmsnorm, rope_inplace, silu_inplace, softmax_inplace_causal};
use forge_runtime::tensor::Tensor;
use std::fs;
use std::path::PathBuf;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn read_f32(name: &str) -> Vec<f32> {
    let bytes = fs::read(fixture_dir().join(name)).unwrap_or_else(|e| {
        panic!(
            "missing fixture {name}: {e}. Run: python tests/reference/phase2_reference.py"
        )
    });
    assert_eq!(bytes.len() % 4, 0);
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn approx_eq(a: &[f32], b: &[f32], atol: f32, rtol: f32) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let tol = atol + rtol * x.abs().max(y.abs());
        assert!(diff <= tol, "idx {i}: {x} vs {y} tol {tol}");
    }
}

#[test]
fn rmsnorm_matches_pytorch() {
    let x = Tensor::from_vec(&[3, 8], read_f32("rmsnorm_x.bin")).unwrap();
    let w = Tensor::from_vec(&[8], read_f32("rmsnorm_w.bin")).unwrap();
    let y = rmsnorm(&x, &w, 1e-6).unwrap();
    approx_eq(y.data(), &read_f32("rmsnorm_y.bin"), 1e-5, 1e-5);
}

#[test]
fn softmax_matches_pytorch() {
    let mut s = Tensor::from_vec(&[1, 2, 3], read_f32("softmax_in.bin")).unwrap();
    softmax_inplace_causal(&mut s, 1, false, 0).unwrap();
    approx_eq(s.data(), &read_f32("softmax_out.bin"), 1e-5, 1e-5);
}

#[test]
fn softmax_causal_matches_pytorch() {
    let mut s = Tensor::from_vec(&[1, 2, 2], read_f32("softmax_causal_in.bin")).unwrap();
    softmax_inplace_causal(&mut s, 1, true, 0).unwrap();
    approx_eq(s.data(), &read_f32("softmax_causal_out.bin"), 1e-5, 1e-5);
}

#[test]
fn silu_matches_pytorch() {
    let mut x = Tensor::from_vec(&[3], read_f32("silu_x.bin")).unwrap();
    silu_inplace(&mut x).unwrap();
    approx_eq(x.data(), &read_f32("silu_y.bin"), 1e-5, 1e-5);
}

#[test]
fn rope_matches_pytorch() {
    let meta = fs::read_to_string(fixture_dir().join("rope.json")).unwrap();
    assert!(meta.contains("10000"));
    let mut x = Tensor::from_vec(&[4, 2 * 8], read_f32("rope_x.bin")).unwrap();
    rope_inplace(&mut x, 2, 8, 0, 10000.0, None).unwrap();
    approx_eq(x.data(), &read_f32("rope_y.bin"), 1e-5, 1e-4);
}

#[test]
fn attention_matches_pytorch() {
    let cfg = AttentionConfig {
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 8,
        rope_theta: 10000.0,
    };
    let t = 5;
    let mut scratch = AttnScratch::new(&cfg, t).unwrap();
    let q = read_f32("attn_q.bin");
    let k = read_f32("attn_k.bin");
    let v = read_f32("attn_v.bin");
    scratch.q.data_mut()[..q.len()].copy_from_slice(&q);
    scratch.k.data_mut()[..k.len()].copy_from_slice(&k);
    scratch.v.data_mut()[..v.len()].copy_from_slice(&v);
    let mut cache = KvCache::new(cfg.n_kv_heads, cfg.head_dim, t).unwrap();
    attend(&cfg, &mut scratch, t, &mut cache, 0).unwrap();
    approx_eq(
        &scratch.ctx.data()[..t * cfg.n_heads * cfg.head_dim],
        &read_f32("attn_ctx.bin"),
        1e-4,
        1e-3,
    );
}

#[test]
fn kernel_version_phase2() {
    let v = kernels::version();
    assert!(
        v.contains("phase2")
            || v.contains("phase5")
            || v.contains("phase6")
            || v.contains("0.2")
            || v.contains("0.5")
            || v.contains("0.6")
            || v.contains("0.7"),
        "{v}"
    );
}
