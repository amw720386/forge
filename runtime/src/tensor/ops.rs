//! High-level tensor ops backed by C++ kernels.

use super::{Tensor, TensorError};
use crate::ffi::kernels;

fn status_to_result(status: i32) -> Result<(), TensorError> {
    kernels::check(status).map_err(TensorError::Kernel)
}

/// Matrix multiply: `A[m,k] @ B[k,n] -> C[m,n]`.
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    if a.ndim() != 2 || b.ndim() != 2 {
        return Err(TensorError::ShapeMismatch {
            expected: vec![2],
            got: vec![a.ndim(), b.ndim()],
        });
    }
    require_contiguous(a)?;
    require_contiguous(b)?;

    let m = a.shape()[0];
    let k = a.shape()[1];
    let k2 = b.shape()[0];
    let n = b.shape()[1];
    if k != k2 {
        return Err(TensorError::MatmulInnerMismatch { a_k: k, b_k: k2 });
    }

    let mut c = Tensor::zeros(&[m, n])?;
    let status = unsafe {
        kernels::forge_kernels_matmul_f32(
            m,
            n,
            k,
            a.data().as_ptr(),
            k,
            b.data().as_ptr(),
            n,
            c.data_mut().as_mut_ptr(),
            n,
        )
    };
    status_to_result(status)?;
    Ok(c)
}

/// Matrix-vector: `A[m,n] @ x[n] -> y[m]`.
pub fn matvec(a: &Tensor, x: &Tensor) -> Result<Tensor, TensorError> {
    if a.ndim() != 2 || x.ndim() != 1 {
        return Err(TensorError::ShapeMismatch {
            expected: vec![2, 1],
            got: vec![a.ndim(), x.ndim()],
        });
    }
    require_contiguous(a)?;
    require_contiguous(x)?;

    let m = a.shape()[0];
    let n = a.shape()[1];
    if x.shape()[0] != n {
        return Err(TensorError::ShapeMismatch {
            expected: vec![n],
            got: x.shape().to_vec(),
        });
    }

    let mut y = Tensor::zeros(&[m])?;
    let status = unsafe {
        kernels::forge_kernels_matvec_f32(
            m,
            n,
            a.data().as_ptr(),
            n,
            x.data().as_ptr(),
            y.data_mut().as_mut_ptr(),
        )
    };
    status_to_result(status)?;
    Ok(y)
}

/// Linear: `Y = X @ W^T + bias?` with W in PyTorch `[out, in]` layout.
pub fn linear(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor, TensorError> {
    if x.ndim() != 2 || weight.ndim() != 2 {
        return Err(TensorError::ShapeMismatch {
            expected: vec![2, 2],
            got: vec![x.ndim(), weight.ndim()],
        });
    }
    require_contiguous(x)?;
    require_contiguous(weight)?;

    let m = x.shape()[0];
    let k = x.shape()[1];
    let n = weight.shape()[0];
    let k_w = weight.shape()[1];
    if k != k_w {
        return Err(TensorError::MatmulInnerMismatch { a_k: k, b_k: k_w });
    }

    let bias_ptr = if let Some(b) = bias {
        require_contiguous(b)?;
        if b.shape() != [n] {
            return Err(TensorError::ShapeMismatch {
                expected: vec![n],
                got: b.shape().to_vec(),
            });
        }
        b.data().as_ptr()
    } else {
        std::ptr::null()
    };

    let mut y = Tensor::zeros(&[m, n])?;
    let status = unsafe {
        kernels::forge_kernels_linear_f32(
            m,
            n,
            k,
            x.data().as_ptr(),
            k,
            weight.data().as_ptr(),
            k,
            bias_ptr,
            y.data_mut().as_mut_ptr(),
            n,
        )
    };
    status_to_result(status)?;
    Ok(y)
}

/// Write linear into an existing buffer `[m, n]` (no alloc).
pub fn linear_into(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    out: &mut Tensor,
) -> Result<(), TensorError> {
    if x.ndim() != 2 || weight.ndim() != 2 || out.ndim() != 2 {
        return Err(TensorError::ShapeMismatch {
            expected: vec![2, 2, 2],
            got: vec![x.ndim(), weight.ndim(), out.ndim()],
        });
    }
    require_contiguous(x)?;
    require_contiguous(weight)?;
    require_contiguous(out)?;

    let m = x.shape()[0];
    let k = x.shape()[1];
    let n = weight.shape()[0];
    if weight.shape()[1] != k || out.shape() != [m, n] {
        return Err(TensorError::ShapeMismatch {
            expected: vec![m, n],
            got: out.shape().to_vec(),
        });
    }

    let bias_ptr = match bias {
        Some(b) => {
            require_contiguous(b)?;
            if b.shape() != [n] {
                return Err(TensorError::ShapeMismatch {
                    expected: vec![n],
                    got: b.shape().to_vec(),
                });
            }
            b.data().as_ptr()
        }
        None => std::ptr::null(),
    };

    let status = unsafe {
        kernels::forge_kernels_linear_f32(
            m,
            n,
            k,
            x.data().as_ptr(),
            k,
            weight.data().as_ptr(),
            k,
            bias_ptr,
            out.data_mut().as_mut_ptr(),
            n,
        )
    };
    status_to_result(status)
}

pub fn rmsnorm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor, TensorError> {
    let mut y = Tensor::zeros(x.shape())?;
    rmsnorm_into(x, weight, eps, &mut y)?;
    Ok(y)
}

pub fn rmsnorm_into(
    x: &Tensor,
    weight: &Tensor,
    eps: f32,
    out: &mut Tensor,
) -> Result<(), TensorError> {
    if x.ndim() != 2 {
        return Err(TensorError::ShapeMismatch {
            expected: vec![2],
            got: vec![x.ndim()],
        });
    }
    require_contiguous(x)?;
    require_contiguous(weight)?;
    require_contiguous(out)?;
    let rows = x.shape()[0];
    let dim = x.shape()[1];
    if weight.shape() != [dim] || out.shape() != x.shape() {
        return Err(TensorError::ShapeMismatch {
            expected: vec![dim],
            got: weight.shape().to_vec(),
        });
    }
    let status = unsafe {
        kernels::forge_kernels_rmsnorm_f32(
            rows,
            dim,
            x.data().as_ptr(),
            dim,
            weight.data().as_ptr(),
            eps,
            out.data_mut().as_mut_ptr(),
            dim,
        )
    };
    status_to_result(status)
}

pub fn silu_inplace(x: &mut Tensor) -> Result<(), TensorError> {
    require_contiguous(x)?;
    let n = x.numel();
    let status =
        unsafe { kernels::forge_kernels_silu_f32(n, x.data().as_ptr(), x.data_mut().as_mut_ptr()) };
    status_to_result(status)
}

pub fn mul_into(a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<(), TensorError> {
    require_contiguous(a)?;
    require_contiguous(b)?;
    require_contiguous(out)?;
    if a.shape() != b.shape() || a.shape() != out.shape() {
        return Err(TensorError::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: b.shape().to_vec(),
        });
    }
    let status = unsafe {
        kernels::forge_kernels_mul_f32(
            a.numel(),
            a.data().as_ptr(),
            b.data().as_ptr(),
            out.data_mut().as_mut_ptr(),
        )
    };
    status_to_result(status)
}

pub fn add_into(a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<(), TensorError> {
    require_contiguous(a)?;
    require_contiguous(b)?;
    require_contiguous(out)?;
    if a.shape() != b.shape() || a.shape() != out.shape() {
        return Err(TensorError::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: b.shape().to_vec(),
        });
    }
    let status = unsafe {
        kernels::forge_kernels_add_f32(
            a.numel(),
            a.data().as_ptr(),
            b.data().as_ptr(),
            out.data_mut().as_mut_ptr(),
        )
    };
    status_to_result(status)
}

pub fn rope_inplace(
    x: &mut Tensor,
    n_heads: usize,
    head_dim: usize,
    position_offset: usize,
    rope_theta: f32,
    positions: Option<&[i32]>,
) -> Result<(), TensorError> {
    // x: [T, n_heads * head_dim]
    if x.ndim() != 2 {
        return Err(TensorError::ShapeMismatch {
            expected: vec![2],
            got: vec![x.ndim()],
        });
    }
    require_contiguous(x)?;
    let seq = x.shape()[0];
    let width = x.shape()[1];
    if width != n_heads * head_dim {
        return Err(TensorError::ShapeMismatch {
            expected: vec![n_heads * head_dim],
            got: vec![width],
        });
    }
    if let Some(p) = positions {
        if p.len() != seq {
            return Err(TensorError::ShapeMismatch {
                expected: vec![seq],
                got: vec![p.len()],
            });
        }
    }
    let pos_ptr = positions.map(|p| p.as_ptr()).unwrap_or(std::ptr::null());
    let status = unsafe {
        kernels::forge_kernels_rope_f32(
            seq,
            n_heads,
            head_dim,
            x.data_mut().as_mut_ptr(),
            pos_ptr,
            position_offset,
            rope_theta,
        )
    };
    status_to_result(status)
}

pub fn softmax_inplace_causal(
    scores: &mut Tensor,
    n_heads: usize,
    causal: bool,
    causal_q_offset: usize,
) -> Result<(), TensorError> {
    // scores: [n_heads, T_q, T_k]
    if scores.ndim() != 3 || scores.shape()[0] != n_heads {
        return Err(TensorError::ShapeMismatch {
            expected: vec![n_heads],
            got: scores.shape().to_vec(),
        });
    }
    require_contiguous(scores)?;
    let rows = scores.shape()[1];
    let cols = scores.shape()[2];
    let status = unsafe {
        kernels::forge_kernels_softmax_f32(
            n_heads,
            rows,
            cols,
            scores.data_mut().as_mut_ptr(),
            if causal { 1 } else { 0 },
            causal_q_offset,
        )
    };
    status_to_result(status)
}

fn require_contiguous(t: &Tensor) -> Result<(), TensorError> {
    if t.is_contiguous() {
        Ok(())
    } else {
        Err(TensorError::ContiguityRequired)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (x - y).abs();
            let scale = x.abs().max(y.abs());
            assert!(
                diff <= tol + 1e-4 * scale,
                "mismatch at {}: {} vs {}",
                i,
                x,
                y
            );
        }
    }

    #[test]
    fn matmul_hand_checked() {
        let a = Tensor::from_vec(&[2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap();
        let b = Tensor::from_vec(&[3, 2], vec![7., 8., 9., 10., 11., 12.]).unwrap();
        let c = matmul(&a, &b).unwrap();
        approx_eq(c.data(), &[58., 64., 139., 154.], 1e-5);
    }

    #[test]
    fn matmul_matches_reference() {
        let m = 11usize;
        let n = 7usize;
        let k = 9usize;
        let a_data: Vec<f32> = (0..m * k).map(|i| 0.01 * i as f32).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| -0.02 * i as f32 + 0.5).collect();
        let a = Tensor::from_vec(&[m, k], a_data.clone()).unwrap();
        let b = Tensor::from_vec(&[k, n], b_data.clone()).unwrap();
        let c = matmul(&a, &b).unwrap();

        let mut ref_c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a_data[i * k + p] * b_data[p * n + j];
                }
                ref_c[i * n + j] = sum;
            }
        }
        approx_eq(c.data(), &ref_c, 1e-4);
    }

    #[test]
    fn matvec_hand_checked() {
        let a = Tensor::from_vec(&[2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap();
        let x = Tensor::from_vec(&[3], vec![1., 0., -1.]).unwrap();
        let y = matvec(&a, &x).unwrap();
        approx_eq(y.data(), &[-2., -2.], 1e-5);
    }

    #[test]
    fn matmul_inner_mismatch() {
        let a = Tensor::zeros(&[2, 3]).unwrap();
        let b = Tensor::zeros(&[4, 2]).unwrap();
        assert!(matches!(
            matmul(&a, &b),
            Err(TensorError::MatmulInnerMismatch { .. })
        ));
    }

    #[test]
    fn linear_matches_xtwt() {
        let x = Tensor::from_vec(&[2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap();
        let w = Tensor::from_vec(&[2, 3], vec![1., 0., 0., 0., 1., 0.]).unwrap();
        let y = linear(&x, &w, None).unwrap();
        approx_eq(y.data(), &[1., 2., 4., 5.], 1e-5);
    }
}
