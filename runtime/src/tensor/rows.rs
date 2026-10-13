//! Zero-copy row-prefix helpers over larger scratch buffers.

use super::TensorError;
use crate::ffi::kernels;

fn check(status: i32) -> Result<(), TensorError> {
    kernels::check(status).map_err(TensorError::Kernel)
}

/// RMSNorm first `rows` of `x` (leading dim `x_cols`) into `out` (leading `out_cols`).
pub fn rmsnorm_rows(
    x: &[f32],
    rows: usize,
    dim: usize,
    x_cols: usize,
    weight: &[f32],
    eps: f32,
    out: &mut [f32],
    out_cols: usize,
) -> Result<(), TensorError> {
    if rows == 0 || dim == 0 || weight.len() < dim {
        return Err(TensorError::InvalidArgument);
    }
    if x.len() < rows * x_cols || out.len() < rows * out_cols || x_cols < dim || out_cols < dim {
        return Err(TensorError::ShapeMismatch {
            expected: vec![rows, dim],
            got: vec![x.len(), out.len()],
        });
    }
    let status = unsafe {
        kernels::forge_kernels_rmsnorm_f32(
            rows,
            dim,
            x.as_ptr(),
            x_cols,
            weight.as_ptr(),
            eps,
            out.as_mut_ptr(),
            out_cols,
        )
    };
    check(status)
}

/// Linear on first `rows`: out[i] = x[i] @ W^T (+ bias).
pub fn linear_rows(
    x: &[f32],
    rows: usize,
    in_dim: usize,
    x_cols: usize,
    weight: &[f32], // [out, in]
    out_dim: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
    out_cols: usize,
) -> Result<(), TensorError> {
    if rows == 0 || in_dim == 0 || out_dim == 0 {
        return Err(TensorError::InvalidArgument);
    }
    if x.len() < rows * x_cols
        || out.len() < rows * out_cols
        || weight.len() < out_dim * in_dim
        || x_cols < in_dim
        || out_cols < out_dim
    {
        return Err(TensorError::ShapeMismatch {
            expected: vec![rows, out_dim],
            got: vec![x.len(), out.len()],
        });
    }
    if let Some(b) = bias {
        if b.len() < out_dim {
            return Err(TensorError::ShapeMismatch {
                expected: vec![out_dim],
                got: vec![b.len()],
            });
        }
    }
    let bias_ptr = bias.map(|b| b.as_ptr()).unwrap_or(std::ptr::null());
    let status = unsafe {
        kernels::forge_kernels_linear_f32(
            rows,
            out_dim,
            in_dim,
            x.as_ptr(),
            x_cols,
            weight.as_ptr(),
            in_dim,
            bias_ptr,
            out.as_mut_ptr(),
            out_cols,
        )
    };
    check(status)
}

pub fn add_rows(
    a: &[f32],
    b: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) -> Result<(), TensorError> {
    let n = rows * cols;
    if a.len() < n || b.len() < n || out.len() < n {
        return Err(TensorError::ShapeMismatch {
            expected: vec![n],
            got: vec![a.len(), b.len(), out.len()],
        });
    }
    let status =
        unsafe { kernels::forge_kernels_add_f32(n, a.as_ptr(), b.as_ptr(), out.as_mut_ptr()) };
    check(status)
}

pub fn copy_rows(
    src: &[f32],
    rows: usize,
    cols: usize,
    dst: &mut [f32],
) -> Result<(), TensorError> {
    let n = rows * cols;
    if src.len() < n || dst.len() < n {
        return Err(TensorError::ShapeMismatch {
            expected: vec![n],
            got: vec![src.len(), dst.len()],
        });
    }
    dst[..n].copy_from_slice(&src[..n]);
    Ok(())
}
