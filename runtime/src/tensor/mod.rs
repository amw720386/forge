//! Contiguous FP32 tensors with shape/stride validation.

use std::fmt;
use std::sync::Arc;

pub mod ops;
pub mod rows;

pub use ops::{
    add_into, linear, linear_into, matmul, matvec, mul_into, rmsnorm, rmsnorm_into, rope_inplace,
    silu_inplace, softmax_inplace_causal,
};
pub use rows::{add_rows, copy_rows, linear_rows, rmsnorm_rows};

/// Errors from tensor construction or operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorError {
    Configuration(String),
    EmptyShape,
    ZeroDimension,
    SizeOverflow,
    ContiguityRequired,
    InvalidArgument,
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    MatmulInnerMismatch {
        a_k: usize,
        b_k: usize,
    },
    Kernel(i32),
    AllocFailed(String),
    BudgetExceeded {
        need_mib: u64,
        budget_mib: u64,
    },
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorError::Configuration(s) => write!(f, "configuration: {s}"),
            TensorError::EmptyShape => write!(f, "tensor shape must be non-empty"),
            TensorError::ZeroDimension => write!(f, "tensor dimensions must be > 0"),
            TensorError::SizeOverflow => write!(f, "tensor element count overflowed"),
            TensorError::ContiguityRequired => {
                write!(f, "operation requires a contiguous row-major tensor")
            }
            TensorError::InvalidArgument => write!(f, "invalid argument"),
            TensorError::ShapeMismatch { expected, got } => {
                write!(f, "shape mismatch: expected {:?}, got {:?}", expected, got)
            }
            TensorError::MatmulInnerMismatch { a_k, b_k } => {
                write!(f, "matmul inner dim mismatch: {} vs {}", a_k, b_k)
            }
            TensorError::Kernel(code) => write!(f, "kernel error status {}", code),
            TensorError::AllocFailed(msg) => write!(f, "allocation failed: {msg}"),
            TensorError::BudgetExceeded {
                need_mib,
                budget_mib,
            } => write!(
                f,
                "memory budget exceeded: need {need_mib} MiB, budget {budget_mib} MiB"
            ),
        }
    }
}

impl std::error::Error for TensorError {}

/// Row-major FP32 tensor. Data is `Arc`-backed so tied embeddings can share storage.
#[derive(Clone, Debug)]
pub struct Tensor {
    shape: Vec<usize>,
    strides: Vec<usize>,
    data: Arc<Vec<f32>>,
}

impl Tensor {
    /// Create a zero-filled contiguous tensor.
    pub fn zeros(shape: &[usize]) -> Result<Self, TensorError> {
        Self::filled(shape, 0.0)
    }

    /// Create a contiguous tensor filled with `value`.
    pub fn filled(shape: &[usize], value: f32) -> Result<Self, TensorError> {
        validate_shape(shape)?;
        let numel = numel(shape)?;
        let strides = row_major_strides(shape);
        Ok(Self {
            shape: shape.to_vec(),
            strides,
            data: Arc::new(vec![value; numel]),
        })
    }

    /// Create from owned data; length must match product of shape.
    pub fn from_vec(shape: &[usize], data: Vec<f32>) -> Result<Self, TensorError> {
        validate_shape(shape)?;
        let n = numel(shape)?;
        if data.len() != n {
            return Err(TensorError::ShapeMismatch {
                expected: shape.to_vec(),
                got: vec![data.len()],
            });
        }
        Ok(Self {
            shape: shape.to_vec(),
            strides: row_major_strides(shape),
            data: Arc::new(data),
        })
    }

    /// Cheap share of the underlying buffer (tied embed / lm_head).
    pub fn share(&self) -> Self {
        Self {
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            data: Arc::clone(&self.data),
        }
    }

    pub fn strong_count(&self) -> usize {
        Arc::strong_count(&self.data)
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [f32] {
        Arc::make_mut(&mut self.data).as_mut_slice()
    }

    /// True when strides match contiguous row-major layout.
    pub fn is_contiguous(&self) -> bool {
        self.strides == row_major_strides(&self.shape)
    }

    pub fn as_slice(&self) -> Result<&[f32], TensorError> {
        if !self.is_contiguous() {
            return Err(TensorError::ContiguityRequired);
        }
        Ok(&self.data)
    }

    pub fn as_mut_slice(&mut self) -> Result<&mut [f32], TensorError> {
        if !self.is_contiguous() {
            return Err(TensorError::ContiguityRequired);
        }
        Ok(Arc::make_mut(&mut self.data).as_mut_slice())
    }

    /// Take ownership of the buffer if uniquely held; otherwise clones.
    pub fn into_vec(self) -> Vec<f32> {
        match Arc::try_unwrap(self.data) {
            Ok(v) => v,
            Err(arc) => (*arc).clone(),
        }
    }
}

fn validate_shape(shape: &[usize]) -> Result<(), TensorError> {
    if shape.is_empty() {
        return Err(TensorError::EmptyShape);
    }
    if shape.contains(&0) {
        return Err(TensorError::ZeroDimension);
    }
    Ok(())
}

fn numel(shape: &[usize]) -> Result<usize, TensorError> {
    shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or(TensorError::SizeOverflow)
}

fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0; shape.len()];
    let mut acc = 1usize;
    for i in (0..shape.len()).rev() {
        strides[i] = acc;
        acc *= shape[i];
    }
    strides
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_and_strides() {
        let t = Tensor::zeros(&[2, 3, 4]).unwrap();
        assert_eq!(t.numel(), 24);
        assert_eq!(t.strides(), &[12, 4, 1]);
        assert!(t.is_contiguous());
    }

    #[test]
    fn rejects_empty_and_zero_dims() {
        assert!(matches!(Tensor::zeros(&[]), Err(TensorError::EmptyShape)));
        assert!(matches!(
            Tensor::zeros(&[2, 0]),
            Err(TensorError::ZeroDimension)
        ));
    }

    #[test]
    fn from_vec_length_check() {
        assert!(Tensor::from_vec(&[2, 2], vec![1.0, 2.0, 3.0]).is_err());
        assert!(Tensor::from_vec(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]).is_ok());
    }

    #[test]
    fn kernel_version_nonempty() {
        let v = crate::ffi::kernels::version();
        assert!(!v.is_empty());
    }
}
