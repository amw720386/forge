use super::quant::{QuantBits, QuantMatrix};
use crate::ffi::kernels;
use crate::tensor::ops::{linear, linear_into};
use crate::tensor::{Tensor, TensorError};

/// Weight storage: FP32 reference or packed weight-only quant.
#[derive(Clone, Debug)]
pub enum WeightStorage {
    Fp32(Tensor), // [out, in]
    Quant(QuantMatrix),
}

impl WeightStorage {
    pub fn out_features(&self) -> usize {
        match self {
            Self::Fp32(t) => t.shape()[0],
            Self::Quant(q) => q.out_features,
        }
    }

    pub fn in_features(&self) -> usize {
        match self {
            Self::Fp32(t) => t.shape()[1],
            Self::Quant(q) => q.in_features,
        }
    }

    pub fn nbytes(&self) -> usize {
        match self {
            Self::Fp32(t) => t.data().len() * 4,
            Self::Quant(q) => q.nbytes(),
        }
    }

    pub fn is_quantized(&self) -> bool {
        matches!(self, Self::Quant(_))
    }
}

/// Affine map with PyTorch-layout weight `[out, in]` and optional bias `[out]`.
#[derive(Clone, Debug)]
pub struct Linear {
    pub weight: WeightStorage,
    pub bias: Option<Tensor>,
}

impl Linear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Result<Self, TensorError> {
        if weight.ndim() != 2 {
            return Err(TensorError::ShapeMismatch {
                expected: vec![2],
                got: vec![weight.ndim()],
            });
        }
        let out = weight.shape()[0];
        if let Some(ref b) = bias {
            if b.shape() != [out] {
                return Err(TensorError::ShapeMismatch {
                    expected: vec![out],
                    got: b.shape().to_vec(),
                });
            }
        }
        Ok(Self {
            weight: WeightStorage::Fp32(weight),
            bias,
        })
    }

    pub fn out_features(&self) -> usize {
        self.weight.out_features()
    }

    pub fn in_features(&self) -> usize {
        self.weight.in_features()
    }

    pub fn nbytes(&self) -> usize {
        self.weight.nbytes() + self.bias.as_ref().map(|b| b.data().len() * 4).unwrap_or(0)
    }

    /// Convert FP32 weights in-place to weight-only quant. Bias stays FP32.
    /// Moves the FP32 buffer out (no extra full clone when uniquely owned via Arc).
    pub fn quantize(&mut self, bits: QuantBits, group_size: usize) -> Result<(), TensorError> {
        let prev = std::mem::replace(
            &mut self.weight,
            // Temporary placeholder; replaced on all paths below.
            WeightStorage::Fp32(Tensor::zeros(&[1, 1])?),
        );
        match prev {
            WeightStorage::Quant(q) => {
                self.weight = WeightStorage::Quant(q);
                Ok(())
            }
            WeightStorage::Fp32(t) => {
                let q = QuantMatrix::from_fp32(&t, bits, group_size)?;
                drop(t);
                self.weight = WeightStorage::Quant(q);
                Ok(())
            }
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor, TensorError> {
        match &self.weight {
            WeightStorage::Fp32(w) => linear(x, w, self.bias.as_ref()),
            WeightStorage::Quant(_) => {
                let m = x.shape()[0];
                let n = self.out_features();
                let mut out = Tensor::zeros(&[m, n])?;
                self.forward_into(x, &mut out)?;
                Ok(out)
            }
        }
    }

    pub fn forward_into(&self, x: &Tensor, out: &mut Tensor) -> Result<(), TensorError> {
        match &self.weight {
            WeightStorage::Fp32(w) => linear_into(x, w, self.bias.as_ref(), out),
            WeightStorage::Quant(_) => {
                let m = x.shape().get(0).copied().unwrap_or(1);
                let k = self.in_features();
                let n = self.out_features();
                let out_cols = out.shape().get(1).copied().unwrap_or(n);
                let x_cols = if x.ndim() == 1 { k } else { x.shape()[1] };
                let rows = if x.ndim() == 1 { 1 } else { m };
                self.apply_rows(x.data(), rows, k, x_cols, out.data_mut(), out_cols)
            }
        }
    }

    /// Apply this linear to the first `rows` of `x` into `out` (fused dequant when quantized).
    pub fn apply_rows(
        &self,
        x: &[f32],
        rows: usize,
        in_dim: usize,
        x_cols: usize,
        out: &mut [f32],
        out_cols: usize,
    ) -> Result<(), TensorError> {
        let n = self.out_features();
        let k = self.in_features();
        if in_dim != k {
            return Err(TensorError::ShapeMismatch {
                expected: vec![k],
                got: vec![in_dim],
            });
        }
        if rows == 0 || x.len() < rows * x_cols || out.len() < rows * out_cols || x_cols < k
            || out_cols < n
        {
            return Err(TensorError::ShapeMismatch {
                expected: vec![rows, n],
                got: vec![x.len(), out.len()],
            });
        }
        let bias_ptr = self
            .bias
            .as_ref()
            .map(|b| b.data().as_ptr())
            .unwrap_or(std::ptr::null());

        let status = match &self.weight {
            WeightStorage::Fp32(w) => unsafe {
                kernels::forge_kernels_linear_f32(
                    rows,
                    n,
                    k,
                    x.as_ptr(),
                    x_cols,
                    w.data().as_ptr(),
                    k,
                    bias_ptr,
                    out.as_mut_ptr(),
                    out_cols,
                )
            },
            WeightStorage::Quant(q) => match q.bits {
                QuantBits::Int8 => unsafe {
                    kernels::forge_kernels_linear_i8_f32(
                        rows,
                        n,
                        k,
                        x.as_ptr(),
                        x_cols,
                        q.qweight.as_ptr() as *const i8,
                        q.scales.as_ptr(),
                        q.group_size,
                        bias_ptr,
                        out.as_mut_ptr(),
                        out_cols,
                    )
                },
                QuantBits::Int4 => unsafe {
                    kernels::forge_kernels_linear_i4_f32(
                        rows,
                        n,
                        k,
                        x.as_ptr(),
                        x_cols,
                        q.qweight.as_ptr(),
                        q.scales.as_ptr(),
                        q.group_size,
                        bias_ptr,
                        out.as_mut_ptr(),
                        out_cols,
                    )
                },
            },
        };
        kernels::check(status).map_err(TensorError::Kernel)
    }
}
