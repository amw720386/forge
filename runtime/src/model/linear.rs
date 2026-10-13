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
    /// Keep the original allocation intact until conversion succeeds.
    pub fn quantize(&mut self, bits: QuantBits, group_size: usize) -> Result<(), TensorError> {
        if let WeightStorage::Fp32(t) = &self.weight {
            let q = QuantMatrix::from_fp32(t, bits, group_size)?;
            self.weight = WeightStorage::Quant(q);
        }
        Ok(())
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
                let m = x.shape().first().copied().unwrap_or(1);
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
        if rows == 0
            || x.len() < rows * x_cols
            || out.len() < rows * out_cols
            || x_cols < k
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failed_quantization_preserves_weights_and_inference() {
        for bits in [QuantBits::Int8, QuantBits::Int4] {
            for group_size in [0, 3, 8] {
                let weight =
                    Tensor::from_vec(&[2, 4], vec![1., -2., 3., -4., 5., 6., -7., 8.]).unwrap();
                let bias = Tensor::from_vec(&[2], vec![0.5, -0.5]).unwrap();
                let mut layer = Linear::new(weight, Some(bias)).unwrap();
                let input = Tensor::from_vec(&[1, 4], vec![0.25, -0.5, 0.75, 1.]).unwrap();
                let expected = layer.forward(&input).unwrap();
                let original = match &layer.weight {
                    WeightStorage::Fp32(w) => w.data().as_ptr(),
                    WeightStorage::Quant(_) => unreachable!(),
                };

                assert_eq!(
                    layer.quantize(bits, group_size),
                    Err(TensorError::InvalidArgument)
                );
                match &layer.weight {
                    WeightStorage::Fp32(w) => {
                        assert_eq!(w.shape(), &[2, 4]);
                        assert_eq!(w.data().as_ptr(), original);
                        assert_eq!(w.data(), &[1., -2., 3., -4., 5., 6., -7., 8.]);
                    }
                    WeightStorage::Quant(_) => panic!("failed conversion changed storage"),
                }
                assert_eq!(layer.bias.as_ref().unwrap().data(), &[0.5, -0.5]);
                assert_eq!(layer.forward(&input).unwrap().data(), expected.data());
                layer.quantize(bits, 4).unwrap();
                assert!(layer.weight.is_quantized());
            }
        }
    }

    #[test]
    fn quantization_of_packed_weights_is_a_noop() {
        let mut layer = Linear::new(Tensor::filled(&[2, 4], 1.).unwrap(), None).unwrap();
        layer.quantize(QuantBits::Int8, 4).unwrap();
        let before = match &layer.weight {
            WeightStorage::Quant(q) => q.clone(),
            WeightStorage::Fp32(_) => unreachable!(),
        };
        layer.quantize(QuantBits::Int4, 0).unwrap();
        match &layer.weight {
            WeightStorage::Quant(q) => {
                assert_eq!(q.bits, before.bits);
                assert_eq!(q.group_size, before.group_size);
                assert_eq!(q.qweight, before.qweight);
                assert_eq!(q.scales, before.scales);
            }
            WeightStorage::Fp32(_) => panic!("packed weights were replaced"),
        }
    }
}
