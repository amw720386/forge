//! Weight-only quantization (INT8 / INT4) — pack FP32 weights, keep activations FP32.
//!
//! # Equations (symmetric, group-wise)
//!
//! For each output row `j` and group `g` covering columns `[g·G, (g+1)·G)`:
//!
//! ```text
//! s[j,g] = max_{k in group} |W[j,k]| / qmax
//! q[j,k] = clamp(round(W[j,k] / s[j,g]), -qmax, qmax)
//! Ŵ[j,k] = q[j,k] · s[j,g]
//! ```
//!
//! - INT8: `qmax = 127`
//! - INT4: `qmax = 7` (signed nibble ∈ [-8, 7]; we clamp to [-7, 7] for symmetry
//!   about zero with `qmax=7`, or [-8,7] with asymmetric max — we use [-8,7] and
//!   `s = max|W|/7` when max abs maps to 7; if a value needs -8 we allow it)
//!
//! Per-channel = `group_size == in_features` (one scale per output row).
//!
//! # INT4 packing
//!
//! Row-major over output channels. Within a row, columns are packed two per byte:
//! low nibble = even `k`, high nibble = odd `k`. Each nibble is 4-bit two's complement.
//! Packed row stride = `(k + 1) / 2` bytes.

use crate::tensor::{Tensor, TensorError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantBits {
    Int8,
    Int4,
}

impl QuantBits {
    pub fn from_u8(bits: u8) -> Result<Self, TensorError> {
        match bits {
            8 => Ok(Self::Int8),
            4 => Ok(Self::Int4),
            _ => Err(TensorError::InvalidArgument),
        }
    }

    pub fn qmax(self) -> f32 {
        match self {
            Self::Int8 => 127.0,
            Self::Int4 => 7.0,
        }
    }

    pub fn bits(self) -> u8 {
        match self {
            Self::Int8 => 8,
            Self::Int4 => 4,
        }
    }
}

#[derive(Clone, Debug)]
pub struct QuantMatrix {
    pub bits: QuantBits,
    pub out_features: usize,
    pub in_features: usize,
    pub group_size: usize,
    /// INT8: `out * in` int8. INT4: `out * ((in+1)/2)` packed bytes.
    pub qweight: Vec<u8>,
    /// Length `out * (in / group_size)`.
    pub scales: Vec<f32>,
}

impl QuantMatrix {
    pub fn n_groups(&self) -> usize {
        self.in_features / self.group_size
    }

    pub fn nbytes(&self) -> usize {
        self.qweight.len() + self.scales.len() * 4
    }

    pub fn from_fp32(
        weight: &Tensor,
        bits: QuantBits,
        group_size: usize,
    ) -> Result<Self, TensorError> {
        if weight.ndim() != 2 {
            return Err(TensorError::ShapeMismatch {
                expected: vec![2],
                got: vec![weight.ndim()],
            });
        }
        let out = weight.shape()[0];
        let inn = weight.shape()[1];
        if group_size == 0 || inn % group_size != 0 {
            return Err(TensorError::InvalidArgument);
        }
        let n_groups = inn / group_size;
        let data = weight.data();
        let mut scales = vec![0.0f32; out * n_groups];
        let qmax = bits.qmax();

        match bits {
            QuantBits::Int8 => {
                let mut qweight = vec![0u8; out * inn];
                for j in 0..out {
                    for g in 0..n_groups {
                        let base = j * inn + g * group_size;
                        let mut amax = 0.0f32;
                        for p in 0..group_size {
                            amax = amax.max(data[base + p].abs());
                        }
                        let s = if amax == 0.0 { 1.0 } else { amax / qmax };
                        scales[j * n_groups + g] = s;
                        for p in 0..group_size {
                            let q = (data[base + p] / s).round().clamp(-127.0, 127.0) as i8;
                            qweight[base + p] = q as u8;
                        }
                    }
                }
                Ok(Self {
                    bits,
                    out_features: out,
                    in_features: inn,
                    group_size,
                    qweight,
                    scales,
                })
            }
            QuantBits::Int4 => {
                let packed_row = (inn + 1) / 2;
                let mut qweight = vec![0u8; out * packed_row];
                for j in 0..out {
                    for g in 0..n_groups {
                        let base = j * inn + g * group_size;
                        let mut amax = 0.0f32;
                        for p in 0..group_size {
                            amax = amax.max(data[base + p].abs());
                        }
                        let s = if amax == 0.0 { 1.0 } else { amax / qmax };
                        scales[j * n_groups + g] = s;
                        for p in 0..group_size {
                            let col = g * group_size + p;
                            let q = (data[base + p] / s).round().clamp(-8.0, 7.0) as i8;
                            let byte_i = j * packed_row + (col >> 1);
                            let nibble = (q as u8) & 0x0f;
                            if (col & 1) == 0 {
                                qweight[byte_i] = (qweight[byte_i] & 0xf0) | nibble;
                            } else {
                                qweight[byte_i] = (qweight[byte_i] & 0x0f) | (nibble << 4);
                            }
                        }
                    }
                }
                Ok(Self {
                    bits,
                    out_features: out,
                    in_features: inn,
                    group_size,
                    qweight,
                    scales,
                })
            }
        }
    }

    /// Reconstruct an approximate FP32 weight matrix (tests / diagnostics only).
    pub fn dequantize_fp32(&self) -> Result<Tensor, TensorError> {
        let out = self.out_features;
        let inn = self.in_features;
        let n_groups = self.n_groups();
        let mut data = vec![0.0f32; out * inn];
        match self.bits {
            QuantBits::Int8 => {
                for j in 0..out {
                    for g in 0..n_groups {
                        let s = self.scales[j * n_groups + g];
                        for p in 0..self.group_size {
                            let idx = j * inn + g * self.group_size + p;
                            let q = self.qweight[idx] as i8;
                            data[idx] = q as f32 * s;
                        }
                    }
                }
            }
            QuantBits::Int4 => {
                let packed_row = (inn + 1) / 2;
                for j in 0..out {
                    for g in 0..n_groups {
                        let s = self.scales[j * n_groups + g];
                        for p in 0..self.group_size {
                            let col = g * self.group_size + p;
                            let byte = self.qweight[j * packed_row + (col >> 1)];
                            let n = if (col & 1) == 0 {
                                byte & 0x0f
                            } else {
                                byte >> 4
                            };
                            let q = ((n as i8) << 4) >> 4;
                            data[j * inn + col] = q as f32 * s;
                        }
                    }
                }
            }
        }
        Tensor::from_vec(&[out, inn], data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int8_roundtrip_small() {
        let w = Tensor::from_vec(
            &[2, 8],
            vec![
                0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.05, -0.05, 0.15, -0.15, 0.25, -0.25,
                0.35, -0.35,
            ],
        )
        .unwrap();
        let q = QuantMatrix::from_fp32(&w, QuantBits::Int8, 8).unwrap();
        let back = q.dequantize_fp32().unwrap();
        for (a, b) in w.data().iter().zip(back.data().iter()) {
            assert!((a - b).abs() < 0.01, "{a} vs {b}");
        }
    }

    #[test]
    fn int4_pack_unpack() {
        let mut data = Vec::new();
        for i in 0..16 {
            data.push((i as f32 - 8.0) * 0.05);
        }
        let w = Tensor::from_vec(&[2, 8], data).unwrap();
        let q = QuantMatrix::from_fp32(&w, QuantBits::Int4, 8).unwrap();
        assert_eq!(q.qweight.len(), 2 * 4);
        let back = q.dequantize_fp32().unwrap();
        for (a, b) in w.data().iter().zip(back.data().iter()) {
            assert!((a - b).abs() < 0.05, "{a} vs {b}");
        }
    }
}
