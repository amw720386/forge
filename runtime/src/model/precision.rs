//! Existing numerical policies, made explicit without changing kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScalarType {
    Fp32,
    Fp64,
    Bf16,
    Int8,
    Int4,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExecutionPrecision {
    CpuFp32,
    CpuInt8,
    CpuInt4,
    CudaFp32,
    CudaBf16,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrecisionContract {
    pub storage: ScalarType,
    pub projection_input: ScalarType,
    pub activation: ScalarType,
    pub accumulation: ScalarType,
    pub kv: ScalarType,
    pub normalization: ScalarType,
    pub normalization_reduction: ScalarType,
    pub attention_reduction: ScalarType,
    pub sampling: ScalarType,
    pub exact_greedy_required: bool,
}
impl ExecutionPrecision {
    pub fn contract(self) -> PrecisionContract {
        use ScalarType::*;
        PrecisionContract {
            storage: match self {
                Self::CpuInt8 => Int8,
                Self::CpuInt4 => Int4,
                Self::CudaBf16 => Bf16,
                _ => Fp32,
            },
            projection_input: if self == Self::CudaBf16 { Bf16 } else { Fp32 },
            activation: Fp32,
            accumulation: Fp32,
            kv: Fp32,
            normalization: Fp32,
            normalization_reduction: if matches!(self, Self::CudaFp32 | Self::CudaBf16) {
                Fp32
            } else {
                Fp64
            },
            attention_reduction: Fp32,
            sampling: Fp32,
            exact_greedy_required: !matches!(self, Self::CpuInt8 | Self::CpuInt4),
        }
    }
    /// Phase 15's conservative BF16 dispatch boundary; no split BF16 reductions.
    pub fn partitioned_attention(self, batch: usize, lengths: &[usize]) -> bool {
        self.partitioned_attention_lengths(batch, lengths.iter().copied())
    }
    pub fn partitioned_attention_lengths(
        self,
        batch: usize,
        lengths: impl Iterator<Item = usize> + Clone,
    ) -> bool {
        lengths.clone().next().is_some()
            && lengths.clone().all(|n| n >= 128)
            && (self != Self::CudaBf16 || batch <= 2 || lengths.clone().all(|n| n <= 512))
    }
}
