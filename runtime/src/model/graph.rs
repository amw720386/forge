//! Replay identity for model, precision, path and stable captured allocations.
use super::{Architecture, ExecutionPrecision, ModelConfig};
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphModelIdentity {
    pub architecture: Architecture,
    pub dimensions: [usize; 8],
    pub rope_theta: u32,
    pub norm_epsilon: u32,
    pub tied_output: bool,
    pub precision: ExecutionPrecision,
    pub generation: u64,
}
impl GraphModelIdentity {
    pub fn new(cfg: &ModelConfig, precision: ExecutionPrecision, generation: u64) -> Self {
        Self {
            architecture: cfg.architecture,
            dimensions: cfg.execution_shape(),
            rope_theta: cfg.rope_theta.to_bits(),
            norm_epsilon: cfg.rms_norm_eps.to_bits(),
            tied_output: cfg.tie_word_embeddings,
            precision,
            generation,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodeGraphKey {
    pub model: GraphModelIdentity,
    pub batch: usize,
    pub kv_block_size: usize,
    pub kv_blocks: usize,
    /// Scratch owns all captured workspaces; model generation owns weights.
    pub allocations: [usize; 5],
    pub workspace: [usize; 19],
    pub reuse_bf16: bool,
    pub gemm_lt: bool,
    pub partitioned_attention: bool,
    /// The captured path is batched paged decode with FP32 KV.
    pub paged_decode_fp32_kv: bool,
}
