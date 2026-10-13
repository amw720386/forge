//! Synthetic / real decoder-only transformer blocks.

mod architecture;
mod batch;
mod graph;
mod precision;
pub use architecture::{parse_hf_config, Architecture, DecoderOp, DecoderPlan};
pub use graph::{DecodeGraphKey, GraphModelIdentity};
pub use precision::{ExecutionPrecision, PrecisionContract, ScalarType};
mod config;
mod decoder;
mod generate;
mod linear;
mod load;
mod mlp;
mod quant;
mod session;
mod tokenizer;
mod transformer;

pub use batch::{BatchError, BatchEvent, BatchScratch, ContinuousBatcher};
pub use config::ModelConfig;
pub use decoder::{DecoderBlock, DecoderWeights};
pub use generate::{generate, GenError, GenerateOutput, GenerateParams};
pub use linear::{Linear, WeightStorage};
// Re-export for checkpoint / tooling.
pub use load::{load_hf_dir, load_model_path, load_qwen2_dir, LoadError};
pub use mlp::{Mlp, MlpWeights};
pub use quant::{QuantBits, QuantMatrix};
pub use session::StreamSession;
pub use tokenizer::{tokenizer_dir, ForgeTokenizer, TokError};
pub use transformer::{ForwardScratch, LayerKvCaches, Transformer, TransformerWeights};
