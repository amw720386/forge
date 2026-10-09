# Phase 16: model-neutral decoder architecture

Forge retains its own transformer, attention, KV, sampling and serving code.
The generalization changes model metadata and loading, not numerical kernels or
scheduler policy. The Phase 15 harness, accepted results and rejected experiments
remain unchanged.

## Configuration and adapters

`Architecture::{Qwen2,Llama}` selects the family adapter. `parse_hf_config`
requires architecture, dimensions, norm epsilon, RoPE theta, SiLU activation,
embedding tying and BOS/EOS metadata. No missing architecture or dimension is
replaced with Qwen defaults. Head dimension is derived from hidden/head dimensions;
an explicit HF head dimension must agree. Scale is the supported causal GQA
scale `1/sqrt(head_dim)`. Configuration rejects zero heads, bad GQA divisibility,
odd RoPE dimensions, non-finite norm/RoPE parameters, invalid token IDs, dimension
overflow, scaled/interleaved/partial RoPE, enabled sliding windows, MLP biases,
unsupported activation, tensor parallel pretraining and incompatible architecture
or attention-bias declarations.

Both adapters map HF `model.layers.N.*` weights in `[out,in]` layout, without a
permutation. Qwen requires Q/K/V biases; the verified Llama adapter requires
bias-free projections. Tensor shapes are checked before allocating converted
weights, then checked again when building decoder blocks. Embeddings, final norm,
MLP projections and output shapes are validated, including quantized output heads.
Single-file Safetensors loading accepts mixed F32, F16 and BF16 tensors. A read-only
mapping avoids a complete checkpoint-byte copy while constructing FP32 host weights.

## Static execution plan

`DecoderPlan` describes the one supported schedule:

1. Embedding lookup.
2. Per layer: RMSNorm, Q/K/V projections, split-half RoPE, causal GQA, attention
   output projection, residual, RMSNorm, SiLU-gated MLP, output projection, residual.
3. Final RMSNorm and tied or independent logits projection.

The plan declares layer count, activation/attention operations, attention scale,
FP32 KV, output normalization/tying and available execution precisions. CPU
`Transformer` and `CudaTransformer` construct the same validated plan. Existing
statically written kernel sequences execute it; this is intentionally not an
operation interpreter or a per-kernel model trait. CPU INT8/INT4 keep using existing
`Linear` storage dispatch. Bias differences are represented in loaded weights.

`Transformer::forward_tokens` now checks context bounds and cache-layer count and
accepts a one-row decode workspace at nonzero positions. Existing full-model CPU
batching and CUDA packed/paged execution share the same family-independent weights.

## Precision and graph contracts

See [PRECISION_CONTRACTS.md](PRECISION_CONTRACTS.md). The Phase 15 BF16 attention
fallback is unchanged and now expressed by `ExecutionPrecision`: partitioned
attention needs contexts at least 128; BF16 batches above two stay on the
conservative path beyond 512. No BF16 split reductions were introduced.

`GraphModelIdentity` includes family, all execution dimensions, RoPE theta, norm
epsilon, tying, precision and a monotonic model allocation generation. A model's
address alone is insufficient because allocator reuse can recycle it.
`DecodeGraphKey` adds batch, paged FP32 KV path, block geometry, pool/workspace
addresses, conversion scratch, Lt dispatch, BF16 input reuse and attention dispatch.
An incompatible model shape rejects an existing batch workspace. Identity changes
drop the old graph and require eager warmup before recapture. Dynamic admission,
cancel, block growth and slot/table replacement still update metadata outside capture.

## Checkpoints and tokenizer

FRG1 version 1 remains readable. Qwen architecture ID 1 is unchanged; Llama adds ID
2. The JSON metadata preserves architecture and special token IDs and adds optional
padding ID and tokenizer SHA-256. Missing optional fields in old checkpoints are
accepted. Header and JSON architecture must agree. A present malformed output head
cannot be replaced silently by tied embeddings. See [CHECKPOINT_FORMAT.md](CHECKPOINT_FORMAT.md).

`ForgeTokenizer::for_model` checks vocabulary/special tokens and a recorded tokenizer
digest. `tokenizer_dir` supports the existing sibling layout, a same-stem directory,
and `FORGE_TOKENIZER_DIR`. New exports carry the tokenizer digest. Legacy checkpoints
without a digest retain their old sibling-directory compatibility, without claiming
cryptographic tokenizer identity. The API continues accepting caller-formatted
prompts; it does not add a chat-template interpreter.

## Validation scope

CPU references run in a separate process before Forge and use raw argmax. Qwen's
HF generation defaults apply a repetition penalty, so `generate(do_sample=False)`
alone is not an equivalent control. Hooks export each layer, full last-token logits,
cached decoding and greedy IDs for four prompts including 160-token contexts.
Real L4 tests cover both dtypes, paged/chunked batching, graphs, cancellation and
model lifetime replacement. No complete inference engine replaces Forge execution.

Phase 16 does not support arbitrary Llama variants, sharded Safetensors, scaled RoPE,
custom attention, MoE, multimodal models, CUDA quantization or multiple devices.
BF16 has bounded-workload numerical qualifications; universal greedy parity is not
claimed. Results and cost evidence are in [PHASE16_RESULTS.md](PHASE16_RESULTS.md).
