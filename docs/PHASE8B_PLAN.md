# Phase 8B Plan — Full Qwen2.5 CUDA Inference

## Goal

Run the complete Qwen2.5-0.5B decoder forward pass and greedy generation on NVIDIA GPUs using Forge’s CUDA backend, while keeping CPU/INT8/INT4 paths unchanged on Windows.

## Current state (post–8A)

| Layer | CPU | CUDA (8A) |
|-------|-----|-----------|
| Memory H2D/D2H | host | device |
| GEMM / matvec | `forge_kernels_*` | cuBLAS |
| RMSNorm, SiLU, add, mul | yes | yes |
| Softmax | yes | yes (device) |
| RoPE | yes | **8B** |
| Linear (batched) | yes | **8B** |
| Embedding gather | CPU slice | **8B** |
| GQA attention | Rust loops | **8B kernel** |
| Full decoder / model | `Transformer` | **`CudaTransformer`** |
| KV cache | host `Tensor` | **device buffers** |
| Generation | `ContinuousBatcher` | **`CudaBatcher` (single-flight)** |
| Go server | CPU default | **`--device cuda`** |

## Architecture

```
Tokenize (CPU) → token IDs H2D
       ↓
CudaTransformer::forward_tokens (GPU)
  embed → [L × decoder] → final RMSNorm → lm_head → logits (device)
       ↓
D2H last logits → greedy (CPU)
       ↓
decode loop (1 token / step, KV on GPU)
```

**Weight ownership:** One H2D upload at load. Weights stay in `Buffer` until `CudaTransformer` drop. Tied embeddings share one device embed buffer for `lm_head` GEMV/GEMM.

**Layout:** Forge row-major `[rows, cols]`. Linear weights stored on GPU as:

- `[out, in]` for `matvec` (decode, `m=1`)
- `[in, out]` transpose (`weight_gemm`) for prefill `matmul`

**Attention:** Custom GQA causal kernel (`forge_cuda_gqa_attn_f32`) — RoPE applied before KV append (matches CPU `attend`).

## Implementation steps

1. **CUDA C API** — `linear`, `rope`, `embedding`, `gqa_attn`, `copy_rows`, `add_rows`
2. **`ComputeBackend` extensions** — wire Rust `CudaBackend`
3. **`runtime/src/cuda_model/`** — weights, KV, `CudaTransformer`, `generate_cuda`, `CudaBatcher`
4. **FFI** — `forge_runtime_load_ex(..., device)`; device `1` = CUDA when built with `--features cuda`
5. **Go** — `forge-server --device cuda` (queue clamped to 1)
6. **Tests** — `cuda_transformer_parity` (skip without GPU)
7. **Bench / gen examples** — `bench_cuda_qwen`, `generate_cuda_qwen`
8. **Remote script** — `scripts/remote_cuda_phase8b.sh` (**no auto-provision**)
9. **Docs** — CUDA_ARCHITECTURE, PHASE8B_RESULTS (pending GPU), ROADMAP, README

## CPU-only operations moving to GPU

| Operation | CPU today | GPU 8B |
|-----------|-----------|--------|
| Embed lookup | `copy_from_slice` | `forge_cuda_embedding_f32` |
| Q/K/V/O/MLP linear | `forge_kernels_linear_f32` | `forge_cuda_linear_f32` / matmul |
| RoPE | `forge_kernels_rope_f32` | `forge_cuda_rope_f32` |
| KV write | `append_rows` host | device memcpy + length |
| Attention matmuls | nested loops | `forge_cuda_gqa_attn_f32` |
| Residual add | `add_rows` | `forge_cuda_add_rows_f32` |
| SwiGLU | SiLU + mul kernels | same CUDA ops |
| Logits | CPU matvec | GPU matvec/matmul + D2H |

## Validation (requires approved Lightning session)

1. Build `forge-runtime` with `--features cuda`
2. `forge_cuda_tests` + new parity tests
3. `forge-cuda-gen` or example on Qwen2.5-0.5B — greedy vs CPU FP32
4. Prefill/decode benchmarks vs CPU FP32
5. `nvidia-smi` / `forge_cuda_mem_info` after load and after drop

## Out of scope (8B)

- GPU continuous batching (`max_batch` forced to 1)
- Quantized GPU weights (FP32 only on device)
- Paged KV
- BF16 compute (BF16 weights may be cast to FP32 at load on CPU before upload)

## Risk / mitigations

| Risk | Mitigation |
|------|------------|
| cuBLAS layout | Separate `weight_gemm` transpose at upload; unit tests vs CPU |
| VRAM (~2GB+ FP32 0.5B) | Report mem at load; OOM errors propagated |
| Status / billing | N/A for 8B local dev; remote script reuses 8A stop checks |
| Numerical drift | Document atol/rtol; compare greedy token ids |
