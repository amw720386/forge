# Phase 11 Design — GPU Performance Engineering

Status: **complete** — see [PHASE11_RESULTS.md](PHASE11_RESULTS.md)

## Goals

1. Recover or exceed Phase 9 B=8 aggregate throughput while retaining Phase 10 paged KV + chunked prefill.
2. Add BF16 weight storage / mixed-precision GEMM on NVIDIA L4.
3. Move greedy sampling onto the GPU (argmax only; transfer token IDs).
4. Cut redundant launch / sync / D2H overhead where profiling justifies it.

Primary stretch target (Phase 9 L4, short prompts, `max_new=16`):

| Batch | Phase 9 | Phase 10 | Phase 11 target |
|---:|---:|---:|---|
| 8 | 41.6 tok/s | 34.4 tok/s | ≥ 41.6 tok/s (FP32 or labeled BF16) |

Correctness and benchmark validity take priority over hitting the stretch number.

## Profiling plan (before/after)

Measure on identical L4 workloads (short + longer prompts, B=1/2/4/8):

| Metric | Method |
|---|---|
| Prefill / decode wall | host clocks + `cudaDeviceSynchronize` boundaries |
| Attention / linear / RMSNorm / SiLU | CUDA events around kernel groups |
| Logits D2H | event around `memcpy_d2h` |
| CPU sampling | host clock around `greedy` |
| Active batch / VRAM | existing batch stats + `mem_info` |

Baseline recorded in [PHASE11_RESULTS.md](PHASE11_RESULTS.md) before claiming wins.

## Optimized paged attention

### Phase 10 baseline (kept as reference)

`paged_gqa_attn_kernel` (naive):

- Grid `(n_heads, t_q)`, **1 thread** per (head, query).
- Three full passes over keys (max → sum → V), recomputing Q·K each time.
- Per-key `block_table[kj / block_size]` lookup.

Exported as `forge_cuda_paged_gqa_attn_naive_f32` for numerical reference.

### Phase 11 default

`paged_gqa_attn_opt_kernel` → called by `forge_cuda_paged_gqa_attn_f32`:

1. **One CUDA block per (head, query)** with a full warp (32 threads).
2. **Online softmax** (FlashAttention-style): single pass over K/V updating `(m, ℓ, acc)`.
3. **Block-oriented KV walk**: load physical id once per logical block; iterate rows inside the block.
4. **Warp reduction** for Q·K; threads partition `head_dim` for V accumulation (Qwen `d=64` → 2 elems/thread).
5. Vectorized `float2`/`float4` loads when alignment allows.

Preserves: GQA mapping, causal mask, RoPE positions (caller-applied), dynamic block growth, no contiguous rebuild.

## BF16 mixed precision

| Tensor | Storage | Compute |
|---|---|---|
| Linear / embed / lm_head weights | BF16 on device | cuBLAS `GemmEx` BF16×FP32→FP32 (FP32 accumulate) |
| Activations, norms, RoPE, attention, KV | FP32 | FP32 |
| Softmax / RMSNorm stats | FP32 | FP32 |

CLI: `forge-server --device cuda --dtype bf16` (FP32 remains default / correctness reference).

No persistent FP32 duplicate of BF16 weights on GPU. Host may still hold FP32 after safetensors promote; upload converts once.

## GPU greedy sampling

- Kernel: row-wise argmax over `[rows, vocab]` logits on device.
- Tie-break: **lowest index wins** (same as CPU `greedy`).
- Hot path transfers `rows × sizeof(int32)` only.
- Temperature / top-k stay on CPU (full logits D2H only when needed).
- Env escape hatch: `FORGE_CPU_GREEDY=1` forces host sampling for A/B.

## Launch / sync reductions

1. Replace naive `linear_f32` tiled kernel with **cuBLAS** GEMM (same path BF16 uses).
2. Fuse argmax into decode so greedy skips full-vocab D2H.
3. Avoid extra `synchronize` between layer ops (rely on stream ordering; sync only at sample / metric boundaries).
4. Persistent scratch already reused (Phase 9/10); keep that.

CUDA Graphs optional; only if event profiling still shows launch-bound decode after the above.

## Compatibility

- Contiguous GQA + `generate_cuda` remain oracles.
- Chunked prefill, admit/cancel, block reclaim unchanged.
- CPU FP32 / INT8 / INT4 paths untouched.

## Validation

- Naive vs opt paged attn (GPU) + contiguous / CPU references.
- BF16 vs FP32: max abs / relative logit error, token agreement (document legitimate divergences).
- GPU argmax ≡ CPU greedy on identical logits.
- `cuda_batch_parity`, chunked stress, B=1..8 benches, mixed lengths.
- Budget: **$6** Phase 11 / **$2** per session / ≤45 min / single L4 `forge-gpu`.
