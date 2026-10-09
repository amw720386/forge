# Phase 9 Design — GPU Continuous Batching

## Goal

Run multiple concurrent requests through **shared CUDA batched linear ops** (`m=B`) while each request keeps an independent device KV cache. Preserve Phase 8B single-request CUDA as the reference forward path.

## Current state (Phase 8B)

| Component | Behavior |
|---|---|
| `CudaTransformer::forward_tokens` | Single sequence; `m=t` linears; one `LayerGpuKvCaches` |
| `CudaBatcher` | API-compatible admit/step/cancel but **`max_batch=1`** |
| CUDA kernels | `linear_f32` / norms / elem already accept `m>1` |
| `gqa_attn_f32` / `kv_append` | One contiguous KV prefix (per call) |
| Go scheduler | Device-agnostic `BatchStep` loop |
| `forge-server --device cuda` | Clamps queue capacity to 1 |

CPU `ContinuousBatcher` is the algorithmic template: multi-slot state machine, serialized prefill, shared decode with per-seq attention.

## Design

### Slot model (mirrors CPU)

```text
CudaBatcher {
  model: CudaTransformer,           // shared weights (device)
  slots: Vec<CudaSlot>,             // max_batch
  batch_scratch: CudaBatchScratch,  // [B, …] activations + [B, vocab] logits
  prefill_scratch: CudaScratch,     // serialized prefill (max_seq)
  stats: CudaBatchStats,            // last B, GEMM shapes, occupancy
}

CudaSlot {
  id, status ∈ {Free, Ready, Decoding},
  caches: LayerGpuKvCaches,         // private GPU KV
  tokens, params, generated, rng,
  last_logits (host), pending_decode,
}
```

### Decode step (iteration-level batching)

1. Sample/emit for all `Ready` slots → stream tokens; mark `Decoding` or free on EOS/`max_tokens`.
2. Collect pending decode tokens for active slots → batch size `B_active`.
3. **`forward_batch_decode`**:
   - Embed `B` tokens → `x[B, H]`
   - Per layer: RMSNorm (`rows=B`) → **batched** Q/K/V (`m=B`) → **per-slot** RoPE + KV append + GQA → pack `ctx` → **batched** O / MLP / residual
   - Final RMSNorm + **batched** `lm_head` (`m=B`, `n=vocab`)
4. D2H logits rows → host `last_logits`; slots → `Ready`.
5. Admit replacements on free slots between steps (Go scheduler already does this).

### Prefill

Serialized, same as CPU: newly admitted requests run `forward_tokens` on `prefill_scratch` into that slot’s KV. Active siblings keep their GPU KV untouched.

**Decode stall:** while a new request prefills, the batcher does not run a shared decode. Documented; chunked prefill is Phase 10.

### Attention strategy

Dense linears must be shared. Attention stays **per-request** because KV lengths differ. Implementation uses single-row device temps + existing `rope_f32` / `kv_append_f32` / `gqa_attn_f32` (no new attention ABI).

### Memory

- Weights: ~1.9 GiB FP32 (shared).
- Per slot KV: `L × 2 × max_seq × kv_width × 4` bytes.
- Batch scratch: `O(B × (H + I + vocab))`.
- Cap `B` via `FORGE_MAX_BATCH` and soft GPU budget env (`FORGE_CUDA_MAX_BATCH` optional override).

Simple contiguous per-request KV (no paging). Phase 10: paged KV + chunked prefill.

### FFI / Go

Reuse existing `forge_batch_*` ABI. `CudaBatcher::max_batch()` returns the configured capacity (≥1). Remove forge-server queue clamp when CUDA capacity > 1. CPU and CUDA coexist via `--device`.

### Instrumentation

`CudaBatchStats` records last active `B`, linear `(m,n,k)` for Q-proj and lm_head, and sync wall time. Exposed via env `FORGE_CUDA_BATCH_TRACE=1` and bench helpers.

### Correctness

- `B=1` matches `forward_tokens` / single-request generate.
- Batched greedy matches sequential per-prompt CUDA (and CPU FP32 within known tolerances).
- Cancel / slot reuse / isolation tests (synthetic where possible; CUDA on NVIDIA).

### Out of scope

Paged attention, chunked prefill scheduling, GPU quant, multi-GPU.
