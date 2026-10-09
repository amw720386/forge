# Phase 5 Results — Performance & Continuous Batching

Host: Windows, 12 logical CPUs (Intel family 6 model 154). Forge FP32, AVX2/FMA.

Command:

```text
cargo run --release --example profile_qwen -- ../models/Qwen2.5-0.5B-Instruct 64 16
```

## Baseline → optimized (single-request)

| Stage | Baseline | Optimized | Speedup |
|---|---|---|---|
| Prefill 64 tok | 4.71 tok/s | **16.58 tok/s** | **3.5×** |
| Decode (steady) | 3.43 tok/s (291 ms/tok) | **7.77 tok/s (129 ms/tok)** | **2.3×** |
| lm_head m=1 | 88.8 ms | **22.2 ms** | **4.0×** |
| gate_proj m=1 | 2.25 ms | **0.88 ms** | **2.6×** |
| down_proj m=1 | 2.37 ms | **0.74 ms** | **3.2×** |

Warm-up / load are reported separately by `profile_qwen` and are excluded from tok/s above.

### What helped (measured)

1. **Persistent thread pool** + parallel `forge_kernels_linear_f32` for ops with ≥5e6 FLOPs (MLP + lm_head; tiny Q/K/V stay serial).
2. **Zero-copy** decoder/attention paths (no per-step `prefix()` / full KV copies).
3. **Kernel mul** for SiLU×up instead of a scalar Rust loop.
4. No temporary `to_vec` of the last hidden row before lm_head.

### What did not / deferred

- Aggressive MT on small projections (q_proj) added overhead — left serial.
- Weight-only quantization not applied this phase (would need separate numerical validation).
- Fused attention kernels not yet; attention remains a secondary cost vs MLP+lm_head.

## Correctness

Greedy Qwen2.5-0.5B chat smoke (`generate_qwen`) still emits HF-matching ids:

```text
[9707, 0, 1084, 594, 6419, 311, 3367, 498, 13, 151645]
Hello! It's nice to meet you.
```

Continuous-batching unit tests (`cargo test batch::`):

- `concurrent_greedy_matches_sequential` — 3 concurrent synthetic requests match solo greedy token streams.
- `cancel_does_not_corrupt_sibling` — cancel mid-batch leaves the survivor bit-exact vs sequential.

## Continuous batching

Rust `ContinuousBatcher`:

- Independent per-request KV caches + scratch.
- Shared decode step packs `B` rows and runs **batched** linear/MLP/lm_head (`m=B`).
- Attention stays per-sequence (different cache lengths) — linears dominate.
- Admit / cancel / free-slot backpressure between steps.
- Not “N parallel single-request loops”.

Go scheduler (`api/internal/scheduler`) now admits into free batch slots and calls `BatchStep` for shared decode. FFI: `forge_batch_admit` / `forge_batch_step` / `forge_batch_cancel` (`FORGE_MAX_BATCH`, default 8).

## Remaining bottlenecks

1. Decode still ~129 ms/tok on this CPU — MLP×24 + lm_head dominate even when parallelized.
2. Prefill is single-sequence; batched prefill of equal-length prompts is future work.
3. Per-slot `ForwardScratch` sized to `max_seq` is memory-heavy at large `FORGE_MAX_BATCH`.
4. Quantization (INT8/INT4 weights) is the next likely win for lm_head/MLP bandwidth.

## Environment notes

- `FORGE_NUM_THREADS` — linear pool size (default = hardware concurrency).
- `FORGE_MAX_BATCH` — continuous batcher capacity (default 8).
