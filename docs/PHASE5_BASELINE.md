# Phase 5 Baseline (pre-optimization)

Host: Windows, 12 logical CPUs (Intel family 6 model 154). Forge FP32, AVX2/FMA single-threaded linear.

Command:

```text
cargo run --release --example profile_qwen -- ../models/Qwen2.5-0.5B-Instruct 64 16
```

## End-to-end

| Stage | Result |
|---|---|
| Model load | 3749 ms |
| Prefill 64 tok | 13580 ms (**4.71 tok/s**) |
| Decode 16 tok | mean **291.3 ms/tok** (**3.43 tok/s**), p50=284, p95=404 |

(Earlier ~0.44 tok/s was likely cold/longer-context; this is the measured steady-state baseline.)

## Microbench: decode linears (m=1)

| Op | Shape (m,n,k) | ms | GFLOP/s |
|---|---|---|---|
| q_proj | 1×896×896 | 0.36 | 4.41 |
| k_proj | 1×128×896 | 0.05 | 4.43 |
| gate_proj | 1×4864×896 | 2.25 | 3.88 |
| down_proj | 1×896×4864 | 2.37 | 3.68 |
| **lm_head** | **1×151936×896** | **88.8** | 3.07 |

## Bottleneck ranking (decode)

1. **lm_head** ≈ 89 ms/tok (~30%)
2. **MLP × 24** ≈ 24×(gate+up+down) ≈ 163 ms/tok (~56%)
3. Other projections / attn / copies — remainder

## Issues observed in code

- Single-threaded linear despite 12 logical CPUs
- Decoder `prefix()` / `project_into()` allocate+copy every step
- Attention copies full K/V cache into scratch each step

## Optimization plan (evidence order)

1. Multithreaded `forge_kernels_linear_f32` (outer-n parallel)
2. Zero-copy prefix views / write QKV directly into scratch
3. Decode attention without K/V buffer copies
4. Re-measure; only then continuous batching

**Status:** completed — see [PHASE5_RESULTS.md](PHASE5_RESULTS.md).
