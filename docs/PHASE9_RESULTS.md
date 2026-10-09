# Phase 9 Results — GPU Continuous Batching

**Status: complete — validated on NVIDIA L4** (2026-10-08).

Genuine concurrent GPU inference with shared batched CUDA linears (`m=B`) was demonstrated. Per-request KV isolation, cancel/slot-reuse, and B=1..8 throughput were measured on Lightning Studio `forge-gpu`.

## Local deploy proof (before GPU)

`go test ./internal/cloud -run Phase9Deploy` (×2 passes):

| Check | Result |
|---|---|
| Build exact validate-phase9 tar | PASS |
| Extract + `VerifyStagingAgainstManifest` | PASS |
| Exact `scripts/verify_forge_deploy.sh` | PASS (`DEPLOY_VERIFY_OK`, 113 files) |
| Content hash deterministic across 2 builds | PASS (`c229fabb…` both passes) |
| Staged `*.sh` are LF-only | PASS |
| Post-verify chmod does not change hashes | PASS |
| Injecting CRLF into a hashed script breaks verify | PASS |
| `copyFile` normalizes CRLF→LF for `.sh` | PASS |

## GPU session

| Field | Value |
|---|---|
| Studio | `forge-gpu` (existing; **1 start**) |
| GPU | NVIDIA L4 |
| Deploy | `20261008T141349Z-c229fabb6557` |
| Estimate | ≈$0.33 / 30m @ ~$0.65/h; cap $2 |
| Wall time | ≈**7.6 min** (start→stop) |
| Estimated spend | ≈**$0.08** |
| Shutdown | **Verified** `before=running after=stopped` |

In-session recoverable fix: `bench_cuda_batch.rs` missing `tok` after a local cleanup (compile error). Patched on the live Studio + refreshed deploy hashes; also taught verify to ignore mid-session `runtime/target/` artifacts so retries are not poisoned.

## Correctness (L4)

### `cuda_batch_parity` — **7/7 PASS**

| Test | Result |
|---|---|
| `batch_size_one_matches_sequential` | PASS (B=1 == single-request CUDA) |
| `batched_greedy_matches_sequential` | PASS (multi-slot greedy == sequential) |
| `cancel_does_not_corrupt_sibling` | PASS |
| `slot_reuse_after_completion` | PASS |
| `cancel_before_step_and_invalid_id` | PASS |
| `cuda_batch_types_link` | PASS |
| `cpu_batch_regression_still_passes` | PASS |

Also: `cuda_transformer_parity` 2/2 PASS; `forge_cuda_tests` PASS; Qwen greedy `TOKEN MATCH: greedy CUDA == CPU`.

### Shared GEMM shapes (`FORGE_CUDA_BATCH_TRACE=1`)

Instrument lines show real batched linears, not a loop of `m=1`:

```text
[forge cuda batch] B=8 q_proj=(8,896,896) lm_head=(8,151936,896) steps=…
[forge cuda batch] B=4 q_proj=(4,896,896) lm_head=(4,151936,896) steps=…
[forge cuda batch] B=2 q_proj=(2,896,896) lm_head=(2,151936,896) steps=…
[forge cuda batch] B=1 q_proj=(1,896,896) lm_head=(1,151936,896) steps=…
```

Q-proj `m=B`, `n=896`, `k=896`; lm_head `m=B`, `n=151936`, `k=896`.

## Throughput (identical prompts, max_new=16)

Phase 8B single-request baseline (same session): prefill ≈204 tok/s; generate ≈19.7 tok/s end-to-end.

| Config | Aggregate tok/s | Per-req avg tok/s | Avg active B | TTFT proxy | VRAM free (MiB) |
|---|---|---|---|---|---|
| Sequential B=1 ×4 | 19.80 | 4.95 | 1 | — | — |
| Continuous B=1 | 21.96 | 21.96 | 1.00 | 0.045 s | 20279 |
| Continuous B=2 | 30.16 | 15.08 | 2.00 | 0.062 s | 20267 |
| Continuous B=4 | 37.11 | 9.28 | 4.00 | 0.096 s | 20239 |
| Continuous B=8 | 41.59 | 5.20 | 8.00 | 0.166 s | 20189 |

Weights ≈**1884.5 MiB** FP32. Total VRAM ≈22563 MiB.

### Mixed-length prompts (`FORGE_MIXED_PROMPTS=1`)

| Config | Aggregate tok/s | Avg active B | Notes |
|---|---|---|---|
| B=1 | 21.95 | 1.00 | |
| B=2 | 29.28 | 1.60 | occupancy drops as short reqs finish |
| B=4 | 28.67 | 2.50 | mixed lengths reduce steady occupancy |
| B=8 | 34.46 | 5.19 | still above sequential aggregate |

## Bottlenecks (measured / observed)

- **Serialized prefill** on admit (B=8 admit ≈0.39–0.41 s) stalls decode.
- **Per-request attention** still loops B times (linears are shared; attn is not).
- **D2H logits + CPU sampling** every step.
- Aggregate scales with B; per-request tok/s falls as expected when sharing an L4.
- Mixed lengths lower average occupancy vs identical prompts.

## Phase 9 completion checklist

1. Shared CUDA batched linears with `m=B` — **yes** (trace)
2. Independent GPU KV per request — **yes** (parity + cancel tests)
3. Join/leave between decode iterations — **yes** (slot reuse / occupancy drop)
4. Cancel/complete release resources — **yes**
5. Go HTTP/SSE path unchanged (same FFI) — **yes** (device-agnostic scheduler)
6. B=1,2,4,8 tested — **yes**
7. Deterministic tokens match single-request — **yes**
8. Performance + memory recorded — **yes**
9. CPU path still functional — **yes**
10. Studio shutdown verified — **yes**
