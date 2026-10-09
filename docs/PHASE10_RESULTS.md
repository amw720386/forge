# Phase 10 Results — Paged KV, Chunked Prefill & Scheduling

Status: **complete** — GPU-validated on NVIDIA L4 (Lightning Studio `forge-gpu`).

## What shipped

| Feature | Behavior |
|---|---|
| Paged GPU KV | Shared block pool; logical→physical tables; block size 16 (configurable) |
| Paged attention | `forge_cuda_paged_gqa_attn_f32` reads pool via block table (no full contiguous rebuild) |
| Contiguous oracle | `GpuKvCache` + `gqa_attn_f32` + `forward_tokens` / `generate_cuda` unchanged |
| Chunked prefill | Admit → `Prefilling`; chunks under `FORGE_PREFILL_CHUNK` / `FORGE_STEP_TOKEN_BUDGET` |
| Scheduling | Decode preferred; round-robin prefill; KV capacity-aware admit |
| Cancellation | Prefill or decode → blocks returned; double-free guarded |
| Metrics | `forge_batch_sched_stats` + `/metrics` KV/sched fields |

## Local (no GPU)

| Check | Result |
|---|---|
| `BlockAllocator` unit tests | 6/6 PASS |
| `cuda_paged_kv` (CPU paged≡contiguous attn, grow/clear) | 3/3 PASS |
| `cuda-stub` compile / link | PASS |

## GPU correctness (L4 session 2026-10-08)

| Suite | Result |
|---|---|
| `forge_cuda_tests` | PASS |
| `cuda_compile_check` | 2/2 PASS |
| `cuda_paged_kv` | 3/3 PASS |
| `cuda_transformer_parity` | 2/2 PASS |
| `cuda_batch_parity` (default chunk) | **7/7 PASS** |
| `cuda_batch_parity` (`FORGE_PREFILL_CHUNK=8`, budget=16) | **7/7 PASS** |
| Qwen generate / benches | Ran successfully |

Greedy batched outputs match sequential contiguous CUDA (synthetic). Shared GEMMs remain `m=B` (e.g. Q-proj `(8,896,896)`).

## Throughput vs Phase 9 baseline

Identical short prompts, `max_new=16`, Qwen2.5-0.5B FP32, L4. **Actual wall-clock measurements** (not synthetic).

| Batch | Phase 9 agg tok/s | Phase 10 agg tok/s | Δ |
|---:|---:|---:|---:|
| 1 | 22.0 | 19.7 | −10% |
| 2 | 30.2 | 26.2 | −13% |
| 4 | 37.1 | 31.3 | −16% |
| 8 | 41.6 | 34.4 | −17% |

Weights ≈ **1884.5 MiB**. VRAM free after B=8 short run ≈ **20.2 GiB**.

Interpretation: paged attention adds overhead; Phase 10’s primary win is **dynamic KV memory + chunked scheduling**, not raw decode throughput. Do not claim a throughput win over Phase 9 for identical short-prompt decode.

### Mixed-length prompts (Phase 10 only)

| B | Aggregate tok/s | Notes |
|---:|---:|---|
| 1 | 19.8 | |
| 2 | 25.0 | |
| 4 | 25.1 | |
| 8 | 29.6 | |

### Long-prompt stress (`FORGE_PREFILL_CHUNK=32`, ~200-word prompt)

Prefill-dominated; aggregate decode tok/s ≈ 1.5–2.0 with TTFT tens of seconds at B=8 — expected. Blocks reclaim to 0 used after completion.

### KV accounting (short B=8)

- Pool: 256 blocks × 16 tokens; peak used 16 during decode growth; **0 used after free**.
- Trace shows `prefill` then `decode` interleaving; `kv_free` / `kv_used` move as expected.

## Config knobs

| Env | Default | Role |
|---|---|---|
| `FORGE_KV_BLOCK_SIZE` | 16 | Tokens per physical block |
| `FORGE_KV_NUM_BLOCKS` | `B × ceil(max_seq/block)` | Pool capacity |
| `FORGE_PREFILL_CHUNK` | 128 | Max tokens per prefill chunk |
| `FORGE_STEP_TOKEN_BUDGET` | 256 | Prefill+decode tokens per `step` |

## Cloud budget ledger

| Session | GPU | Wall | Rate | Est. cost | Stop |
|---|---|---:|---:|---:|---|
| validate-phase10 2026-10-08 | L4 | ~7.1 min (planned 30m / $0.33) | $0.65/h | **≈ $0.08** | **verified** `running→stopped` |

Cumulative Phase 10 spend ≈ **$0.08** of **$8** cap. Remaining ≈ **$7.92**.

## Remaining bottlenecks

1. Naive paged GQA (per-key block lookup; no FlashAttention).
2. Block-table H2D on grow (synced per session; not every layer after first).
3. Host sampling + D2H logits every decode step (same as Phase 9).
4. Long-prompt TTFT dominated by chunked prefill compute, not paging metadata.

Phase 11 candidates: coalesced paged attn, CUDA Graphs, GPU weight quant — not required for Phase 10 DoD.
