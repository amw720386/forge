# Phase 6 Results — Quantization, Benchmarks, Scheduling

## Reproducible methodology

| Item | Value |
|---|---|
| Host | Windows 10, Intel family 6 model 154, **12 logical CPUs** |
| Build | `cargo run --release` (MSVC, `/arch:AVX2`, FMA enabled) |
| Threads | `forge_kernels_num_threads()` → **12** (override: `FORGE_NUM_THREADS`) |
| Model | `Qwen2.5-0.5B-Instruct` (BF16 weights → FP32 load, then optional in-memory quant) |
| Profiler | `profile_qwen <dir> <prefill_len> <decode_n>` — reports load / warm-up / prefill / decode separately |
| Bench | `bench_phase6` — FP32 / INT8 / INT4 + batch B=1..8; `FORGE_BENCH_PREFILL`, `FORGE_BENCH_DECODE` |

Always compare backends with **identical prompts, token counts, weights source, and hardware**.

## Why 0.44 vs 3.43 vs later numbers?

| Figure | Source | What it actually measured |
|---|---|---|
| **~0.44 tok/s** | Early Phase 3 (`PHASE3_RESULTS`) | Unoptimized single-threaded path; often **cold** (load/caches) and/or longer context folded into a coarse wall-clock |
| **3.43 tok/s** | [PHASE5_BASELINE](PHASE5_BASELINE.md) | Steady-state **decode only** after warm-up; prefill_len=64; **single-threaded** linear; AVX path incomplete on MSVC |
| **7.77 tok/s** | [PHASE5_RESULTS](PHASE5_RESULTS.md) | + persistent thread pool + MT MLP/lm_head + zero-copy |
| **12.3 tok/s** | Phase 6 FP32 `profile_qwen 64 16` | Same method as Phase 5 baseline, plus **MSVC FMA enable** in `simd.hpp` (`__AVX2__` ⇒ FMA on MSVC) |

**Takeaway:** 0.44 was not a comparable steady-state decode baseline. Prefer `profile_qwen` / `bench_phase6` warm decode ms/tok.

---

## Single-request results (`bench_phase6`, short chat prompt ≈15 tok prefill, 32 decode)

| Backend | Weight RAM | Warm prefill | Decode | vs FP32 decode |
|---|---|---|---|---|
| **FP32** | 2403.9 MiB | 33.2 tok/s | **11.1 tok/s** (90 ms/tok) | 1.0× |
| **INT8 g=64** | **1020.1 MiB** (−58%) | 56.2 tok/s | **23.0 tok/s** (43.5 ms/tok) | **2.1×** |
| INT4 g=32 | 814.0 MiB | 8.1 tok/s | 4.2 tok/s | 0.38× (slower) |
| INT4 g=64 | 784.6 MiB | 7.3 tok/s | 4.3 tok/s | 0.39× |
| INT4 g=128 | 769.9 MiB | 7.8 tok/s | 4.5 tok/s | 0.40× |

Apples-to-apples `profile_qwen 64 16` FP32 decode: **12.3 tok/s** (81 ms/tok), prefill **37.2 tok/s**.

### INT4 note (original Phase 6)

Those INT4 numbers used an unpack-to-`tmp[]` then INT8-AVX path. Prefer the **re-measurement below** after the fused INT4 kernel.

---

## INT4 investigation (safe re-bench, separate processes)

Methodology: `forge-int4-diag` / `scripts/bench_int4_compare.ps1` — **one backend per process**, same prompt, `group=64`, `prefill=64`, `decode=32`, `B=1`, `max_seq=512`, conservative `FORGE_BENCH_MEM_CAP_MIB` + available-RAM abort. Host: Windows, **15.7 GiB** RAM, ~3 GiB free during runs, **12** threads.

### Kernel microbench (synthetic Qwen shapes, no model load)

| Shape | Before (unpack→tmp) INT4/INT8 | After (fused) INT4/INT8 | Δfaults |
|---|---|---|---|
| q_proj 1×896×896 | **11.05×** | **2.01×** | ~0–1 |
| gate_proj 1×4864×896 | 9.65× | **1.03×** | ~0 |
| down_proj 1×896×4864 | 7.58× | **0.83×** (INT4 faster) | 0 |
| lm_head (n capped 24576) | 7.36× | **0.99×** | 0 |

### Full-model decode (separate processes, 2 reps)

| Backend | Weights | Warm prefill | Decode mean | Steady RSS | Δfaults during decode |
|---|---|---|---|---|---|
| INT8 g=64 (`.forge`) | 1020 MiB | ~52–68 tok/s | **25.7 tok/s** | ~1070 MiB | **1–2** |
| INT4 g=64 (in-process quant) **before** fused kernel | 785 MiB | ~7.5–8.4 tok/s | **4.4 tok/s** | ~836 MiB | **0–1** |
| INT4 g=64 **after** fused kernel | 785 MiB | ~35–40 tok/s | **20.1 tok/s** | ~836 MiB | **0–5** |

**Verdict:** The Phase 6 INT4 slowdown was **kernel inefficiency**, not OS paging. Decode page-fault deltas were negligible for both INT8 and INT4 while INT4 was 5× slower end-to-end. Fused in-register nibble unpack recovered **~4.6×** decode throughput (4.4 → 20.1 tok/s), approaching INT8.

FP32 was skipped in this re-bench because available physical RAM was only ~3 GiB (loading FP32 risks thrash). Peak commit during INT4 load+quant still reaches ~2.8 GiB — prefer offline `.forge` loads when free RAM is tight.

### Commands

```powershell
.\scripts\bench_int4_compare.ps1 -SkipFp32
# or micro only:
cargo run --release --bin forge-int4-diag -- --model ../models/Qwen2.5-0.5B-Instruct --backend int8 --group 64 --skip-model --min-avail-mib 1500
```

---

## Correctness

### Synthetic unit tests (`cargo test --release`)

- INT8/INT4 pack round-trip
- Quantized continuous batch ↔ sequential greedy parity
- Bounded logit error vs FP32 twin

### Qwen first-token agreement (`bench_phase6`, INT8 g=64, 4 prompts)

| Prompt | FP32 tok | INT8 tok | Match | MAD (logits) |
|---|---|---|---|---|
| 0 | 9707 | 9707 | yes | 0.055 |
| 1 | 17 | 17 | yes | 0.078 |
| 2 | 6033 | 6033 | yes | 0.060 |
| 3 | 59039 | 59039 | yes | 0.052 |

### Multi-token chat smoke (`generate_qwen`, Hello)

| Backend | Tokens (greedy) | Text |
|---|---|---|
| FP32 | `[9707, 0, 2585, 646, 358, 7789, 498, 3351, 30, 151645]` | Hello! How can I assist you today? |
| INT8 g=64 | **same as FP32** | same |
| INT4 g=64 | `[9707, 0, 2585, 646, 358, 1492, 498, 3351, 30, 151645]` | Hello! How can I **help** you today? |

INT4 diverges at token index 5 (`1492` vs `7789`) — expected under coarser quantization; output remains coherent.

Norms / Softmax / RoPE / attention stay FP32 for all backends.

---

## Continuous batching (INT8 g=64, max_tokens=16)

| B | Admit (prefill all) | Decode wall | Aggregate tok/s | Step p50 / p95 |
|---|---|---|---|---|
| 1 | 234 ms | 294 ms | 34.0 | 32.9 / 36.2 |
| 2 | 504 ms | 510 ms | 37.2 | 57.8 / 65.5 |
| 4 | 1015 ms | 1068 ms | 34.6 | 91.9 / 100.6 |
| 8 | 1841 ms | 2002 ms | **39.5** | 143.6 / 159.6 |

Shared `m=B` linears; attention still per-sequence. Prefills remain serialized on admit (fairness: FIFO admit into free slots).

### Scheduling / memory improvements this phase

- Removed **per-slot** `ForwardScratch(max_seq)` — one shared prefill scratch + shared `BatchScratch(max_batch)`.
- KV caches remain per-request; cancel frees one slot only.

---

## Remaining bottlenecks

1. **lm_head + MLP** still dominate decode even under INT8 / fused INT4.
2. INT4 is now competitive with INT8 on this CPU; further gains need VNNI/`dpbusd`-style integer dot products or better cache blocking.
3. Prefill is still **one request at a time** (batched prefill of equal lengths not yet).
4. Offline quantized checkpoint format delivered in Phase 7 — prefer `.forge` loads to avoid FP32+quant peaks.

## Commands

```powershell
cargo test --release
cargo run --release --example profile_qwen -- ../models/Qwen2.5-0.5B-Instruct 64 16
cargo run --release --example bench_phase6 -- ../models/Qwen2.5-0.5B-Instruct
cargo run --release --example generate_qwen -- ../models/Qwen2.5-0.5B-Instruct --bits 8 --group 64
```
