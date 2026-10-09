# Phase 11 Results — GPU Performance Engineering

Status: **complete** — GPU-validated on NVIDIA L4 (Lightning Studio `forge-gpu`).

## What shipped

| Feature | Behavior |
|---|---|
| Optimized paged GQA | Warp (32 threads) + online softmax + block-cached phys ids; default `forge_cuda_paged_gqa_attn_f32` |
| Naive reference | `forge_cuda_paged_gqa_attn_naive_f32` / `FORGE_PAGED_ATTN_NAIVE=1` |
| cuBLAS linear | `forge_cuda_linear_f32` uses `cublasSgemm` (replaces naive tiled kernel) |
| GPU greedy | `forge_cuda_argmax_rows_f32`; batcher/generate skip full-vocab D2H unless `FORGE_CPU_GREEDY=1` |
| BF16 weights | Device BF16 linears+embed; FP32 acts/KV/norms; `GemmEx` BF16×BF16→FP32; `--dtype bf16` / device=2 / `FORGE_CUDA_DTYPE=bf16` |
| Contiguous oracle | `gqa_attn_f32` + `generate_cuda` retained |

## Local (no GPU)

| Check | Result |
|---|---|
| `cargo check --features cuda-stub` | PASS |
| `cuda_paged_kv` (cuda-stub) | 3/3 PASS |
| `go build ./cmd/forge-cloud` | PASS |
| `prepare-deploy` | PASS |

## GPU correctness (L4)

| Suite | Result |
|---|---|
| `forge_cuda_tests` (incl. opt vs naive paged, argmax, cuBLAS linear) | PASS |
| `cuda_compile_check` | 2/2 PASS |
| `cuda_paged_kv` | 3/3 PASS |
| `cuda_transformer_parity` | 2/2 PASS |
| `cuda_batch_parity` (default + chunked) | **7/7 + 7/7 PASS** |
| FP32 greedy CUDA == CPU (`bench_cuda_qwen`) | **TOKEN MATCH** |
| BF16 vs FP32 greedy text (Hello prompt, 16 tok) | **identical tokens** `BF16_TOKEN_TEXT_MATCH=1` |

### BF16 numerical notes

- Same prompt/tokens as FP32 for the short “Hello!” chat template (10 new tokens).
- Exact FP32 logit parity is **not** claimed; agreement on this prompt is legitimate but not a guarantee for all prompts.
- Weights on device: **942.4 MiB** BF16 vs **1884.6 MiB** FP32 (~2× reduction).
- Longer multi-prompt BF16 vs FP32 divergence survey left for Phase 12 if needed.

## Throughput (NVIDIA L4, short prompts, `max_new=16`)

### Historical baselines (unchanged)

| Batch | Phase 9 | Phase 10 |
|---:|---:|---:|
| 1 | 22.0 | 19.7 |
| 2 | 30.2 | 26.2 |
| 4 | 37.1 | 31.3 |
| 8 | 41.6 | 34.4 |

### Phase 11 measured (aggregate tok/s)

**Important:** Session “baseline” already includes **cuBLAS linear** (Phase 11 change). Pure Phase 10 used the old tiled linear kernel.

| Config | B=1 | B=2 | B=4 | B=8 |
|---|---:|---:|---:|---:|
| A. Naive attn + CPU greedy (+ cuBLAS) | 31.9 | 36.3 | 39.4 | **41.3** |
| B. Opt attn + CPU greedy | 99.0 | 159.3 | 239.5 | **330.9** |
| C. Opt attn + GPU argmax (FP32) | 101.0 | 164.4 | 253.8 | **357.8** |
| D. BF16 weights + opt + GPU argmax | 166.7 | 261.6 | 399.5 | **543.4** |

Mixed prompts (FP32 combined): B=8 ≈ **304.7** agg tok/s.  
Long-prompt stress (`FORGE_PREFILL_CHUNK=32`): B=8 ≈ **88.2** agg tok/s (prefill-dominated).

### Attribution (B=8 short)

| Step | Agg tok/s | Δ vs prior |
|---|---:|---|
| Phase 10 historical | 34.4 | — |
| + cuBLAS linear (A vs Phase 10) | 41.3 | +20% (recovers Phase 9) |
| + optimized paged attn (B vs A) | 330.9 | **~8×** (naive 1-thread triple-pass was dominant after GEMM fixed) |
| + GPU argmax (C vs B) | 357.8 | +8% |
| + BF16 (D vs C) | 543.4 | +52% (**precision change — label when comparing**) |

**Primary objective met:** Phase 9 B=8 (41.6) surpassed while retaining paged KV + chunked prefill. FP32 combined ≈ **8.6×** Phase 9; BF16 higher still.

### Single-request notes

- Contiguous `generate_cuda` path still uses legacy contiguous GQA (not the paged opt kernel): ~31 tok/s end-to-end including prefill for Hello (TOKEN MATCH vs CPU).
- Serving path is paged batcher (optimized).

### VRAM

| Mode | Weights | Free after B=8 short (approx) |
|---|---:|---:|
| FP32 | 1884.6 MiB | ~20.3 GiB / 22.6 GiB total |
| BF16 | 942.4 MiB | ~21.2 GiB free |

KV pool reclaim: used → 0 after completion in traces.

## Cloud budget ledger

| Session | GPU | Planned | Est. | Actual wall (approx) | Est. cost | Stop |
|---|---|---:|---:|---:|---:|---|
| validate-phase11 2026-10-08 | L4 | 30m / $0.33 | $0.65/h | ~5 min GPU work | **≈ $0.05** | **verified** `running→stopped` |
| BF16 bench focused | L4 | 15m / $0.16 | $0.65/h | ~2–3 min | **≈ $0.03** | **verified** |
| BF16 numerical (token compare) | L4 | 12m / $0.13 | $0.65/h | ~2 min | **≈ $0.02** | **verified** |

Cumulative Phase 11 spend ≈ **$0.10** of **$6** cap. Remaining ≈ **$5.90**.

Failed local deploy attempt before first start: **$0** (no GPU).

## Remaining bottlenecks / Phase 12 candidates

1. Contiguous `gqa_attn` (single-seq oracle) still naive — optional port of warp/online softmax.
2. Per-request attention launches (not fused across batch heads).
3. BF16 activation cast scratch before each `linear_bf16_w` (extra traffic).
4. Broader BF16 vs FP32 token-divergence survey on adversarial / long prompts.
5. CUDA Graphs / FlashAttention-class kernels if launch overhead still shows in nsys.
6. GPU INT8/INT4 weight kernels.

## Config knobs (new)

| Env / flag | Role |
|---|---|
| `FORGE_PAGED_ATTN_NAIVE=1` | Use Phase 10 reference paged kernel |
| `FORGE_CPU_GREEDY=1` | Force full logits D2H + CPU greedy |
| `FORGE_CUDA_DTYPE=bf16` | Bench examples: BF16 weight upload |
| `forge-server --device cuda --dtype bf16` | Server BF16 path (device code 2) |
