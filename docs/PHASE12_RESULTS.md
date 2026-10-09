# Phase 12 Results — Competitive Benchmarking

Status: **complete** (with disclosed gaps). GPU work on NVIDIA L4, Studio `forge-gpu`.

## Methodology

Audited harness: `runtime/examples/bench_phase12.rs`  
Rules: [BENCHMARK_METHODOLOGY.md](BENCHMARK_METHODOLOGY.md)  
Design: [PHASE12_DESIGN.md](PHASE12_DESIGN.md)

Phase 11 throughput tables remain **historical** and are not rewritten. Phase 12 numbers below are independently remasured with warm-up, median of trials, `fixed_len` (EOS disabled), and end-of-region `synchronize()`.

## Forge remasured (short Hello, max_new=16, fixed_len)

| Dtype | B=1 | B=2 | B=4 | B=8 | Weights |
|---|---:|---:|---:|---:|---:|
| FP32 aggregate tok/s | 100.8 | 169.3 | 269.8 | **398.8** | 1884.6 MiB |
| BF16 aggregate tok/s | 166.9 | 266.8 | 418.9 | **585.3** | 942.4 MiB |
| FP32 decode≈ | 108.5 | 191.5 | 329.7 | 543.7 | |
| BF16 decode≈ | 179.0 | 297.7 | 498.8 | 752.5 | |
| BF16 TTFT p50 (s) | 0.012 | 0.019 | 0.033 | 0.059 | |

vs Phase 11 historical (same GPU, different harness): BF16 B=1 matches (~167); B=8 is modestly higher (585 vs 543) under `fixed_len` + warm-up/median. Do not treat the delta as a Phase 12 kernel win.

### Additional Forge workloads (BF16)

| Workload | B | max_new | Agg tok/s | TTFT p50 |
|---|---:|---:|---:|---:|
| medium (~193–256 tok prompt) | 1 | 64 | 112.7 | 0.045 s |
| medium | 4 | 64 | 138.2 | 0.165 s |
| long (~1024 tok prompt) | 1 | 64 | 52.9 | 0.209 s |
| long | 2 | 64 | 39.9 | 0.414 s |
| mixed stagger | 8 | 32 | 472.3 | 0.118 s (occ≈6.9) |

## Controlled experiments (FP32, B=8, short)

| Config | Agg tok/s |
|---|---:|
| Naive paged attn + CPU greedy | 35.3 |
| Opt paged attn + CPU greedy | 373.1 |
| Opt paged attn + GPU argmax | 398.8 |

**Ranking (measured):** optimized paged attention dominates (~10× vs naive after cuBLAS). GPU argmax adds ~7% on this workload.

## BF16 correctness survey

Reference: CUDA FP32 greedy. Candidate: CUDA BF16. `FORGE_CPU_GREEDY=1` for logit D2H.

| Metric | Value |
|---|---|
| Sequence agreement | **6/7 (85.7%)** |
| Token agreement | **85/103 (82.5%)** |
| Batch B=1/2/4/8 (hello) | **all agree** |
| Chunked prefill (chunk=32, hello) | **agree** |

Sole divergence: **medium** prompt (~193 tok), first diverge at generated index 14.  
At shared prefix: `max_abs_logit_err≈0.125`, `top2_margin_fp32≈0.019`, `top2_margin_bf16≈0.015` (competing logits). Treated as **legitimate BF16 rounding**, not a confirmed kernel bug.

HF Transformers spot-check: **skipped** (deps unavailable on Studio).

## llama.cpp (required reference)

| Field | Value |
|---|---|
| Commit | `71ad0590f4808b6202f9213d166913858c73b1bc` |
| Precision | F16 GGUF |
| Tool | `llama-bench` CUDA |
| Model | Qwen2.5-0.5B-Instruct (GGUF F16, 1.17 GiB, 630M params) |

| Test | tok/s |
|---|---:|
| pp32 (prefill) | 4492 ± 1829 |
| **tg16 (decode)** | **213.6 ± 1.0** |
| pp1+tg16 | 214.0 ± 0.6 |

Summary JSON: [`benchmarks/results/summaries/llama_cpp_short_f16.json`](../benchmarks/results/summaries/llama_cpp_short_f16.json)

### Forge vs llama.cpp (fairness)

| Comparison | Forge | llama.cpp | Winner |
|---|---:|---:|---|
| Single-stream decode, short, BF16/F16 | ~179 decode / 167 agg | ~214 tg | **llama.cpp** (~+19% decode) |
| Concurrent B=8 aggregate BF16 | **585** | not collected (parallel pg incomplete) | Forge (serving) — no direct llama parallel number |
| Prefill microbench | Forge prefill≈825–1350 (batch-integrated) | llama pp32≈4492 | **llama.cpp** (microbench vs integrated; not apples-to-apples) |

Caveats: Forge numbers are end-to-end continuous-batcher wall time; llama-bench `tg` is a decode microbench. Precision class matched (BF16 vs F16), not INT4.

## vLLM

**Unavailable** on this Studio: venv creation blocked (`max 1` conda env). Blocker recorded in [`benchmarks/results/summaries/vllm_blocker.json`](../benchmarks/results/summaries/vllm_blocker.json). Not extrapolated.

## Profiling

- `nsys` CLI installed mid-session; produced `.qdstrm` but **importer missing** → no kernel table.
- Bottleneck ranking therefore from **controlled experiments + architecture**, not nsys:

1. Naive paged attention (when enabled) — dominant  
2. cuBLAS / BF16 GEMM path (remaining after attn fix)  
3. BF16 activation cast scratch before linears  
4. Per-request attention launches (not fused across batch)  
5. CPU scheduler / step loop overhead at small B  

See [PERFORMANCE_ANALYSIS.md](PERFORMANCE_ANALYSIS.md).

## Cloud

Ledger: [PHASE12_COST_LEDGER.md](PHASE12_COST_LEDGER.md)

| | |
|---|---|
| Sessions | 1a–1c, 2, 3b, 4, 5, 6 (plus aborted Tee starts) |
| Cumulative est. spend | **≈$1.5** of **$6** |
| Final stop | **verified** after last session |

## Reproducibility

```powershell
go run ./api/cmd/forge-cloud prepare-deploy
go run ./api/cmd/forge-cloud validate-phase12 --gpu l4 --duration 40m --mode forge --confirm
go run ./api/cmd/forge-cloud validate-phase12 --gpu l4 --duration 35m --mode compare --confirm
```

Scripts: `scripts/remote_cuda_phase12.sh`, `benchmarks/forge/*`, `benchmarks/llama_cpp/build_and_bench.sh`.
