# Phase 14 results

Implemented and GPU-validated on one NVIDIA L4 session, 2026-10-08. Instrumented
HTTP, shared ragged chunked prefill, partitioned prefill attention, cached
selective GEMM policy and early token delivery preserve old oracles and CPU.
Medium B=8 TTFT and short throughput targets were reached in these workloads.
Single-stream compute is largely unchanged. Phase 15 was not started.

## Controlled HTTP comparison

Fresh optimized Phase 13 baseline versus final Phase 14: same HF source and BF16
execution, FP32 KV, raw prompts 10/190/910 tokens, greedy, EOS enabled, max output
64, eight slots and 2048 context/request. One full-concurrency warmup, three
trials/cell; client POST through SSE DONE excludes load. No concurrent engine,
compiler or correctness work during accepted throughput runs. Final flags:
packed+block attention+Lt+early emit, chunk 128, budget 1024; baseline budget 256.
TTFT/P50/P95 are medians of trial statistics. Small-sample P95 is descriptive.

| workload | C | Phase 13 tok/s | Phase 14 tok/s | change | TTFT ms | P50 ms | P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| short | 1 | 183.18 | 184.22 | +0.6% | 6.9 | 54.3 | 54.3 |
| short | 2 | 305.87 | 343.57 | +12.3% | 7.1 | 58.0 | 58.2 |
| short | 4 | 509.17 | 559.01 | +9.8% | 8.2 | 71.2 | 71.5 |
| short | 8 | 695.63 | 992.33 | +42.7% | 23.1 | 80.0 | 80.4 |
| medium | 1 | 130.77 | 131.50 | +0.6% | 16.4 | 486.7 | 486.7 |
| medium | 2 | 195.07 | 202.47 | +3.8% | 22.6 | 632.0 | 632.1 |
| medium | 4 | 366.01 | 392.11 | +7.1% | 38.1 | 652.5 | 652.8 |
| medium | 8 | 642.04 | 738.02 | +14.9% | 73.7 | 693.1 | 693.8 |
| long | 1 | 56.41 | 59.49 | +5.5% | 114.2 | 1075.9 | 1075.9 |
| long | 2 | 68.85 | 74.81 | +8.7% | 206.8 | 1710.6 | 1710.9 |
| long | 4 | 115.36 | 133.88 | +16.1% | 401.5 | 1911.9 | 1912.2 |
| long | 8 | 174.33 | 221.41 | +27.0% | 795.0 | 2311.8 | 2312.4 |
| mixed | 1 | 184.19 | 184.83 | +0.3% | 6.8 | 54.1 | 54.1 |
| mixed | 2 | 143.05 | 145.51 | +1.7% | 9.6 | 95.8 | 508.5 |
| mixed | 4 | 85.76 | 91.07 | +6.2% | 15.3 | 229.8 | 1625.0 |
| mixed | 8 | 174.00 | 193.56 | +11.2% | 64.5 | 1661.6 | 1807.9 |

Per-cell throughput ranges are in summary.json/raw trials. Earlier same-config
B=8 repeat: short 1025.09, medium 734.04, long 220.31, mixed 192.59 tok/s. Do not
present one point as universal. Final matrix sampled VRAM peaks at **1978 MiB**,
versus baseline ~1.5–1.6 GiB. Persistent buffers/graph shapes grow across cells;
100 ms nvidia-smi sampling can miss peaks and is not allocation-exact. KV stats,
queue means and sampled active mean/peak are retained in JSON/metrics.
Mixed arrivals at 10 ms spacing, C=8, three trials: **185.91 tok/s**. No fresh
staggered baseline was collected, so no controlled staggered speedup is claimed.

## Ablations

B=8 output tok/s. Packed uses original attention/budget 256; prefill-only adds
partitioned attention. Combined budget 256 retains old emission. budget1024 also
enables early emit; these two scheduling choices are not separately attributed.

| config | short | medium | long | mixed | medium TTFT ms |
|---|---:|---:|---:|---:|---:|
| phase13 | 695.63 | 642.04 | 174.33 | 174.00 | 170.4 |
| packed | 1024.15 | 692.74 | 186.65 | 179.40 | 110.6 |
| prefill-only | 974.61 | 708.12 | 216.55 | 188.67 | 94.4 |
| gemm-only | 738.02 | 650.53 | 175.49 | 176.44 | 169.6 |
| combined | 1050.67 | 717.35 | 217.16 | 192.84 | 93.9 |
| combined-budget1024 | 1025.09 | 734.04 | 220.31 | 192.59 | 76.6 |
| final | 992.33 | 738.02 | 221.41 | 193.56 | 73.7 |

GEMM benefits are small end-to-end. Packed-only C=2 short regressed to 286.77
tok/s; final scheduling reaches 343.57. C=1 short remains near historical 184.
Intermediate experiments and regressions are retained.

## Internal benchmark and timing caveat

Same 10-token prompt, fixed 16 outputs, EOS disabled; one warmup/three trials.

| B | baseline aggregate | final aggregate | baseline reported steady | final reported steady |
|---:|---:|---:|---:|---:|
| 1 | 187.48 | 187.50 | 203.04 | 189.52 |
| 2 | 323.55 | 350.94 | 370.76 | 354.76 |
| 4 | 574.56 | 694.48 | 738.68 | 708.64 |
| 8 | 940.08 | 1356.55 | 1471.45 | 1408.30 |

Reported steady regresses: B=1 203.04 -> 189.52. Early emit moves TTFT before
the next decode round; the old boundary excluded that already-computed round
from steady timing. This is not an isolated compute A/B. Do not hide the raw
regression or claim a single-stream compute gain from TTFT/aggregate. Short C=1
stream profile is ~5.102 -> 5.113 ms; C=8 ~5.566 -> 5.440 ms (profile overhead).

## Correctness

- GPU-required synthetic packed/chunked logits, FP32/BF16 B=1/2/4/8, different
  chunks/positions/block crossings: max 5.96e-8 / 2.98e-8 under unchanged
  1e-4 + 1e-3 relative tolerance.
- Mixed lifecycle: eight unequal prompts/generation lengths, partial-prefill
  cancellation, slot reuse, starvation bound, freed blocks and capacity exhaustion;
  legacy/packed greedy outputs agree.
- Existing batch 7/7, paged KV 3/3, transformer 2/2, graph parity and CUDA kernel
  suites pass. Chunk-8/budget-16/early-emit batch rerun passes. Graph parity covers
  B=1/2/4/8, B=9 fallback, replacements and cast-buffer growth. EOS occurs in HTTP
  short completion; Go verifies terminal tokens are delivered.
- Full checkpoint: FP32 max error 3.73e-5; BF16 **0.181134**; zero disagreements
  in 120 greedy prefix decisions per dtype. This test measures real errors and
  asserts greedy stability, not tight BF16 logit equality.
- BF16 attribution, modes packed / packed+attention / packed+Lt / combined:
  B=1 errors 0 / 0.085318 / 0.099360 / 0.099360; B=8 0.134038 / 0.134038 /
  0.181134 / 0.181134. Tested greedy decisions all agree. Shape/reduction changes
  feeding BF16 rounding are consistent with these results; standalone attention
  error <=5e-8 and GEMM accumulation remains FP32. Universal output parity is not
  proved. Original single-request paths remain oracles.
- Seven-prompt BF16 vs CUDA FP32 survey at max 16: 7/7 sequences, 71/71 tokens,
  with chunked/batch cases. Phase 13 used max 32; this shorter run does not prove
  its 6/7 result universally fixed. Baseline mixed HTTP outputs varied, while
  final repeats were stable in these trials. Precision sensitivity remains.
- Windows CPU Rust release/Go suites pass; CUDA stub tests compile. CPU-only
  feature-gated zero-test runs are not counted as GPU passes.

No tolerance was silently relaxed. Forced graph OOM/capture-failure injection
and exhaustive pretrained-prefix parity remain untested inherited edge cases.

## Matched-source llama.cpp

Pinned commit `71ad0590f4808b6202f9213d166913858c73b1bc`. Fresh GGUF converted
with official Qwen2 conversion from Forge HF revision
`7ae557604adf67be50417f59c2c2f167def9a775`. All 290 tensors exactly match after
storage conversion; all 151665 tokenizer entries, chat template, RoPE 1e6 and
BOS/EOS 151643/151645 agree. Native IDs for the actual 10/190/910-token prompts
match. Full hashes/checks: full-identity.json, input-identity.json. The old cached
GGUF's provenance remains unresolved and is excluded from these claims.

Reference: -ngl 99 -np 8 -c 16384 -ctk f32 -ctv f32, prompt caching disabled in
requests and globally (--no-cache-prompt --cache-ram 0 --no-cache-idle-slots).
Verbose audit: **25/25 layers offloaded**, CUDA model 942.43 MiB, KV 384 MiB.
Engines ran sequentially on the same L4. Old Phase 13 reference results had
unresolved source/cache semantics and are not a strict reference trend line.

| workload | C | Forge BF16 tok/s | llama F16 tok/s | Forge TTFT/P95 ms | llama TTFT/P95 ms |
|---|---:|---:|---:|---:|---:|
| short | 1 | 184.22 | 168.58 | 6.9/54.3 | 8.0/59.3 |
| short | 2 | 343.57 | 266.99 | 7.1/58.2 | 9.2/74.9 |
| short | 4 | 559.01 | 371.23 | 8.2/71.5 | 10.5/107.5 |
| short | 8 | 992.33 | 540.28 | 23.1/80.4 | 26.8/147.5 |
| medium | 1 | 131.50 | 173.52 | 16.4/486.7 | 12.0/368.8 |
| medium | 2 | 202.47 | 269.56 | 22.6/632.1 | 12.4/474.6 |
| medium | 4 | 392.11 | 415.85 | 38.1/652.8 | 16.1/615.2 |
| medium | 8 | 738.02 | 584.43 | 73.7/693.8 | 59.5/875.8 |
| long | 1 | 59.49 | 161.53 | 114.2/1075.9 | 31.1/396.2 |
| long | 2 | 74.81 | 241.19 | 206.8/1710.9 | 29.8/530.6 |
| long | 4 | 133.88 | 349.29 | 401.5/1912.2 | 58.9/732.8 |
| long | 8 | 221.41 | 446.90 | 795.0/2312.4 | 181.3/1145.0 |
| mixed | 1 | 184.83 | 163.38 | 6.8/54.1 | 8.1/61.2 |
| mixed | 2 | 145.51 | 188.33 | 9.6/508.5 | 8.3/392.9 |
| mixed | 4 | 91.07 | 177.52 | 15.3/1625.0 | 15.4/833.6 |
| mixed | 8 | 193.56 | 245.94 | 64.5/1807.9 | 92.1/1422.9 |

Forge leads short and medium C=8 throughput here; llama leads medium C=1/2/4,
long/mixed throughput. Medium C=8 Forge TTFT remains slower (73.7 vs 59.5 ms).
BF16/F16 activations/weight storage and engine sampling/EOG implementation differ.
This is audited matched source/inputs/configuration, not bitwise precision
equivalence or a general competitive win. Raw output/counts/native timings
remain visible; no reference outputs are changed to manufacture agreement.

## Cloud and source

One existing forge-gpu L4, $1.68/hour. Conservative estimate **$1.16**, remaining
**$4.84**; actual invoice unavailable. Absolute local/remote watchdogs and auto-sleep
900 s; requested 40-minute MaxRuntime was not visibly enforced. Conservative
provision/stop bound 41m08s, below 45 min. Actual stop/unreachable and authenticated
requested=null/inUse=null for all visible Studios verified. See cost ledger.

Final runtime deploy `20261008T192014Z-e4331fde9a23`, content hash
`e4331fde9a239d9e064ec786fc867fc0573b757c314acfc03cd71c8e649dd87a`.
Every deploy was manifest verified; standalone diagnostic source used SHA-256
verification. Missing Lt link/conversion dependency errors were fixed within the
active session. Failed builds/interrupted eager runs are not accepted performance
evidence. Offline cleanup narrows the Lt gate to exact L4 and adds CLI aliases;
neither changes executed L4 math. Documentation was completed after shutdown.

## Next three improvements supported by profiling

1. Long decode attention (~23.7 ms B=8 vs ~5.4 ms short): prefill alone cannot
   close long completion latency.
2. Shared query/KV tiles in prefill: current partitioning still rereads KV across
   queries; long TTFT 795 ms versus reference 181 ms.
3. Separate early-emit/budget ablations and mixed fairness policy: mixed C=8
   194 vs reference 246 tok/s, sampled active mean ~5.18.
