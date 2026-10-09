# Phase 13 results

Execution optimizations implemented and GPU validated. Serving comparison is
provisional: cached GGUF checkpoint identity is not fully verified. This report
does not declare all Phase 13 measurement requirements complete.
One NVIDIA L4 session, 2026-10-08; raw JSON/CSV and selected logs are under
`benchmarks/results/phase13`. Historical Phase 12 records remain unchanged.

## Controlled internal A/B

Same HF Qwen2.5-0.5B-Instruct checkpoint, BF16 weights, FP32 KV/activations/norms,
FP32 GEMM accumulation. Hello prompt (10 input tokens), 16 generated tokens,
fixed length (EOS disabled), one warm-up and three trials, synchronized end of
region, model loading excluded. This is an internal batch benchmark, not HTTP.
Aggregate output tok/s (prefill + decode):

| Configuration | B=1 | B=2 | B=4 | B=8 |
|---|---:|---:|---:|---:|
| Phase 12 eager | 166.08 | 265.48 | 417.53 | 584.76 |
| Batched attention | 173.32 | 299.87 | 536.83 | 888.73 |
| + BF16 reuse | 175.94 | 304.03 | 544.11 | 896.46 |
| + CUDA graphs | 188.38 | 324.35 | 575.64 | 939.79 |

Steady decode aggregate tok/s (first-token prefill excluded):

| Configuration | B=1 | B=2 | B=4 | B=8 |
|---|---:|---:|---:|---:|
| Phase 12 eager | 178.09 | 296.35 | 497.55 | 752.34 |
| Combined | 204.08 | 371.47 | 740.08 | 1469.38 |

Combined end-to-end gains are 13.4%, 22.2%, 37.9%, and 60.7% at B=1/2/4/8.
B=1 steady decode improves from 178.09 to **204.08 tok/s**, below the historical
214 tok/s stretch target. B=8 steady decode reaches **1469.38 aggregate tok/s**.
BF16 reuse adds only about 1–2%; attention removal accounts for most B=8 gain.
FP32 baseline aggregate: 101.01/169.14/269.57/398.45; combined:
106.11/187.15/319.62/520.62 (B=1/2/4/8). Full timing/occupancy fields are in JSON.

## Correctness

GPU-required `cuda_phase13` compares Phase 12 eager against combined graph path
for FP32 and BF16, B=1/2/4/8 and unsupported B=9, 24 decode steps per case,
variable prompt lengths, physical block crossings, request replacement at
steps 9 and 15, and prefill-driven shared cast-buffer growth. Maximum absolute
logit error was **0 in all ten cases**, tolerance unchanged (1e-4 + 1e-3 relative).
Graphs captured and replayed at B=1/2/4/8; B=9 used eager fallback.

Existing CUDA batch parity 7/7, transformer parity 2/2, paged KV 3/3 and kernel
checks passed during the session. Batch parity rerun with chunk size 8 and
prefill budget 16 passed 7/7, including cancellation before execution,
cancellation without sibling corruption, and slot reuse. Capacity exhaustion
and cleanup remain covered by existing paged allocator tests; no dedicated
graph-OOM or forced capture-failure test was run. New logit test directly covers
combined graphs versus baseline, rather than a separate three-way eager logit
test. Attention-only and reuse-only were exercised in successful A/B generation.

BF16 versus FP32 real-model survey remains **6/7 sequences**, **85/103 tokens
(82.5%)**. Medium prompt first diverges at token 14; small top-two margin and
approximately 0.125 maximum logit difference explain precision sensitivity.
Graph parity does not imply BF16 and FP32 generate identical sequences.
Windows CPU Rust release tests passed; CUDA-stub tests compile. GPU tests are
feature gated and are not counted as passing on the CPU-only machine.

## HTTP serving

`bench_http.py` uses loopback POST to streaming completion end, excludes model
load, includes transport, admission, prefill and decode. Identical raw prompt
strings, temperature 0, EOS enabled, max output 64, one full-concurrency warm-up,
three measured trials at concurrency 1/2/4/8. Short ends after 10 generated
token events including EOS; medium and long reach 64. Mixed cycles short,
medium and long. Forge has eight CUDA slots; llama-server `-np 8 -ngl 99 -c
16384`, 2048 context/slot, F16 GGUF and F32 K/V. Full offload log confirms
25/25 layers. Engines ran sequentially. A 5ms observer confirmed each measured
matrix cell attained requested active concurrency. Peak is not average occupancy.

Forge BF16 and llama F16 differ in weight/input GEMM precision. Both use FP32 KV.
llama.cpp commit `71ad0590f4808b6202f9213d166913858c73b1bc`; HF revision
`7ae557604adf67be50417f59c2c2f167def9a775`. GGUF metadata identifies the same
Qwen architecture/name and unquantized F16, but embedding sample matched while
Q-projection sample failed checkpoint comparison. An attempted RoPE permutation
also failed and was removed. No full tensor/revision equivalence was established.
The reference is therefore comparable in model family/configuration, with an
unresolved checkpoint identity limitation. It is not a proven exact-revision
comparison. `verify_model.py` now retains diagnostics and exits nonzero on mismatch.

Median aggregate output rate; TTFT is median of trial per-request P50;
P95 is median of trial P95, with only three rounds and at most eight requests
per round. These small samples are descriptive, not production tail estimates.

| Workload | Concurrency | Forge tok/s | llama tok/s | Forge TTFT / P95 ms | llama TTFT / P95 ms |
|---|---:|---:|---:|---:|---:|
| short | 1 | 184.39 | 183.63 | 12.0 / 54.2 | 7.3 / 54.5 |
| short | 2 | 305.96 | 298.39 | 18.8 / 65.3 | 9.0 / 66.9 |
| short | 4 | 479.69 | 517.72 | 31.1 / 83.2 | 9.9 / 77.3 |
| short | 8 | 727.52 | 876.09 | 61.8 / 109.9 | 12.8 / 90.8 |
| medium | 1 | 130.89 | 191.31 | 28.0 / 489.0 | 10.9 / 334.5 |
| medium | 2 | 195.29 | 329.54 | 47.2 / 655.4 | 14.0 / 388.3 |
| medium | 4 | 367.43 | 542.39 | 84.8 / 696.7 | 30.6 / 472.0 |
| medium | 8 | 642.77 | 888.72 | 170.1 / 796.3 | 45.3 / 575.6 |
| long | 1 | 56.46 | 174.17 | 197.8 / 1133.5 | 30.6 / 367.5 |
| long | 2 | 68.93 | 281.05 | 373.1 / 1856.9 | 56.0 / 455.4 |
| long | 4 | 115.54 | 430.45 | 723.6 / 2215.6 | 57.4 / 594.6 |
| long | 8 | 174.32 | 579.55 | 1436.1 / 2937.0 | 157.9 / 883.4 |
| mixed | 1 | 184.03 | 181.31 | 12.0 / 54.3 | 7.6 / 55.2 |
| mixed | 2 | 142.95 | 206.09 | 30.7 / 517.7 | 8.7 / 359.1 |
| mixed | 4 | 86.98 | 311.46 | 71.5 / 1707.0 | 20.1 / 475.0 |
| mixed | 8 | 172.89 | 564.08 | 136.0 / 2000.9 | 79.4 / 620.2 |

Clean current Forge eager HTTP baseline short: 165.30/259.29/373.35/519.15;
medium: 121.45/131.45/161.90/182.39 tok/s. The combined path improves both,
especially medium at B=8 (642.77 vs 182.39). Forge and llama are close on short
at C=1/2; Forge's C=2 point lead is small. llama leads at C=4/8 and on medium,
long and mixed. No general competitive win is demonstrated.

Independent VRAM-observed repeat short Forge: 183.64/306.91/479.56/698.44;
llama: 178.17/304.04/512.33/811.26. Medium Forge:
130.81/195.30/367.17/644.24; llama: 188.01/321.30/520.48/848.17.
Sampled peak device memory (nvidia-smi ~100ms, can miss short peaks):
Forge short 1500/1536/1550/1562 MiB, medium 1568 MiB at all concurrencies;
llama short 1588/1604/1604/1604 MiB, medium 1606/1608/1614/1614 MiB.
BF16 weights remain approximately 942.4 MiB. No sampled VRAM result was collected
for long/mixed; no allocation-precise peak is claimed.

Per-request rates and P50 latency are preserved in raw JSON. Queue wait is null
because neither endpoint exposes it per request; TTFT includes it and cannot
separate it. `cache_prompt=false` was sent to llama, but native slot routing logs
show prefix retention behavior, so warmed prompt reuse is a fairness limitation.
Prompt-eval counters were inspected, but cache semantics were not exhaustively
matched. No controlled staggered-arrival throughput rerun was collected; existing
dynamic admission tests and mixed concurrent HTTP are distinct evidence.

An early Forge HTTP run mistakenly linked the old shared library after building
only examples; retained as `forge_http_baseline_extended.json`, excluded from
optimized claims. It briefly overlapped correctness work and is not used as a
controlled baseline. Final optimized library was explicitly rebuilt, and capture
logs verify graphs were active. Fresh clean baseline short/medium is used above.

## Launches, graphs and profiler

See [profile](PHASE13_PROFILE.md). B=8 attention preparation + attention drops
from 768 to 48 launches/decode round (93.75%); attention alone 192 to 24 (87.5%).
Four staging copies × 8 rows × 24 layers = 768 D2D copies removed/round.
BF16 reuse saves 72 casts/round. A decode round generates B tokens; do not label
these round counts as per-token counts. Captured body has one host graph replay
per round after warm-up; GPU nodes still execute and sampling remains outside.

Real-model isolated captures were approximately 3.7–12.1ms; serving captures
2.6–14.4ms. Initial captures for all eight serving shapes reduced device free
memory by 46 MiB in total (4–8 MiB each); later invalidation/recapture
showed 0 incremental reserved-memory delta. These allocator deltas are not exact
graph allocation accounting. Node tracing affects launch latency, so throughput
is from unprofiled runs.

## Cloud and remaining work

One session, conservative estimated $1.03, remaining $4.97 of $6 compute budget.
Actual instance stop and empty provider inUse/requested state verified. See
[ledger](PHASE13_COST_LEDGER.md). No GPU remains running.

Measurement gaps: exact GGUF revision equivalence, per-request queue wait,
isolated host scheduler time, controlled staggered throughput, long/mixed peak
VRAM, forced graph failure/OOM. These are not marked successful or complete.

Recommended Phase 14 priorities, without implementing Phase 14:
1. Shape-specific dense GEMM tuning: cuBLAS is approximately 77% of optimized
   profiled GPU kernel time, including LM-head work.
2. Batched/chunked prefill: long C=8 TTFT is 1436ms vs llama 158ms; decode graphs
   do not improve sequential per-request prefill.
3. Profile remaining cast/norm/sampling and host admission costs in a decode-only
   trace before choosing further fusion (casts 5.5%, argmax 2.7% of whole trace).
Resolve reference identity/cache equivalence before treating small serving leads
as competitive wins. No Phase 14 changes were made.
