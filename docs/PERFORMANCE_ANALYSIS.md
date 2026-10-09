# Performance Analysis (Phase 12)

Evidence-backed bottleneck ranking for Forge CUDA on NVIDIA L4 (Qwen2.5-0.5B-Instruct).

## Measurement basis

| Source | Role |
|---|---|
| `bench_phase12` medians | End-to-end aggregate / decode / TTFT |
| Controlled flags | Isolate attn / greedy / dtype |
| llama-bench F16 | External decode reference |
| nsys | Attempted; `.qdstrm` only (importer missing) — **not** used for kernel % |

## Top bottlenecks (ranked by measured contribution)

### 1. Paged attention kernel quality (when naive)

Controlled B=8 FP32 short: naive **35.3** tok/s → optimized **373** tok/s (~**10.6×**).  
After Phase 11 opt, this is no longer the default path, but is the largest measured swing of any single flag.

### 2. Batched GEMM (cuBLAS / GemmEx BF16)

With opt attn, remaining work is dominated by QKV/O/MLP/lm_head linears.  
BF16 B=8 **585** vs FP32 **399** (~**+47%**) at matched workload shows weight bandwidth / Tensor Core path still matters.

### 3. BF16 activation conversion traffic

Design note (Phase 11): FP32 activations cast before each BF16 weight GEMM. Not isolated with a dedicated flag in Phase 12; listed as a **Phase 13 hypothesis**, not a measured %.

### 4. Per-request attention launches

Batch occupancy reaches configured B, but attention still runs per sequence (shared linears only). Small-B (B=1) leaves GPU underfilled: BF16 decode≈179 vs B=8 decode≈753.

### 5. Host step / scheduling overhead

TTFT and B=1 rates include chunked prefill scheduling and CPU `step()` loops. Absolute overhead is small at B=8 but visible at B=1 relative to llama-bench `tg` microbench (214 vs Forge 179 decode).

## Competitive interpretation

| Scenario | Observation |
|---|---|
| Single-stream short decode | llama.cpp F16 **~214 tok/s** beats Forge BF16 **~179 decode / 167 agg** |
| Concurrent short serving B=8 | Forge BF16 **585 agg** — continuous batching is the architectural win; llama parallel not fully measured here |
| Long prefill | Forge long B=1 **53 agg** (prefill-dominated); llama pp32 microbench is much higher but not e2e-equivalent |

## Phase 13 candidates (hypotheses, not measurements)

1. Fuse / batch paged attention across active sequences (reduce launch + improve occupancy).  
2. Eliminate or fuse BF16 activation casts (persistent BF16 acts or fused epilogue).  
3. CUDA Graphs for decode-only steady state at fixed B.  
4. Port warp/online softmax to contiguous single-seq oracle path.  
5. Optional FlashAttention-class kernel if profiling confirms attn still ≥30% after (1).

Do **not** implement these in Phase 12.


## Phase 13 measured update

Batched preparation/attention removes the dominant row staging and attention overhead. BF16 internal B=8 aggregate improves 584.76→939.79 tok/s; B=1 steady decode reaches 204.08. Small BF16 reuse gain is 1–2%. Optimized whole-trace cuBLAS share is ~77%; long HTTP TTFT remains a prefill bottleneck. llama-server still leads most workloads. Exact GGUF identity and cache equivalence remain unresolved. See [PHASE13_RESULTS.md](PHASE13_RESULTS.md) and [PHASE13_PROFILE.md](PHASE13_PROFILE.md); previous measurements are retained as history.

## Phase 14 measured update

See [profile](PHASE14_PROFILE.md) for shape-specific GEMM/attention measurements
and [results](PHASE14_RESULTS.md) for the full HTTP matrix and ablations. At C=8,
short throughput rose 695.63 to 992.33 tok/s, medium 642.04 to 738.02, long
174.33 to 221.41, mixed 174.00 to 193.56. Medium TTFT fell 170.4 to 73.7 ms;
long 1437.1 to 795.0 ms. Sampled final peak VRAM reached 1978 MiB.

Packed prefill is the major change; selective Lt alone yields modest overall
gains. Single-stream decode is essentially unchanged. Synchronized long B=8
decode remains about 23.7 ms versus short about 5.4 ms. Host scheduler/SSE costs
are much smaller. Internal reported steady rates regress under a changed
first-token boundary; no isolated decode speedup is claimed.

The audited llama.cpp reference still leads long and mixed throughput and long
TTFT. Prioritize long decode attention, shared query/KV tiles during prefill,
and isolated scheduling/early-emission ablations. See the Phase 15 results below.

## Phase 15 — precision-aware long-context decode

[Full results](PHASE15_RESULTS.md) and [profile](PHASE15_PROFILE.md) separate
accepted measurements from rejected candidates. With a common pinned worker,
BF16 C=8 medium rises 735.6→1030.9 tok/s (P95 695.9→496.5 ms). Long C=1/2
rises 59.5→104.2 and 74.9→149.4 tok/s. Long C=8 stays 221.3→221.4 tok/s
and TTFT 796.3→795.1 ms because the faster unrestricted BF16 kernel failed a
64-step greedy check. Mixed C=8 improves modestly, 194.0→202.2 tok/s and
P95 1803.6→1730.1 ms; stretch targets remain unmet. Nine-trial short C=8
medians are 1055.0→1053.6 tok/s.

FP32 permits larger long batches: C=8 long 166.5→228.8 and mixed
151.4→231.2 tok/s, with 960 teacher-forced greedy decisions per dtype
matching the reference. Attention remains the leading measured prefill kernel
cost; shared KV tiles and adaptive caps did not justify retention. Pool
pressure reached 95.3% without a new allocator. Actual SM occupancy/bandwidth
counters were unavailable; active-slot samples are distinct from SM occupancy.
Three sessions estimate $2.98; all GPU instances are verified stopped.
