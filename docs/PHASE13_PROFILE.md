# Phase 13 profile

Fresh NVIDIA L4 BF16 Phase 12 baseline, 2026-10-08. Same cached Phase 12
binary, Hello prompt (10 tokens), 16 generated tokens, EOS disabled, one
warmup and three trials. Model loading excluded, synchronized wall clocks.

| Batch | Aggregate tok/s | Steady decode tok/s |
|---|---:|---:|
| 1 | 166.08 | 178.09 |
| 2 | 265.48 | 296.35 |
| 4 | 417.53 | 497.55 |
| 8 | 584.76 | 752.34 |

Nsight Systems executable and matching QdstrmImporter were already bundled
under `/opt/nvidia/nsight-compute/2025.3.1/host`. Import failed because
`libdw.so.1` was missing. Installing `libdw1t64` fixed import; `.nsys-rep` and
SQLite reports generated and inspected. The trace covers loading, warmup,
prefill and decode; these percentages are not decode-only measurements.

B=8 kernel totals: BF16 cuBLAS kernels ~56%, paged attention 23.7%, RoPE 5.1%,
activation casts 4.8%, KV append 2.4%, argmax 2.0%. Cast median ~1.12 us;
its aggregate includes a load-time outlier. Host CUDA API time: cudaMemcpy
47.2%, runtime kernel launch 26.0%, driver launches 4.5%. There are 23,056
device-to-device copies (~25.4 ms GPU time) in the trace.

First experiment: shared prepare/attention launches with direct batched Q/K/V
access, eliminating per-row staging copies. Both attention kernel latency and
host copy overhead support this choice. Baseline comparison must isolate the
effect with identical benchmark settings. BF16 casts merit a smaller follow-up;
do not attribute all cuBLAS time to conversion.

Raw evidence fetched to `benchmarks/results/phase13/`: baseline and optimized
CSV summaries plus local `.nsys-rep` artifacts (reports are not committed).

## Optimized trace and limits

Use `graphs_nodes_nsys_b8_stats.csv`: graph-node tracing was explicitly enabled.
The earlier `graphs_nsys_b8` report misses graph kernel nodes and must not be
used to infer absent GEMM/attention execution. Both reports were imported and
inspected; no conclusions rely on an unimported qdstrm.

Matched B=8 whole trace: attention GPU total 90.01ms baseline versus 11.46ms
batched + 2.23ms retained prefill; median batched kernel 15.90us versus 14.93us
per-row old kernel (eight old launches per layer/round). Prepare median 3.39us.
cuBLAS accounts for ~77.3% optimized kernel time, casts 5.5%, batched attention
4.2%, argmax 2.7%, prepare 0.9%. Cast count 7943→5783, consistent with saving
72 × 30 decode rounds = 2160 casts. Runtime memcpy calls 23470→460;
runtime kernel launch calls 44817→9561, plus driver calls; 29 graph replays
after one eager warm decode. Counts include loading/prefill and cannot be
presented as all per-token hot-path work.

Launch arithmetic at B=8 for 24 layers: old attention/RoPE/append 768 calls per
decode round→48 shared prepare/attention calls; attention alone 192→24.
Four D2D staging copies/row/layer (768/round) disappear. This explains the
attention-only end-to-end improvement 584.76→888.73 tok/s. Cast reuse adds
888.73→896.46; graph replay adds 896.46→939.79. Whole-trace API percentages
include synchronous model transfers and graph instrumentation; they are not
isolated host scheduler or synchronization costs per token.

CUDA event microbench: 1024-cubed GEMM 0.128ms, 64MiB H2D 5.298ms, D2H
5.279ms, SiLU launch device interval 2.3us. This validates bounded kernel/event
timing, but these synthetic shapes do not directly predict model GEMM time.
Profiler timestamps provide attention/cast latency; unprofiled synchronized
wall clocks provide end-to-end comparisons. Exact scheduler-only cost and
event-based timings for each individual real-model operator remain unmeasured.

The fresh baseline agrees with historical Phase 12. Optimized GEMM dominates
the remaining GPU work; HTTP long-prompt TTFT identifies prefill as a separate
end-to-end bottleneck. No decode-only breakdown by all batch sizes was collected.
See [results](PHASE13_RESULTS.md) for numerical results, A/B rates and gaps.
