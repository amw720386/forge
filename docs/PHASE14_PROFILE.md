# Phase 14 serving profile

NVIDIA L4, 2026-10-08. Real HTTP was measured before changing prefill/compute.
`phase13-http.json` is the fresh optimized baseline. The initial partial
`baseline-http.json` omitted Phase 13's opt-in flags and is an eager reference,
excluded from Phase 13 improvement claims. Its interruption is documented.

## Timing boundaries

| Stage | Instrumentation | Boundary |
|---|---|---|
| HTTP receive/parse | http_parse | Handler entry through JSON validation; excludes earlier network time |
| Queue admission | queue_admission | Bounded queue publication |
| Scheduler waiting | queue_wait | Enqueue through scheduler admission |
| Tokenization | runtime_admit_tokenize, Rust tokenize log | FFI admit overall; Rust isolates encode |
| GPU prefill | prefill_stream_region_ms | CUDA stream events, synchronized at end |
| Decode scheduling | scheduling_host_ns, scheduler_host | Rust sample/plan; Go iteration excluding BatchStep |
| CUDA execution | decode stream_region_ms | Stream region including staging/launch gaps; capture separately logged |
| Token D2H | d2h host_ns and bytes | Blocking copy wall time; sample_transfer also includes argmax |
| SSE streaming | sse_write_flush | Token serialization/write/flush |
| Final completion | http_completion | Handler entry through response; client separately measures POST through DONE |

`/metrics.stages` exposes count/total_ns/max_ns; benchmark records per-trial
deltas and means. Stage sums overlap and cannot be added as a disjoint waterfall.
Iteration contains runtime; completion contains all request work. Rust scheduling
includes token sampling/decoding and planning. Synchronized event/host profiling
adds logging and waits and is excluded from throughput claims. Stream intervals
include launch gaps and staging, not a sum of pure kernel durations.
Existing prefill_stall_ns is host-observed submission/result time while decode
is active; use synchronized profile intervals to measure actual elapsed stalls.

## Measured bottleneck ranking

1. Prefill dominates TTFT: medium B=8 170.4 ms, long 1437.1 ms. Medium B=8
   Go scheduler host mean ~0.042 ms/iteration, SSE ~0.039 ms/token cannot explain it.
2. Long-prefix decode: profiled B=8 stream median 23.92 ms vs short 5.57 ms.
   Final: 23.74 vs 5.44 ms. Decode attention remains the major long limitation.
3. Dense shapes: selective Lt improves small projections and MLP down at M=32/64.
   LM head does not show a useful broad gain. GEMM-only medium B=8 is 650.5
   vs 642.0 tok/s; most end-to-end benefit is prefill.
4. Short admission ramp/emission boundaries: sampled active mean 4.24 -> 6.80,
   peak eight in both; queue mean 5.04 -> 4.73 ms. Active includes prefill/decode
   and observer idle samples, not exact mean device decode occupancy.
5. D2H/sampling: profiled blocking copy median ~0.204 ms, sample+transfer ~0.275 ms.

Across the same synchronized profiling mix (short/medium/long, C=1/8, one warm-up
and one trial), prefill stream totals 3727.8 -> 2124.1 ms, regions 97 -> 47.
Region medians 34.19 -> 19.99 ms span different row counts. No fresh Nsight system
profiler was available; these are CUDA-event and host traces.

## Internal versus HTTP

Fresh internal B=8 reproduces 940.08 aggregate tok/s and 1471.45 reported steady
decode. HTTP baseline is 695.63. Internal uses fixed 16 outputs with EOS disabled,
all eight admitted before execution; HTTP short generates 10 outputs with EOS
enabled and dynamic admission. Shorter output amortization and admission/graph
shape warm-up differ. The whole gap cannot be attributed to HTTP transport.
Client, internal generation, stream intervals and CPU submission are distinct.

## GEMM shapes

BF16 inputs/weights, FP32 output/accumulation, L4 CUDA 13. Three event-timed trials,
40 calls/trial after ten warmups. Microseconds below; fastest passing of up to
eight heuristics is not necessarily accepted by the runtime policy. Synthetic
data/launch gaps limit extrapolation. Lookup time is separately in shapes.jsonl.

| M | N | K | GemmEx us | best Lt us | rank |
|---:|---:|---:|---:|---:|---:|
| 1 | 896 | 896 | 5.33 | 4.56 | 1 |
| 1 | 128 | 896 | 4.43 | 3.43 | 0 |
| 1 | 4864 | 896 | 11.81 | 10.85 | 4 |
| 1 | 896 | 4864 | 11.19 | 10.55 | 1 |
| 1 | 151936 | 896 | 1104.62 | 1079.94 | 1 |
| 2 | 896 | 896 | 11.21 | 10.37 | 4 |
| 2 | 128 | 896 | 10.87 | 6.30 | 5 |
| 2 | 4864 | 896 | 11.96 | 11.96 | 0 |
| 2 | 896 | 4864 | 15.16 | 13.23 | 1 |
| 2 | 151936 | 896 | 1084.21 | 1083.55 | 0 |
| 4 | 896 | 896 | 10.60 | 8.14 | 0 |
| 4 | 128 | 896 | 8.39 | 6.43 | 5 |
| 4 | 4864 | 896 | 10.82 | 10.83 | 0 |
| 4 | 896 | 4864 | 14.59 | 13.44 | 1 |
| 4 | 151936 | 896 | 1094.45 | 1094.55 | 0 |
| 8 | 896 | 896 | 8.43 | 8.22 | 2 |
| 8 | 128 | 896 | 7.64 | 5.93 | 5 |
| 8 | 4864 | 896 | 10.80 | 10.65 | 0 |
| 8 | 896 | 4864 | 15.00 | 13.53 | 1 |
| 8 | 151936 | 896 | 1108.63 | 1108.38 | 0 |
| 32 | 896 | 896 | 9.24 | 8.09 | 3 |
| 32 | 128 | 896 | 7.45 | 6.20 | 0 |
| 32 | 4864 | 896 | 13.41 | 11.42 | 3 |
| 32 | 896 | 4864 | 22.46 | 11.92 | 2 |
| 64 | 896 | 896 | 9.69 | 7.45 | 1 |
| 64 | 128 | 896 | 6.55 | 5.73 | 1 |
| 64 | 4864 | 896 | 11.55 | 11.62 | 0 |
| 64 | 896 | 4864 | 25.50 | 13.59 | 4 |
| 128 | 896 | 896 | 8.19 | 8.32 | 0 |
| 128 | 128 | 896 | 8.27 | 6.99 | 2 |
| 128 | 4864 | 896 | 21.50 | 20.91 | 0 |
| 128 | 896 | 4864 | 21.94 | 21.35 | 1 |
| 256 | 896 | 896 | 10.65 | 10.57 | 0 |
| 256 | 128 | 896 | 9.16 | 8.04 | 1 |
| 256 | 4864 | 896 | 39.71 | 36.77 | 1 |
| 256 | 896 | 4864 | 33.97 | 33.82 | 0 |

## Prefill attention

Three trials, 20 calls/trial, five warmups; 14 Q heads/2 KV heads/dim 64, causal
offsets and reversed physical blocks. FP32 online softmax; original warp oracle.

| queries | KV | warp us | four-warp us | ratio | max error |
|---:|---:|---:|---:|---:|---:|
| 8 | 128 | 55.5 | 18.6 | 2.98x | 1.0e-08 |
| 32 | 128 | 59.8 | 22.4 | 2.67x | 1.0e-08 |
| 128 | 128 | 80.6 | 60.2 | 1.34x | 1.0e-08 |
| 8 | 256 | 109.1 | 34.9 | 3.13x | 1.0e-08 |
| 32 | 256 | 119.0 | 42.9 | 2.77x | 1.0e-08 |
| 128 | 256 | 214.1 | 138.5 | 1.55x | 1.0e-08 |
| 8 | 512 | 227.7 | 68.8 | 3.31x | 3.0e-08 |
| 32 | 512 | 242.5 | 85.9 | 2.82x | 3.0e-08 |
| 128 | 512 | 502.9 | 321.6 | 1.56x | 3.0e-08 |
| 8 | 1024 | 506.6 | 143.5 | 3.53x | 5.0e-08 |
| 32 | 1024 | 491.7 | 180.3 | 2.73x | 5.0e-08 |
| 128 | 1024 | 1131.8 | 741.4 | 1.53x | 5.0e-08 |

Raw JSON, event traces and server logs are under benchmarks/results/phase14.
