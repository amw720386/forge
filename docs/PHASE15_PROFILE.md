# Phase 15 profile

## Timing boundaries and baseline

The first fresh Phase 14 L4 C=8 baseline measured 978.16 short, 728.94 medium,
220.58 long, and 192.87 mixed output tokens/s. Final matched controls use the
pinned worker in both arms; their values are in PHASE15_RESULTS.md. Input counts
are 10/190/910 with output cap 64 and EOS enabled. Client time starts at POST
after the arrival delay and ends at SSE completion. Model load is excluded.

Host clocks measure tokenization/admission, queue wait, planning, sample transfer,
and SSE emission. FORGE_KERNEL_PROFILE uses synchronized CUDA events for kernel
regions; FORGE_SERVING_PROFILE adds stream regions and host stage clocks.
These diagnostic runs are separate from ordinary serving trials. Diagnostic
synchronization changes execution timing, and stream intervals include launch gaps.
GEMM/cast regions overlap; they must not be added together. Capture skips kernel
diagnostics, so initial per-kernel decode samples are incomplete for graph replay.
The final eager context profile supplies additional decode/event evidence.

## Measured device contribution

Initial long/mixed diagnostic totals, including their warmups:

| Region | Samples | Summed CUDA event ms | Mean ms/sample |
|---|---:|---:|---:|
| prefill_attention | 4704 | 1522.05 | 0.3236 |
| gemm | 9595 | 624.96 | 0.0651 |
| gemm_cast | 2755 | 279.12 | 0.1013 |
| kv_append | 4704 | 23.95 | 0.0051 |
| decode_attention | 168 | 77.94 | 0.4639 |
| rope_kv_append | 168 | 1.40 | 0.0084 |

Prefill attention exceeds measured GEMM time by about 2.4x. KV append is small
beside attention. These are sampled kernel-region contributions, not an additive
decomposition of client latency. GPU launch gaps, synchronization, and sampling
also contribute; block lookup is included inside attention rather than separately
attributed. Allocator/upload host intervals are recorded in the final profile log.
No allocator replacement was justified.

## Decode length scaling

Direct FP32 attention uses 14 query heads, two KV heads, head dimension 64,
reversed physical block tables, warmup and three CUDA-event trials. Values below
are tq=1 (one decode row); tq>1 measurements retain the Phase 14 prefill path.

| KV tokens | Original warp ms | Partitioned ms | Ratio |
|---|---:|---:|---:|
| 31 | 0.0158 | 0.0183 | 0.86x |
| 128 | 0.0584 | 0.0175 | 3.34x |
| 512 | 0.2273 | 0.0621 | 3.66x |
| 1024 | 0.5266 | 0.1310 | 4.02x |
| 2048 | 1.0103 | 0.2485 | 4.07x |
| 4096 | 1.8647 | 0.4816 | 3.87x |

The raw short-context partitioned wrapper is slower, so production launches the
original kernel there. Four warps expose more parallel work than one warp for
a long head/row; this does not establish measured SM occupancy. Shared-prefill
tiles 8/16 were slower: at tq=128, tk=1024, approximately 0.696 ms for the existing
kernel versus 1.086/1.395 ms; at tk=4096, 3.368 versus 4.502/5.363 ms.
No claimed memory-traffic reduction is inferred from those timings.

## Audited context counts

The plain-text bodies contain 128/512/1024/2048/4096 tokens. The server's chat
wrapper adds 19, giving actual model inputs of 147/531/1043/2067/4115 tokens.
Saved runtime tokenization logs confirm these counts. The original harness
counted the bodies only; six JSON reports were corrected after GPU shutdown,
with originals preserved under `uncorrected-`. Wire prompts and timing samples
are unchanged; see `context-token-count-audit.json` and its reproducible audit
script. Future harness runs count the complete server wrapper.

FORGE_MAX_SEQ=4608, output cap eight, one full-concurrency warmup and three
trials per cell. TTFT includes prefill, queueing, tokenization, first sampling
and transport. Direct CUDA tests separately cover exact 128–4096 lengths.

| Actual input tokens | C | Baseline TTFT ms | Final TTFT ms | Final effective input tok/s |
|---|---:|---:|---:|---:|
| 147 | 1 | 14.9 | 14.9 | 9850 |
| 147 | 8 | 57.6 | 58.3 | 20157 |
| 531 | 1 | 52.0 | 52.3 | 10144 |
| 531 | 8 | 329.7 | 332.0 | 12797 |
| 1043 | 1 | 140.6 | 142.6 | 7314 |
| 1043 | 8 | 988.9 | 1000.8 | 8337 |
| 2067 | 1 | 446.5 | 455.0 | 4543 |
| 2067 | 8 | 3346.9 | 3369.3 | 4908 |
| 4115 | 1 | 1590.8 | 1597.1 | 2577 |
| 4115 | 8 | 12231.7 | 12386.8 | 2658 |

The effective input rate divides total input tokens by median client TTFT; it is
not pure device prefill throughput. Identical lengths still vary with concurrency,
packed GEMM shapes, chunk size and output work. Long TTFT remains a prefill issue;
the retained decode optimization cannot bring C=8 long TTFT below 500 ms.

Nsight tooling was unavailable in this Studio. Actual SM occupancy, hardware
bandwidth, and register-pressure counters were not measured. HTTP active-slot
samples describe batch occupancy, not SM occupancy. Five-millisecond metrics
polls can miss scheduled-token peaks (even reporting zero during short decode);
those values are lower bounds. Fixed-policy tests assert the actual counts after
every step. The existing prefill_stall_ns counter is host prefill time while
decode waits, not a CUDA-exclusive stall measurement.

Raw evidence: profile-phase14-server.log, accepted-context-profile-server.log,
accepted-attention.jsonl, rejected-shared-tiles.jsonl and accepted-context-*.json
under benchmarks/results/phase15. Numerical and serving qualifications appear
in PHASE15_RESULTS.md.
