# Phase 15 design

Forge retains its own transformer forward passes, packed causal prefill,
paged FP32 KV pool, cuBLAS/Lt, GPU sampling, and decode graphs. CPU Windows
execution and existing APIs remain intact. No Phase 16 implementation was added.

## Retained attention dispatch

Long decode now reuses the Phase 14 four-warp online-softmax implementation.
Each warp processes every fourth logical KV block, loading its physical index
once per block. Four FP32 maxima, denominators, and value accumulators merge
using a common maximum. There is no attention matrix or new device scratch.
GQA mapping, append-before-attention, RoPE positions, and request-specific block
tables are preserved. Prefill remains the original eager causal four-warp path.

`FORGE_DECODE_ATTN_BLOCK=0` selects the original one-warp batched decode.
Otherwise the transformer makes a host decision before launching attention:

| Shape / precision | Retained kernel |
|---|---|
| Any row shorter than 128 KV tokens | Original one-warp |
| FP32, all rows at least 128 tokens | Four-warp partitioned |
| BF16 weights, batch 1 or 2, all rows at least 128 | Four-warp partitioned |
| BF16 weights, larger batch, every row between 128 and 512 | Four-warp partitioned |
| Other BF16 batches | Original one-warp |

A larger long BF16 batch failed a new 64-token teacher-forced greedy check:
at decision 14, a baseline margin of 0.004341 changed sign (-0.002647).
Tiny FP32 attention differences can cross subsequent BF16 rounding boundaries.
That configuration is rejected. The host fallback launches 32 threads directly,
avoiding the slower intermediate implementation that launched 128 threads with
three idle warps. FP32 passed the larger-batch checks and retains that optimization.
The original C entry point remains for standalone references; the new explicit
dispatch entry point lets the transformer enforce the precision policy.

The dispatch bit is part of the existing graph identity along with model, pools,
BF16 scratch address/reuse, and Lt policy. Crossing a dispatch threshold invalidates
the graph and warmup state. Dynamic metadata and KV growth remain outside capture;
prefill is not forced into a graph. Existing unsupported-shape fallbacks remain.

## One CUDA execution thread

The Go inference worker now locks its OS thread for admission, stepping, and
cancellation. CUDA kernels, cuBLAS and capture use the per-thread default stream;
goroutine migration was an unsafe ownership boundary. An output audit found
corrupted first long-batch completions even with graphs disabled. Pinning removed
that corruption across repeated serving checks. Both accepted baseline and
candidate use the thread fix, so its effect is not attributed to attention.
It also confines the thread-local Lt workspace/cache to the worker.

Cargo watches the actual CUDA archive, ensuring an incremental kernel rebuild
relinks Rust instead of silently using an older static library.

## Scheduling

`FORGE_SCHED_POLICY=decode-first` remains the default and preserves Phase 14
behavior. `fixed` enforces a hard combined decode-plus-prefill token budget.
It reserves progress for waiting prefill, rotates decode selection when the
budget cannot serve all decoders, and retains round-robin prefill chunks.
At budget one, it alternates prefill and decode turns. Admission still uses
existing FIFO backpressure and conservative KV capacity reservations.
Legacy decode-first can exceed an exceptionally tiny budget on decode; select
fixed when that hard bound and starvation prevention are required.

The experimental adaptive policy limited query-KV work and prefill tokens when
decode was active. It produced no reliable serving advantage and was removed.
No heuristic predictor or additional production scheduler state remains.

## Prefill and KV decisions

Shared KV tiles across two adjacent query groups, tile sizes 8 and 16, were
implemented and measured. Shared-memory synchronization and additional work
lost to the existing implementation. Softmax score tiles 4 and 8 also failed
to justify their serving complexity. Both experiments were removed; their
sources and measurements are retained as rejected evidence.

Packed prefill grows all selected sessions before uploading their tables.
Optional diagnostics separate host allocation and table-upload costs. The allocator,
physical layout, free list, and capacity-aware admission were retained because
allocation was not a dominant measured cost. Cancellation and reuse reclaim KV
blocks; no stale data or additional full attention storage was introduced.

## Reproduction

Use the existing bounded Lightning workflow and independently verified watchdogs
before GPU execution. `forge-cloud inspect-phase15`, `deploy-phase15`, and
`fetch-phase15` reuse the existing infrastructure. `scripts/phase15_validate.sh`
builds and runs the accepted matrix, regressions, context/pressure/policy tests,
and limited reference comparison. `phase15_fp32_validate.sh` validates FP32
and the BF16 fallback. These scripts do not provision or authorize a GPU.

Measured serving flags: BF16 by default (FP32 via FORGE_BENCH_DTYPE), FP32 KV,
FORGE_BATCHED_PAGED=1, FORGE_REUSE_BF16=1, FORGE_CUDA_GRAPHS=1,
FORGE_PACKED_PREFILL=1, FORGE_PREFILL_ATTN_BLOCK=1, FORGE_GEMM_LT=1,
FORGE_EARLY_EMIT=1, step budget 1024, prefill chunk 128, eight slots.
Use FORGE_MAX_SEQ=4608 for the HTTP context sweep (147–4115 actual input tokens). The ordinary serving
matrix used the existing effective 1024-token runtime cap, despite CLI max-seq
2048; its 910+64-token longest request fits that cap.
