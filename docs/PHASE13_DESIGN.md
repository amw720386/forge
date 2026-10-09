# Phase 13 execution design

GPU validated on one NVIDIA L4. All three optimizations are opt-in for controlled
A/B measurement: `FORGE_BATCHED_PAGED=1`, `FORGE_REUSE_BF16=1`,
`FORGE_CUDA_GRAPHS=1`. Unset flags preserve the Phase 12 eager reference.
Graphs require batched paged attention. CPU builds need no CUDA toolkit.

## Batched paged decode

Dense projections already share GEMMs. New CUDA prepare/attention kernels share
launches across active sequences; they never assemble a contiguous KV cache.
Each metadata row holds three uint64 values: device logical-block-table address,
RoPE position, and existing KV length. Rust grows sessions and uploads dirty
tables, then copies 24 bytes per active row before execution.

Prepare uses a 256-thread CTA per sequence: split-half RoPE on Q/K and direct
writes to that sequence's physical K/V blocks. It removes Q/K/V staging plus
context-return copies. Attention maps a warp across each query head and batch
row, reads the row's logical table, uses existing GQA head mapping and online
softmax, and attends through existing length plus the newly appended token.
Future allocated KV entries are never read. Existing causal chunked prefill
and the single-request optimized attention kernel remain intact.

Supported shared decode: B≤8, even head dimension≤128. B>8, incompatible
dimensions, and `FORGE_PAGED_ATTN_NAIVE` use the old row path. Different cache
lengths and physical block ownership remain per session.

## BF16 activation reuse

Keep FP32 activations, KV, norms, reductions and GEMM accumulation. Convert
normalized input once for Q/K/V, and once for gate/up, into scratch-owned BF16
storage. New `linear_bf16_xw` accepts these already converted activations.
Other projections retain existing casting. This saves three casts/layer,
72 per Qwen decode round, with approximately 1–2% incremental throughput gain.
There is no global conversion of KV or every activation to BF16.

## CUDA graphs and lifetimes

CUDA compilation uses the per-thread default stream; both cuBLAS handles use
that same stream. Warm one eager decode per shape before capture. Capture the
layer loop, final normalization and LM head. Token upload, embedding, table
growth/upload, metadata upload, greedy sampling and host scheduling stay outside.
Graph nodes read current device metadata, so request replacement and new logical
tables do not bake request positions or table pointers into kernel arguments.

Scratch owns graph executables with RAII destruction, cached by active batch
size (including intermediate B=3/5/6/7). Key includes model address, K/V pool
addresses, BF16 reuse mode and shared BF16 cast buffer address. If prefill grows
the latter, invalidate the cached executable, warm and capture again. Stable
scratch/model/pool ownership is required throughout replay. Graph invalidation
destroys old executables before replacement.

Unsupported shapes execute eagerly. Capture begin/end/body failures discard
capture and disable graph capture for that scratch/shape, then execute eagerly;
they do not trigger a retry each token. Replay errors propagate as request
errors rather than silently advancing KV twice. Cancellation, admission and EOS
remain outside capture. Allocator capacity checks remain before execution.
Capture failure injection and graph-specific forced OOM were not GPU tested;
B=9 eager fallback, block growth, request replacement, and cast-buffer growth
were tested. No graph-memory pressure heuristic was introduced; opt out using
the feature flag when needed.

## Serving and build fixes

HTTP handlers previously discarded the final generated token when its event
also marked completion. Both SSE and non-stream responses now include it;
regression tests cover the behavior. C scheduler stats typedef is now
`forge_batch_sched_stats_t`, avoiding a Linux C typedef/function name collision
while preserving the exported function symbol. Deployment verification now
checks manifest-listed result fixtures instead of accidentally ignoring them.

## Reproduction

Build the actual CUDA shared library, not only examples:

```sh
cd runtime
cargo build --release --features cuda
cargo build --release --features cuda --examples
FORGE_BATCHED_PAGED=1 FORGE_REUSE_BF16=1 FORGE_CUDA_GRAPHS=1 \
  cargo test --release --features cuda --test cuda_phase13 -- --test-threads=1 --nocapture
```

Run environment-mutating CUDA tests serialized. Use `bench_phase12 --help` for
the unchanged internal harness. For real serving use `benchmarks/forge/bench_http.py`.
Read [results](PHASE13_RESULTS.md) for timing boundaries and fairness limits.
