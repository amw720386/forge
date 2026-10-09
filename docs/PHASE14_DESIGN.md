# Phase 14 design

Opt-in packed prefill, prefill attention, L4 GEMM selection and earlier token
delivery preserve the CPU backend and existing CUDA paths. See
[profile](PHASE14_PROFILE.md) and [results](PHASE14_RESULTS.md) for measurements.

## Packed ragged prefill

`CudaTransformer::forward_prefill_paged` packs selected request chunks into token
rows without padding. One transformer layer loop runs shared normalization,
Q/K/V, output and MLP GEMMs. Only RoPE positions, paged KV append and causal
attention operate per request; there are no per-request whole-model loops.

Bounds-checked borrowed `BufferView` ranges retain parent allocation ownership.
Shapes, token IDs, session capacities and aggregate KV availability are checked
before mutation. Session tables and lengths remain independent. Only completed
prompts gather their last hidden row and run final norm plus shared LM head.
Partial chunks skip unused vocabulary logits. Existing GPU greedy argmax and
non-greedy sampling handle completed rows.

Scratch is bounded by token budget and chunk-size times batch limit, with at
least one chunk. Scratch owns reusable BF16 casts. Normalization, softmax
accumulation and KV storage retain FP32. Existing round-robin prefill selection
and decode priority remain in force.

## Attention and GEMM

`FORGE_PREFILL_ATTN_BLOCK=1` selects four warps per query/head for multi-token
prefill, KV length at least 128 and head dimension at most 128. Warps traverse
disjoint logical blocks, apply causal bounds and combine online-softmax maxima,
denominators and weighted values in FP32. Other shapes use the original kernel.
This still rereads KV across queries and does not change single-token decode.
Reversed physical block mappings are tested.

`FORGE_GEMM_LT=1` enables offline-measured BF16 shape choices only on exact
`NVIDIA L4`. Plans cache descriptors/heuristics by M/N/K/strides and reuse a
32 MiB thread-local workspace. Missing plans during graph capture, unsupported
shapes/devices and library failures fall back to GemmEx. No runtime timing search
is introduced. LM head is unchanged; gains concentrate in selected small KV
projections and MLP-down shapes.

## Delivery and graphs

`FORGE_EARLY_EMIT=1` returns ready events before another GPU round; pending work
runs next iteration. This changes TTFT and the internal benchmark's steady-rate
denominator, so that rate is not an isolated decode-compute comparison.
Packed prefill runs eagerly. Decode retains graph replay, with Lt policy and
BF16 cast address in the graph key. Existing scratch-growth invalidation prevents
stale captured pointers.

## Validated configuration

```bash
FORGE_BATCHED_PAGED=1
FORGE_REUSE_BF16=1
FORGE_CUDA_GRAPHS=1
FORGE_PACKED_PREFILL=1
FORGE_PREFILL_ATTN_BLOCK=1
FORGE_GEMM_LT=1
FORGE_EARLY_EMIT=1
FORGE_PREFILL_CHUNK=128
FORGE_STEP_TOKEN_BUDGET=1024
FORGE_CUDA_MAX_BATCH=8
```

Flags are opt-in. The final run uses token budget 1024; fresh Phase 13 uses 256.
Ablations report this explicitly. `FORGE_SERVING_PROFILE=1` adds synchronization
and timers and must be disabled in throughput runs.

## Verification

Real L4 tests cover FP32/BF16 B=1/2/4/8, unequal chunks, block crossings, partial
prompts, cancellation/replacement, released KV capacity and graph B=9 fallback.
Full-checkpoint prefixes additionally test greedy stability. Synthetic tolerance
is unchanged; real BF16 logits are not bitwise equivalent. Arbitrary driver OOM
recovery is not established by capacity-admission tests.

```bash
cargo test --release --features cuda --test cuda_phase14 -- --test-threads=1
FORGE_TEST_MODEL=/path/to/model cargo test --release --features cuda \
  --test cuda_phase14 -- --ignored --test-threads=1
cargo run --release --features cuda --example phase14_bf16_diagnostic -- /path/to/model
```

CPU tests and CUDA-stub compilation are regression checks, not GPU proof. The
diagnostic feature guard and exact-L4 predicate were cleaned up after shutdown;
they preserve tested CUDA arithmetic and the executed L4 branch.
