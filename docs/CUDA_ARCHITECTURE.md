# CUDA Backend Architecture (Phase 8A–11)

## Phase 16 model-neutral integration

`CudaTransformer` and CPU `Transformer` construct the same validated `DecoderPlan`.
The Qwen and Llama adapters load family-specific biases and metadata into the existing
decoder weights; CUDA forward, paged KV, packed prefill, batching and sampling kernels
are shared. SmolLM2-135M-Instruct is tested on an actual NVIDIA L4 in FP32 and BF16;
BF16 has experimental numerical qualifications in [PRECISION_CONTRACTS.md](PRECISION_CONTRACTS.md).

Graph keys now include architecture, dimensions, RoPE/norm constants, tying,
precision, unique model allocation generation, batch/path, KV geometry and captured
buffer addresses. A recycled object address cannot identify a new weight allocation.
Incompatible workspace dimensions fail before execution. Identity/dispatch changes
invalidate capture, followed by eager warmup; dynamic session metadata remains outside
the graph. Existing cancellation, table replacement and KV growth behavior is retained.
L4 tests explicitly reuse graph scratch across Qwen destruction and Llama creation.
See [PHASE16_DESIGN.md](PHASE16_DESIGN.md) and [PHASE16_RESULTS.md](PHASE16_RESULTS.md).

## Goals

Add an independently testable **NVIDIA CUDA compute foundation** beside the existing CPU kernels, without changing the Go→Rust→C++ ownership model or requiring the CUDA toolkit on Windows developer machines.

**Phase 8A:** device memory + FP32 primitives (validated on L4).  
**Phase 8B:** full Qwen2.5 decoder on GPU — see `runtime/src/cuda_model/`.  
**Phase 9:** GPU continuous batching (`m=B` linears, per-request contiguous KV) — [PHASE9_RESULTS.md](PHASE9_RESULTS.md).  
**Phase 10:** paged KV + chunked prefill + token-budget scheduling — [PHASE10_DESIGN.md](PHASE10_DESIGN.md), [PHASE10_RESULTS.md](PHASE10_RESULTS.md).  
**Phase 11:** optimized paged attn, BF16 weights, GPU greedy argmax — [PHASE11_DESIGN.md](PHASE11_DESIGN.md), [PHASE11_RESULTS.md](PHASE11_RESULTS.md).

## Layering (unchanged ownership)

```text
Client ──HTTP/SSE──► Go API ──C ABI──► Rust Runtime ──C ABI──► Kernels
                                              │                    │
                                         compute::Backend     CPU (always)
                                         DeviceBuffer         CUDA (optional)
```

| Layer | Role |
|---|---|
| Go | Scheduling, SSE, **Lightning orchestration** (`forge-cloud`) — never CUDA details |
| Rust | `compute::Backend` trait; owns host tensors / future device tensors; selects CPU vs CUDA |
| C++ CPU | Existing `forge_kernels_*` (unchanged hot path for production CPU) |
| C++/CUDA | Optional `forge_cuda_*` library: device memory + FP32 primitives |

## Compute backend interface (Rust)

Minimal surface — not a full tensor framework:

- Device identity / availability
- Device buffer allocate / free (RAII)
- H2D / D2H copies + synchronize
- `matmul_f32`, `matvec_f32`, `rmsnorm_f32`, `softmax_f32`, `silu_f32`, `add_f32`, `mul_f32`
- Phase 8B: `linear_f32`, `rope_f32`, `embedding_f32`, `gqa_attn_f32`, `kv_append_f32`, `memcpy_d2d`
- Phase 10: `paged_gqa_attn_f32`, `paged_kv_append_f32` (block-table KV; contiguous kernels remain the oracle)
- Phase 11: optimized `paged_gqa_attn_f32` (+ `paged_gqa_attn_naive_f32` reference), `linear_bf16_w`, `f32_to_bf16`, `embedding_bf16`, `argmax_rows_f32`
- Error mapping to `TensorError` / `ComputeError`

CPU backend wraps today’s FFI. CUDA backend is compiled only with `--features cuda`.

## CUDA library layout

Existing CPU sources stay in `kernels/src/`. New code lives under:

```text
kernels/cuda/
  include/forge_cuda.h      # C ABI
  src/
    error.hpp / device.cu / memory.cu
    matmul.cu / matvec.cu / rmsnorm.cu / softmax.cu / activation.cu
  tests/                    # optional; needs GPU
  bench/                    # CUDA event microbenches
  CMakeLists.txt            # FORGE_BUILD_CUDA=ON
```

### Design choices

| Topic | Choice |
|---|---|
| GEMM | **cuBLAS** `cublasSgemm` initially (correct, fast baseline) |
| Matvec / norms / activations | Custom CUDA kernels (educational + low launch overhead) |
| Softmax | Custom kernel with shared-memory reduction (rows independent) |
| Memory | `cudaMalloc` / `cudaFree` behind RAII `ForgeCudaBuffer`; explicit sync |
| Errors | `forge_cuda_status` + `forge_cuda_last_error_string()`; never silent ignore |
| Build | CMake CUDA language; Rust `cc`/`cmake` feature gate; Windows default = CPU-only |

## CPU-specific assumptions to dig out later (8B)

- Persistent CPU thread pool (`ThreadPool`) for parallel row ranges
- AVX2/FMA paths in `simd.hpp`
- Weight-only INT8/INT4 fused CPU GEMV
- Host-resident `.forge` weight buffers passed as raw pointers to kernels
- Continuous batching assumes host scratch layouts

Phase 8A does **not** move the transformer forward onto GPU. It only makes device primitives available for tests and future wiring.

## Feature flags

| Flag | Effect |
|---|---|
| (default) | CPU kernels only — works on Windows without CUDA |
| Rust `--features cuda` | Link `forge_cuda`, enable `compute::CudaBackend` |
| CMake `-DFORGE_BUILD_CUDA=ON` | Build CUDA lib + optional tests/benches |

## Numerical validation policy

- Host CPU reference remains the source of truth for FP32 primitives.
- CUDA vs CPU tests use tolerances ~`1e-4` relative / `1e-5` absolute for GEMM-sized work.
- **Do not claim GPU parity until tests execute on a real GPU** (Lightning Studio).
- Local Windows CI runs CPU scaffolding + “CUDA unavailable” skip paths only.

## Lightning remote workflow

See [LIGHTNING.md](LIGHTNING.md). Preferred path:

1. Develop on Windows (CPU); secrets in repo-root `.env`.
2. `forge-cloud auth` (non-billable).
3. `forge-cloud validate-cuda --gpu l4 --confirm` — upload `kernels/`, build, test, bench, **always stop**.
4. Or: `start` → Git/`UploadFolder` → `exec` → `stop`.

**Status caveat:** `GetStudio().Status()` is cloudspace-level (`CLOUD_SPACE_STATE_READY`). Instance readiness = successful `Studio.Run` after `Start()`.

Phase 8A GPU proof: [PHASE8A_RESULTS.md](PHASE8A_RESULTS.md).  
Phase 8B plan/results: [PHASE8B_PLAN.md](PHASE8B_PLAN.md), [PHASE8B_RESULTS.md](PHASE8B_RESULTS.md).

### Phase 8B execution path

1. Load FP32 weights on CPU (`load_model_path`); BF16/F16 cast to FP32 at load; reject quant for `--device cuda`.
2. `CudaTransformer::from_cpu` — one H2D upload; tied `lm_head` uses embed device buffer (no duplicate).
3. `forward_tokens` — embed → decoder layers (device KV) → final norm → D2D last row → lm_head; logits D2H for sampling only.
4. `generate_cuda` — single-sequence reference path (Phase 8B).
5. **Phase 9** contiguous-KV batching (reference).  
6. **Phase 10** `CudaBatcher` — shared `GpuKvBlockPool` + per-slot `PagedKvSession` block tables; chunked prefill; `forward_batch_decode_paged` / `forward_tokens_paged`. Contiguous `LayerGpuKvCaches` remains for `generate_cuda` / parity.

**Memory ownership:** `Buffer` RAII frees device memory on drop. Weights live for the lifetime of `CudaTransformer`. Phase 10: one device K/V pool for all slots; sessions own block-table entries; cancel/complete returns blocks to the free list. Prefill scratch sized to `FORGE_PREFILL_CHUNK`.

**Layouts:** row-major `[rows, cols]`; linear weights `[out, in]` on device. Paged pool: `k_pool/v_pool[n_layers][num_blocks][block_size][kv_width]`.

**Batch decode (Phase 10):** embed B → batched linears → per-row RoPE + `paged_kv_append` + `paged_gqa_attn` → batched O/MLP/lm_head. Prefill chunks interleaved under `FORGE_STEP_TOKEN_BUDGET`. Instrument via `CudaBatchStats` / `FORGE_CUDA_BATCH_TRACE=1` / `forge_batch_sched_stats`.

Remote validation: `scripts/remote_cuda_phase10.sh` / `scripts/remote_cuda_phase11.sh`.  
CLI: `forge-cloud validate-phase10|validate-phase11 --confirm` (budgets — see [LIGHTNING.md](LIGHTNING.md)).

## Out of scope (through Phase 11)

- Full FlashAttention library / CUDA Graphs / speculative decoding / GPU INT8–INT4 weights (Phase 12+)
- GPU INT8/INT4 weight kernels
- Multi-GPU / distributed


## Phase 13 shared decode and graph execution

The opt-in path uses compact device per-row table/position/length metadata, shared RoPE+KV prepare and warp/head/row online-softmax attention. BF16 normalized inputs are reused across Q/K/V and gate/up. CUDA graph instances are owned by batch scratch, keyed by shape and model/pool/cast addresses; host metadata and sampling stay outside capture. Per-thread streams align kernels and cuBLAS. Prefill remains causal and eager; unsupported shapes retain the existing reference. See [PHASE13_DESIGN.md](PHASE13_DESIGN.md) for flags, lifetime rules and fallback limits.

## Phase 14 execution paths

[PHASE14_DESIGN.md](PHASE14_DESIGN.md) describes shared ragged prefill, borrowed
buffer ranges, aggregate KV preflight and completed-prompt-only LM head. Scratch
owns reusable BF16 casts. Four-warp causal prefill attention combines online
softmax in FP32; decode keeps its original kernel. Selective L4 cuBLASLt plans
cache descriptors/heuristics and a 32 MiB workspace with GemmEx fallback.
Packed prefill is eager; decode graphs retain scratch invalidation and now key
on Lt policy and BF16 cast address. All new paths are opt-in.

Measured flags: batched paged + BF16 reuse + graphs + packed prefill + block
attention + Lt + early emit; chunk 128, step budget 1024, max batch 8. See the
design for exact environment names. GPU tests cover mixed lifecycle and graph
fallback; real-model BF16 differences remain documented, not bitwise parity.

## Phase 15 dispatch and worker ownership

[Design](PHASE15_DESIGN.md) extends four-warp online softmax to paged decode
without an attention matrix or additional device buffers. Host dispatch uses
the original kernel for contexts below 128, and for BF16-weight batches above
two with any context beyond 512. FP32 can use the partitioned path at all
supported B=1..8 shapes above the short-context threshold.

The dispatch bit joins the graph identity, so growth or replacement across a
threshold invalidates capture/warmup state. Metadata remains dynamically
uploaded and prefill remains eager. The original C entry point is retained;
the explicit dispatch entry point carries the host decision.

The Go scheduler pins its execution thread because kernels, cuBLAS and graph
capture use the per-thread default stream. Admission, stepping and cancellation
share that owner. Cargo watches the selected CUDA archive so incremental
rebuilds relink. Shared prefill/softmax tile experiments were removed; the
Phase 14 prefill kernel and paged allocator remain. CPU Windows paths are
preserved. BF16 numerical qualifications and actual L4 regressions are in
[results](PHASE15_RESULTS.md).
