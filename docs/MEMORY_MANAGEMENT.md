# Memory management

## What consumed RAM in Phase 6 benches

1. **Multiple full models in one process** (`bench_phase6` originally kept FP32 + INT8 + several INT4 live) → multi-GiB peaks.
2. **Tied embedding clone** → extra ~519 MiB FP32 (fixed in Phase 7 via `Arc` share).
3. **Safetensors load** keeps the BF16 file buffer while building FP32 tensors (~0.9 + 2.4 GiB transient).
4. **Prefill attention scores** `O(n_heads · S²)` dominate large `max_seq`.
5. **Continuous batcher** pre-allocates `B` KV caches at construct time.

## Phase 7 controls

| Variable | Default | Role |
|---|---|---|
| `FORGE_MAX_SEQ` | 1024 | Cap sequence / scratch / KV depth |
| `FORGE_MAX_BATCH` | 4 | Continuous-batcher slots |
| `FORGE_MEM_BUDGET_MIB` | 8192 | Soft estimate; load fails if exceeded |
| `FORGE_MEM_RAM_FRACTION` | 0.55 | Also fail if estimate > fraction of host RAM |
| `FORGE_BENCH_MEM_CAP_MIB` | 6144 | Bench refusal threshold |
| `FORGE_BENCH_RAM_FRACTION` | 0.45 | Bench vs physical RAM |
| `FORGE_MMAP` | true | mmap `.forge` files for I/O (data still copied into owned buffers) |

`MemoryBudget::estimate_runtime_bytes` sums weights + prefill scratch + `B·KV` + batch scratch (+15% headroom).

## Windows accounting

Prefer **working set** and **commit charge** (`ProcessMemory` via `GetProcessMemoryInfo`) over virtual address size. mmap can inflate virtual space without resident pages.

## Limits of in-process estimates

Forge **cannot** hard-cap the OS. Job Objects (Windows) or cgroups (Linux) are required for true enforcement. Budgets refuse unsafe configs; they do not replace OS limits.

## Ownership

| Object | Owner | Freed when |
|---|---|---|
| Weights | Rust `Transformer` | `forge_runtime_free` / drop |
| KV / scratch | `ContinuousBatcher` | drop runtime |
| `.forge` mmap | `LoadedCheckpoint` (optional `Arc<Mmap>`) | drop after tensors copied |
| Go `Runtime` | caller | `Close()` |

Kernels receive borrowed pointers valid for the call only; weight storage must outlive all inference.

## Phase 10 GPU paged KV

CUDA serving no longer allocates `max_seq` contiguous KV per slot at construct time.

| Object | Layout | Freed when |
|---|---|---|
| `GpuKvBlockPool` | Fixed pool: `n_layers × num_blocks × block_size × kv_width × 4` × 2 (K,V) | Drop batcher |
| `PagedKvSession` | Host block table + small device u32 mirror | Cancel / complete returns physical ids to free list |

Defaults: `FORGE_KV_BLOCK_SIZE=16`, `FORGE_KV_NUM_BLOCKS = max_batch × ceil(max_seq/block)`. Qwen2.5-0.5B: **384 KiB per block** (all layers). Admit fails with `KvExhausted` if free blocks < `ceil(prompt_len/block)`.

Contiguous `LayerGpuKvCaches` remains for single-request CUDA / parity oracles only.

Metrics (no pointers): `kv_num_blocks`, `kv_free_blocks`, `kv_allocated_blocks`, `active_prefill`, `active_decode`, `scheduled_*_tokens`.

## Safe local defaults (16–32 GiB)

- Prefer **INT8 `.forge`** checkpoints (no re-quant at load).
- `FORGE_MAX_SEQ=1024`, `FORGE_MAX_BATCH=4`.
- One backend per bench process (`forge-bench --backend …`, or `forge-int4-diag` / `scripts/bench_int4_compare.ps1`).
- INT4’s historical ~4 tok/s was **kernel unpack overhead**, not decode-time paging (Δfaults ≈ 0); see PHASE6_RESULTS.
- Never run unrestricted multi-model stress tests on the laptop.
- Avoid full-model GPU benches on Windows (no NVIDIA GPU); use Lightning within Phase 11 budget ($6 / $2/session).

## Phase 11 GPU weight dtype

| Mode | Weight storage | Activations / KV / norms |
|---|---|---|
| FP32 (default) | FP32 on device | FP32 |
| BF16 (`--dtype bf16` / device=2) | BF16 linears + embed | FP32 (mixed; FP32 accumulate in GEMM) |

BF16 roughly halves dense weight VRAM (~1.9 GiB → ~0.95 GiB for Qwen2.5-0.5B eligible tensors). KV block pool remains FP32.
