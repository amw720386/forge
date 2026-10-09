# Phase 7 Results — Checkpoints, Memory Safety, Lightning Prep

## Delivered

1. **Memory audit + safeguards** — `MemoryBudget`, RSS/commit reporting, bench caps, Arc-shared tied embeddings, sequential benches, conservative defaults (`FORGE_MAX_SEQ=1024`, `FORGE_MAX_BATCH=4`).
2. **Native `.forge` v1** — versioned header, JSON+table metadata, INT8/INT4/FP32 payloads, SHA-256, atomic write — [CHECKPOINT_FORMAT.md](CHECKPOINT_FORMAT.md).
3. **`forge-quantize`** — offline HF → quantized `.forge` (temp file + rename).
4. **Direct load** — `load_model_path` / `load_forge_path` with optional mmap I/O; **no re-quantization**.
5. **Server** — `--model path/to.forge` supported via FFI `load_model_path`.
6. **Tests** — FP32/INT8/INT4 round-trips, corruption rejection, batch after load, existing suites green.
7. **Lightning prep** — `forge-cloud status` only; no paid APUs — [LIGHTNING.md](LIGHTNING.md).

## Correctness

| Test | Result |
|---|---|
| `checkpoint_roundtrip` (5) | pass |
| INT8 forge load == in-memory quant greedy | pass |
| INT4 forge load == in-memory quant greedy | pass |
| Invalid magic / truncate / data SHA | rejected |
| Full `cargo test --release` | pass (after Phase 7 changes) |

## Startup / memory (measured, Windows, 15.7 GiB host)

| Path | Load wall | Peak working set | Steady weights |
|---|---|---|---|
| HF dir + in-process INT8 quant | **4835 ms** | **2833 MiB** | 1020 MiB |
| `.forge` INT8 direct load | **1555 ms** (~3.1× faster) | **2047 MiB** | 1020 MiB |

Decode after either path ≈ **29–30 tok/s** (16 tokens, short prompt).  
`forge-quantize` peak during FP32 materialize ≈ **2826 MiB**; after quant RSS ≈ **1021 MiB**.  
Tied-embedding `Arc` share dropped reported FP32 weights from ~2404 → **1885 MiB**.

## Memory methodology notes

- In-process estimates ≠ OS hard limits (Job Objects / cgroups needed for enforcement).
- Windows: report working set + commit via `GetProcessMemoryInfo`.
- Do **not** load FP32+INT8+INT4 simultaneously; use `forge-bench --backend …`.

## Commands

```powershell
# Offline quantize (writes models/*.forge — gitignored)
cargo run --release --bin forge-quantize -- `
  --model ../models/Qwen2.5-0.5B-Instruct --bits 8 --group-size 64 `
  --output ../models/qwen2.5-int8.forge

# Single-backend bench
cargo run --release --bin forge-bench -- --model ../models/qwen2.5-int8.forge --backend forge

# Server
go run ./api/cmd/forge-server -model models/qwen2.5-int8.forge -max-seq 1024

# Lightning (no spend)
go run ./api/cmd/forge-cloud status
```

## INT4 follow-up (pre–Phase 8)

Safe re-measurement with `forge-int4-diag` (separate processes, page-fault logging) showed the Phase 6 **~4.3 tok/s INT4** figure was **not** caused by paging after memory fixes — decode Δfaults ≈ 0 while kernel microbenches were **7–11×** slower than INT8.

Targeted fix: fused AVX2 INT4 unpack in `kernels/src/linear_quant.cpp` (no `tmp[]` store/reload). Post-fix INT4 decode **~20 tok/s** vs INT8 **~26 tok/s** (same methodology). Details in [PHASE6_RESULTS.md](PHASE6_RESULTS.md).

Harness: `runtime/src/bin/forge_int4_diag.rs`, `scripts/bench_int4_compare.ps1`.

## Known limitations

- `.forge` tensors are copied into owned buffers (mmap is I/O assist, not zero-copy kernels yet).
- Tokenizer still loaded from an HF directory beside the checkpoint.
- INT4 is now near INT8 on CPU; integer VNNI paths still unused.
- Lightning start/benchmark not implemented (by design this phase).

## Phase 8 candidates

- Zero-copy mmap weight views with explicit lifetimes.
- Batched prefill; lazy KV allocation.
- Optional VNNI / further INT4+INT8 kernel polish.
- Lightning provision + remote bench with spend limits.
- CUDA backend (separate from orchestration).
