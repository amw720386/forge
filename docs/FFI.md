# FFI Contract

Chain: **Go → Rust C ABI → C++ kernels**.

## Error codes

Shared integer codes (`forge_status`):

| Code | Meaning |
|---|---|
| 0 | OK |
| 1 | Invalid argument (null, bad shape) |
| 2 | Dimension mismatch |
| 3 | Out of memory |
| 4 | Internal error |
| 5 | Busy / backpressure (batch full) |
| 6 | Not found |
| 7 | Finished |

## C++ kernels (`forge_kernels.h`)

- Buffers are non-owning `float*`.
- Caller guarantees alignment/lifetime for the duration of the call.
- Functions are pure compute; no heap allocation in hot paths (Phase 1).
- Thread-safe given disjoint output buffers.

## C++ vs Rust symbol names

| Layer | Matmul symbol |
|---|---|
| C++ kernels | `forge_kernels_matmul_f32` |
| Rust C ABI (for Go) | `forge_matmul_f32` |

## Rust runtime C ABI (`forge_runtime.h`)

Exports:

- `forge_version()` — null-terminated version string (static)
- `forge_matmul_f32` / `forge_matvec_f32` — Phase 1 compute smoke tests
- `forge_runtime_load(dir, max_seq, err, err_len)` → opaque handle (owns model + continuous batcher)
- `forge_runtime_free(handle)`
- **Continuous batching:** `forge_batch_admit` / `forge_batch_step` / `forge_batch_cancel` / `forge_batch_capacity|active|free_slots`
- **Compat single-flight:** `forge_generate_begin` / `forge_generate_next` / `forge_generate_cancel` (implemented on top of the batcher)

`temperature_milli` is `temperature * 1000` (integer) to avoid Windows x64 float syscall ABI issues.
`FORGE_MAX_BATCH` (default 8) sets continuous-batcher capacity; `FORGE_NUM_THREADS` sets the linear thread pool.

## Go binding strategy

- **Unix:** cgo links `libforge_runtime` directly.
- **Windows:** loads `forge_runtime.dll` at runtime (`FORGE_RUNTIME_DIR` or `runtime/target/release`) to avoid MSVC/MinGW link mismatches.

### Memory

- Rust allocates and frees model/runtime objects.
- Go passes raw pointers only for inputs it owns (prompt bytes, host float buffers in smoke tests).
- Outputs: copy into Go-owned memory or poll short-lived slots before the next generate call.

### Blocking

- `forge_runtime_load` / `forge_generate_begin` / `forge_generate_next` **block** the calling goroutine.
- Token streaming is poll-based (`generate_next`), not callbacks.

### Cancellation

`forge_batch_cancel(id)` / `forge_generate_cancel` free that request's slot without touching siblings. No async callbacks into Go.

### Continuous batching

The Go scheduler admits into free slots and runs shared `forge_batch_step` rounds (batched `m=B` linears). Queue wait + `FORGE_ERR_BUSY` are backpressure — not thread-per-request fake parallelism.
