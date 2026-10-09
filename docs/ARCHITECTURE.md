# Architecture

```
Client ──HTTP/SSE──► Go API ──C ABI──► Rust Runtime ──C ABI──► C++ Kernels
                         │                    │
                    queue/scheduler      tensors, model,
                    metrics/lifecycle   KV cache, sampling
```

## Ownership

| Layer | Owns |
|---|---|
| Go | HTTP connections, request IDs, queues, SSE framing, metrics |
| Rust | Model weights (FP32 or packed INT8/INT4), activations, KV cache, sampling / batcher state |
| C++ | Numerical kernels over caller-provided buffers (FP32 + fused quant GEMV); persistent thread pool |

Go never mutates Rust-owned tensors. Token text/IDs cross the boundary as copied scalars or short borrowed buffers with an explicit lifetime (poll until consumed).

## Scheduling

1. Request admitted into a bounded Go queue (backpressure → 503 or wait).
2. Scheduler fills free continuous-batcher slots (`forge_batch_admit` = per-request prefill).
3. Shared `forge_batch_step` runs one decode round for all active sequences (`m=B` linears).
4. Events are routed back to the owning HTTP/SSE stream; cancel frees one slot only.

Calls into Rust **block the scheduling goroutine**. Throughput concurrency is batched matmuls, not N independent generate loops.

Prefill uses one **shared** `ForwardScratch`; decode uses shared `BatchScratch` with `m=B`. Per-request state is KV + logits + token list only.

Weight-only quantization (optional): see [QUANTIZATION.md](QUANTIZATION.md). Activations and norms remain FP32.

Checkpoints: HF safetensors **or** native `.forge` (see [CHECKPOINT_FORMAT.md](CHECKPOINT_FORMAT.md)). Memory budgets: [MEMORY_MANAGEMENT.md](MEMORY_MANAGEMENT.md).

## Thread safety

- C++ kernels: persistent thread pool for wide linears; safe if output row ranges don't overlap.
- Rust runtime handle: mutated by the Go scheduler goroutine only.
- Go: one scheduler owns the runtime handle.
