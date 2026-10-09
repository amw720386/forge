# Phase 8B Results — Full Qwen CUDA Inference

**Status: GPU-verified (2026-10-08)** — Qwen2.5-0.5B greedy CUDA matches CPU FP32 on NVIDIA L4.

## Attempt 5 — development session (authorized ≤$2 / 30m)

| Item | Value |
|------|-------|
| GPU | **NVIDIA L4** |
| Studio | `forge-gpu` (single start) |
| Estimate | L4 30m ≈ **$0.33** @ $0.65/h (cap **$2**) |
| Deploy | Isolated tar `20261008T131454Z-e8a399deb4e4` — **DEPLOY_VERIFY_OK** |
| Shutdown | **Verified**: `before=running after=stopped (instance unreachable)` |

### Stages

| Stage | Result |
|-------|--------|
| Source deploy + content-hash verify | **PASS** |
| CMake `libforge_cuda` + `forge_cuda_tests` | **PASS** |
| Rust `--features cuda` link | **PASS** (after `CUDA_SEPARABLE_COMPILATION OFF` + `-fuse-ld=bfd`) |
| `cuda_compile_check` | **PASS** (2) |
| `cuda_transformer_parity` | **PASS** (2) |
| Qwen2.5-0.5B GPU load | **PASS** (~1884 MiB weights) |
| Greedy generate | **PASS** |
| CPU FP32 token parity | **PASS** — `TOKEN MATCH` |
| Prefill / decode bench | **PASS** |
| Studio stop | **PASS** |

### Generated tokens (greedy, prompt `Hello!`)

```
tokens=[9707, 0, 2585, 646, 358, 7789, 498, 3351, 30, 151645]
text: Hello! How can I assist you today?
```

Stopped at EOS (`151645`) after **10** new tokens (max_new=16). CUDA == CPU.

### Throughput (wall-clock + `synchronize`, L4)

| Metric | Value |
|--------|-------|
| GPU prefill | **0.049 s** ≈ **204 tok/s** (10 prompt tokens) |
| GPU generate (prefill+decode+sample) | **0.507 s**, 10 new tokens ≈ **19.7 tok/s** |
| CPU generate (same prompt) | **0.691 s**, 10 new tokens ≈ **14.5 tok/s** |
| GPU weight upload | **0.38 s**, ~1884 MiB |
| VRAM free/total after load | ~20337 / 22563 MiB |

### Fixes applied during the session (synced locally)

1. CLI reattach: ID-based Studio handle with `Status: "running"` so `exec`/`upload` work after a separate `start`.
2. `CUDA_SEPARABLE_COMPILATION OFF` — static `libforge_cuda.a` no longer needs nvcc device-link under cargo.
3. `build.rs`: `-fuse-ld=bfd` for Linux CUDA links (avoid rust-lld CUDA archive issues).
4. `bench_cuda_qwen`: import `ComputeBackend` for `synchronize()`.
5. Session workflow: keep GPU up across recoverable deploy/build errors; stop only at end / budget.

## Prior attempts

| # | Outcome |
|---|---------|
| 1 | CRLF + missing Rust → stopped |
| 2 | Private import E0603 → stopped; fixed locally |
| 3 | Stale Studio upload → same E0603 → stopped |
| 4 | Tar uploaded but install looked wrong path / raced → stopped early |
| **5** | **Success** — deploy verify, CUDA link, Qwen greedy parity, bench, stop |

## Phase 8B DoD

| Criterion | Met? |
|-----------|------|
| CUDA kernels compile on NVIDIA | **Yes** |
| Rust links real CUDA | **Yes** |
| Transformer numerical tests on GPU | **Yes** |
| Qwen GPU load + generate | **Yes** |
| Greedy token agreement vs CPU FP32 | **Yes** |
| Prefill/decode measured | **Yes** |
| Studio stopped | **Yes** |

**Phase 8B full-model GPU path: verified on NVIDIA L4.**
