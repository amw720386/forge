# Phase 8A Results — CUDA Foundation & Lightning Orchestration

## Configuration

| Item | Result |
|---|---|
| Repo-root `.env` loading (`godotenv`) | **Works** (process env > `.env` > defaults) |
| `--env-file` override | Supported |
| Discovery from `api/` subdirectory | **Works** |
| Lightning auth | **OK** — org `amw720386-org`, teamspace `general` |
| Credentials in Git | **Never tracked**; `.env` gitignored |
| API key logging | Redacted (`set(len=N)` only) |

**Security incident (mitigated):** A PowerShell UTF-8 BOM rewrite briefly caused `godotenv` to embed `.env` contents in an error string. Loaders now **never** wrap godotenv parse errors. **Rotate `LIGHTNING_API_KEY` in the Lightning UI** if this log may have been retained.

## CUDA validation (executed on real GPU)

| Item | Value |
|---|---|
| GPU | **NVIDIA L4** (23034 MiB) |
| Driver | 580.178.04 |
| CUDA toolkit | **13.0** (`nvcc` 13.0.88) |
| Host compiler | GCC 13.3.0 |
| Build | CMake `-DFORGE_BUILD_CUDA=ON`, arches default 75;86;89 |
| `forge_cuda_tests` | **All passed** on device |
| `forge_cuda_bench` | **Ran** (see below) |
| Full-model inference | **Not run** (out of Phase 8A scope) |

### Correctness

Remote `forge_cuda_tests` on NVIDIA L4:

- Alloc/free cycles, invalid-arg checks
- FP32 GEMM (hand-checked 2×2 + larger non-square vs CPU when linked)
- Matvec, RMSNorm, SiLU, causal Softmax
- H2D/D2H + `cudaMemGetInfo`

Result: `All forge_cuda tests passed`

### Microbenchmarks (CUDA events)

| Metric | Result |
|---|---|
| Alloc/free 4 MiB | 0.190 ms/iter |
| H2D 64 MiB | 5.336 ms (**11.71 GiB/s**) |
| D2H 64 MiB | 5.370 ms (**11.64 GiB/s**) |
| GEMM 1024³ FP32 | 0.128 ms/iter (**~16745 GFLOP/s** GPU event) |
| Matvec 4096×4096 | 0.780 ms/iter |
| SiLU launch | 0.0023 ms/launch |
| Free / total VRAM | 22369 / 22563 MiB |

These are **kernel microbenches**, not end-to-end model tok/s.

## Cloud usage

| Item | Value |
|---|---|
| Paid sessions started | **2 Studio starts** on the same `forge-gpu` (first attempt aborted after status-semantics bug; second completed validation) |
| Successful validation wall | ~2.5 minutes compute work after ready |
| First attempt waste | ~8 minutes waiting on misinterpreted `CLOUD_SPACE_STATE_READY` |
| Planned estimate | L4 20m ≈ **$0.22** at $0.65/h |
| Spend limit | **$3** |
| Actual invoice | **Not available via SDK** — treat ≈ **$0.25–$0.45** as estimate for both starts; confirm in Lightning billing |
| Shutdown | **Verified**: `Stop` API ok, instance unreachable (`PublicIP` clear / `Run` fails) |
| Second session after success | **Not started** |

### Shutdown notes

- `GetStudio().Status()` often stays `CLOUD_SPACE_STATE_READY` (cloudspace exists) even when compute is stopped — **do not** treat that as “still billing.”
- Verification uses `studio.Stop()` success + instance probes (`PublicIP` / failed `Run`).
- Provider auto-sleep (15 min) and `MaxRuntime` (20 min) remain backstops if the CLI dies.

## Project deliverables

- `.env` / `.env.example`, `godotenv`, `--env-file`
- `forge-cloud auth|validate-cuda` + fixed instance wait/stop
- CUDA lib + remote GPU proof
- Docs: LIGHTNING, CUDA_ARCHITECTURE, ROADMAP, README

## Known limitations

- No full Qwen CUDA forward (Phase 8B)
- Hourly rates are planning estimates
- First validation attempt burned extra minutes due to status-string bug (fixed)
- Rotate API key after the BOM error exposure

## Phase 8B readiness

CUDA primitives are **GPU-validated**. Next: wire transformer weights/activations through `CudaBackend`, device KV, then token generation — only after another explicit spend approval.
