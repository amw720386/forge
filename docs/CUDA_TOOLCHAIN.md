# CUDA toolchain (pinned for Phase 8B)

Reproducible Linux GPU build expectations. Match Lightning Studio L4 images where possible.

## Pinned / validated versions (Lightning L4, 2026-10-08)

| Component | Version |
|-----------|---------|
| GPU | NVIDIA L4 (24 GiB) |
| Driver | 580.178.04 |
| CUDA toolkit / nvcc | **13.0.88** |
| cmake | ≥ **3.28** |
| g++ | **13.x** (Ubuntu 24.04) |
| Rust / cargo | **stable 1.99+** (rustup) |
| cuBLAS | from CUDA toolkit |

Override architecture list if needed:

```bash
cmake -S kernels -B kernels/build-cuda \
  -DFORGE_BUILD_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="89" \
  -DCMAKE_BUILD_TYPE=Release
```

L4 is sm_89; default CMake list is `75;86;89`.

## Cargo features

| Feature | Links | Purpose |
|---------|-------|---------|
| _(none)_ | CPU kernels only | Windows / default CI |
| `cuda-stub` | `forge_cuda_stub.c` | Compile/link check of Rust CUDA path **without** nvcc/GPU |
| `cuda` | `libforge_cuda.a` + cudart + cublas | Real backend; **build fails** if lib missing |

Do **not** enable `cuda` and `cuda-stub` together.

```bash
# Local / CI (no GPU)
cargo build --release --features cuda-stub
cargo test --release --features cuda-stub --test cuda_compile_check

# Linux with toolkit (GPU optional for link)
cmake -S kernels -B kernels/build-cuda -DFORGE_BUILD_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build kernels/build-cuda -j
export FORGE_CUDA_LIB_DIR=$PWD/kernels/build-cuda/cuda
cargo build --release --features cuda
```

Helper: `scripts/verify_cuda_compile.sh`

## Env

| Variable | Meaning |
|----------|---------|
| `FORGE_CUDA_LIB_DIR` | Directory containing `libforge_cuda.a` |
| `CUDA_HOME` / `CUDA_PATH` | Toolkit root for linker search |
| `FORGE_MODEL_DIR` | Qwen HF directory for remote full-model |
| `FORGE_MAX_NEW_TOKENS` | Cap generate (default 16) |

## Status labels

- **Compile-verified (stub):** Rust CUDA modules typecheck/link against stub.
- **Link-verified (real):** `libforge_cuda` built and Rust `--features cuda` links.
- **GPU-verified:** tests executed on NVIDIA hardware (see `PHASE8B_RESULTS.md`).

Full-model Qwen inference is **not** claimed working until GPU-verified.

## Studio upload caveat

Lightning `UploadFolder` into a reused tree may leave stale files (Phase 8B attempt 3).
`forge-cloud validate-phase8b` now uses an **isolated deploy**:

1. Local content-hashed tar (`prepare-deploy` / `BuildDeployBundle`) — no `.env` or weights  
2. Extract to fresh `forge-deploys/<deploy_id>/` (not in-place overwrite)  
3. `scripts/verify_forge_deploy.sh` — manifest hashes + obsolete-import rejection (`DEPLOY_VERIFY_OK`)  
4. Symlink `forge` → active deploy; models cached under `forge-cache/`  

If verification fails, the session aborts before wasting GPU compile time.
