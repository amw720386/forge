#!/usr/bin/env bash
set -euo pipefail
export PATH="${HOME}/.cargo/bin:/usr/local/cuda/bin:${PATH}"
. "${HOME}/.cargo/env" 2>/dev/null || true
ROOT=/teamspace/studios/this_studio/forge-current
test -d "$ROOT" || ROOT=/teamspace/studios/this_studio/forge
export FORGE_CUDA_LIB_DIR="$ROOT/kernels/build-cuda/cuda"
export LD_LIBRARY_PATH="${FORGE_CUDA_LIB_DIR}:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${FORGE_CUDA_LIB_DIR}:${LIBRARY_PATH:-}"
export RUSTFLAGS="-C link-arg=-Wl,-rpath,${FORGE_CUDA_LIB_DIR}"
MODEL=/teamspace/studios/this_studio/forge-cache/models/Qwen2.5-0.5B-Instruct
cd "$ROOT/runtime"
echo LABEL=phase11_bf16_focused
export FORGE_CUDA_DTYPE=bf16
cargo run --release --features cuda --example bench_cuda_batch -- "$MODEL" 16
echo PHASE11_BF16_OK
