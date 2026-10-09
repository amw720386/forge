#!/usr/bin/env bash
# Focused Phase 11 BF16 validation on an already-started Studio (no provision).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
MODEL_DIR="${FORGE_MODEL_DIR:-/teamspace/studios/this_studio/forge-cache/models/Qwen2.5-0.5B-Instruct}"
MAX_NEW="${FORGE_MAX_NEW_TOKENS:-16}"

export PATH="${HOME}/.cargo/bin:/usr/local/cuda/bin:${PATH}"
. "$HOME/.cargo/env" 2>/dev/null || true

export FORGE_CUDA_LIB_DIR="$ROOT/kernels/build-cuda/cuda"
export LIBRARY_PATH="${FORGE_CUDA_LIB_DIR}:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${FORGE_CUDA_LIB_DIR}:${LD_LIBRARY_PATH:-}"
export RUSTFLAGS="${RUSTFLAGS:-} -C link-arg=-Wl,-rpath,${FORGE_CUDA_LIB_DIR}"

cd "$ROOT/runtime"
echo "LABEL=phase11_bf16_focused"
export FORGE_CUDA_DTYPE=bf16
cargo run --release --features cuda --example bench_cuda_batch -- "$MODEL_DIR" "$MAX_NEW"
# Spot-check: FP32 generate token match still on default path
unset FORGE_CUDA_DTYPE
cargo run --release --features cuda --example bench_cuda_qwen -- "$MODEL_DIR" "$MAX_NEW"
echo "PHASE11_BF16_OK"
