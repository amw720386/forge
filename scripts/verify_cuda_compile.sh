#!/usr/bin/env bash
# Compile/link checks only. Does not provision hardware or run GPU inference.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
cargo test --release --manifest-path runtime/Cargo.toml --features cuda-stub
if ! command -v nvcc >/dev/null 2>&1; then
  echo "CUDA stub checks complete; real CUDA link skipped (nvcc unavailable)."
  exit 0
fi
cmake -S kernels -B kernels/build-cuda -DCMAKE_BUILD_TYPE=Release \
  -DFORGE_BUILD_CUDA=ON -DFORGE_CUDA_BUILD_TESTS=OFF -DFORGE_CUDA_BUILD_BENCH=OFF
cmake --build kernels/build-cuda --parallel
export FORGE_CUDA_LIB_DIR="$ROOT/kernels/build-cuda/cuda"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
cargo build --release --manifest-path runtime/Cargo.toml --features cuda
cargo test --release --manifest-path runtime/Cargo.toml --features cuda --test cuda_compile_check
echo "Real CUDA linkage checked; GPU execution was not tested."
