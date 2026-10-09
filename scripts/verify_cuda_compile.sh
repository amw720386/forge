#!/usr/bin/env bash
# Compile-only CUDA verification — no Lightning, no model download, no GPU required.
# 1) cargo build/test --features cuda-stub  (always)
# 2) if nvcc+cmake available: build libforge_cuda + cargo build --features cuda (link only)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Ensure LF on this script's siblings when invoked from a fresh clone on Windows mounts
if command -v sed >/dev/null 2>&1; then
  sed -i 's/\r$//' scripts/*.sh 2>/dev/null || true
fi

bash scripts/preflight_phase8b.sh

echo "=== [1/2] cuda-stub compile + tests (no GPU) ==="
cd "$ROOT/runtime"
cargo build --release --features cuda-stub
cargo test --release --features cuda-stub --test cuda_compile_check -- --nocapture
cargo test --release --features cuda-stub --test cuda_transformer_parity -- --nocapture

echo "=== [2/2] real --features cuda (needs nvcc + libforge_cuda) ==="
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP real cuda feature: nvcc not available on this host"
  echo "verify_cuda_compile: PARTIAL OK (stub only)"
  exit 0
fi

cd "$ROOT"
cmake -S kernels -B kernels/build-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DFORGE_BUILD_CUDA=ON \
  -DFORGE_KERNELS_BUILD_TESTS=OFF \
  -DFORGE_CUDA_BUILD_TESTS=OFF \
  -DFORGE_CUDA_BUILD_BENCH=OFF
cmake --build kernels/build-cuda -j"$(nproc 2>/dev/null || echo 4)"

export FORGE_CUDA_LIB_DIR="$ROOT/kernels/build-cuda/cuda"
if [[ ! -f "$FORGE_CUDA_LIB_DIR/libforge_cuda.a" ]]; then
  echo "FAIL: libforge_cuda.a not produced under $FORGE_CUDA_LIB_DIR"
  exit 1
fi
export LIBRARY_PATH="${FORGE_CUDA_LIB_DIR}:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${FORGE_CUDA_LIB_DIR}:${LD_LIBRARY_PATH:-}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

cd "$ROOT/runtime"
cargo build --release --features cuda
cargo test --release --features cuda --test cuda_compile_check -- --nocapture

echo "verify_cuda_compile: OK (stub + real link). GPU inference still unverified."
