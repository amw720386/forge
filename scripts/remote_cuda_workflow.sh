#!/usr/bin/env bash
# Run **on the Lightning Studio** (via forge-cloud exec), not on Windows.
# Sync via git first: git pull in the cloned Forge repo.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== Forge remote CUDA workflow ==="
nvidia-smi || { echo "nvidia-smi failed — wrong machine?"; exit 1; }

cmake -S kernels -B kernels/build-cuda \
  -DFORGE_BUILD_CUDA=ON \
  -DFORGE_KERNELS_BUILD_TESTS=ON \
  -DFORGE_CUDA_BUILD_TESTS=ON \
  -DFORGE_CUDA_BUILD_BENCH=ON

cmake --build kernels/build-cuda -j"$(nproc 2>/dev/null || echo 4)"

echo "--- CPU kernel tests ---"
ctest --test-dir kernels/build-cuda --output-on-failure -R forge_kernels || true

echo "--- CUDA tests ---"
if [[ -x kernels/build-cuda/cuda/forge_cuda_tests ]]; then
  ./kernels/build-cuda/cuda/forge_cuda_tests
elif [[ -x kernels/build-cuda/cuda/Release/forge_cuda_tests ]]; then
  ./kernels/build-cuda/cuda/Release/forge_cuda_tests
else
  ctest --test-dir kernels/build-cuda --output-on-failure -R forge_cuda
fi

echo "--- CUDA microbench ---"
if [[ -x kernels/build-cuda/cuda/forge_cuda_bench ]]; then
  ./kernels/build-cuda/cuda/forge_cuda_bench
fi

echo "done"
