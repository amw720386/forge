#!/usr/bin/env bash
set -euo pipefail
ROOT=$(readlink -f /teamspace/studios/this_studio/forge-current)
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
cd "$ROOT"
bash scripts/verify_forge_deploy.sh
cmake -S kernels -B kernels/build-cuda -DCMAKE_BUILD_TYPE=Release -DFORGE_BUILD_CUDA=ON -DFORGE_CUDA_BUILD_TESTS=ON -DFORGE_CUDA_BUILD_BENCH=ON
cmake --build kernels/build-cuda -j8
export FORGE_CUDA_LIB_DIR="$ROOT/kernels/build-cuda/cuda"
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH="$ROOT/runtime/target/release:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
cargo build --manifest-path runtime/Cargo.toml --release --features cuda --examples
cargo build --manifest-path runtime/Cargo.toml --release --features cuda
cd api
go build -o "$ROOT/forge-server" ./cmd/forge-server
printf '%s\n' PHASE14_BUILD_OK
