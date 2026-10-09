#!/usr/bin/env bash
set -euo pipefail
ROOT=$(readlink -f /teamspace/studios/this_studio/forge-current)
OUT=/teamspace/studios/this_studio/forge-cache/phase15-results
cd "$ROOT"
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH" CUDA_HOME=/usr/local/cuda
export FORGE_CUDA_LIB_DIR="$ROOT/kernels/build-cuda/cuda" LD_LIBRARY_PATH="$ROOT/runtime/target/release:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export FORGE_TEST_MODEL="$ROOT/models/Qwen2.5-0.5B-Instruct"
cargo test --manifest-path runtime/Cargo.toml --release --features cuda --test cuda_phase15 -- --test-threads=1 --nocapture > "$OUT/phase15-tests.log" 2>&1
cargo test --manifest-path runtime/Cargo.toml --release --features cuda --test cuda_phase15 real_long_greedy_agreement -- --ignored --test-threads=1 --nocapture > "$OUT/phase15-real-tests.log" 2>&1
FORGE_DECODE_ATTN_BLOCK=1 cargo test --manifest-path runtime/Cargo.toml --release --features cuda --test cuda_batch_parity --test cuda_paged_kv --test cuda_transformer_parity --test cuda_phase13 --test cuda_phase14 -- --test-threads=1 --nocapture > "$OUT/gpu-regressions.log" 2>&1
kernels/build-cuda/cuda/forge_cuda_tests > "$OUT/cuda-kernels.log" 2>&1
printf '%s\n' PHASE15_CORRECTNESS_OK
