#!/usr/bin/env bash
set -euo pipefail
ROOT=$(readlink -f /teamspace/studios/this_studio/forge-current)
OUT=/teamspace/studios/this_studio/forge-cache/phase15-results
cd "$ROOT"
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH" CUDA_HOME=/usr/local/cuda
export FORGE_CUDA_LIB_DIR="$ROOT/kernels/build-cuda/cuda" LD_LIBRARY_PATH="$ROOT/runtime/target/release:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
cmake --build kernels/build-cuda -j8
cargo build --manifest-path runtime/Cargo.toml --release --features cuda
bash scripts/phase15_correctness.sh
FORGE_TEST_MODEL="$ROOT/models/Qwen2.5-0.5B-Instruct" FORGE_GEMM_LT=1 cargo test --manifest-path runtime/Cargo.toml --release --features cuda --test cuda_phase15 real_http_long_teacher_forced -- --ignored --test-threads=1 --nocapture > "$OUT/natural-long-tests.log" 2>&1
FORGE_DECODE_ATTN_BLOCK=1 bash scripts/phase15_http.sh final-bf16-check --concurrency 8 --workloads short,medium,long,mixed --trials 1
FORGE_BENCH_DTYPE=fp32 FORGE_DECODE_ATTN_BLOCK=0 bash scripts/phase15_http.sh fp32-baseline --concurrency 1,8 --workloads long,mixed --trials 3
FORGE_BENCH_DTYPE=fp32 FORGE_DECODE_ATTN_BLOCK=1 bash scripts/phase15_http.sh fp32-final --concurrency 1,8 --workloads long,mixed --trials 3
python3 scripts/phase15_source_identity.py > "$OUT/retained-source-hashes.json"
printf '%s\n' PHASE15_FP32_DONE
