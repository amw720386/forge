#!/usr/bin/env bash
set -euo pipefail
ROOT=$(readlink -f /teamspace/studios/this_studio/forge-current)
CACHE=/teamspace/studios/this_studio/forge-cache
OUT="$CACHE/phase16-results"
cd "$ROOT"
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH" CUDA_HOME=/usr/local/cuda
export FORGE_CUDA_LIB_DIR="$ROOT/kernels/build-cuda/cuda"
export LD_LIBRARY_PATH="$ROOT/runtime/target/release:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
FORGE_REFERENCE_TEXT_ONLY=1 python3 tests/reference/phase16_reference.py models/SmolLM2-135M-Instruct "$CACHE/phase16-smollm-refs" > "$OUT/smollm-hf.log" 2>&1
runtime/target/release/examples/phase16_parity models/SmolLM2-135M-Instruct "$CACHE/phase16-smollm-refs" cuda > "$OUT/smollm-parity.log" 2>&1
export FORGE_SMOLLM_MODEL="$ROOT/models/SmolLM2-135M-Instruct" FORGE_SMOLLM_REFS="$CACHE/phase16-smollm-refs"
cargo test --manifest-path runtime/Cargo.toml --release --features cuda --test cuda_phase16 -- --ignored --test-threads=1 --nocapture > "$OUT/smollm-batch.log" 2>&1
FORGE_DECODE_ATTN_BLOCK=1 bash scripts/phase15_http.sh phase16-after-short-c1 --workloads short --concurrency 1 --trials 3
FORGE_DECODE_ATTN_BLOCK=1 bash scripts/phase15_http.sh phase16-after-c8 --workloads short,medium,long,mixed --concurrency 8 --trials 3
cp "$CACHE"/phase15-results/phase16-after-* "$OUT/"
printf '%s\n' PHASE16_SECOND_MODEL_DONE
