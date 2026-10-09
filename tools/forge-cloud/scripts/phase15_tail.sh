#!/usr/bin/env bash
set -euo pipefail
ROOT=$(readlink -f /teamspace/studios/this_studio/forge-current)
OUT=/teamspace/studios/this_studio/forge-cache/phase15-results
cd "$ROOT"
FORGE_DECODE_ATTN_BLOCK=0 bash scripts/phase15_http.sh short-repeat-baseline --workloads short --concurrency 4,8 --trials 9
FORGE_DECODE_ATTN_BLOCK=1 bash scripts/phase15_http.sh short-repeat-final --workloads short --concurrency 4,8 --trials 9
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH" CUDA_HOME=/usr/local/cuda FORGE_CUDA_LIB_DIR="$ROOT/kernels/build-cuda/cuda" LD_LIBRARY_PATH="$ROOT/runtime/target/release:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}" FORGE_TEST_MODEL="$ROOT/models/Qwen2.5-0.5B-Instruct"
cargo test --manifest-path runtime/Cargo.toml --release --features cuda --test cuda_phase14 real_model_packed_logits -- --ignored --test-threads=1 --nocapture > "$OUT/phase14-real-regression.log" 2>&1
nvcc --default-stream=per-thread -O3 -I kernels/cuda/include kernels/cuda/bench/bench_phase15_attention.cu kernels/build-cuda/cuda/libforge_cuda.a -lcublas -lcublasLt -o /tmp/phase15-attention-final
/tmp/phase15-attention-final > "$OUT/attention-final.jsonl"
printf '%s\n' PHASE15_TAIL_DONE
