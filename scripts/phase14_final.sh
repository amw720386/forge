#!/usr/bin/env bash
set -euo pipefail
ROOT=$(readlink -f /teamspace/studios/this_studio/forge-current)
OUT=/teamspace/studios/this_studio/forge-cache/phase14-results
cd "$ROOT"
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="$ROOT/runtime/target/release:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}" FORGE_RUNTIME_DIR="$ROOT/runtime/target/release"
export FORGE_BATCHED_PAGED=1 FORGE_REUSE_BF16=1 FORGE_CUDA_GRAPHS=1 FORGE_CUDA_MAX_BATCH=8
serve() {
 label=$1; shift
 ./forge-server --model models/Qwen2.5-0.5B-Instruct --device cuda --dtype bf16 --max-seq 2048 > "$OUT/$label-server.log" 2>&1 &
 pid=$!
 trap 'kill "$pid" 2>/dev/null || true' EXIT
 for i in $(seq 1 90); do curl -sf http://localhost:8080/health >/dev/null && break; sleep 1; done
 python3 benchmarks/forge/bench_http.py --runtime forge --out "$OUT/$label-http.json" "$@"
 curl -sf http://localhost:8080/metrics > "$OUT/$label-metrics.json"
 kill "$pid";wait "$pid" || true
 trap - EXIT
}
export FORGE_PACKED_PREFILL=1 FORGE_PREFILL_ATTN_BLOCK=1 FORGE_GEMM_LT=1 FORGE_EARLY_EMIT=1 FORGE_STEP_TOKEN_BUDGET=1024
serve final --workloads short,medium,long,mixed --trials 3
serve staggered --workloads mixed --concurrency 8 --trials 3 --arrival-gap-ms 10
export FORGE_SERVING_PROFILE=1
export FORGE_PACKED_PREFILL=0 FORGE_PREFILL_ATTN_BLOCK=0 FORGE_GEMM_LT=0 FORGE_EARLY_EMIT=0 FORGE_STEP_TOKEN_BUDGET=256
serve profile-final-baseline --workloads short,medium,long --concurrency 1,8 --trials 1
export FORGE_PACKED_PREFILL=1 FORGE_PREFILL_ATTN_BLOCK=1 FORGE_GEMM_LT=1 FORGE_EARLY_EMIT=1 FORGE_STEP_TOKEN_BUDGET=1024
serve profile-final --workloads short,medium,long --concurrency 1,8 --trials 1
unset FORGE_SERVING_PROFILE
export FORGE_PACKED_PREFILL=0 FORGE_PREFILL_ATTN_BLOCK=0 FORGE_GEMM_LT=0 FORGE_EARLY_EMIT=0 FORGE_STEP_TOKEN_BUDGET=256
runtime/target/release/examples/bench_phase12 --model models/Qwen2.5-0.5B-Instruct --dtype bf16 --batch 1,2,4,8 --max-new 16 --trials 3 --warmup 1 --token-policy fixed_len --label phase13-internal --json-out "$OUT/phase13-internal.json"
export FORGE_PACKED_PREFILL=1 FORGE_PREFILL_ATTN_BLOCK=1 FORGE_GEMM_LT=1 FORGE_EARLY_EMIT=1 FORGE_STEP_TOKEN_BUDGET=1024
runtime/target/release/examples/bench_phase12 --model models/Qwen2.5-0.5B-Instruct --dtype bf16 --batch 1,2,4,8 --max-new 16 --trials 3 --warmup 1 --token-policy fixed_len --label phase14-internal --json-out "$OUT/phase14-internal.json"
runtime/target/release/examples/bf16_correctness_survey models/Qwen2.5-0.5B-Instruct 16 --json-out "$OUT/bf16-correctness.json"
printf '%s\n' PHASE14_FINAL_GPU_DONE
