#!/usr/bin/env bash
set -euo pipefail
ROOT=$(readlink -f /teamspace/studios/this_studio/forge-current)
OUT=/teamspace/studios/this_studio/forge-cache/phase15-results
cd "$ROOT"
export LD_LIBRARY_PATH="$ROOT/runtime/target/release:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}" FORGE_RUNTIME_DIR="$ROOT/runtime/target/release"
export FORGE_MAX_SEQ=4608
export FORGE_CUDA_MAX_BATCH=8 FORGE_BATCHED_PAGED=1 FORGE_REUSE_BF16=1 FORGE_CUDA_GRAPHS=${FORGE_CUDA_GRAPHS:-1}
export FORGE_PACKED_PREFILL=1 FORGE_PREFILL_ATTN_BLOCK=1 FORGE_GEMM_LT=1 FORGE_EARLY_EMIT=1 FORGE_STEP_TOKEN_BUDGET=1024 FORGE_PREFILL_CHUNK=128
export FORGE_DECODE_ATTN_BLOCK=${FORGE_DECODE_ATTN_BLOCK:-1}
label=${1:-context-final}
./forge-server --model models/Qwen2.5-0.5B-Instruct --device cuda --dtype bf16 --max-seq 4608 >"$OUT/$label-server.log" 2>&1 &
pid=$!
trap 'kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true' EXIT
for i in $(seq 1 90); do curl -sf http://localhost:8080/health >/dev/null && break; sleep 1; done
python3 benchmarks/forge/phase15_context.py --model models/Qwen2.5-0.5B-Instruct --out "$OUT/$label.json" --trials "${FORGE_CONTEXT_TRIALS:-3}" --concurrency "${FORGE_CONTEXT_CONCURRENCY:-1,8}"
curl -sf http://localhost:8080/metrics > "$OUT/$label-metrics.json"
