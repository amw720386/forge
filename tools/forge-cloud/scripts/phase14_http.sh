#!/usr/bin/env bash
set -euo pipefail
ROOT=$(readlink -f /teamspace/studios/this_studio/forge-current)
OUT=/teamspace/studios/this_studio/forge-cache/phase14-results
mkdir -p "$OUT"
cd "$ROOT"
export LD_LIBRARY_PATH="$ROOT/runtime/target/release:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export FORGE_RUNTIME_DIR="$ROOT/runtime/target/release"
export FORGE_CUDA_MAX_BATCH=8
export FORGE_BATCHED_PAGED=1 FORGE_REUSE_BF16=1 FORGE_CUDA_GRAPHS=1
label=${1:-baseline}
./forge-server --model models/Qwen2.5-0.5B-Instruct --device cuda --dtype bf16 --max-seq 2048 >"$OUT/$label-server.log" 2>&1 &
pid=$!
trap 'kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true' EXIT
for i in $(seq 1 90); do curl -sf http://localhost:8080/health >/dev/null && break; sleep 1; done
python3 benchmarks/forge/bench_http.py --runtime forge --workloads short,medium,long,mixed --trials 3 --out "$OUT/$label-http.json"
curl -sf http://localhost:8080/metrics > "$OUT/$label-metrics.json"
