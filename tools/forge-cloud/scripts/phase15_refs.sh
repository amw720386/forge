#!/usr/bin/env bash
set -euo pipefail
ROOT=$(readlink -f /teamspace/studios/this_studio/forge-current)
OUT=/teamspace/studios/this_studio/forge-cache/phase15-results
CACHE=/teamspace/studios/this_studio/forge-cache/llama.cpp
cd "$ROOT"
git -C "$CACHE" rev-parse HEAD > "$OUT/llama-commit.txt"
"$CACHE/build/bin/llama-server" -m /teamspace/studios/this_studio/forge-cache/gguf/forge-phase14-f16.gguf -ngl 99 -np 8 -c 16384 -ctk f32 -ctv f32 --slots --no-cache-prompt --cache-ram 0 --no-cache-idle-slots --port 8080 > "$OUT/llama-server.log" 2>&1 &
pid=$!
trap 'kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true' EXIT
for i in $(seq 1 90); do curl -sf http://localhost:8080/health >/dev/null && break; sleep 1; done
python3 benchmarks/forge/bench_http.py --runtime llama --workloads long,mixed --concurrency 1,8 --trials 3 --out "$OUT/llama-http.json"
