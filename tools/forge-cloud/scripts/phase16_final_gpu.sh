#!/usr/bin/env bash
set -euo pipefail
ROOT=$(readlink -f /teamspace/studios/this_studio/forge-current)
OUT=/teamspace/studios/this_studio/forge-cache/phase16-results
cd "$ROOT"
python3 - <<'PY' > "$OUT/final-source-verify.log"
import hashlib,json,pathlib
expected=json.load(open('phase16-source-identity.json'))
bad=[p for p,h in expected.items() if hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()!=h]
print('FINAL_SOURCE_MISMATCHES',bad)
assert not bad
print('FINAL_SOURCE_VERIFIED',len(expected))
PY
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH" CUDA_HOME=/usr/local/cuda
export FORGE_CUDA_LIB_DIR="$ROOT/kernels/build-cuda/cuda"
export LD_LIBRARY_PATH="$ROOT/runtime/target/release:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
cargo build --manifest-path runtime/Cargo.toml --release --features cuda --examples > "$OUT/final-build.log" 2>&1
cargo build --manifest-path runtime/Cargo.toml --release --features cuda >> "$OUT/final-build.log" 2>&1
export FORGE_SMOLLM_MODEL="$ROOT/models/SmolLM2-135M-Instruct" FORGE_SMOLLM_REFS=/teamspace/studios/this_studio/forge-cache/phase16-smollm-refs
cargo test --manifest-path runtime/Cargo.toml --release --features cuda --test cuda_phase16 --test phase16_architecture -- --include-ignored --test-threads=1 --nocapture > "$OUT/final-model-graphs.log" 2>&1
runtime/target/release/examples/phase16_parity "$FORGE_SMOLLM_MODEL" "$FORGE_SMOLLM_REFS" cuda > "$OUT/smollm-final-parity.log" 2>&1
FORGE_DECODE_ATTN_BLOCK=1 bash scripts/phase15_http.sh phase16-after-short-repeat --workloads short --concurrency 1,8 --trials 9
cp /teamspace/studios/this_studio/forge-cache/phase15-results/phase16-after-short-repeat-* "$OUT/"
./forge-server --model models/SmolLM2-135M-Instruct --device cuda --dtype bf16 --max-seq 256 > "$OUT/smollm-server.log" 2>&1 &
pid=$!
trap 'kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true' EXIT
for i in $(seq 1 30); do curl -sf http://localhost:8080/health >/dev/null && break; sleep 1; done
curl --fail -sS http://localhost:8080/v1/completions -H 'Content-Type: application/json' -d '{"prompt":"<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n","max_tokens":12,"temperature":0,"stream":false}' > "$OUT/smollm-http.json"
printf '%s\n' PHASE16_FINAL_GPU_DONE
