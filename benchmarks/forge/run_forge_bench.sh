#!/usr/bin/env bash
# Run audited Forge Phase 12 benches. Expects CUDA build env already set
# (see scripts/remote_cuda_phase12.sh).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODEL_DIR="${FORGE_MODEL_DIR:-$ROOT/models/Qwen2.5-0.5B-Instruct}"
OUT_DIR="${FORGE_BENCH_OUT:-$ROOT/benchmarks/results}"
mkdir -p "$OUT_DIR"
cd "$ROOT/runtime"

export FORGE_PREFILL_CHUNK="${FORGE_PREFILL_CHUNK:-64}"
export FORGE_STEP_TOKEN_BUDGET="${FORGE_STEP_TOKEN_BUDGET:-256}"

run_one() {
  local label="$1"; shift
  echo "LABEL=$label"
  cargo run --release --features cuda --example bench_phase12 -- "$@"
}

# P0 short remasure (fixed_len) — FP32 + BF16
run_one forge_short_fp32 \
  --model "$MODEL_DIR" --dtype fp32 --batch 1,2,4,8 --max-new 16 \
  --trials 3 --warmup 1 --token-policy fixed_len --workload short_hello \
  --label forge_short_fp32 \
  --json-out "$OUT_DIR/forge_short_fp32.json"

run_one forge_short_bf16 \
  --model "$MODEL_DIR" --dtype bf16 --batch 1,2,4,8 --max-new 16 \
  --trials 3 --warmup 1 --token-policy fixed_len --workload short_hello \
  --label forge_short_bf16 \
  --json-out "$OUT_DIR/forge_short_bf16.json"

run_one forge_short64_bf16 \
  --model "$MODEL_DIR" --dtype bf16 --batch 1,4,8 --max-new 64 \
  --trials 2 --warmup 1 --token-policy fixed_len --workload short_64 \
  --label forge_short64_bf16 \
  --json-out "$OUT_DIR/forge_short64_bf16.json"

# P0 medium
run_one forge_medium_fp32 \
  --model "$MODEL_DIR" --dtype fp32 --batch 1,4 --max-new 64 --medium \
  --trials 2 --warmup 1 --token-policy fixed_len \
  --label forge_medium_fp32 \
  --json-out "$OUT_DIR/forge_medium_fp32.json"

run_one forge_medium_bf16 \
  --model "$MODEL_DIR" --dtype bf16 --batch 1,4 --max-new 64 --medium \
  --trials 2 --warmup 1 --token-policy fixed_len \
  --label forge_medium_bf16 \
  --json-out "$OUT_DIR/forge_medium_bf16.json"

# Controlled experiments (B=8 short BF16)
echo "LABEL=forge_ctrl_naive_attn"
FORGE_PAGED_ATTN_NAIVE=1 FORGE_CPU_GREEDY=1 \
  cargo run --release --features cuda --example bench_phase12 -- \
  --model "$MODEL_DIR" --dtype fp32 --batch 8 --max-new 16 \
  --trials 2 --warmup 1 --token-policy fixed_len \
  --label forge_ctrl_naive_cpu_greedy \
  --json-out "$OUT_DIR/forge_ctrl_naive_cpu_greedy.json"

echo "LABEL=forge_ctrl_opt_cpu_greedy"
FORGE_CPU_GREEDY=1 \
  cargo run --release --features cuda --example bench_phase12 -- \
  --model "$MODEL_DIR" --dtype fp32 --batch 8 --max-new 16 \
  --trials 2 --warmup 1 --token-policy fixed_len \
  --label forge_ctrl_opt_cpu_greedy \
  --json-out "$OUT_DIR/forge_ctrl_opt_cpu_greedy.json"

echo "FORGE_BENCH_OK"
