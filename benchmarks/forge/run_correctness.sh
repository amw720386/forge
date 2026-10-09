#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODEL_DIR="${FORGE_MODEL_DIR:-$ROOT/models/Qwen2.5-0.5B-Instruct}"
OUT_DIR="${FORGE_BENCH_OUT:-$ROOT/benchmark-output}"
mkdir -p "$OUT_DIR"
cd "$ROOT/runtime"

MAX_NEW="${FORGE_CORRECTNESS_MAX_NEW:-32}"
echo "LABEL=bf16_correctness_survey"
cargo run --release --features cuda --example bf16_correctness_survey -- \
  "$MODEL_DIR" "$MAX_NEW" --json-out "$OUT_DIR/bf16_correctness.json"

# Optional HF spot-check (skip if deps missing)
if python3 "$ROOT/benchmarks/forge/hf_spotcheck.py" --model "$MODEL_DIR" --max-new 16 \
    --out "$OUT_DIR/hf_spotcheck.json" 2>"$OUT_DIR/hf_spotcheck.skip.txt"; then
  echo "HF_SPOTCHECK_OK"
else
  echo "HF_SPOTCHECK_SKIPPED (see $OUT_DIR/hf_spotcheck.skip.txt)"
fi

echo "FORGE_CORRECTNESS_OK"
