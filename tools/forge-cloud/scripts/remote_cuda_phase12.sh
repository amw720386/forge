#!/usr/bin/env bash
# Phase 12 remote validation ON an already-started Lightning GPU Studio.
# Modes via FORGE_PHASE12_MODE:
#   forge      (default) — build, tests, audited benches, correctness, controlled exps
#   refs       — llama.cpp + optional nsys + medium/long Forge
#   vllm       — best-effort vLLM
#   compare    — llama.cpp + vLLM only (no Forge rebuild; uses cached CUDA build if present)
#   all        — forge then refs (skip vllm unless FORGE_PHASE12_TRY_VLLM=1)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODEL_DIR="${FORGE_MODEL_DIR:-$ROOT/models/Qwen2.5-0.5B-Instruct}"
# Prefer durable cache path so mid-session JSON does not poison deploy hash verify.
OUT_DIR="${FORGE_BENCH_OUT:-/teamspace/studios/this_studio/forge-cache/phase12-results}"
if [[ ! -d "$(dirname "$OUT_DIR")" ]]; then
  OUT_DIR="${FORGE_BENCH_OUT:-$ROOT/benchmarks/results}"
fi
MODE="${FORGE_PHASE12_MODE:-forge}"
COMPILE_OK=0
mkdir -p "$OUT_DIR"

die() { echo "FAIL: $*" >&2; exit 1; }

echo "=== STAGE -1: deploy source identity ==="
echo "BUILD_ROOT=$ROOT mode=$MODE"
# Keep incremental cargo target across recoverable retries in the same deploy tree.
# Only wipe if FORGE_PHASE12_CLEAN=1.
if [[ "${FORGE_PHASE12_CLEAN:-0}" == "1" ]]; then
  rm -rf "$ROOT/runtime/target" 2>/dev/null || true
fi
if [[ -f "$ROOT/.forge-deploy-manifest.json" ]]; then
  bash scripts/verify_forge_deploy.sh || die "deploy verification failed — refusing compile"
else
  echo "NOTE: no .forge-deploy-manifest.json (local/dev run); continuing without deploy gate"
fi
chmod +x scripts/*.sh benchmarks/forge/*.sh benchmarks/llama_cpp/*.sh benchmarks/vllm/*.sh 2>/dev/null || true
sed -i "s/\r$//" scripts/*.sh benchmarks/forge/*.sh benchmarks/llama_cpp/*.sh benchmarks/vllm/*.sh 2>/dev/null || true

echo "=== STAGE 0: preflight ==="
export PATH="${HOME}/.cargo/bin:/usr/local/cuda/bin:${PATH}"
bash scripts/preflight_phase8b.sh || die "preflight failed"

command -v rustc >/dev/null || die "rustc missing"
command -v cargo >/dev/null || die "cargo missing"
command -v nvcc >/dev/null || die "nvcc missing"
command -v nvidia-smi >/dev/null || die "nvidia-smi missing"
nvidia-smi -L || true

ensure_model() {
  if [[ -f "$MODEL_DIR/model.safetensors" || -f "$MODEL_DIR/model.safetensors.index.json" ]]; then
    return 0
  fi
  echo "=== downloading Qwen2.5-0.5B-Instruct ==="
  mkdir -p "$MODEL_DIR"
  pip3 install -q huggingface_hub
  python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', local_dir='$MODEL_DIR')"
}

build_forge() {
  echo "=== STAGE 1: CUDA library build ==="
  if [[ "${FORGE_PHASE12_CLEAN:-0}" == "1" ]]; then
    rm -rf kernels/build-cuda
  fi
  if [[ ! -f kernels/build-cuda/cuda/libforge_cuda.a ]]; then
    cmake -S kernels -B kernels/build-cuda \
      -DCMAKE_BUILD_TYPE=Release \
      -DFORGE_BUILD_CUDA=ON \
      -DFORGE_KERNELS_BUILD_TESTS=ON \
      -DFORGE_CUDA_BUILD_TESTS=ON \
      -DFORGE_CUDA_BUILD_BENCH=ON
  fi
  cmake --build kernels/build-cuda -j"$(nproc 2>/dev/null || echo 4)"

  export FORGE_CUDA_LIB_DIR="$ROOT/kernels/build-cuda/cuda"
  [[ -f "$FORGE_CUDA_LIB_DIR/libforge_cuda.a" ]] || die "libforge_cuda.a missing"
  export LIBRARY_PATH="${FORGE_CUDA_LIB_DIR}:${LIBRARY_PATH:-}"
  export LD_LIBRARY_PATH="${FORGE_CUDA_LIB_DIR}:${LD_LIBRARY_PATH:-}"
  export RUSTFLAGS="${RUSTFLAGS:-} -C link-arg=-Wl,-rpath,${FORGE_CUDA_LIB_DIR}"
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

  echo "=== STAGE 2: forge_cuda_tests ==="
  if [[ -x "$FORGE_CUDA_LIB_DIR/forge_cuda_tests" ]]; then
    "$FORGE_CUDA_LIB_DIR/forge_cuda_tests"
  elif [[ -x "$FORGE_CUDA_LIB_DIR/Release/forge_cuda_tests" ]]; then
    "$FORGE_CUDA_LIB_DIR/Release/forge_cuda_tests"
  else
    ctest --test-dir kernels/build-cuda --output-on-failure -R forge_cuda
  fi

  echo "=== STAGE 3: Rust --features cuda ==="
  cd "$ROOT/runtime"
  cargo build --release --features cuda
  cargo test --release --features cuda --test cuda_compile_check -- --nocapture
  COMPILE_OK=1

  echo "=== STAGE 4: parity suites ==="
  cargo test --release --features cuda --test cuda_paged_kv -- --nocapture
  cargo test --release --features cuda --test cuda_transformer_parity -- --nocapture
  cargo test --release --features cuda --test cuda_batch_parity -- --nocapture
  export FORGE_PREFILL_CHUNK=8 FORGE_STEP_TOKEN_BUDGET=16 FORGE_KV_BLOCK_SIZE=16
  cargo test --release --features cuda --test cuda_batch_parity -- --nocapture
  unset FORGE_PREFILL_CHUNK FORGE_STEP_TOKEN_BUDGET || true
  cd "$ROOT"
}

run_forge_benches() {
  ensure_model
  export FORGE_MODEL_DIR="$MODEL_DIR"
  export FORGE_BENCH_OUT="$OUT_DIR"
  export FORGE_PREFILL_CHUNK="${FORGE_PREFILL_CHUNK:-64}"
  export FORGE_STEP_TOKEN_BUDGET="${FORGE_STEP_TOKEN_BUDGET:-256}"

  echo "=== STAGE 5: audited Forge benches ==="
  bash benchmarks/forge/run_forge_bench.sh

  echo "=== STAGE 6: BF16 correctness survey ==="
  bash benchmarks/forge/run_correctness.sh

  # Capture env metadata
  {
    echo "{"
    echo "\"gpu\": \"$(nvidia-smi -L | head -1 | sed 's/"/\\"/g')\","
    echo "\"driver\": \"$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)\","
    echo "\"cuda\": \"$(nvcc --version | tail -1 | sed 's/"/\\"/g')\","
    echo "\"mode\": \"$MODE\""
    echo "}"
  } > "$OUT_DIR/env_meta.json"
}

run_refs() {
  ensure_model
  export FORGE_MODEL_DIR="$MODEL_DIR"
  export FORGE_BENCH_OUT="$OUT_DIR"

  # Ensure forge built for long/medium + profile
  if [[ "$COMPILE_OK" -ne 1 ]]; then
    build_forge
  fi

  echo "=== STAGE 7: Forge medium/long (if not already) ==="
  cd "$ROOT/runtime"
  cargo run --release --features cuda --example bench_phase12 -- \
    --model "$MODEL_DIR" --dtype bf16 --batch 1,2 --max-new 64 --long \
    --trials 2 --warmup 1 --token-policy fixed_len \
    --label forge_long_bf16 \
    --json-out "$OUT_DIR/forge_long_bf16.json" || true

  FORGE_STAGGER_ADMIT=1 FORGE_MIXED_PROMPTS=1 \
  cargo run --release --features cuda --example bench_phase12 -- \
    --model "$MODEL_DIR" --dtype bf16 --batch 8 --max-new 32 --mixed \
    --max-seq 1024 --trials 2 --warmup 1 --token-policy fixed_len \
    --workload mixed_stagger --label forge_mixed_bf16 \
    --json-out "$OUT_DIR/forge_mixed_bf16.json" || true
  cd "$ROOT"

  echo "=== STAGE 8: llama.cpp ==="
  bash benchmarks/llama_cpp/build_and_bench.sh || echo "LLAMA_CPP_BENCH_FAILED (continuing)"

  echo "=== STAGE 9: nsys profile (bounded) ==="
  if ! command -v nsys >/dev/null 2>&1; then
    echo "Attempting nsight-systems install (best-effort, bounded)…"
    sudo apt-get update -qq >/tmp/nsys_apt.txt 2>&1 || true
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq nsight-systems-cli 2>>/tmp/nsys_apt.txt \
      || sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq nsight-systems 2>>/tmp/nsys_apt.txt || true
  fi
  if command -v nsys >/dev/null 2>&1; then
    cd "$ROOT/runtime"
    nsys profile -o "$OUT_DIR/forge_nsys_b1" --force-overwrite true \
      cargo run --release --features cuda --example bench_phase12 -- \
      --model "$MODEL_DIR" --dtype bf16 --batch 1 --max-new 16 \
      --trials 1 --warmup 1 --token-policy fixed_len --label nsys_b1 \
      --json-out "$OUT_DIR/forge_nsys_b1.json" || true
    nsys profile -o "$OUT_DIR/forge_nsys_b8" --force-overwrite true \
      cargo run --release --features cuda --example bench_phase12 -- \
      --model "$MODEL_DIR" --dtype bf16 --batch 8 --max-new 16 \
      --trials 1 --warmup 1 --token-policy fixed_len --label nsys_b8 \
      --json-out "$OUT_DIR/forge_nsys_b8.json" || true
    nsys stats --report cuda_gpu_kern_sum "$OUT_DIR/forge_nsys_b8.nsys-rep" \
      > "$OUT_DIR/forge_nsys_b8_kern.txt" 2>/dev/null || true
    nsys stats --report cuda_gpu_kern_sum "$OUT_DIR/forge_nsys_b1.nsys-rep" \
      > "$OUT_DIR/forge_nsys_b1_kern.txt" 2>/dev/null || true
    cd "$ROOT"
    echo "NSYS_OK"
  else
    echo "NSYS_SKIPPED (nsys not installed)"
    # Fallback: controlled A/B already ranks attn vs greedy; capture nvidia-smi utilization sample.
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv \
      > "$OUT_DIR/nvidia_smi_sample.csv" 2>/dev/null || true
    echo "PROFILE_FALLBACK=controlled_experiments+nvidia_smi" | tee "$OUT_DIR/profile_fallback.txt"
  fi

  if [[ "${FORGE_PHASE12_TRY_VLLM:-0}" == "1" ]]; then
    echo "=== STAGE 10: vLLM best-effort ==="
    bash benchmarks/vllm/try_bench.sh || true
  fi

  python3 benchmarks/analysis/summarize_results.py \
    --results-dir "$OUT_DIR" \
    --out-md "$OUT_DIR/comparison.md" \
    --out-csv "$OUT_DIR/comparison.csv" || true
}

run_vllm() {
  ensure_model
  export FORGE_MODEL_DIR="$MODEL_DIR"
  export FORGE_BENCH_OUT="$OUT_DIR"
  bash benchmarks/vllm/try_bench.sh
}

case "$MODE" in
  forge)
    build_forge
    run_forge_benches
    ;;
  refs)
    run_refs
    ;;
  vllm)
    run_vllm
    ;;
  compare)
    ensure_model
    export FORGE_MODEL_DIR="$MODEL_DIR"
    export FORGE_BENCH_OUT="$OUT_DIR"
    echo "=== compare mode: llama.cpp + vLLM only ==="
    bash benchmarks/llama_cpp/build_and_bench.sh
    bash benchmarks/vllm/try_bench.sh || true
    python3 benchmarks/analysis/summarize_results.py \
      --results-dir "$OUT_DIR" \
      --out-md "$OUT_DIR/comparison.md" \
      --out-csv "$OUT_DIR/comparison.csv" || true
    ;;
  all)
    build_forge
    run_forge_benches
    run_refs
    if [[ "${FORGE_PHASE12_TRY_VLLM:-0}" == "1" ]]; then
      run_vllm
    fi
    ;;
  *)
    die "unknown FORGE_PHASE12_MODE=$MODE"
    ;;
esac

echo "Phase 12 remote script finished OK (mode=$MODE)."
echo "PHASE12_WORK_OK"
