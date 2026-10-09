#!/usr/bin/env bash
# Phase 10 remote validation ON an already-started Lightning GPU Studio.
# Does NOT provision machines. Prefer: forge-cloud validate-phase10 --confirm
#
# Stages: deploy-verify → preflight → cmake CUDA → forge_cuda_tests → cargo --features cuda
#         → paged KV + batch parity → optional Qwen benches (B=1,2,4,8) + mixed/chunked
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODEL_DIR="${FORGE_MODEL_DIR:-$ROOT/models/Qwen2.5-0.5B-Instruct}"
MAX_NEW="${FORGE_MAX_NEW_TOKENS:-16}"
COMPILE_OK=0

die() { echo "FAIL: $*" >&2; exit 1; }

echo "=== STAGE -1: deploy source identity ==="
echo "BUILD_ROOT=$ROOT"
rm -rf "$ROOT/runtime/target" 2>/dev/null || true
if [[ -f "$ROOT/.forge-deploy-manifest.json" ]]; then
  bash scripts/verify_forge_deploy.sh || die "deploy verification failed — refusing compile"
else
  echo "NOTE: no .forge-deploy-manifest.json (local/dev run); continuing without deploy gate"
fi
chmod +x scripts/*.sh 2>/dev/null || true

echo "=== STAGE 0: preflight ==="
export PATH="${HOME}/.cargo/bin:/usr/local/cuda/bin:${PATH}"
bash scripts/preflight_phase8b.sh || die "preflight failed — refusing to continue"

command -v rustc >/dev/null || die "rustc missing after preflight"
command -v cargo >/dev/null || die "cargo missing"
command -v nvcc >/dev/null || die "nvcc missing — cannot build real CUDA backend"
command -v nvidia-smi >/dev/null || die "nvidia-smi missing — not a GPU machine"

echo "=== STAGE 1: CUDA library build (fail-fast) ==="
cmake -S kernels -B kernels/build-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DFORGE_BUILD_CUDA=ON \
  -DFORGE_KERNELS_BUILD_TESTS=ON \
  -DFORGE_CUDA_BUILD_TESTS=ON \
  -DFORGE_CUDA_BUILD_BENCH=ON
cmake --build kernels/build-cuda -j"$(nproc 2>/dev/null || echo 4)"

export FORGE_CUDA_LIB_DIR="$ROOT/kernels/build-cuda/cuda"
[[ -f "$FORGE_CUDA_LIB_DIR/libforge_cuda.a" ]] || die "libforge_cuda.a missing after cmake"
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

echo "=== STAGE 4: paged KV + transformer + batch parity ==="
cargo test --release --features cuda --test cuda_paged_kv -- --nocapture
cargo test --release --features cuda --test cuda_transformer_parity -- --nocapture
cargo test --release --features cuda --test cuda_batch_parity -- --nocapture

# Chunked-prefill stress with small chunk / budget
export FORGE_PREFILL_CHUNK=8
export FORGE_STEP_TOKEN_BUDGET=16
export FORGE_KV_BLOCK_SIZE=16
cargo test --release --features cuda --test cuda_batch_parity -- --nocapture
unset FORGE_PREFILL_CHUNK FORGE_STEP_TOKEN_BUDGET

if [[ "$COMPILE_OK" -ne 1 ]]; then
  die "internal: COMPILE_OK not set"
fi

echo "=== STAGE 5: Qwen full-model (only if weights present) ==="
if [[ ! -f "$MODEL_DIR/model.safetensors" && ! -f "$MODEL_DIR/model.safetensors.index.json" ]]; then
  echo "SKIP full-model: weights missing at $MODEL_DIR"
  echo "Phase 10 remote script finished: COMPILE+KERNEL+PARITY path done; full-model SKIPPED"
  exit 0
fi

export FORGE_CUDA_BATCH_TRACE=1
export FORGE_PREFILL_CHUNK="${FORGE_PREFILL_CHUNK:-64}"
export FORGE_STEP_TOKEN_BUDGET="${FORGE_STEP_TOKEN_BUDGET:-256}"

cargo run --release --features cuda --example generate_cuda_qwen -- "$MODEL_DIR" "$MAX_NEW"
cargo run --release --features cuda --example bench_cuda_qwen -- "$MODEL_DIR" "$MAX_NEW"
cargo run --release --features cuda --example bench_cuda_batch -- "$MODEL_DIR" "$MAX_NEW"
FORGE_MIXED_PROMPTS=1 cargo run --release --features cuda --example bench_cuda_batch -- "$MODEL_DIR" "$MAX_NEW"

# Long-prompt / small-chunk stress
FORGE_PREFILL_CHUNK=32 FORGE_STEP_TOKEN_BUDGET=64 \
  FORGE_PROMPT="$(python3 -c 'print("<|im_start|>user\\n"+"word "*200+"\\n<|im_end|>\\n<|im_start|>assistant\\n")')" \
  cargo run --release --features cuda --example bench_cuda_batch -- "$MODEL_DIR" "$MAX_NEW" || true

echo "Phase 10 remote script finished OK."
