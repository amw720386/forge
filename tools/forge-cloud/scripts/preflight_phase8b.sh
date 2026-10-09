#!/usr/bin/env bash
# Non-billable Phase 8B preflight. Does NOT start Lightning or download models.
# Exit 0 only when compile prerequisites look sane.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
FAIL=0

say() { echo "=== $* ==="; }
ok() { echo "OK: $*"; }
bad() { echo "FAIL: $*"; FAIL=1; }

say "line endings (shell scripts must be LF)"
for f in scripts/remote_cuda_phase8b.sh scripts/preflight_phase8b.sh scripts/verify_cuda_compile.sh scripts/remote_cuda_workflow.sh scripts/verify_forge_deploy.sh; do
  if [[ ! -f "$f" ]]; then
    continue
  fi
  if grep -q $'\r' "$f" 2>/dev/null; then
    bad "$f contains CR (CRLF) — run dos2unix or rewrite LF"
  else
    ok "$f is LF"
  fi
done

say "toolchain commands"
for c in cmake cargo rustc; do
  if command -v "$c" >/dev/null 2>&1; then
    ok "$c=$($c --version 2>&1 | head -1)"
  else
    bad "$c not on PATH"
  fi
done
if command -v nvcc >/dev/null 2>&1; then
  ok "nvcc=$(nvcc --version | tail -1)"
else
  echo "NOTE: nvcc not on PATH (required for --features cuda real build; cuda-stub OK without it)"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  ok "nvidia-smi present"
  nvidia-smi -L || true
else
  echo "NOTE: nvidia-smi absent — GPU execution tests will stay unverified"
fi

say "source layout"
for p in kernels/CMakeLists.txt kernels/cuda/include/forge_cuda.h kernels/cuda/src/forge_cuda_stub.c \
         runtime/Cargo.toml runtime/src/cuda_model/mod.rs scripts/remote_cuda_phase8b.sh \
         scripts/verify_forge_deploy.sh; do
  if [[ -f "$p" ]]; then ok "$p"; else bad "missing $p"; fi
done

say "model files (optional for compile; required for remote full-model)"
MODEL_DIR="${FORGE_MODEL_DIR:-$ROOT/models/Qwen2.5-0.5B-Instruct}"
if [[ -f "$MODEL_DIR/model.safetensors" || -f "$MODEL_DIR/model.safetensors.index.json" ]]; then
  ok "model weights under $MODEL_DIR"
elif [[ -d "$MODEL_DIR" ]]; then
  echo "NOTE: model dir exists but weights not found yet: $MODEL_DIR"
else
  echo "NOTE: model dir missing (OK for compile-only): $MODEL_DIR"
fi

say "CUDA build artifact (required for --features cuda)"
LIB_A="$ROOT/kernels/build-cuda/cuda/libforge_cuda.a"
if [[ -f "$LIB_A" ]]; then
  ok "found $LIB_A"
else
  echo "NOTE: $LIB_A missing — use cuda-stub locally or cmake -DFORGE_BUILD_CUDA=ON first"
fi

if [[ "$FAIL" -ne 0 ]]; then
  echo "preflight FAILED"
  exit 1
fi
echo "preflight OK"
exit 0
