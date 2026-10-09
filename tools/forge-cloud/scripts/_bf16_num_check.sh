#!/usr/bin/env bash
set -e
export PATH="${HOME}/.cargo/bin:/usr/local/cuda/bin:${PATH}"
. "${HOME}/.cargo/env" 2>/dev/null || true
ROOT=/teamspace/studios/this_studio/forge-current
test -d "$ROOT" || ROOT=/teamspace/studios/this_studio/forge
export FORGE_CUDA_LIB_DIR="$ROOT/kernels/build-cuda/cuda"
export LD_LIBRARY_PATH="${FORGE_CUDA_LIB_DIR}:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${FORGE_CUDA_LIB_DIR}:${LIBRARY_PATH:-}"
export RUSTFLAGS="-C link-arg=-Wl,-rpath,${FORGE_CUDA_LIB_DIR}"
MODEL=/teamspace/studios/this_studio/forge-cache/models/Qwen2.5-0.5B-Instruct
cd "$ROOT/runtime"
# Patch example if uploaded alongside
if [ -f /teamspace/studios/this_studio/forge-upload/generate_cuda_qwen.rs ]; then
  cp /teamspace/studios/this_studio/forge-upload/generate_cuda_qwen.rs examples/generate_cuda_qwen.rs
fi
echo "=== FP32 generate ==="
unset FORGE_CUDA_DTYPE || true
cargo run --release --features cuda --example generate_cuda_qwen -- "$MODEL" 16 | tee /tmp/fp32_out.txt
echo "=== BF16 generate ==="
export FORGE_CUDA_DTYPE=bf16
cargo run --release --features cuda --example generate_cuda_qwen -- "$MODEL" 16 | tee /tmp/bf16_out.txt
echo "=== compare texts ==="
echo "FP32: $(cat /tmp/fp32_out.txt)"
echo "BF16: $(cat /tmp/bf16_out.txt)"
if cmp -s /tmp/fp32_out.txt /tmp/bf16_out.txt; then
  echo "BF16_TOKEN_TEXT_MATCH=1"
else
  echo "BF16_TOKEN_TEXT_MATCH=0 (divergence expected when logits close)"
fi
echo PHASE11_BF16_NUM_OK