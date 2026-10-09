#!/usr/bin/env bash
set -euo pipefail
ROOT=$(readlink -f /teamspace/studios/this_studio/forge-current)
OUT=/teamspace/studios/this_studio/forge-cache/phase15-results
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
cd "$ROOT"
bash scripts/phase14_build.sh
nvcc --default-stream=per-thread -O3 -I kernels/cuda/include kernels/cuda/bench/bench_phase15_attention.cu kernels/build-cuda/cuda/libforge_cuda.a -lcublas -lcublasLt -o /tmp/phase15-attention
/tmp/phase15-attention > "$OUT/attention-experiment.jsonl"
FORGE_DECODE_ATTN_BLOCK=1 bash scripts/phase15_http.sh decode-only --workloads short,long,mixed --concurrency 1,8 --trials 3
