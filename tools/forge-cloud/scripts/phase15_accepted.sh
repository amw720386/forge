#!/usr/bin/env bash
set -euo pipefail
ROOT=$(readlink -f /teamspace/studios/this_studio/forge-current)
OUT=/teamspace/studios/this_studio/forge-cache/phase15-results
cd "$ROOT"
bash scripts/phase15_correctness.sh
FORGE_DECODE_ATTN_BLOCK=0 bash scripts/phase15_http.sh accepted-baseline --concurrency 1,2,4,8 --workloads short,medium,long,mixed --trials 3
FORGE_DECODE_ATTN_BLOCK=1 bash scripts/phase15_http.sh accepted-final --concurrency 1,2,4,8 --workloads short,medium,long,mixed --trials 3
FORGE_DECODE_ATTN_BLOCK=0 bash scripts/phase15_context.sh accepted-context-baseline
FORGE_DECODE_ATTN_BLOCK=1 bash scripts/phase15_context.sh accepted-context-final
FORGE_DECODE_ATTN_BLOCK=0 FORGE_KV_NUM_BLOCKS=512 bash scripts/phase15_http.sh accepted-pressure-baseline --concurrency 8 --workloads long,mixed --trials 3 --arrival-gap-ms 10
FORGE_DECODE_ATTN_BLOCK=1 FORGE_KV_NUM_BLOCKS=512 bash scripts/phase15_http.sh accepted-pressure-final --concurrency 8 --workloads long,mixed --trials 3 --arrival-gap-ms 10
bash scripts/phase15_scheduling.sh
FORGE_CUDA_MAX_BATCH=4 FORGE_DECODE_ATTN_BLOCK=1 FORGE_SCHED_POLICY=fixed bash scripts/phase15_http.sh accepted-queued-four-slots --concurrency 8 --workloads mixed --trials 3 --arrival-gap-ms 10
FORGE_DECODE_ATTN_BLOCK=0 bash scripts/phase15_http.sh accepted-short-repeat-baseline --concurrency 4,8 --workloads short --trials 9
FORGE_DECODE_ATTN_BLOCK=1 bash scripts/phase15_http.sh accepted-short-repeat-final --concurrency 4,8 --workloads short --trials 9
nvcc --default-stream=per-thread -O3 -I kernels/cuda/include kernels/cuda/bench/bench_phase15_attention.cu kernels/build-cuda/cuda/libforge_cuda.a -lcublas -lcublasLt -o /tmp/phase15-attention-final
/tmp/phase15-attention-final > "$OUT/accepted-attention.jsonl"
FORGE_TEST_MODEL="$ROOT/models/Qwen2.5-0.5B-Instruct" FORGE_GEMM_LT=1 cargo test --manifest-path runtime/Cargo.toml --release --features cuda --test cuda_phase15 real_http_long_teacher_forced -- --ignored --test-threads=1 --nocapture > "$OUT/natural-long-tests.log" 2>&1
FORGE_DECODE_ATTN_BLOCK=0 FORGE_SCHED_POLICY=fixed bash scripts/phase15_http.sh accepted-scheduling-only --concurrency 1,8 --workloads mixed --trials 3
FORGE_DECODE_ATTN_BLOCK=1 FORGE_SCHED_POLICY=fixed bash scripts/phase15_http.sh accepted-combined-fixed --concurrency 1,8 --workloads mixed --trials 3
FORGE_CUDA_GRAPHS=0 FORGE_DECODE_ATTN_BLOCK=1 bash scripts/phase15_http.sh accepted-eager --concurrency 8 --workloads long,mixed --trials 3
FORGE_CONTEXT_TRIALS=1 FORGE_CONTEXT_CONCURRENCY=1 FORGE_SERVING_PROFILE=1 FORGE_KERNEL_PROFILE=1 FORGE_CUDA_GRAPHS=0 FORGE_DECODE_ATTN_BLOCK=1 bash scripts/phase15_context.sh accepted-context-profile
printf '%s\n' PHASE15_ACCEPTED_DONE
