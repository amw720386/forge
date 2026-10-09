#!/usr/bin/env bash
set -euo pipefail
ROOT=$(readlink -f /teamspace/studios/this_studio/forge-current)
OUT=/teamspace/studios/this_studio/forge-cache/phase15-results
cd "$ROOT"
FORGE_DECODE_ATTN_BLOCK=0 FORGE_KV_NUM_BLOCKS=512 bash scripts/phase15_http.sh pressure-baseline --workloads long,mixed --concurrency 8 --trials 3 --arrival-gap-ms 10
FORGE_DECODE_ATTN_BLOCK=1 FORGE_KV_NUM_BLOCKS=512 bash scripts/phase15_http.sh pressure-final --workloads long,mixed --concurrency 8 --trials 3 --arrival-gap-ms 10
FORGE_DECODE_ATTN_BLOCK=0 bash scripts/phase15_context.sh context-baseline
FORGE_DECODE_ATTN_BLOCK=1 bash scripts/phase15_context.sh context-final
FORGE_DECODE_ATTN_BLOCK=1 FORGE_CONTEXT_TRIALS=1 FORGE_CONTEXT_CONCURRENCY=1 FORGE_SERVING_PROFILE=1 FORGE_KERNEL_PROFILE=1 FORGE_CUDA_GRAPHS=0 bash scripts/phase15_context.sh context-profile-final
bash scripts/phase15_refs.sh
bash scripts/phase15_tail.sh
bash scripts/phase15_scheduling.sh
FORGE_DECODE_ATTN_BLOCK=1 FORGE_CUDA_MAX_BATCH=4 FORGE_SCHED_POLICY=fixed bash scripts/phase15_http.sh queued-four-slots --workloads mixed --concurrency 8 --trials 3 --arrival-gap-ms 10
printf '%s\n' PHASE15_SESSION2_DONE