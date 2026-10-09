#!/usr/bin/env bash
set -euo pipefail
ROOT=$(readlink -f /teamspace/studios/this_studio/forge-current)
cd "$ROOT"
for policy in decode-first fixed; do
 for budget in 256 1024; do
  for chunk in 64 128; do
   FORGE_DECODE_ATTN_BLOCK=1 FORGE_SCHED_POLICY="$policy" FORGE_STEP_TOKEN_BUDGET="$budget" FORGE_PREFILL_CHUNK="$chunk" bash scripts/phase15_http.sh "policy-$policy-budget$budget-chunk$chunk" --workloads mixed --concurrency 8 --trials 3 --arrival-gap-ms 10
  done
 done
done
