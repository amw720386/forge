#!/usr/bin/env bash
set -euo pipefail
ROOT=$(readlink -f /teamspace/studios/this_studio/forge-current)
cd "$ROOT"
bash scripts/phase14_build.sh
bash scripts/phase15_final.sh
