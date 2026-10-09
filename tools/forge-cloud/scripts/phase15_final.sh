#!/usr/bin/env bash
set -euo pipefail
ROOT=$(readlink -f /teamspace/studios/this_studio/forge-current)
cd "$ROOT"
bash scripts/phase15_accepted.sh
bash scripts/phase15_refs.sh
