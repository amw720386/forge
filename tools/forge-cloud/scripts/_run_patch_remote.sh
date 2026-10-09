#!/usr/bin/env bash
set -euo pipefail
P=""
for d in /teamspace/studios/this_studio/forge-upload "$HOME/forge-upload"; do
  if [[ -f "$d/_patch_phase9_remote.py" ]]; then
    P="$d/_patch_phase9_remote.py"
    break
  fi
done
echo "PATCH_SRC=$P"
[[ -n "$P" ]] || { echo "FAIL: patch script not found"; exit 1; }
python3 "$P"
