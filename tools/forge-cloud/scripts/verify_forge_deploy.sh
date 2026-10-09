#!/usr/bin/env bash
# Verify Forge deploy identity BEFORE any cargo/cmake work.
# Expects to run from the deploy root (directory containing .forge-deploy-manifest.json).
# Exit 0 only when source identity matches the manifest and obsolete markers are absent.
set -euo pipefail

ROOT="$(pwd)"
MANIFEST="${ROOT}/.forge-deploy-manifest.json"
IDFILE="${ROOT}/.forge-deploy-id"

die() { echo "FAIL: $*" >&2; exit 1; }
ok() { echo "OK: $*"; }

echo "=== FORGE DEPLOY VERIFY ==="
echo "cwd=$ROOT"

[[ -f "$MANIFEST" ]] || die "missing $MANIFEST"
[[ -f "$IDFILE" ]] || die "missing $IDFILE"

command -v sha256sum >/dev/null 2>&1 || die "sha256sum required"
command -v python3 >/dev/null 2>&1 || die "python3 required for manifest JSON parse"

DEPLOY_ID="$(tr -d '[:space:]' <"$IDFILE")"
# Use relative paths so native Windows python3 (via Git Bash) and Linux agree.
read -r MAN_ID CONTENT_HASH GIT_REV < <(python3 -c "
import json
m=json.load(open('.forge-deploy-manifest.json'))
print(m['deploy_id'], m['content_hash'], m.get('git_revision') or '-')
")

[[ -n "$DEPLOY_ID" ]] || die "empty deploy id"
[[ "$DEPLOY_ID" == "$MAN_ID" ]] || die "deploy id mismatch: file=$DEPLOY_ID manifest=$MAN_ID"

echo "deploy_id=$DEPLOY_ID"
echo "content_hash=$CONTENT_HASH"
if [[ "$GIT_REV" != "-" && -n "$GIT_REV" ]]; then
  echo "git_revision=$GIT_REV"
fi

REAL="$(readlink -f "$ROOT" 2>/dev/null || pwd -P)"
case "$REAL" in
  */forge-deploys/*) ok "build root is isolated deploy: $REAL" ;;
  *) echo "NOTE: path $REAL is not under forge-deploys (allowed for local staging checks)" ;;
esac

REQUIRED=(
  runtime/Cargo.toml
  runtime/build.rs
  runtime/src/lib.rs
  runtime/src/cuda_model/mod.rs
  runtime/src/cuda_model/batcher.rs
  runtime/src/cuda_model/transformer.rs
  runtime/src/cuda_model/generate.rs
  runtime/src/cuda_model/kv.rs
  runtime/src/cuda_model/weights.rs
  kernels/CMakeLists.txt
  kernels/cuda/include/forge_cuda.h
  scripts/remote_cuda_phase8b.sh
  scripts/verify_forge_deploy.sh
  scripts/preflight_phase8b.sh
)
for f in "${REQUIRED[@]}"; do
  [[ -f "$ROOT/$f" ]] || die "missing required file $f"
done
ok "required source files present"

if grep -R --include='*.rs' -n 'use crate::model::batch::' runtime/src 2>/dev/null; then
  die "obsolete private import use crate::model::batch:: still present"
fi
if grep -R --include='*.rs' -n 'use crate::model::generate::GenError' runtime/src 2>/dev/null; then
  die "obsolete private import use crate::model::generate::GenError still present"
fi
ok "no known obsolete cuda_model import markers"

# Per-file hashes + content_hash (must match Go cloud.hashFileMap)
python3 - <<'PY'
import hashlib, json, os, sys

root = os.getcwd()
man = json.load(open(".forge-deploy-manifest.json"))
expected = {
    k: v for k, v in man["files"].items()
    if k not in (".forge-deploy-manifest.json", ".forge-deploy-id")
}
errors = []
actual = {}
for dirpath, _, filenames in os.walk(root):
    for name in filenames:
        path = os.path.join(dirpath, name)
        rel = os.path.relpath(path, root).replace("\\", "/")
        if rel in (".forge-deploy-manifest.json", ".forge-deploy-id"):
            continue
        # Ignore build / benchmark artifacts created mid-session in the deploy tree.
        if (
            rel.startswith("runtime/target/")
            or "/target/" in rel
            or "build-cuda" in rel
            or (rel.startswith("benchmarks/results/") and rel not in expected)
            or rel.endswith(".qdstrm")
            or rel.endswith(".nsys-rep")
        ):
            continue
        actual[rel] = hashlib.sha256(open(path, "rb").read()).hexdigest()

for rel, want in sorted(expected.items()):
    got = actual.get(rel)
    if got is None:
        errors.append(f"missing:{rel}")
    elif got != want:
        errors.append(f"hash:{rel}")
for rel in actual:
    if rel not in expected:
        errors.append(f"unexpected:{rel}")
if len(actual) != len(expected):
    errors.append(f"count:{len(actual)}!={len(expected)}")

lines = "".join(f"{k}\t{actual[k]}\n" for k in sorted(actual))
got_hash = hashlib.sha256(lines.encode()).hexdigest()
if got_hash != man["content_hash"]:
    errors.append(f"content_hash:{got_hash}!={man['content_hash']}")

if errors:
    print("FAIL: " + ", ".join(errors[:24]), file=sys.stderr)
    sys.exit(1)
print(f"OK: per-file hashes match manifest ({len(expected)} files)")
print(f"OK: content_hash matches")
PY

echo "DEPLOY_VERIFY_OK id=$DEPLOY_ID content_hash=$CONTENT_HASH"
