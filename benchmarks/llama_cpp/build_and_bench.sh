#!/usr/bin/env bash
# Build official llama.cpp and bench Qwen2.5-0.5B against Forge workloads.
# Does not modify upstream sources. Records exact commit in results JSON.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_DIR="${FORGE_BENCH_OUT:-$ROOT/benchmarks/results}"
CACHE="${LLAMA_CPP_CACHE:-/teamspace/studios/this_studio/forge-cache/llama.cpp}"
MODEL_DIR="${FORGE_MODEL_DIR:-/teamspace/studios/this_studio/forge-cache/models/Qwen2.5-0.5B-Instruct}"
GGUF_DIR="${GGUF_DIR:-/teamspace/studios/this_studio/forge-cache/gguf}"
mkdir -p "$OUT_DIR" "$GGUF_DIR" "$(dirname "$CACHE")"

echo "=== llama.cpp setup ==="
need_clone=0
if [[ ! -d "$CACHE/.git" ]]; then
  need_clone=1
elif ! git -C "$CACHE" rev-parse HEAD >/dev/null 2>&1; then
  need_clone=1
fi
if [[ "$need_clone" -eq 1 ]]; then
  echo "cloning fresh llama.cpp into $CACHE"
  rm -rf "$CACHE"
  git clone --depth 1 https://github.com/ggerganov/llama.cpp.git "$CACHE"
fi
cd "$CACHE"
COMMIT="$(git rev-parse HEAD)"
echo "llama_cpp_commit=$COMMIT"

echo "=== llama.cpp CUDA build ==="
cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc 2>/dev/null || echo 4)" --target llama-cli 2>/dev/null || true
cmake --build build -j"$(nproc 2>/dev/null || echo 4)" --target llama-bench 2>/dev/null || true
cmake --build build -j"$(nproc 2>/dev/null || echo 4)"

GGUF_F16="$GGUF_DIR/Qwen2.5-0.5B-Instruct-f16.gguf"
if [[ ! -f "$GGUF_F16" ]]; then
  echo "=== acquiring GGUF (download F16/Q8 or convert) ==="
  python3 -m pip install -q --upgrade pip huggingface_hub 2>/tmp/gguf_deps.txt || true
  export GGUF_F16_OUT="$GGUF_F16" MODEL_DIR
  python3 - <<'PY' 2>>/tmp/gguf_deps.txt
import os, shutil, sys
from huggingface_hub import hf_hub_download, list_repo_files
out = os.environ["GGUF_F16_OUT"]
repos = [
    "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
    "bartowski/Qwen2.5-0.5B-Instruct-GGUF",
]
for repo in repos:
    try:
        files = list_repo_files(repo)
    except Exception as e:
        print("list failed", repo, e, file=sys.stderr)
        continue
    cands = [f for f in files if f.endswith(".gguf") and ("f16" in f.lower() or "fp16" in f.lower())]
    if not cands:
        cands = [f for f in files if f.endswith(".gguf") and "q8_0" in f.lower()]
    if not cands:
        continue
    # Prefer smaller path names; take first sorted fp16 then q8
    fn = sorted(cands, key=lambda s: (0 if "f16" in s.lower() or "fp16" in s.lower() else 1, len(s)))[0]
    print("using", repo, fn)
    path = hf_hub_download(repo_id=repo, filename=fn)
    shutil.copy(path, out)
    print("copied to", out)
    raise SystemExit(0)
raise SystemExit("no prebuilt GGUF found")
PY
fi
if [[ ! -f "$GGUF_F16" ]]; then
  echo "=== converting HF -> GGUF F16 (fallback) ==="
  python3 -m pip install -q sentencepiece protobuf 'numpy<2' transformers 2>>/tmp/gguf_deps.txt \
    || python3 -m pip install -q sentencepiece protobuf numpy transformers 2>>/tmp/gguf_deps.txt
  export PYTHONPATH="$CACHE/gguf-py:${PYTHONPATH:-}"
  python3 convert_hf_to_gguf.py "$MODEL_DIR" --outfile "$GGUF_F16" --outtype f16
fi
[[ -f "$GGUF_F16" ]] || { echo "FAIL: missing $GGUF_F16"; tail -c 2000 /tmp/gguf_deps.txt 2>/dev/null || true; exit 1; }
# Record which file we use (may be Q8 if F16 unavailable)
echo "gguf=$GGUF_F16 size=$(du -h "$GGUF_F16" | awk '{print $1}')"

PROMPT_HELLO=$'<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n'
CLI=""
for c in build/bin/llama-cli build/llama-cli; do
  [[ -x "$c" ]] && CLI="$c" && break
done
BENCH=""
for c in build/bin/llama-bench build/llama-bench; do
  [[ -x "$c" ]] && BENCH="$c" && break
done
[[ -n "$CLI" ]] || { echo "FAIL: llama-cli missing"; ls -la build/bin 2>/dev/null | head; exit 1; }
echo "cli=$CLI bench=${BENCH:-none}"

RESULTS="$OUT_DIR/llama_cpp_short_f16.json"
{
  echo "{"
  echo "\"runtime\":\"llama.cpp\","
  echo "\"commit\":\"$COMMIT\","
  echo "\"precision\":\"f16\","
  echo "\"gguf\":\"$GGUF_F16\","
  echo "\"gpu\":\"cuda\","
  echo "\"workloads\":["
  first=1
  for n_predict in 16 64; do
    for parallel in 1 4 8; do
      [[ $first -eq 1 ]] || echo ","
      first=0
      echo -n "  {\"max_new\":$n_predict,\"parallel\":$parallel"
      if [[ -n "$BENCH" ]]; then
        out="$("$BENCH" -m "$GGUF_F16" -p 32 -n "$n_predict" -pg "$parallel,$n_predict" -r 3 2>&1 || true)"
        echo -n ",\"llama_bench_raw\":$(python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))' <<<"$out")"
        # Parse tg tok/s lines when present
        tg=$(python3 - <<'PY' <<<"$out"
import re,sys
text=sys.stdin.read()
# llama-bench prints lines with tg (tokens/second decode)
vals=[]
for m in re.finditer(r"tg\s*\S*\s+([0-9.]+)", text):
    try: vals.append(float(m.group(1)))
    except: pass
for m in re.finditer(r"([0-9.]+)\s*t/s", text):
    try: vals.append(float(m.group(1)))
    except: pass
print(vals[-1] if vals else "null")
PY
)
        echo -n ",\"parsed_tok_s\":$tg"
      fi
      if [[ "$parallel" == "1" ]]; then
        # Steady generate without counting load: warm once then time
        "$CLI" -m "$GGUF_F16" -p "$PROMPT_HELLO" -n "$n_predict" --temp 0 -ngl 99 -c 2048 >/dev/null 2>&1 || true
        t0=$(date +%s%N)
        "$CLI" -m "$GGUF_F16" -p "$PROMPT_HELLO" -n "$n_predict" --temp 0 -ngl 99 -c 2048 >/tmp/llama_cli_out.txt 2>/tmp/llama_cli_err.txt || true
        t1=$(date +%s%N)
        wall=$(python3 -c "print( ($t1-$t0)/1e9 )")
        echo -n ",\"cli_wall_s\":$wall,\"cli_note\":\"second_run_after_warmup_includes_model_reload_unless_server\""
      fi
      echo -n "}"
    done
  done
  echo
  echo "]"
  echo "}"
} > "$RESULTS"
echo "Wrote $RESULTS"

if [[ -n "$BENCH" ]]; then
  echo "LABEL=llama_cpp_medium_long_f16"
  "$BENCH" -m "$GGUF_F16" -p 256 -n 64 -pg 1,64 -pg 4,64 -r 3 2>&1 | tee "$OUT_DIR/llama_cpp_medium.txt" || true
  "$BENCH" -m "$GGUF_F16" -p 1024 -n 64 -pg 1,64 -pg 2,64 -r 3 2>&1 | tee "$OUT_DIR/llama_cpp_long.txt" || true
fi

echo "LLAMA_CPP_BENCH_OK"
echo "llama_cpp_commit=$COMMIT"
