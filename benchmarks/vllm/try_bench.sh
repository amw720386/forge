#!/usr/bin/env bash
# Best-effort vLLM benchmark for Qwen2.5-0.5B on L4 (Lightning conda constraints).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_DIR="${FORGE_BENCH_OUT:-$ROOT/benchmarks/results}"
MODEL_DIR="${FORGE_MODEL_DIR:-/teamspace/studios/this_studio/forge-cache/models/Qwen2.5-0.5B-Instruct}"
mkdir -p "$OUT_DIR"
BLOCKER="$OUT_DIR/vllm_blocker.json"
deadline=$((SECONDS + 600))
export MODEL_DIR BLOCKER

skip() {
  local reason="$1"
  python3 - <<PY
import json, os
open(os.environ["BLOCKER"],"w").write(json.dumps({
  "runtime": "vllm",
  "status": "unavailable",
  "reason": """$reason""",
  "note": "Not a Forge failure; comparison omitted per Phase 12 fairness rules."
}, indent=2))
print("VLLM_SKIPPED:", """$reason""")
PY
  exit 0
}

python3 -V || skip "python3 missing"
if [[ $SECONDS -gt $deadline ]]; then skip "deadline before install"; fi

# Lightning forbids extra venvs; install into the active conda/base env.
if ! python3 -c "import vllm" 2>/dev/null; then
  echo "Installing vLLM into active Python env (no venv)…"
  python3 -m pip install -U pip wheel setuptools >/tmp/vllm_pip1.txt 2>&1 || skip "pip upgrade failed: $(tail -c 400 /tmp/vllm_pip1.txt)"
  if ! python3 -m pip install "vllm>=0.6,<0.9" >/tmp/vllm_pip2.txt 2>&1; then
    skip "vllm pip install failed: $(tail -c 800 /tmp/vllm_pip2.txt)"
  fi
fi

python3 - <<'PY' >"$OUT_DIR/vllm_short.json" 2>"$OUT_DIR/vllm_short.err"
import json, os, time, sys
try:
    from vllm import LLM, SamplingParams
    import vllm
except Exception as e:
    print(f"import_failed: {e}", file=sys.stderr)
    sys.exit(3)

model = os.environ["MODEL_DIR"]
prompt = "<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n"
sp = SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True)
results = []
llm = LLM(model=model, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.7)
for bs in [1, 2, 4, 8]:
    prompts = [prompt] * bs
    llm.generate(prompts, sp)  # warmup
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    wall = time.time() - t0
    ntok = sum(len(o.outputs[0].token_ids) for o in outs)
    results.append({
        "batch": bs,
        "wall_s": wall,
        "total_tokens": ntok,
        "aggregate_tok_s": ntok / wall if wall > 0 else 0,
        "dtype": "bfloat16",
    })
print(json.dumps({
    "runtime": "vllm",
    "version": getattr(vllm, "__version__", "unknown"),
    "model": model,
    "precision": "bfloat16",
    "token_policy": "ignore_eos",
    "results": results,
}, indent=2))
PY

if [[ ! -s "$OUT_DIR/vllm_short.json" ]] || grep -q import_failed "$OUT_DIR/vllm_short.err" 2>/dev/null; then
  skip "vLLM runtime failed: $(tail -c 800 "$OUT_DIR/vllm_short.err" 2>/dev/null || true)"
fi
echo "VLLM_BENCH_OK"
