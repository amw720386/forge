# Phase 15 evidence map

The accepted serving comparisons use `accepted-baseline-http.json` and
`accepted-final-http.json`. Both include the CUDA worker thread fix. The
control disables partitioned decode attention; the candidate enables the
precision-aware dispatch. `summary.json` includes labeled experiments as well
as accepted runs; consult `docs/PHASE15_RESULTS.md` before comparing its arms.

- `accepted-short-repeat-*.json`: nine-trial short-request regression checks.
- `accepted-context-*.json`: HTTP input-length sweep and eager profiling.
- `accepted-pressure-*.json`, `accepted-scheduling-only-http.json`,
  `accepted-combined-fixed-http.json`, policy and queued runs: controlled
  capacity and scheduling experiments, with confounds disclosed in results.
- `fp32-*-http.json`: matched FP32 long/mixed comparisons.
- `final-bf16-*-http.json`: final-source BF16 behavior check.
- `llama-*.json`, `llama-offload-validation.log`: limited F16 reference;
  different weight precision, verified 25/25 layer offload.
- Output `*-audit.json`, GPU test/regression logs and CPU/Go logs: correctness
  evidence. Rejected numerical failures are preserved, not counted as passes.
- `accepted-attention.jsonl`: direct attention timings. Shared tiles,
  adaptive scheduling, unrestricted BF16 dispatch and intermediate device
  fallback were rejected; their raw experiments remain separately labeled.
  Initial, pinned-final, final-http, decode-only, prefill-only, combined-adaptive
  and interim-device-fallback measurements are development evidence, not the
  final production comparison.
- `retained-source-hashes.json`: final executed GPU source identity.
  `bf16-measurement-source-hashes.json`: source at the main BF16 matrix.
  `final-source-verification.json`: 22 matching final local paths and the sole
  post-GPU context-reporting correction.
- `context-token-count-audit.json`: runtime-log proof that the server adds 19
  tokens to the plain-text context bodies. Corrected reports retain all timing
  samples; `uncorrected-*.json` preserves the original body-only metadata.
  Reproduce with `python benchmarks/analysis/audit_phase15_context_counts.py`.
- `session*-cost.json`, preflight/stop proofs and `final-cloud-state.txt`:
  conservative compute estimate and independently verified stopped state.

Runtime source and ordinary HTTP prompts were unchanged by the metadata audit.
No additional GPU execution occurred after shutdown.
