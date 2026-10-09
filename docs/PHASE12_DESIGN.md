# Phase 12 Design — Competitive Benchmarking

## Goals

1. Audit and replace unreliable throughput measurement with a reproducible harness.
2. Remasure Forge FP32/BF16 on NVIDIA L4 under documented workloads.
3. Compare against llama.cpp (required) and vLLM (best-effort).
4. Broaden BF16 greedy correctness vs CUDA FP32.
5. Profile CUDA bottlenecks; propose Phase 13 work from evidence only.

## Non-goals

- New kernels, FlashAttention, CUDA Graphs, speculative decoding, multi-GPU
- Replacing Forge’s forward pass with another framework
- Rewriting historical Phase 11 benchmark tables

## Architecture

```
benchmarks/
  forge/          run_forge_bench.sh, run_correctness.sh, hf_spotcheck.py
  llama_cpp/      build_and_bench.sh
  vllm/           try_bench.sh
  workloads/      phase12_matrix.json
  results/        gitignored raw JSON
  analysis/       summarize_results.py
runtime/examples/
  bench_phase12.rs
  bf16_correctness_survey.rs
scripts/remote_cuda_phase12.sh
api/cmd/forge-cloud/validate_phase12.go
```

## Workload priorities

| Pri | Name | Prompt | Gen | B | Dtype |
|---|---|---|---|---|---|
| P0 | short | ~16–32 tok | 16–64 | 1,2,4,8 | fp32, bf16 |
| P0 | medium | ~256 tok | 64 | 1,4 | fp32, bf16 |
| P1 | long | ~1024 tok | 64 | 1,2 | bf16 |
| P1 | mixed | staggered | varied | 8 | bf16 |
| P2 | full sweep | — | — | — | **omitted** |

## Cloud budget

- Studio: `forge-gpu`, GPU: L4 only
- ≤45 min / session, ≤$2 / session, ≤$6 cumulative Phase 12
- Ledger: [PHASE12_COST_LEDGER.md](PHASE12_COST_LEDGER.md)
- Do not roll unused prior-phase budget into Phase 12

## Session plan

1. Forge remasure + BF16 survey + controlled flag A/B
2. llama.cpp + nsys + medium/long
3. Reserve: vLLM or gaps

## Success criteria

See Definition of Done in the Phase 12 task brief / [PHASE12_RESULTS.md](PHASE12_RESULTS.md).
