# Phase 12 Cloud Cost Ledger

Non-secret spending log for Lightning Studio `forge-gpu` (NVIDIA L4).

| Field | Value |
|---|---|
| Cap (cumulative new Phase 12) | **$6.00** |
| Cap per session | **$2.00** |
| Max session wall | **45 min** |
| Planning rate | ≈ **$0.65 / h** |

## Sessions

| # | Date | Planned | Est. $ | Actual wall | Est. actual $ | Stop verified | Notes |
|---|---|---:|---:|---:|---:|---|---|
| 1a | 2026-10-08 | 40m | 0.43 | ~13 min | ≈0.14 | yes | serde_json recursion_limit |
| 1b | 2026-10-08 | 40m | 0.43 | ~10 min | ≈0.11 | yes | missing ComputeBackend import |
| 1c | 2026-10-08 | 40m | 0.43 | ~7.5 min | ≈0.08 | yes | Forge remasure + BF16 survey OK |
| 2 | 2026-10-08 | 40m | 0.43 | ~10 min | ≈0.11 | yes | long/mixed; llama convert deps; no nsys |
| 3b | 2026-10-08 | 40m | 0.43 | ~8 min | ≈0.09 | yes | nsys qdstrm; venv blocked for vLLM |
| 4 | 2026-10-08 | 35m | 0.38 | ~9 min | ≈0.10 | yes | bad llama cache .git |
| 5 | 2026-10-08 | 35m | 0.38 | ~24 min | ≈0.26 | yes | transformers missing on convert |
| 6 | 2026-10-08 | 30m | 0.33 | ~8 min | ≈0.09 | yes | llama-bench tg≈214; JSON/hash retry noise |
| aborted | 2026-10-08 | — | — | ≤2 min | ≈0.05 | yes | Tee-Object file locks / early stop |

## Totals

| | |
|---|---|
| Cumulative estimated spend | **≈$1.03** |
| Remaining budget | **≈$4.97** |

All paid instances stopped and verified (`running→stopped` / unreachable) after sessions.
