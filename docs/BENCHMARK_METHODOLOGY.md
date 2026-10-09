# Benchmark Methodology (Phase 12)

This document defines how Forge measures inference performance and how those
measurements compare to reference runtimes. Historical Phase 9–11 throughput
tables remain in their phase results docs and are **not rewritten** here.

## Audit of `bench_cuda_batch` (pre–Phase 12)

Source: [`runtime/examples/bench_cuda_batch.rs`](../runtime/examples/bench_cuda_batch.rs).

| Check | Finding |
|---|---|
| Generated tokens only | **Pass** — counts `BatchEvent`s (emitted tokens). No padded/inactive slots in GEMM. |
| EOS handling | **Inconsistent for fair compare** — EOS enabled; early stop changes token count and wall time. |
| End-to-end sync | **Mostly pass** — `synchronize()` after step loop; argmax D2H also drains work. |
| Async after timer stop | **Pass** for aggregate wall (sync included). TTFT taken without sync at first event (slightly optimistic). |
| Excluded expensive ops | **Partial** — admit/tokenize outside batch wall; sequential baseline includes admit. Weight upload outside B wall but **re-uploaded per B**. |
| Batch occupancy | **Measured** via `last_batch_size` average. |
| Warm-up | **Fail** — no excluded warm-up; single trial. |
| Model loaded before steady-state | **Partial** — load before B loop, but new GPU upload each B. |
| Token vs request throughput | **Ambiguous** — “per-req tok/s” is `aggregate/B`, not mean of per-request rates. |
| TTFT definition | **Doc bug** — header says “from admit”; code starts clock after admit. |
| Long prompts | **Blocked** — `max_seq = 512` hardcoded. |
| Machine-readable output | **Fail** — stderr only; `LABEL=` log scraping. |

**Verdict:** Useful engineering signal; insufficient alone for competitive claims.
Use [`bench_phase12`](../runtime/examples/bench_phase12.rs) for Phase 12 numbers.

## Phase 12 harness rules

### Timing boundaries

1. **Setup (untimed):** CPU load, GPU weight upload, batcher construction, tokenizer init.
2. **Warm-up (excluded):** ≥1 full generation pass at the measured `(B, dtype, workload)`.
3. **Trial (timed):**
   - `synchronize()` immediately before start.
   - Admit all requests; record `t_admit_start`.
   - Step until idle; record per-request first-token and completion times.
   - `synchronize()` after idle; stop wall clock.
4. **Trials:** default 3; report **median** with min/max (and IQR when ≥3).

### Token accounting

| Mode | Policy | Use |
|---|---|---|
| `fixed_len` | `eos_token_id = None`; emit exactly `max_new` tokens | Competitive throughput |
| `with_eos` | Production EOS; report actual emitted count | Realism / correctness |

EOS tokens are counted **if emitted**. Prefill/prompt tokens are **never** counted in output tok/s.

### Metric definitions

| Metric | Definition |
|---|---|
| Aggregate output tok/s | `sum(emitted_tokens) / e2e_wall_s` |
| Per-request output tok/s | For each request: `emitted_i / (t_done_i - t_admit_i)`; report mean of those rates |
| Prefill tok/s | `sum(prompt_tokens) / (t_last_first_token - t_admit_start)` |
| Steady decode tok/s | `(total_emitted - n_req) / (t_all_done - t_last_first_token)` when `max_new > 1` |
| TTFT | `t_first_token_i - t_admit_i` (from admit, includes chunked prefill) |
| E2E latency | `t_done_i - t_admit_i` |
| P50 / P95 latency | Across requests in a trial (or pooled trials) |
| Occupancy | Mean `last_batch_size` over steps with `last_batch_size > 0` |
| Request throughput | `n_req / e2e_wall_s` |

### Fairness vs llama.cpp / vLLM

- Same model revision: `Qwen/Qwen2.5-0.5B-Instruct`
- Same chat-template prompts and verified tokenized lengths
- Same `max_new`, concurrency, GPU (NVIDIA L4)
- Compare BF16↔F16/BF16 and FP32↔FP32 separately; never claim categorical wins across quant formats
- Model-only benches exclude HTTP; serving benches state the boundary explicitly
- Record runtime version/commit, CUDA toolkit/driver, flags, KV policy, warm-up, trials

### What Phase 12 does **not** claim

- Historical Phase 11 numbers are independently validated (they are remasured)
- Synthetic kernel microbench ≡ end-to-end inference
- Different precision / scheduler knobs are strictly equivalent

## Phase 14 serving audit

[Phase 14 results](PHASE14_RESULTS.md) records all 16 HTTP cells (C=1/2/4/8,
short/medium/long/mixed), fresh Phase 13 baseline, three trials after one
full-concurrency warmup. Rates use actual emitted tokens and POST-to-SSE-DONE
wall time, with EOS enabled and max 64. Latencies are median trial statistics;
small-sample P95 is descriptive. Raw JSON preserves request counts, text and
stage deltas. Never normalize emitted counts to requested maximums.

GPU profiling synchronizes and is excluded from throughput runs. Stream event
regions include staging/launch gaps, not just summed kernel durations. Server
stage timers overlap and are not an additive latency decomposition. Sampled
active sessions include prefill and idle observations, not decode occupancy.
The internal steady-rate boundary changes under early emission; its reported
regression remains published. Peak VRAM is sampled every 100 ms and may miss
brief peaks. Retained grown scratch makes matrix order relevant.

Fresh official GGUF conversion from Forge's exact HF revision passed all 290
weight tensors, tokenizer/template/rope/special-token and native prompt-ID
checks. Reference CUDA offload is verified 25/25 layers. Forge BF16 and llama.cpp
F16 execution differ in precision despite the audited common source; this is
not a bitwise-equivalent arithmetic comparison. Engines ran sequentially with
reference prompt reuse disabled. See the results for flags and hashes. Earlier
phase evidence is preserved rather than retroactively relabeled.
