# Phase 15 results

Phase 15 is implemented and validated with precision-aware attention dispatch,
a pinned CUDA worker, graph dispatch invalidation, and an optional fair fixed
token budget. Shared prefill tiles and the adaptive heuristic were evaluated
and rejected. All GPU work used the existing L4; it is stopped and verified.

## Accepted methodology

Same Qwen2.5-0.5B-Instruct revision
`7ae557604adf67be50417f59c2c2f167def9a775`, same tokenizer/prompts (10/190/910
actual input tokens), FP32 KV, same flags and client POST-to-SSE-DONE boundary.
EOS enabled, cap 64, full-concurrency warmup plus three trials per cell.
Tables report medians of trial metrics; ranges are trial throughput min/max.
Mixed assigns short/medium/long cyclically: C=1 is short only; C=2 has no long
request; C=4 has 2/1/1 and C=8 has 3/3/2 requests. The staggered trace delays
successive arrivals by ten milliseconds; that delay precedes each POST timer.

Both accepted arms include the OS-thread fix. The original one-warp attention
is selected by flag zero in the control; flag one uses the validated dispatch.
The BF16 matrix preceded the final FP32-only widening of dispatch. Its exact
sources are in bf16-measurement-source-hashes.json; the final BF16 behavior was
rechecked afterward. Final retained-source-hashes.json matches 22 of the 23
listed local sources, including every runtime/kernel path and bench_http.py.
The sole later change is phase15_context.py: it now counts the server chat
wrapper without changing the wire prompts. final-source-verification.json
records that reporting-only correction. The HTTP context sweep therefore
reports audited model inputs of 147–4115 tokens; original body-only metadata
and every timing sample are preserved. See PHASE15_PROFILE.md for details.

## BF16 HTTP matrix

| Workload | C | Phase 14 control tok/s | Phase 15 tok/s | Baseline/final TTFT ms | Baseline P50/P95 ms | Final P50/P95 ms |
|---|---:|---:|---:|---:|---:|---:|
| short | 1 | 183.01 | 183.16 | 7.0/6.9 | 54.6/54.6 | 54.6/54.6 |
| short | 2 | 340.31 | 343.66 | 7.4/7.3 | 58.6/58.8 | 57.9/58.1 |
| short | 4 | 563.72 | 564.56 | 8.2/19.2 | 65.3/70.9 | 70.1/70.8 |
| short | 8 | 1032.35 | 1010.03 | 22.1/22.7 | 76.6/77.3 | 74.7/78.9 |
| medium | 1 | 131.51 | 166.82 | 16.2/16.4 | 486.6/486.6 | 383.6/383.6 |
| medium | 2 | 202.24 | 292.38 | 22.3/22.0 | 632.8/632.9 | 437.4/437.8 |
| medium | 4 | 391.81 | 561.02 | 37.7/37.0 | 652.9/653.4 | 455.9/456.3 |
| medium | 8 | 735.60 | 1030.89 | 74.3/72.8 | 695.2/695.9 | 496.0/496.5 |
| long | 1 | 59.50 | 104.21 | 113.1/114.4 | 1075.7/1075.7 | 614.1/614.1 |
| long | 2 | 74.95 | 149.36 | 205.2/208.1 | 1707.3/1707.8 | 856.7/856.9 |
| long | 4 | 134.01 | 134.06 | 398.6/399.4 | 1909.9/1910.2 | 1909.4/1909.5 |
| long | 8 | 221.31 | 221.43 | 796.3/795.1 | 2312.9/2313.2 | 2311.5/2312.3 |
| mixed | 1 | 183.76 | 183.68 | 6.8/6.9 | 54.4/54.4 | 54.4/54.4 |
| mixed | 2 | 145.11 | 175.84 | 9.7/10.0 | 96.2/510.0 | 97.5/420.8 |
| mixed | 4 | 90.54 | 174.45 | 15.4/15.5 | 230.6/1634.4 | 234.5/848.2 |
| mixed | 8 | 194.04 | 202.21 | 66.4/66.3 | 1657.2/1803.6 | 1667.8/1730.1 |

Medium C=8 improves about 40%, with P95 696→496 ms. Long C=1/2 throughput rises
about 75%/99%; larger long BF16 batches preserve the original kernel and latency.
Mixed C=4 P95 falls 1634→848 ms; C=8 improves only about 4% (1804→1730 ms).
The C=8 stretch targets—long TTFT <500 ms, mixed >250 tok/s and P95 <1.4 s—
were **not met**. The unrestricted candidate's larger apparent gains are rejected
because it failed BF16 greedy agreement. They are not production results.

All 180 accepted BF16 completions match control text exactly. A final-source
C=8 check adds 32 matching completions. This is bounded workload evidence,
not a claim of bitwise BF16 logit equivalence for arbitrary prompts.

## Short-request regression checks

| Arm | Workload | C | Tok/s median (min–max) | TTFT ms | P50/P95 ms |
|---|---|---:|---:|---:|---:|
| accepted-short-repeat-baseline | short | 4 | 571.32 (495.77–674.11) | 8.3 | 64.7/69.8 |
| accepted-short-repeat-baseline | short | 8 | 1054.95 (989.67–1088.84) | 21.5 | 73.6/75.6 |
| accepted-short-repeat-final | short | 4 | 565.91 (465.28–672.16) | 8.5 | 65.1/70.6 |
| accepted-short-repeat-final | short | 8 | 1053.61 (936.04–1087.44) | 22.4 | 74.9/75.7 |

These are nine trials per cell. C=8 throughput is essentially unchanged at
1055→1054 tok/s, above the prior reported 992. C=4 is 571→566 tok/s, about -1%.
Its main three-trial TTFT sample rose to 19 ms; the nine-trial medians are
8.3→8.5 ms, with visible outliers. The earlier isolated C=4 baseline at 671 tok/s
was not a stable basis for a regression claim. Variability is retained.

## FP32 long/mixed comparison

| Arm | Workload | C | Tok/s median (min–max) | TTFT ms | P50/P95 ms |
|---|---|---:|---:|---:|---:|
| fp32-baseline | long | 1 | 33.68 (33.63–33.69) | 185.4 | 1900.1/1900.1 |
| fp32-baseline | long | 8 | 166.51 (166.46–166.81) | 1259.2 | 3074.3/3074.7 |
| fp32-baseline | mixed | 1 | 105.01 (104.98–105.19) | 11.6 | 95.2/95.2 |
| fp32-baseline | mixed | 8 | 151.43 (151.37–151.93) | 147.1 | 2138.9/2310.9 |
| fp32-final | long | 1 | 60.37 (60.30–60.37) | 184.8 | 1060.2/1060.2 |
| fp32-final | long | 8 | 228.77 (228.19–228.89) | 1263.8 | 2237.2/2237.6 |
| fp32-final | mixed | 1 | 104.81 (104.37–104.87) | 11.7 | 95.4/95.4 |
| fp32-final | mixed | 8 | 231.16 (230.55–234.15) | 152.3 | 1406.8/1513.5 |

FP32 can retain partitioning for long larger batches: C=8 long throughput
improves about 37%, mixed about 53%. Both arms use FP32 weights and otherwise
matched settings; these are not substituted for the BF16 baseline. All 54
FP32 completions match their FP32 control text exactly.

## Ablations and scheduling

The accepted attention-only arm is accepted-final (legacy scheduling, unchanged
prefill). No prefill optimization was retained: its production path is the
control. The following fixed-policy runs use the same unstaggered mixed trace:

| Arm | Workload | C | Tok/s median (min–max) | TTFT ms | P50/P95 ms |
|---|---|---:|---:|---:|---:|
| accepted-scheduling-only | mixed | 1 | 182.89 (182.45–183.65) | 6.9 | 54.7/54.7 |
| accepted-scheduling-only | mixed | 8 | 192.57 (192.44–192.85) | 70.4 | 1667.8/1817.0 |
| accepted-combined-fixed | mixed | 1 | 182.32 (182.28–182.99) | 7.0 | 54.8/54.8 |
| accepted-combined-fixed | mixed | 8 | 203.16 (201.49–204.07) | 70.4 | 1660.1/1722.3 |

Scheduling-only (original attention + fixed) does not beat the control reliably.
Combined fixed is close to attention-only, so it is not credited for the gain.
The retained fixed policy is justified by hard-budget fairness, not average speed.
Staggered C=8 trials compare both policies, chunks 64/128 and budgets 256/1024:

| Policy | Budget | Chunk | Tok/s | TTFT ms | P95 ms | Max request TTFT ms |
|---|---:|---:|---:|---:|---:|---:|
| decode-first | 256 | 64 | 190.14 | 46.5 | 1799.1 | 433.0 |
| decode-first | 256 | 128 | 188.73 | 39.5 | 1812.8 | 343.3 |
| decode-first | 1024 | 64 | 189.93 | 46.7 | 1800.9 | 441.5 |
| decode-first | 1024 | 128 | 191.48 | 40.1 | 1786.4 | 336.8 |
| fixed | 256 | 64 | 189.96 | 46.7 | 1800.7 | 432.9 |
| fixed | 256 | 128 | 195.08 | 42.8 | 1773.8 | 343.9 |
| fixed | 1024 | 64 | 188.53 | 46.7 | 1814.9 | 437.7 |
| fixed | 1024 | 128 | 191.47 | 45.4 | 1786.3 | 335.8 |

The four-active-slot trace queues eight arrivals; throughput and waiting tradeoffs
are reported below. Queue-wait maxima and host prefill-stall totals are in each
matching metrics JSON (including warmup), not misrepresented as per-request GPU
stall counters. Scheduled-token samples are lower bounds; every-step hard bounds
and starvation prevention are asserted in the runtime tests.

## KV pressure, occupancy and memory

| Arm | Workload | C | Tok/s median (min–max) | TTFT ms | P50/P95 ms |
|---|---|---:|---:|---:|---:|
| accepted-pressure-baseline | long | 8 | 206.59 (205.94–206.66) | 817.2 | 2402.3/2418.2 |
| accepted-pressure-baseline | mixed | 8 | 185.66 (185.18–186.02) | 55.8 | 1657.2/1840.0 |
| accepted-pressure-final | long | 8 | 212.52 (210.23–213.25) | 756.9 | 2344.2/2358.8 |
| accepted-pressure-final | mixed | 8 | 191.21 (191.19–192.51) | 41.1 | 1667.9/1788.8 |
| accepted-queued-four-slots | mixed | 8 | 149.89 (149.81–150.24) | 200.1 | 1820.2/2265.1 |

The pressure case uses 512 sixteen-token blocks, the same pool capacity as the
ordinary effective-1024-context matrix, with staggered long/mixed arrivals.
It is a near-exhaustion test, not a claim that the pool was halved. Long C=8
observes 488/512 blocks (95.3125%); mixed observes about 33.2%. This is KV-pool
utilization, not the fraction of GPU VRAM used. Default pool storage is 192 MiB;
the 4608-context sweep uses 2304 blocks / 864 MiB. All pools return to zero
allocated blocks at the end of completed traces; cancellation/reuse tests agree.

| Precision / scenario | Sampled peak VRAM MiB |
|---|---:|
| BF16 accepted matrix | 1658 |
| FP32 accepted subset | 2586 |
| Queued four slots | 1486 |
| BF16 4608-cap context sweep | 2302 |

NVIDIA reports 23034 MiB total. VRAM is sampled around 100 ms and may miss brief
peaks. Pinned baseline/candidate both use one Lt worker cache; the older reported
1978 MiB should not be used to attribute memory savings to the attention kernel.
Active-slot occupancy and request throughput are in summary.json. They are not
hardware SM occupancy. Effective post-first-token output rates are also derived
there, including serving interference rather than pure kernel throughput.

## Graph/eager serving

| Arm | Workload | C | Tok/s median (min–max) | TTFT ms | P50/P95 ms |
|---|---|---:|---:|---:|---:|
| accepted-eager | long | 8 | 218.01 (217.99–218.13) | 792.5 | 2347.7/2348.3 |
| accepted-eager | mixed | 8 | 197.31 (196.83–197.51) | 76.5 | 1706.2/1773.6 |

Compare with the graphed accepted matrix above. Dynamic prefill stays eager.
Graphs preserve correctness after block growth, replacement and dispatch changes;
they do not solve the dominant long-prefill latency.

## Operational metrics

Accepted BF16 C=8 trial medians. Effective decode divides output tokens after
the first by the longest request's post-first-token interval; it includes serving
interference. Active slots are five-millisecond samples, not SM occupancy.

| Workload | Requests/s | Effective post-first-token tok/s | Mean active slots | Mean queue wait ms | Scheduler host ms/step | Peak KV utilization |
|---|---:|---:|---:|---:|---:|---:|
| short | 101.00 | 1183.9 | 6.79 | 3.836 | 0.0175 | 3.12% |
| medium | 16.11 | 1190.4 | 7.73 | 0.679 | 0.0232 | 25.00% |
| long | 3.46 | 332.3 | 7.90 | 4.338 | 0.0714 | 95.31% |
| mixed | 4.62 | 213.5 | 5.34 | 1.643 | 0.0329 | 33.20% |

Policy metrics below come from each fresh server's final counters across one
warmup plus three measured traces. Host prefill stall is divided by those four
traces; it is a mean including warmup, not a median or a GPU-exclusive stall.
Queue maximum is an actual recorded host-stage maximum over those requests.

| Policy | Budget | Chunk | Mean queue wait ms | Max queue wait ms | Host prefill stall ms/trace | End allocated KV blocks |
|---|---:|---:|---:|---:|---:|---:|
| decode-first | 256 | 64 | 18.123 | 91.830 | 125.23 | 0 |
| decode-first | 256 | 128 | 20.756 | 93.245 | 129.63 | 0 |
| decode-first | 1024 | 64 | 17.443 | 86.897 | 112.27 | 0 |
| decode-first | 1024 | 128 | 21.798 | 96.823 | 116.78 | 0 |
| fixed | 256 | 64 | 19.099 | 94.242 | 127.53 | 0 |
| fixed | 256 | 128 | 19.465 | 88.747 | 85.29 | 0 |
| fixed | 1024 | 64 | 18.892 | 87.654 | 111.48 | 0 |
| fixed | 1024 | 128 | 20.842 | 96.172 | 116.01 | 0 |

The four-slot trace's maximum recorded queue wait was
1850.4 ms. Its default pool also shrank to
256 blocks, so this tests the joint slot/capacity configuration
rather than attributing all effects to active slots alone. All requests completed;
KV allocation returned to zero.

## Correctness

- Windows CPU: 43 Rust tests, CPU C++ kernel suite, Go API/cloud and Go bindings pass.
- cuda-stub is a compile/link check only, never counted as a GPU pass.
- Actual L4: existing batch, paged-KV, transformer, Phase 13 graph/fallback and
  Phase 14 packed-prefill/lifecycle suites pass unchanged; CUDA kernel suite passes.
- New synthetic tests cover B=1/2/4/8, 128–4096 tokens, graph growth/replacement,
  and 300 greedy decisions per dtype. Maximum FP32 full-path error is 1.97e-6;
  BF16 measured full-path error is 0.002388, with no greedy disagreements.
- Full checkpoint at 512/4096: 120 matching greedy decisions per dtype. FP32
  maximum error 2.193451e-5; BF16 old/new maximum 0.108668804.
- Actual HTTP long prompt, 64 teacher-forced steps, B=1/2/4/8: 960 matching
  decisions per dtype. FP32 maximum 4.053116e-5; BF16 0.20844984. Long B>=3
  BF16 stays on the exact reference. These compare same-dtype execution paths;
  they are not BF16-versus-FP32 errors.
- Phase 14's full-model BF16-versus-FP32 difference (about 0.181) remains separately
  documented; the real packed-model regression and 7/7 short correctness survey
  are retained. BF16 logits are not bitwise identical.
- Fixed-policy tests assert every-step caps at budgets 1/2/16, cancel prefill and
  decode, replace slots, require finite progress and complete KV reclamation.
  CPU planning tests serve every decoder/prefill slot under continuous competition.

Existing numerical tolerances were not changed. An initial new large-synthetic
BF16 assertion reused FP32-style full-logit tolerances and failed; it is retained
as bf16-diagnostic-failure.log, not counted as a pass. Full-path BF16 validation
uses the established measured-error/greedy contract; standalone attention and
FP32 retain strict tolerances. The later real BF16 greedy failure was resolved
by fallback, without relaxing its zero-disagreement assertion.

## Rejected experiments and evidence boundaries

Shared-prefill tile sizes 8/16 and softmax tiles 4/8 were removed after direct
and serving experiments. Adaptive cost caps were removed after no reliable
benefit. Rejected sources are stored in rejected-*.txt and numerical failure logs.
Historical final-http, pinned-final-http, decode-only/prefill-only and combined
experiment files describe earlier candidates; their larger gains are not accepted.
The unpinned corruption reproduced with graphs off. Interim device-fallback
files are also excluded from accepted conclusions. No combined gain is credited
to a rejected tile or adaptive policy.

## Limited llama.cpp reference

Pinned commit `71ad0590f4808b6202f9213d166913858c73b1bc`, Phase 14 audited GGUF
from the same model revision/tokenizer/input IDs, same L4, cap 64, EOS enabled,
same client boundary and concurrency. F16 weights versus Forge BF16/FP32 are
explicit precision differences. CUDA offload is verified as **25/25 layers**
in llama-offload-validation.log; benchmark startup requested -ngl 99, FP32 KV,
eight slots, prompt caching disabled. Long prompt logs account for 910 evaluated
input tokens on measured requests. Actual C=8 output totals match at 512 long
and 350 mixed tokens per trial. No numerical equivalence is claimed.

| Reference | Workload | C | Tok/s median (min–max) | TTFT ms | P50/P95 ms |
|---|---|---:|---:|---:|---:|
| llama | long | 1 | 155.58 (154.90–156.56) | 33.6 | 411.4/411.4 |
| llama | long | 8 | 448.88 (445.87–454.05) | 187.9 | 1121.0/1139.7 |
| llama | mixed | 1 | 165.64 (163.55–165.96) | 8.0 | 60.4/60.4 |
| llama | mixed | 8 | 436.86 (431.36–439.29) | 89.0 | 791.1/800.3 |

llama.cpp remains faster on long/mixed serving. No short-reference repetitions
were added. Conservative new cloud estimate is **$2.98** across three
sessions; final API state is stopped. See PHASE15_COST_LEDGER.md for margin and
invoice qualifications.

## Highest-impact remaining work and Phase 16 direction

1. Long causal prefill attention still dominates; a different query/KV execution
   structure needs evidence beyond the rejected shared-memory tiles.
2. Larger long BF16 split reductions cross near-tied greedy decisions. Precision
   contracts and numerically stable optimized execution need deliberate design.
3. Mixed requests still serialize expensive prefill with decode; simple token
   budgets do not remove that interference or the remaining reference gap.

For Phase 16, recommend a model-neutral decoder execution plan with explicit
attention capabilities, precision contracts, persistent worker/scratch ownership,
and graph dispatch identities. Add model families through that interface and
prepare release validation/documentation. This is a recommendation only.

Full machine-readable medians, trial ranges, request throughput, sampled memory,
batch occupancy and host stage means are in benchmarks/results/phase15/summary.json.
Output audits and final-source hashes accompany the raw evidence.
