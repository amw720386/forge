# Phase 16 results

The model-neutral architecture and second-model CPU/CUDA FP32 objectives are
implemented and validated. SmolLM2 BF16 has experimental numerical qualifications;
it is not claimed to have tight FP32 logit parity or universal greedy agreement.
No Phase 17 implementation was started.

## Architecture

Qwen and Llama family adapters use one validated `ModelConfig`, `DecoderPlan`, CPU
transformer and CUDA transformer. Shared kernels and scheduler policy are unchanged.
Family-specific biases, tensor mapping, token IDs, tying, context and tokenizer
identity are explicit. Loader dimensions/shapes are validated before allocation.
Precision contracts preserve FP32 KV and sensitive operations. Graph identities
include architecture/configuration, model allocation generation, precision, dispatch,
batch/path, block geometry and captured workspace addresses. Actual L4 scratch reuse
across destroyed Qwen weights and new Llama weights recaptures safely and matches CPU.

## CPU reference and checkpoint results

Four prompts per target, including 160-token context, were exported from HF in
separate reference processes. Forge's raw argmax controls deliberately bypass HF
generation defaults, including Qwen's repetition penalty.

| Target | Layer comparisons | Max full-logit error on Windows | Greedy tokens | Cached/full | CPU shared batch |
|---|---:|---:|---:|---|---|
| Qwen2.5-0.5B | 96 | 4.792e-5 | 42/42 | Last logits identical in these cases | All four controls match |
| SmolLM2-135M | 120 | 4.196e-5 | 44/44 | Last logits identical in these cases | All four controls match |

SmolLM2 FP32 resident weight accounting is 513.13 MiB; INT8 244.40 MiB; INT4
180.27 MiB. Actual FP32, INT8 and INT4 `.forge` round trips preserve logits exactly
after dropping the original model before reload. Generated examples include
"The capital of France is Paris". Quantized generations can differ from FP32 and
are not represented as HF-exact quality results. The pre-existing Qwen INT8 checkpoint
also loads, generates the expected Paris answer and round-trips exactly with new code.
FRG1 version 1 and Qwen architecture ID 1 are preserved.

Informational Windows SmolLM2 decode measurements were approximately 13–32 tok/s
in the first recorded four-prompt run, with later runs varying. These are not a
standardized CPU performance claim. On the Studio's CPU the four-case decode control
was roughly 37–41 tok/s. No separate second-model GPU throughput benchmark was run.

Local Rust release suites, CUDA stub compilation/tests, C++ kernels, Go API/cloud
tests and Go FFI tests pass. Stub skip/zero-test results are not GPU evidence.
New tests cover invalid configuration, mixed Safetensors dtypes, missing/invalid
tensors, tokenizer mismatch, quantized Llama checkpoints and graph identities.

## Actual L4 validation

- Full existing Rust CUDA regression suite, C++ CUDA/CPU kernels and pretrained
  Qwen Phase 15 long greedy tests pass. Qwen long same-dtype dispatch comparisons
  have 120 matching greedy decisions per dtype in the focused run.
- SmolLM2 CUDA FP32 passes all four HF controls, strict full logits (maximum
  7.248e-5), KV decoding and 44 greedy decisions. BF16 also matches all 44 decisions.
- Both dtypes pass mixed-length paged/chunked batches with graphs disabled/enabled,
  cancellation and slot reuse. Graph captures at B=3 and B=1 are recorded.
- A separate model replacement test destroys the original GPU model and reuses
  graph scratch with a new Llama architecture/weight generation. Both families
  match CPU after eager warmup, capture and replay.
- SmolLM2 serves a real CUDA BF16 HTTP request through Go and returns the Paris
  answer. The final library includes tokenizer identity and startup context validation.

BF16 maximum HF FP32 difference is 0.205344; RMS maximum 0.088606. An explicit
rounding oracle verifies the isolated BF16 output projection to 4.91e-5 max error.
The incomplete earlier-layer attribution and failed initial proposed RMS bound are
disclosed in [PRECISION_CONTRACTS.md](PRECISION_CONTRACTS.md), with failure evidence
retained. Existing Phase 15 numerical tolerances and fallback boundaries are unchanged.

## Qwen performance protection

Same L4 session, same Phase 15 BF16 HTTP harness, prompts, warmup, EOS, max output
64, eight slots, context 2048 and three trials. The first baseline uses the preserved
Phase 15 server/library before deploying Phase 16. All first after-run numbers are
retained; short was repeated with the existing nine-trial method to investigate variability.

| Workload | C | Before tok/s | First after tok/s | Change | Nine-trial after repeat |
|---|---:|---:|---:|---:|---:|
| Short | 1 | 181.96 | 183.54 | +0.87% | 183.85 |
| Short | 8 | 992.69 | 937.13 | -5.60% | 1062.39 |
| Medium | 8 | 1032.23 | 1035.48 | +0.31% | — |
| Long | 8 | 223.31 | 223.39 | +0.03% | — |
| Mixed | 8 | 202.79 | 203.96 | +0.58% | — |

Short C=8 ranges overlap: before 920–1044, first after 889–1013, repeat 924–1156
tok/s. The repeated result does not establish an architectural speedup; it shows
the first short dip is not persistent. No benchmark method was changed to hide it.
Long C=8 TTFT is 771.6 ms; mixed C=8 P95 is 1715.5 ms; sampled peak VRAM is
1658 MiB. All **180/180** candidate/repeat HTTP completions match baseline text.
Phase 15's accepted 1010/1031/221/202 matrix, 266 audited completions and 1380
greedy decisions per dtype remain preserved as historical references.

## Cloud and reproducibility

One NVIDIA L4 session, existing `forge-gpu`, on-demand catalog rate $1.68/hour.
Conservative reservation-through-independent-shutdown duration: 24.75 minutes.
Estimated spend rounded up **$0.70**; actual invoice unavailable; remaining estimated
Phase 16 budget **$1.30**. Local and remote absolute-deadline watchdogs were armed;
explicit stop succeeded and independent API inspection returned `requested=null`,
`inUse=null` for every visible Studio. The local watchdog was cancelled only afterward.
See [PHASE16_COST_LEDGER.md](PHASE16_COST_LEDGER.md).

Deployment manifest was verified before the initial build. Generated fixtures were
missing from the source bundle, then regenerated on the Studio without weakening
comparisons. The Studio's incompatible optional torchvision import was disabled
only in the text-only HF reference process. Incremental source repairs were hashed;
the final independent audit verifies all 98 runtime/kernel/server source entries.
The final numerical attribution utility was compiled after its update. The retained
local source audit separately records later comment/help cleanup, formatting and
CPU-only error-path validation; these do not alter CUDA kernels or valid-input
numerics. Raw failures,
stage logs, full benchmark cells and audit summaries are retained.

Reproduce local references and checks:

```powershell
python tests/reference/phase16_reference.py models/SmolLM2-135M-Instruct tests/fixtures/phase16/smollm2
cargo run --release --manifest-path runtime/Cargo.toml --example phase16_parity -- models/SmolLM2-135M-Instruct tests/fixtures/phase16/smollm2
cargo test --release --manifest-path runtime/Cargo.toml
cargo test --release --features cuda-stub --manifest-path runtime/Cargo.toml
python benchmarks/analysis/summarize_phase16.py
```

## Remaining release-readiness priorities

Keep BF16 experimental for the second model until broader prompts/contexts and
earlier-layer precision attribution are audited. Other family-compatible checkpoints,
full 8192/32768 contexts and other GPUs are unvalidated. Sharded HF loading, scaled
RoPE, alternate attention/activations, CUDA quantization and automatic chat templates
are unsupported. Legacy checkpoints without tokenizer hashes cannot prove tokenizer
identity. For Phase 17, prioritize packaging/install reproducibility, model provenance,
clear CLI diagnostics, a small CI reference matrix and qualification of supported
precision combinations. Do not expand this into large-model or distributed optimization.
