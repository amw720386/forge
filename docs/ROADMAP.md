# Forge Roadmap

## Phase 16 status

Model-neutral configuration, family adapters, static decoder plans, precision
contracts, safe graph identities and SmolLM2-135M-Instruct are implemented. CPU HF
layer/logit/cache/greedy references and actual L4 FP32, paged batching, graphs,
cancellation and slot reuse pass. SmolLM2 BF16 is experimentally qualified, with
measured differences and partial precision attribution. Existing Qwen correctness,
checkpoints and representative performance are preserved; the short C=8 dip was
investigated with the existing repeat harness. See [PHASE16_RESULTS.md](PHASE16_RESULTS.md).

Phase 17 is not started. Recommended release-readiness work: reproducible packaging,
model/tokenizer provenance, precise supported-combination reporting, a small CI
reference matrix and broader BF16 qualification. No large-model, multi-GPU,
distributed, speculative or multimodal implementation belongs to Phase 16.

Event-driven LLM inference runtime: **Go** (API/scheduling) → **Rust** (runtime) → **C++** (kernels).

Target model: **Qwen2.5-0.5B-Instruct** (`Qwen2ForCausalLM`).

| Field | Value |
|---|---|
| hidden_size | 896 |
| num_hidden_layers | 24 |
| num_attention_heads | 14 |
| num_key_value_heads | 2 (GQA) |
| head_dim | 64 |
| intermediate_size | 4864 |
| vocab_size | 151936 |
| rope_theta | 1_000_000 |
| rms_norm_eps | 1e-6 |
| tied embeddings | true |
| dtype (upstream) | bfloat16 (we run FP32 initially) |

## Execution strategy (important)

Concurrent HTTP requests ≠ concurrent inference.

Phase 1–4: the scheduler admits many requests into a bounded queue, but the Rust runtime executes **one generation job at a time**. Clients still see async SSE; queue wait is explicit.

Phase 5: continuous batching groups compatible decode steps into shared kernel work (`m=B` linears) — not fake concurrency via threads per request.

---

## Phase 1 — Compute foundation ✅ complete

- [x] Contiguous FP32 tensors with shape/stride validation (Rust)
- [x] FP32 matmul + matvec (C++), C ABI
- [x] Optional AVX2/FMA path when available
- [x] C++ unit tests vs reference GEMM
- [x] Rust FFI bindings + tensor ops
- [x] Go → Rust C ABI smoke test
- [x] Cross-language build (`make build` / `make test` / `scripts/*.ps1`)
- [x] Micro-benchmarks (synthetic) — see [PHASE1_RESULTS.md](PHASE1_RESULTS.md)

**Exit criteria met:** builds/tests pass; matmul matches reference within 1e-4 relative error.

---

## Phase 2 — Transformer fundamentals ✅ complete

- [x] RMSNorm, Softmax, RoPE, linear, SwiGLU
- [x] Causal self-attention + GQA
- [x] Decoder block + tiny synthetic transformer
- [x] PyTorch fixtures + KV-cache vs full-prefill parity — see [PHASE2_RESULTS.md](PHASE2_RESULTS.md)
- [x] Layout docs — [LAYOUTS.md](LAYOUTS.md)

---

## Phase 3 — Real model inference ✅ complete

- [x] Safetensors loader + config parsing (BF16→FP32)
- [x] Qwen2.5-0.5B forward + KV cache
- [x] Tokenizer (`tokenizer.json`)
- [x] Autoregressive generation (greedy, temperature, top-k)
- [x] Exact greedy id/text parity vs HF Transformers FP32 — [PHASE3_RESULTS.md](PHASE3_RESULTS.md)

---

## Phase 4 — Inference server ✅ (implemented)

- [x] Go admission queue, single-flight scheduler, SSE `/v1/completions`
- [x] `/health`, `/metrics`, cancellation on disconnect, graceful shutdown
- [x] Lifecycle: Queued → Running → Streaming → Completed|Cancelled|Failed
- [x] C ABI streaming generate (`forge_generate_begin/next/cancel`)

---

## Phase 5 — Optimization ✅ (implemented)

- [x] Profile prefill vs decode, per-op microbenches, Qwen shapes — [PHASE5_BASELINE.md](PHASE5_BASELINE.md)
- [x] Evidence-driven opts: persistent thread pool, MT linear (MLP/lm_head), zero-copy paths — [PHASE5_RESULTS.md](PHASE5_RESULTS.md)
- [x] Genuine continuous batching (`ContinuousBatcher` + Go scheduler `BatchStep`)
- [x] Admit / cancel / backpressure; concurrent greedy parity + cancel isolation tests

---

## Phase 6 — Weight-only quant + measurable CPU runtime ✅ (implemented)

- [x] Reproducible benches; document 0.44 vs 3.43 tok/s discrepancy — [PHASE6_RESULTS.md](PHASE6_RESULTS.md)
- [x] INT8 + INT4 weight-only quant, fused dequant GEMV — [QUANTIZATION.md](QUANTIZATION.md)
- [x] Same Rust `Linear` interface for FP32 / INT8 / INT4; norms stay FP32
- [x] Logit MAD, next-token agreement, multi-prompt, batched quant parity tests
- [x] Memory / TTFT / decode / batch B=1..8 / p50–p95 measurements
- [x] Shared prefill scratch (no per-request `max_seq` activation buffers)
- [x] Persist offline quantized checkpoints (`.forge` v1) — [CHECKPOINT_FORMAT.md](CHECKPOINT_FORMAT.md)
- [x] Memory budgets / RSS reporting / safe benches — [MEMORY_MANAGEMENT.md](MEMORY_MANAGEMENT.md)
- [x] Lightning AI auth/status prep (no paid GPUs) — [LIGHTNING.md](LIGHTNING.md)
- [ ] Faster INT4 when it beats FP32 on target CPUs
- [ ] Zero-copy mmap kernel views; Lightning provision + CUDA (Phase 8+)

---

## Phase 7 — Checkpoints & memory safety ✅ (implemented)

- [x] Native `.forge` format + `forge-quantize` + direct load (no re-quant)
- [x] Round-trip / corruption / inference parity tests — [PHASE7_RESULTS.md](PHASE7_RESULTS.md)
- [x] Conservative memory defaults and bench safeguards
- [x] Server loads HF dirs or `.forge` files
- [x] `forge-cloud status` (credentials redacted; no provisioning)

---

## Phase 8A — CUDA foundation & Lightning orchestration ✅ (code; GPU pending)

- [x] Compute backend abstraction (CPU + optional CUDA) — [CUDA_ARCHITECTURE.md](CUDA_ARCHITECTURE.md)
- [x] CUDA memory + FP32 primitives (cuBLAS GEMM + custom kernels)
- [x] Optional CMake/Rust feature flags (Windows stays CPU-only by default)
- [x] CUDA vs CPU tests (execute on Lightning GPU; skip locally without device)
- [x] `forge-cloud` start/exec/stop/benchmark with cost controls — [LIGHTNING.md](LIGHTNING.md)
- [x] Remote Git + CMake workflow scripts
- [x] **GPU validation session** — NVIDIA L4; `forge_cuda_tests` + microbench; stop verified — [PHASE8A_RESULTS.md](PHASE8A_RESULTS.md)
- [x] Phase 8B: Qwen GPU generate + CPU greedy token parity + prefill/decode benches on L4 — [PHASE8B_RESULTS.md](PHASE8B_RESULTS.md)
- [x] Phase 9: GPU continuous batching validated on L4 — [PHASE9_DESIGN.md](PHASE9_DESIGN.md), [PHASE9_RESULTS.md](PHASE9_RESULTS.md)
- [x] Phase 10: paged KV + chunked prefill validated on L4 — [PHASE10_DESIGN.md](PHASE10_DESIGN.md), [PHASE10_RESULTS.md](PHASE10_RESULTS.md)
- [x] Phase 11: optimized paged attn, BF16 weights, GPU greedy argmax — [PHASE11_DESIGN.md](PHASE11_DESIGN.md), [PHASE11_RESULTS.md](PHASE11_RESULTS.md)
- [x] Phase 12: competitive benchmarking vs llama.cpp, BF16 survey, audited harness — [PHASE12_DESIGN.md](PHASE12_DESIGN.md), [PHASE12_RESULTS.md](PHASE12_RESULTS.md), [BENCHMARK_METHODOLOGY.md](BENCHMARK_METHODOLOGY.md)
- [ ] Phase 13+: evidence-backed opts (fused attn, BF16 cast removal, CUDA Graphs); INT8/INT4 GPU weights

Results: [PHASE8A_RESULTS.md](PHASE8A_RESULTS.md)

---

## Phase 9 — GPU continuous batching ✅ complete

- [x] Multi-slot `CudaBatcher` with per-request GPU KV
- [x] Shared batched CUDA linears (`m=B`) for QKV/O/MLP/lm_head
- [x] Per-request RoPE + GQA against private caches
- [x] Iteration-level admit / cancel / complete / slot reuse
- [x] Same FFI `forge_batch_*` + Go scheduler (no CUDA→CPU routing)
- [x] Tests + `bench_cuda_batch` + `validate-phase9` workflow
- [x] NVIDIA L4 validation: `cuda_batch_parity` 7/7; B=1..8 benches; `m=B` trace; stop verified — [PHASE9_RESULTS.md](PHASE9_RESULTS.md)

---

## Phase 10 — Paged KV + chunked prefill ✅ complete

- [x] Design — [PHASE10_DESIGN.md](PHASE10_DESIGN.md)
- [x] Block allocator + free list + capacity-aware admit
- [x] Paged GQA / paged KV append CUDA kernels (contiguous oracle preserved)
- [x] Chunked prefill + per-step token budget scheduling
- [x] Metrics / FFI `forge_batch_sched_stats`
- [x] Local CPU paged-vs-contiguous attention + allocator tests
- [x] Lightning L4: parity 7/7 (+ chunked re-run); B=1..8 benches; stop verified — [PHASE10_RESULTS.md](PHASE10_RESULTS.md)

---

## Phase 11 — GPU performance engineering ✅ complete

- [x] Design — [PHASE11_DESIGN.md](PHASE11_DESIGN.md)
- [x] Optimized paged GQA (warp + online softmax); naive retained as reference
- [x] cuBLAS `linear_f32`; BF16 weight upload + `GemmEx`
- [x] GPU greedy argmax (full-vocab D2H off hot path)
- [x] Local cuda-stub / CLI wiring (`--dtype bf16`, `validate-phase11`)
- [x] Lightning L4: parity, B=1..8 FP32/BF16 benches, token match, stop verified — [PHASE11_RESULTS.md](PHASE11_RESULTS.md)

---

## Phase 12 — Competitive benchmarking ✅ complete

- [x] Audit `bench_cuda_batch`; ship `bench_phase12` + methodology docs
- [x] Remasure Forge FP32/BF16 on L4 (short/medium/long/mixed)
- [x] BF16 vs CUDA FP32 correctness survey (6/7 sequences)
- [x] llama.cpp F16 `llama-bench` reference (commit `71ad059`)
- [x] vLLM attempted; Studio venv policy blocker documented
- [x] Controlled experiments + performance analysis for Phase 13
- [x] Cost ledger within $6; shutdowns verified — [PHASE12_COST_LEDGER.md](PHASE12_COST_LEDGER.md)

## Constraints

- No PyTorch/ONNX/candle/llama.cpp for production forward passes
- No committed model weights or binaries
- CPU default; CUDA is an optional backend (toolkit not required on Windows)
- Stable C ABI at language boundaries; Rust owns model/runtime memory


## Phase 13 — GPU execution optimization

- [x] Fresh baseline and imported/inspected Nsight reports
- [x] Shared batched paged attention, BF16 reuse, graph capture/replay
- [x] L4 B=1/2/4/8 parity, block crossings, replacement and B=9 fallback
- [x] HTTP matrix and sampled short/medium VRAM; CPU regression checks
- [x] Bounded cost ledger and verified GPU shutdown
- [ ] Exact-revision GGUF comparison, isolated scheduler/queue measurements and controlled staggered throughput

Implementation is GPU validated; benchmark validation remains qualified. See [PHASE13_RESULTS.md](PHASE13_RESULTS.md). Phase 14 addresses the audit gaps below; historical Phase 13 results remain qualified.

## Phase 14 - Prefill, GEMM and serving optimization

- [x] Real HTTP stage instrumentation, queue/scheduler and stream-region profiling
- [x] Shared packed ragged chunked prefill; partial prompts skip LM head
- [x] Four-warp prefill attention and measured selective L4 GEMM policy
- [x] Mixed prefill/decode, cancellation/replacement, KV and graph regression tests
- [x] Synthetic unchanged-tolerance parity; full-model greedy checks with BF16 differences reported
- [x] Fresh B=1/2/4/8 HTTP matrix, ablations, VRAM and staggered arrival run
- [x] Exact-source GGUF weights/tokenizer/input audit and full-offload comparison
- [x] CPU regression checks; one bounded session; verified shutdown

See [results](PHASE14_RESULTS.md), [design](PHASE14_DESIGN.md),
[profile](PHASE14_PROFILE.md) and [ledger](PHASE14_COST_LEDGER.md).
Real-model BF16 logits are not bitwise equivalent; arbitrary driver OOM
recovery is unverified. Reference long-context serving remains faster.

## Phase 15 — long context and mixed scheduling (complete)

- [x] Fresh L4 baselines, CUDA-event profiles and audited 147–4115-token HTTP sweep
- [x] Four-warp decode with precision-aware host dispatch and original-kernel fallback
- [x] Shared-prefill tiles and adaptive cost caps evaluated; unhelpful variants removed
- [x] Fair hard-budget policy, cancellation/reuse, capacity pressure and starvation tests
- [x] Pinned CUDA worker, graph dispatch invalidation and incremental CUDA relinking
- [x] Full-model 64-step greedy checks at B=1/2/4/8; unchanged existing regressions
- [x] Matched BF16 matrix, FP32 subset, repeated short checks and limited GPU reference
- [x] Three bounded sessions; estimated $2.98 new compute; verified final shutdown

See [results](PHASE15_RESULTS.md), [design](PHASE15_DESIGN.md),
[profile](PHASE15_PROFILE.md) and [ledger](PHASE15_COST_LEDGER.md).
Medium serving and smaller long batches improve; larger long BF16 batches retain
the original kernel because an unrestricted split reduction failed greedy parity.
The C=8 long/mixed stretch targets were not met.

Phase 16 implemented the model-neutral decoder plan, explicit precision/attention
capabilities, safe graph identities and verified SmolLM2 expansion. See the Phase 16
status and qualification notes above; release preparation remains Phase 17 work.
