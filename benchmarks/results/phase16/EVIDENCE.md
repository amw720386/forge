# Phase 16 evidence

All performance cells use the unchanged Phase 15 HTTP harness. `summary.json`
includes medians, trial ranges, TTFT/P95, sampled VRAM and an exact output audit.
Do not replace the first short result with the repeat when reporting this phase.

- `phase16-before-*`: preserved Phase 15 binary before deployment, same-session L4.
- `phase16-after-*`: initial candidate matrix plus explicit nine-trial short repeat.
- `qwen-c8-output-audit.json`, `qwen-c1-output-audit.json`, `summary.json`: 180/180
  candidate/repeat completions match same-session controls.
- `regressions.log`, `qwen-real.log`, `cpu-kernels.log`, `cuda-kernels.log`: actual
  L4 existing regression and pretrained long-context results.
- `smollm-hf.log`, `smollm-parity.log`, `smollm-batch.log`: independent HF controls,
  CPU/FP32/BF16 greedy/logit tests, paged/chunked batches, graphs, cancel and reuse.
- `final-model-graphs.log`: real graph scratch reuse across Qwen model destruction
  and new Llama model allocation; mixed dtype/missing shape/tokenizer checks.
- `smollm-attribution.log`: final compiled numerical utility; strict FP32 checks,
  BF16 measured maximum/RMS and independent explicit-rounding projection oracle.
- `bf16-rms-qualification-failure.log`: retained failure of the initial new RMS
  bound 0.04. This was not an inherited regression tolerance. BF16 is experimentally
  qualified, not represented as tightly matching FP32. Earlier-layer attribution
  remains incomplete.
- `smollm-http.json`, `smollm-server.log`, `smollm-http-stage.log`: actual Go server
  loads SmolLM2 on CUDA BF16 and returns a pretrained Paris answer.
- `final-source-identity.json`, `final-source-verify.log`: 98 GPU-tested runtime,
  kernel and server source hashes independently verified on the Studio. Initial
  source manifest verification preceded generation of test/build artifacts.
- `retained-source-identity.json`: final local source identity after comment encoding
  cleanup, formatting, CLI help text and the CPU-only overflowing-position error fix/test.
  These final cleanup changes do not change CUDA kernels or valid-input execution.
- `cost-session1.json`, `preflight.txt`, `watchdog.log`, `stop.txt`,
  `final-cloud-state.txt`: $0.70 conservative estimate, one L4, independent stop.

Local CPU references use Transformers 5.19.0 / PyTorch 2.8.0+cpu. The Studio uses
Transformers 4.57.6 / PyTorch 2.11.0+cpu. References run separately from Forge;
torchvision is disabled only for the Studio's text-only reference import because
its optional installation is incompatible with that CPU PyTorch build.

Windows CPU layer/logit/cache/greedy/batch reference logs and real checkpoint round
trips are copied here. Generated binary fixtures and weights remain gitignored;
regenerate with `tests/reference/phase16_reference.py`. The full tensor manifest and
token controls are retained under `tests/fixtures/phase16/`.
