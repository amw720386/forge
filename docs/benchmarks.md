# Benchmarks

## Run locally

The kernel benchmark is synthetic and reports matrix throughput, not model decode
speed. From the root, use `make bench` on Linux or:

```powershell
.\scripts\bench.ps1
```

The existing CPU harness measures one model/backend per process:

```sh
cargo run --release --manifest-path runtime/Cargo.toml --bin forge-bench -- --model models/Qwen2.5-0.5B-Instruct --backend fp32 --prefill 64 --decode 16
```

Use `--backend int8`, `int4`, or `forge` (for a `.forge` checkpoint). This is an
internal forward/decode timing, excludes load, and does not stop at EOS. Its
Qwen prompt can tokenize to fewer than `--prefill` tokens; use the reported count.
For repeated CPU quantization comparisons, retain `scripts/bench_int4_compare.ps1`.

For end-to-end measurements against an already running local server:

```sh
mkdir -p benchmarks/results/local
python benchmarks/forge/bench_http.py --runtime forge --url http://127.0.0.1:8080 --concurrency 1,8 --workloads short,medium,long,mixed --max-new 64 --trials 3 --out benchmarks/results/local/http.json
```

Create `benchmarks/results/local` with `New-Item -ItemType Directory -Force` on
PowerShell. The harness uses only the Python standard library; optional VRAM
sampling invokes `nvidia-smi`. Timing runs from client POST through final SSE
completion, includes queueing/prefill/decode/transport, excludes load, and stops
at natural EOS. It performs one full-concurrency warmup per cell and reports
median aggregate output tokens/s across trials. Record hardware, model revision,
dtype, context/slot limits and environment settings alongside new results.

## Retained evidence

The README uses the first after-run cells in
[`phase16/summary.json`](../benchmarks/results/phase16/summary.json), derived from
`phase16-after-short-c1-http.json` and `phase16-after-c8-http.json`. The raw files
include every trial, token count and completion. `python benchmarks/results/phase16/summarize.py` regenerates the summary and audits
candidate/repeat text against the baseline (180/180 exact matches historically).

These are NVIDIA L4 Qwen2.5-0.5B-Instruct CUDA BF16 HTTP results with context 2048,
eight slots, max output 64 and natural EOS. The first short C=8 result is 937.13
tok/s (range 888.72-1013.23); the nine-trial repeat is 1062.39 (924.04-1156.30).
The ranges overlap. The repeat does not establish a speedup. No standardized
SmolLM2 GPU throughput result is claimed; its BF16 support is experimental.

Raw results, failed/rejected experiments, server traces, source hashes and audits
remain under [`benchmarks/results/`](../benchmarks/results/). Earlier phase evidence
is preserved in [`docs/history/`](history/), including
[the methodology](history/BENCHMARK_METHODOLOGY.md),
[context qualifications](history/PHASE15_RESULTS.md) and
[model qualifications](history/PHASE16_RESULTS.md). Historical reports describe
older source states and commands; use build.md for the current checkout.

Original Nsight `.nsys-rep` captures remain local in `benchmarks/results/phase13`
and are ignored by Git; their exported statistics CSVs are versioned. They are
profiling supplements, not the source of the README HTTP throughput numbers.
Historical result names and source hashes retain their original provenance.
Active harnesses use feature names: `bench_cpu_quantization`,
`bench_cuda_inference`, `context_workloads.py`, `summarize_http.py`, and
`audit_outputs.py`. The last two require `--root` to select saved results.
CUDA kernel microbenchmarks remain in `kernels/cuda/bench/`.
