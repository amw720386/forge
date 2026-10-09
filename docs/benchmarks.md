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
mkdir -p benchmark-output
python benchmarks/forge/bench_http.py --runtime forge --url http://127.0.0.1:8080 --concurrency 1,4 --workloads short,medium --max-new 16 --trials 3 --out benchmark-output/local-http.json
```

Create `benchmark-output` with `New-Item -ItemType Directory -Force` on
PowerShell. The harness uses only the Python standard library; optional VRAM
sampling invokes `nvidia-smi`. Timing runs from client POST through final SSE
completion, includes queueing/prefill/decode/transport, excludes load, and stops
at natural EOS. It performs one full-concurrency warmup per cell and reports
median aggregate output tokens/s across trials. Record hardware, model revision,
dtype, context/slot limits and environment settings alongside new results.

## Output and analysis

Generated output belongs in the ignored root `benchmark-output/` directory,
separate from benchmark source. `FORGE_BENCH_OUT` overrides this location for
the shell harnesses. Historical benchmark runs and numbered reports are removed.

Use `python benchmarks/analysis/summarize_http.py --root benchmark-output` to
summarize HTTP files named `*-http.json`. Compare generated text between two runs
with `audit_outputs.py --root benchmark-output --baseline before-http.json --candidate after-http.json`.
CUDA model benchmarking uses
`bench_cuda_inference`; CUDA kernel microbenchmarks are in `kernels/cuda/bench/`
and require NVIDIA hardware.
