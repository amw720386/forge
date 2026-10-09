# Forge

Event-driven LLM inference runtime: **Go** (HTTP + scheduling) â†’ **Rust** (model/runtime) â†’ **C++** (kernels).

Not a wrapper around llama.cpp / candle / ONNX Runtime. The forward pass is our own.

## Status

**Phases 1-16 implemented, with benchmark and numerical qualifications documented.** Phase 16 adds a validated model-neutral decoder plan, Qwen/Llama adapters and SmolLM2-135M-Instruct CPU and real L4 inference. Qwen regressions pass; 180 audited HTTP completions match controls. SmolLM2 BF16 remains experimentally qualified because logits differ from FP32. Phase 15's conservative attention dispatch, pinned worker, graph invalidation and fair fixed-budget scheduling remain intact.

See [Phase 16 results](docs/PHASE16_RESULTS.md), [design](docs/PHASE16_DESIGN.md), [supported architectures](docs/MODEL_ARCHITECTURES.md), [precision contracts](docs/PRECISION_CONTRACTS.md) and [cost ledger](docs/PHASE16_COST_LEDGER.md). New Phase 16 compute cost was estimated at $0.70; the Studio is verified stopped. Previous Phase 15 evidence remains preserved.

See [Phase 15 results](docs/PHASE15_RESULTS.md), [design](docs/PHASE15_DESIGN.md), [profile](docs/PHASE15_PROFILE.md), [cost ledger](docs/PHASE15_COST_LEDGER.md), [roadmap](docs/ROADMAP.md) and [benchmark methodology](docs/BENCHMARK_METHODOLOGY.md). [Phase 14 results](docs/PHASE14_RESULTS.md) remain preserved.

## Layout

```
api/         Go HTTP server, scheduler, forge-cloud (Lightning)
runtime/     Rust inference core + compute backend + C ABI
kernels/     C++ CPU kernels + optional kernels/cuda/
bindings/go  Go cgo bindings to Rust
docs/        Architecture & roadmap
models/      Local weights (gitignored)
```

## Prerequisites

- Go 1.22+
- Rust stable (Cargo)
- CMake 3.16+
- C++17 compiler (clang++/g++/MSVC)
- On Windows: LLVM-MinGW (or MinGW-w64) for cgo

## Quick start

```bash
make build
make test
make run
```

### Windows (PowerShell)

```powershell
.\scripts\test.ps1
.\scripts\bench.ps1

# Inference server (download model first â€” see models/README.md)
cd runtime; cargo build --release; cd ..
$env:FORGE_RUNTIME_DIR = "$PWD\runtime\target\release"
go run ./api/cmd/forge-server -model models/Qwen2.5-0.5B-Instruct -addr :8080
```

```powershell
# Non-streaming completion
Invoke-RestMethod -Method Post http://127.0.0.1:8080/v1/completions `
  -ContentType application/json `
  -Body '{"prompt":"Explain what a KV cache is","max_tokens":64,"temperature":0,"stream":false}'
```

## Execution model

Concurrent HTTP clients share a **continuous batcher**: independent KV per request, shared batched decode steps (`BatchStep`). Queue wait / full batch is intentional backpressure â€” not thread-per-request fake parallelism.

### Quantized inference (CPU)

```powershell
# Offline INT8 checkpoint (recommended â€” no re-quant on load)
cargo run --release --bin forge-quantize -- `
  --model ../models/Qwen2.5-0.5B-Instruct --bits 8 --group-size 64 `
  --output ../models/qwen2.5-int8.forge

cargo run --release --bin forge-bench -- --model ../models/qwen2.5-int8.forge --backend forge

# Server can take a .forge file
go run ./api/cmd/forge-server -model models/qwen2.5-int8.forge -max-seq 1024
```

Defaults: `FORGE_MAX_SEQ=1024`, `FORGE_MAX_BATCH=4`, `FORGE_MEM_BUDGET_MIB=8192` â€” see [MEMORY_MANAGEMENT.md](docs/MEMORY_MANAGEMENT.md).

### CUDA / Lightning (Phase 8Aâ€“12)

- Local Windows: CPU-only by default (no CUDA toolkit required).
- GPU server (Linux + CUDA runtime build): `forge-server --device cuda [--dtype fp32|bf16] --model â€¦`
- Copy `.env.example` â†’ `.env` with Lightning keys (gitignored). See [LIGHTNING.md](docs/LIGHTNING.md).
- Phase 12 benchmarking â€” [PHASE12_RESULTS.md](docs/PHASE12_RESULTS.md), [BENCHMARK_METHODOLOGY.md](docs/BENCHMARK_METHODOLOGY.md).

```powershell
go run ./api/cmd/forge-cloud status
go run ./api/cmd/forge-cloud auth
go run ./api/cmd/forge-cloud prepare-deploy
# Phase 12 billable ($6 total / $2 session; always stops):
# go run ./api/cmd/forge-cloud validate-phase12 --gpu l4 --duration 40m --mode forge --confirm
# go run ./api/cmd/forge-cloud validate-phase12 --gpu l4 --duration 30m --mode compare --confirm
```

Local: `make cuda-stub-check` / `cargo test --features cuda-stub --test cuda_paged_kv`.

## License

MIT (unless noted otherwise in subcomponents).
