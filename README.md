# Forge

Forge is an LLM inference runtime written in Go, Rust, C++ and CUDA.

It loads local Hugging Face Safetensors or native `.forge` checkpoints and runs
its own transformer forward pass on CPU or NVIDIA CUDA. A Go HTTP server exposes
text completions with streaming and continuous batching. `forge-cli` provides
interactive local chat through the same runtime.

## Features

- CPU FP32 and groupwise INT8/INT4 weight quantization.
- CUDA FP32/BF16 inference, paged KV caching, chunked prefill and CUDA Graphs.
- Shared batched decoding, bounded admission and request cancellation.
- Checkpoint validation and offline quantization with `forge-quantize`.
- `/v1/completions`, SSE streaming, `/health` and JSON `/metrics`.

## Build

Requires Rust stable, Go 1.22+, CMake and a C++17 compiler. Default x86-64 CPU
builds require AVX2/FMA. Windows uses MSVC for Rust and LLVM-MinGW for CMake.

```powershell
# Windows, from the repository root
.\scripts\build.ps1
.\scripts\test.ps1
```

```sh
# Linux CPU
make build
make test
```

For optional CUDA builds, toolchain details and individual test commands, see
[build.md](docs/build.md). CUDA requires the toolkit and a separate kernel build;
`cuda-stub` only checks compilation and linkage without a GPU.

## Usage

Supply a local model directory containing `config.json`, `model.safetensors` and
`tokenizer.json`. Model weights are not included.

```sh
# Interactive chat; Windows: .\bin\forge-cli.exe with the same flags
./bin/forge-cli -m models/Qwen2.5-0.5B-Instruct

# Start the server on Linux; Windows: .\bin\forge-server.exe with the same flags
./bin/forge-server -model models/Qwen2.5-0.5B-Instruct -addr 127.0.0.1:8080
```

```sh
curl http://127.0.0.1:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"The capital of France is","max_tokens":16,"temperature":0,"stream":false}'
```

CLI options: `-m/--model`, `-c/--ctx-size` (1024), `-n/--predict` (128),
`--device` (cpu or cuda), and `--dtype` (fp32; CUDA also supports bf16).
Generation is greedy. `/clear` resets history; `/help` lists commands; `/exit`
exits. Ctrl+C stops generation, or exits while waiting for input. Cancellation
takes effect after the current runtime operation completes, including prefill.
The CLI applies `tokenizer_config.json`'s chat template to the entire conversation
and preserves previous turns. A full context asks you to clear history; output
is capped by the remaining context and stops at EOS. `FORGE_MAX_SEQ` also caps
the requested context. Models must include `tokenizer_config.json` for chat.

The server retains its text completions API: plain prompts receive the existing
Qwen-style wrapper; already formatted prompts pass through. For checkpoints,
use `-m models/qwen2.5-int8.forge` in the CLI or `-model` in the server and keep
the matching tokenizer directory available (see [build.md](docs/build.md)).

## Supported models

| Model | CPU | CUDA |
|---|---|---|
| Qwen2.5-0.5B-Instruct | FP32, INT8, INT4 | FP32, BF16 |
| SmolLM2-135M-Instruct | FP32, INT8, INT4 | FP32; BF16 experimental |

CUDA correctness was previously tested on NVIDIA L4. Quantized CPU output is
approximate; CUDA INT8/INT4 is unsupported. SmolLM2 BF16 logits differ from FP32
even where tested greedy outputs agree. Other Qwen/Llama-family checkpoints and
hardware configurations are unvalidated.

## Performance

Historical NVIDIA L4 measurements of Qwen2.5-0.5B-Instruct, CUDA BF16, eight
available slots, context limit 2048, at most 64 output tokens and natural EOS.
These are median aggregate output tokens/s over three warmed loopback HTTP/SSE
trials, including request processing and excluding model load.

| Hardware | Model | Dtype | Workload | Concurrency | HTTP output tok/s |
|---|---|---|---|---:|---:|
| NVIDIA L4 | Qwen2.5-0.5B | BF16 | Short | 1 | 183.54 |
| NVIDIA L4 | Qwen2.5-0.5B | BF16 | Short | 8 | 937.13 |
| NVIDIA L4 | Qwen2.5-0.5B | BF16 | Medium | 8 | 1035.48 |
| NVIDIA L4 | Qwen2.5-0.5B | BF16 | Long | 8 | 223.39 |

Short C=8 varied between runs; a nine-trial repeat measured 1062.39 tok/s.
These are retained results, not a new GPU benchmark or a claim of superiority
over other engines. [Methodology and raw evidence](docs/benchmarks.md).

## Architecture

```text
forge-cli --------------------------------> Rust model runtime -> C++ / CUDA kernels
forge-server / scheduler -> Go bindings -> Rust C ABI ---------^
```

`api/` handles admission, scheduling and streaming. `bindings/go/` bridges the
Rust C ABI using cgo on Unix and DLL loading on Windows. `runtime/` owns model
loading, transformer execution, checkpoints and KV state. `kernels/` contains
CPU kernels and the optional `cuda/` backend. Separate Go modules keep this
platform-specific binding independent of the server. `tools/cli/` is the thin
Rust chat interface; the server stays in `api/cmd/forge-server/` for simple Go builds.

## Limitations

Only the two model targets above are qualified. Sharded Safetensors, scaled RoPE,
sliding-window attention are unsupported. CLI chat is qualified for the two
models above; arbitrary tokenizer templates are unvalidated. Full
model context limits have not been audited. The HTTP API implements text
completions, not the full OpenAI API. There is no authentication or TLS; bind to
loopback for local use.
