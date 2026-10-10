# Forge

A lightweight LLM inference runtime built with Rust, Go, C++ and CUDA.

Forge runs local models on CPU or NVIDIA GPUs, with an interactive CLI and an OpenAI-compatible chat endpoint.

## Features

- CPU inference with FP32, INT8 and INT4 weights
- CUDA inference with FP32 and BF16 weights
- Continuous batching, paged KV cache and CUDA Graphs
- Interactive terminal chat (`forge-cli`) and HTTP serving (`forge-server`)
- Safetensors and `.forge` checkpoints

## Quickstart

**Windows:** Install Rust (MSVC), Go 1.22+, CMake, Visual Studio C++ Build Tools and LLVM-MinGW. See [build instructions](docs/build.md) for details.

```powershell
git clone https://github.com/amw720386/forge.git
cd forge
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build.ps1 -Jobs 1
```

Download `config.json`, `model.safetensors`, `tokenizer.json` and `tokenizer_config.json` from [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) into `models/Qwen2.5-0.5B-Instruct/`.

**Chat in your terminal:**

```powershell
.\bin\forge-cli.exe -m models/Qwen2.5-0.5B-Instruct --device cpu
```

**Start the local server:**

```powershell
.\bin\forge-server.exe -model models/Qwen2.5-0.5B-Instruct -device cpu
```

**Send a chat request** (in another PowerShell window):

```powershell
$body = @{ model = 'Qwen2.5-0.5B-Instruct'; messages = @(@{role='user';content='Hello!'}); max_tokens = 64 } | ConvertTo-Json -Depth 5
Invoke-RestMethod http://127.0.0.1:8080/v1/chat/completions -Method Post -ContentType 'application/json' -Body $body
```

## Supported models

| Model | CPU | CUDA |
|---|---|---|
| Qwen2.5-0.5B-Instruct | FP32, INT8, INT4 | FP32, BF16 |
| SmolLM2-135M-Instruct | FP32, INT8, INT4 | FP32, BF16 (experimental) |

Forge currently supports text-only greedy generation. The HTTP server implements a subset of OpenAI Chat Completions, including streaming. Model weights are downloaded separately; GGUF is not supported.

## Benchmarks

For measured performance and reproducible commands, see [benchmarks](docs/benchmarks.md).

## Build and tests

See [docs/build.md](docs/build.md) for Linux, CUDA, tests and troubleshooting.
