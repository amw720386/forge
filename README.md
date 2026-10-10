# Forge

Forge is an LLM inference runtime written in Go, Rust, C++ and CUDA.

It loads local Hugging Face Safetensors or native `.forge` checkpoints and runs
its own transformer forward pass on CPU or NVIDIA CUDA. A Go HTTP server exposes
text and chat completions with streaming and continuous batching. `forge-cli` provides
interactive local chat through the same runtime.

## Features

- CPU FP32 and groupwise INT8/INT4 weight quantization.
- CUDA FP32/BF16 inference, paged KV caching, chunked prefill and CUDA Graphs.
- Shared batched decoding, bounded admission and request cancellation.
- Checkpoint validation and offline quantization with `forge-quantize`.
- `/v1/chat/completions`, `/v1/models`, legacy `/v1/completions`, SSE streaming,
  `/health` and JSON `/metrics`.

## Windows quickstart

Install Git, Rust stable (`stable-x86_64-pc-windows-msvc`), Go 1.22+, CMake 3.16+,
Visual Studio C++ Build Tools (Desktop development with C++), and LLVM-MinGW.
Put `git`, `cargo`, `go`, `cmake`, `clang++` and `mingw32-make` on PATH.
Default x86-64 CPU builds require AVX2/FMA. Windows uses MSVC for Rust and
LLVM-MinGW for CMake; see [build.md](docs/build.md) for toolchain troubleshooting.

```powershell
git clone https://github.com/amw720386/forge.git
cd forge
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build.ps1 -Jobs 1
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\test.ps1
```

This is a private repository: GitHub access is required. The build needs no model
weights. `-Jobs 1` limits memory use. Obtain the original Hugging Face files from
[Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/main)
or [HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/tree/main).
For example, download the four required Qwen files in PowerShell:

```powershell
New-Item -ItemType Directory -Force models/Qwen2.5-0.5B-Instruct | Out-Null
foreach ($file in @("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")) {
  Invoke-WebRequest "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/$file" -OutFile "models/Qwen2.5-0.5B-Instruct/$file"
}
.\bin\forge-cli.exe -m models/Qwen2.5-0.5B-Instruct --device cpu
# Exit CLI with /exit, then start the server:
.\bin\forge-server.exe -model models/Qwen2.5-0.5B-Instruct -device cpu -addr 127.0.0.1:8080
```

Keep upstream licenses and attribution. Forge loads single-file Safetensors,
not GGUF; weights remain external and are never committed. For SmolLM2, use the
same filenames, its repository URL and `models/SmolLM2-135M-Instruct` directory.

In a second PowerShell, send a non-streaming chat request:

```powershell
Invoke-RestMethod http://127.0.0.1:8080/v1/models
$body = '{"model":"Qwen2.5-0.5B-Instruct","messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"Explain CUDA warps."}],"max_tokens":128,"stream":false}'
Invoke-RestMethod -Method Post http://127.0.0.1:8080/v1/chat/completions -ContentType application/json -Body $body
```

For streaming, use `curl.exe` (rather than PowerShell's `curl` alias) and a
UTF-8 request file to avoid Windows command-line JSON quoting differences:

```powershell
$body = '{"model":"Qwen2.5-0.5B-Instruct","messages":[{"role":"user","content":"Explain CUDA warps."}],"max_tokens":128,"stream":true,"stream_options":{"include_usage":true}}'
$requestFile = Join-Path $env:TEMP "forge-chat.json"
[System.IO.File]::WriteAllText($requestFile, $body, (New-Object System.Text.UTF8Encoding($false)))
curl.exe -N http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" --data-binary "@$requestFile"
```

## Chat API subset

One model is loaded per process. The API model ID defaults to the model directory
or checkpoint filename; `-model-id` overrides it. Use the ID from `/v1/models`.
Send the entire history each time: optional first `system`, then alternating
`user`/`assistant` turns ending with `user`. Content must be a string; an empty
user string is allowed, an empty messages list is rejected. Both qualified models
use their own `tokenizer_config.json` chat template, including default system text.

Generation is greedy. Supported request fields are `model`, `messages`, `stream`,
`max_tokens` (1–2048, default 64), `temperature=0`, `top_p=1`, `top_k=1`, `n=1`
and `stream_options.include_usage` for streaming. Omitted sampling fields also
select greedy generation. Unknown fields, stochastic settings, tools, functions,
multimodal content, logprobs, custom stop sequences and structured output are
rejected. This implements a subset of OpenAI Chat Completions, not full API compatibility.

Non-streaming returns `chat.completion` with one assistant message and actual
usage. Streaming returns `chat.completion.chunk` events: role, content deltas,
finish reason, optional final usage with empty choices, then `[DONE]`. EOS means
`stop`; output budget termination means `length`. Sampled EOS counts toward
completion usage even though it is not displayed. Input usage counts the entire
formatted conversation. Input plus requested output must fit the actual runtime
context (default 1024, also capped by `FORGE_MAX_SEQ`); overflow returns HTTP 400
before streaming. A runtime failure yields an error, without a successful finish
or `[DONE]`. Disconnects cancel generation after the current runtime operation.

An OpenAI SDK can use `base_url="http://127.0.0.1:8080/v1"` and a placeholder
`api_key="local-unused"`; no OpenAI service or key is needed. Keep requests within
the fields above. The server defaults to localhost with no authentication/TLS and
no wildcard CORS. A separately hosted browser frontend needs a deliberate origin
policy or a same-origin server integration; direct cross-origin browser requests
are not enabled by default. No frontend is included.

The existing `/v1/completions` endpoint, health and metrics remain available.
Plain legacy prompts retain the Qwen-style wrapper; formatted prompts pass through.

## Other builds and validation

```sh
# Linux CPU (not revalidated in this phase)
make build
make test
./bin/forge-cli -m models/Qwen2.5-0.5B-Instruct --device cpu
./bin/forge-server -model models/Qwen2.5-0.5B-Instruct -device cpu
```

After building, with Python 3.10+ and explicitly supplied model weights:

```powershell
python tests/cli_smoke.py --model models/Qwen2.5-0.5B-Instruct
python tests/chat_smoke.py --model models/Qwen2.5-0.5B-Instruct
python tests/chat_smoke.py --model models/SmolLM2-135M-Instruct
```

These launch real CPU processes. Chat checks include HTTP/SSE deterministic parity,
model templates and token counts, context limits, EOS, errors, disconnects,
concurrent clients and restart. They use the standard library and additionally
exercise the OpenAI SDK if it is already installed.
For optional CUDA builds and individual test commands, see [build.md](docs/build.md).
CUDA requires the toolkit and a separate kernel build; `cuda-stub` only checks
compilation and linkage without a GPU.

CLI options: `-m/--model`, `-c/--ctx-size` (1024), `-n/--predict` (128),
`--device` (cpu or cuda), and `--dtype` (fp32; CUDA also supports bf16).
Generation is greedy. `/clear` resets history; `/help` lists commands; `/exit`
exits. Ctrl+C stops generation, or exits while waiting for input. Cancellation
takes effect after the current runtime operation completes, including prefill.
The CLI applies `tokenizer_config.json`'s chat template to the entire conversation
and preserves previous turns. A full context asks you to clear history; output
is capped by the remaining context and stops at EOS. `FORGE_MAX_SEQ` also caps
the requested context. Models must include `tokenizer_config.json` for chat.
Run `python tests/cli_smoke.py --model models/Qwen2.5-0.5B-Instruct` after
building to check local CLI conversations and HTTP/SSE behavior.

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

Local CPU measurements on an Intel Core i5-1235U, Windows, October 9, 2026.
Median decode tokens/s across three separate runs per backend, using group size
64, a short prompt and 16 fixed decode steps. Model loading is excluded; these
are internal runtime timings. HTTP serving includes additional work.

| Model | FP32 tok/s | INT8 tok/s | INT4 tok/s |
|---|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | 10.38 | 24.17 | 16.99 |
| SmolLM2-135M-Instruct | 32.30 | 48.94 | 29.02 |

See [benchmarks.md](docs/benchmarks.md) for commands and timing definitions.
Generated output stays in the ignored `benchmark-output/` directory.

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
model context limits have not been audited. The HTTP API implements a small text/chat
completion subset of the OpenAI API. There is no authentication or TLS; bind to
loopback for local use.
