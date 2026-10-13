# Build and run

Run commands from the repository root unless stated otherwise. Weights stay local:
provide `config.json`, one `model.safetensors`, and `tokenizer.json` in
`models/Qwen2.5-0.5B-Instruct` or `models/SmolLM2-135M-Instruct`. Preserve upstream
model licenses and attribution. Forge does not download models automatically.
For CLI and HTTP chat also include the upstream `tokenizer_config.json`.

## CPU

Use Rust stable, Go 1.22+, CMake 3.16+ and a C++17 compiler. Cargo compiles the CPU
kernels itself; CMake separately builds their unit tests. Default x86-64 builds
use AVX2/FMA. Other CPU architectures are not qualified.

On Windows, install Visual Studio 2022 C++ Build Tools (Desktop development with
C++, including a Windows SDK) and CMake 3.21+ on PATH. Visual Studio 2026 Build
Tools requires CMake 4.2+. The scripts use the Visual Studio x64 generator and
MSVC for both Cargo's kernels and standalone tests/benchmarks; LLVM-MinGW is
not required. The build script imports MSVC environment
variables from common Build Tools locations. For another installation, use a
Visual Studio developer PowerShell. It uses your selected Rust toolchain and
requires `stable-x86_64-pc-windows-msvc`; it does not change your global default.

```powershell
.\scripts\build.ps1
.\scripts\test.ps1
python tests/cli_smoke.py --model models/Qwen2.5-0.5B-Instruct
python tests/chat_smoke.py --model models/Qwen2.5-0.5B-Instruct
.\bin\forge-cli.exe -m models/Qwen2.5-0.5B-Instruct -c 1024 -n 128
$env:FORGE_RUNTIME_DIR = "$PWD\runtime\target\release"
.\bin\forge-server.exe -model models/SmolLM2-135M-Instruct -addr 127.0.0.1:8080
```

In a second PowerShell:

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8080/v1/completions -ContentType application/json -Body '{"prompt":"The capital of France is","max_tokens":16,"temperature":0,"stream":false}'
```

The legacy `/v1/completions` endpoint wraps plain prompts in a fixed Qwen-style
ChatML conversation with the system message `You are a helpful assistant.`,
a user turn and an assistant generation prefix. Prompts containing
`<|im_start|>` pass through unchanged. This wrapper does not use the loaded
model's chat template. Use `/v1/chat/completions` for model-specific formatting.

Chat accepts `max_tokens` or `max_completion_tokens` (1–2048, default 64).
Both may be supplied if equal; conflicting values are rejected. They have
identical generation and usage-accounting semantics, and the full formatted
input plus resolved output limit must fit the context. Generation remains
greedy: only `temperature=0`, `top_p=1`, `top_k=1`, `n=1`,
`frequency_penalty=0`, `presence_penalty=0` and `repetition_penalty=1` are
accepted when supplied. Optional `response_format` supports only
`{"type":"text"}`. Unknown fields, explicit nulls, non-neutral penalties,
tools and constrained-output formats are rejected.

On Linux with GCC/Clang and GNU Make:

```sh
make build
make test
./bin/forge-server -model models/Qwen2.5-0.5B-Instruct -addr 127.0.0.1:8080
```

The underlying commands are:

```sh
cargo build --release --manifest-path runtime/Cargo.toml
go -C api build -o ../bin/forge-server ./cmd/forge-server
cmake -S kernels -B kernels/build -DCMAKE_BUILD_TYPE=Release
cmake --build kernels/build --config Release
ctest --test-dir kernels/build --output-on-failure -C Release
cargo test --release --manifest-path runtime/Cargo.toml
go -C bindings/go test -count=1 ./...
go -C api test -count=1 ./...
```

Unix bindings link the runtime in `runtime/target/release` using cgo and an rpath.
The two Go modules use `github.com/amw720386/forge`; `go.work` resolves both
locally. From the root run `go test ./api/... ./bindings/go/...` (root `./...`
is not a module). The API's local replace also supports `GOWORK=off`.
Windows builds default to serial compilation; `scripts/build.ps1 -Jobs 1`
is the low-memory option. Increase `-Jobs` for CMake and Cargo parallelism.
Windows CMake output is in `kernels/cmake-build-msvc/Release`; older MinGW
caches in `kernels/build` are left untouched. On Unix pass `-j 1` to Cargo and `--parallel 1`
to CMake when memory is limited.
Windows bindings load `forge_runtime.dll`; set `FORGE_RUNTIME_DIR` when running
outside the root with `go run`. Built server binaries also search relative to
their executable. Model paths are relative to the current directory; use an
absolute `-model`/`-m` path from other directories. The CLI links the runtime
directly and needs no DLL search setting. Build it alone with:

```sh
cargo build --release --manifest-path runtime/Cargo.toml --bin forge-cli
```

Ordinary Cargo builds include the CLI, quantizer and benchmark tool, but omit
the development-only INT4 diagnostic. `scripts/bench_int4_compare.ps1` enables
it explicitly; to invoke it directly:

```sh
cargo run --release -j 1 --manifest-path runtime/Cargo.toml --features diagnostics --bin forge-int4-diag -- --model models/Qwen2.5-0.5B-Instruct --backend int4 --reps 1
```

For an unbuilt server, use the module-aware command:

```sh
go -C api run ./cmd/forge-server -model ../models/Qwen2.5-0.5B-Instruct -addr 127.0.0.1:8080
```

## CUDA (Linux)

Requires NVIDIA's CUDA toolkit, nvcc, cuBLAS and a compatible C++ compiler.
Historical L4 validation used CUDA 13.0.88 and driver 580.178.04; this is not a
claim that all toolkit versions or GPUs work. CMake 3.18+ is required for CUDA.

```sh
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
cmake -S kernels -B kernels/build-cuda -DCMAKE_BUILD_TYPE=Release -DFORGE_BUILD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build kernels/build-cuda --parallel
export FORGE_CUDA_LIB_DIR="$PWD/kernels/build-cuda/cuda"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
cargo build --release --manifest-path runtime/Cargo.toml --features cuda
go -C api build -o ../bin/forge-server ./cmd/forge-server
./bin/forge-server -model models/Qwen2.5-0.5B-Instruct -device cuda -dtype bf16 -addr 127.0.0.1:8080
```

Use `-dtype fp32` for CUDA FP32. SmolLM2 BF16 remains experimental. The server
rejects unsupported devices/dtypes rather than emulating GPU inference.
`cuda` and `cuda-stub` are mutually exclusive. Always rebuild the default CPU
runtime before CPU validation after a stub build.

```sh
# No toolkit or GPU required
cargo test --release --manifest-path runtime/Cargo.toml --features cuda-stub
# With a real CUDA build and NVIDIA GPU
ctest --test-dir kernels/build-cuda --output-on-failure -C Release
cargo test --release --manifest-path runtime/Cargo.toml --features cuda
```

Some pretrained GPU checks are ignored by default and require model/reference
environment variables. Keep them opt-in. `scripts/verify_cuda_compile.sh` checks
the stub and, when nvcc is present, real linkage without running GPU inference.

## Quantization and checkpoints

```sh
cargo run --release --manifest-path runtime/Cargo.toml --bin forge-quantize -- --model models/Qwen2.5-0.5B-Instruct --bits 8 --group-size 64 --output models/qwen2.5-int8.forge
```

Use `--bits 4` for INT4. Keep the matching HF tokenizer directory in the standard
sibling layout, or set `FORGE_TOKENIZER_DIR` to it. The server and `forge-bench`
accept `.forge` paths. Existing FRG1 v1 checkpoints remain supported; legacy
checkpoints without tokenizer hashes cannot verify tokenizer identity.
Default limits are `FORGE_MAX_SEQ=1024`, `FORGE_MAX_BATCH=4` and
`FORGE_MEM_BUDGET_MIB=8192`. `-max-seq` is also capped by `FORGE_MAX_SEQ`.

## Pretrained CPU checks

Synthetic configuration/loading and checkpoint tests run in the normal Rust
suite. Existing HF reference exports can be regenerated sequentially with Python,
PyTorch, Transformers and Safetensors:

```sh
python tests/reference/pretrained_reference.py models/SmolLM2-135M-Instruct tests/fixtures/pretrained/smollm2
cargo run --release --manifest-path runtime/Cargo.toml --example pretrained_parity -- models/SmolLM2-135M-Instruct tests/fixtures/pretrained/smollm2
cargo run --release --manifest-path runtime/Cargo.toml --example checkpoint_parity -- models/SmolLM2-135M-Instruct models/smollm2-int8.forge models/SmolLM2-135M-Instruct 8
python tests/cli_smoke.py --model models/Qwen2.5-0.5B-Instruct
python tests/chat_smoke.py --model models/Qwen2.5-0.5B-Instruct
cargo test --release -j 1 --manifest-path runtime/Cargo.toml --test chat_templates -- --include-ignored --test-threads=1
```

Repeat with Qwen and `tests/fixtures/pretrained/qwen2`. Generated activation binaries
are ignored; the reference generator, case metadata and tensor manifests are kept.
The checkpoint example drops the original model before reloading and checks exact
logit preservation. Omit the final `8` for FP32, or use `4` for INT4.

The Python smoke entry point exercises the actual CLI and server, including
greedy formatted-input parity, HTTP/SSE, concurrent clients and clean restart.
It needs only Python's standard library; absent weights are explicitly skipped.
On Windows its interrupt check uses Ctrl+Break, which reaches the console
interrupt handler. Rebuild the server and runtime together after ABI changes.

## Common setup failures

- Script execution blocked: use the process-scoped `powershell -NoProfile
  -ExecutionPolicy Bypass -File .\scripts\build.ps1 -Jobs 1` invocation in README.
- Missing MSVC/link.exe or Visual Studio generator: install the C++ workload
  and Windows SDK, update CMake, or use a developer PowerShell.
- Incompatible CMake cache in `kernels/cmake-build-msvc`: use a fresh build
  directory or rename that generated directory before rebuilding. The script
  stops on generator conflicts and does not delete caches or user files.
- Native build failure or low memory: the script stops at the failing command;
  use `-Jobs 1`, close memory-heavy processes, and rebuild.
- DLL not found or missing `forge_chat_prepare`: rebuild runtime and server
  together; remove a stale `FORGE_RUNTIME_DIR` or point it at this checkout's
  `runtime/target/release`. Do not copy an older DLL to make a build pass.
- Model load failed: check the path and the original four files listed in README.
  `.forge` checkpoints additionally need their matching tokenizer directory.
  Chat requires a string `chat_template` in `tokenizer_config.json`.
- Chat context overflow: reduce history/output or increase `-max-seq` and
  `FORGE_MAX_SEQ` within model and memory limits. The full formatted input counts.
- Address already in use: stop the previous process with Ctrl+C or select another
  localhost port using `-addr`. Restart tests verify process cleanup.

A clean source build uses only tracked source plus installed compilers and normal
Cargo/Go package resolution. Model-dependent tests require separately supplied
weights. Existing package download caches may be reused; compiled outputs from
another checkout are not required. Windows CPU is the validated installation
path; Linux and CUDA require verification on their respective platforms.
