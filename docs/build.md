# Build and run

Run commands from the repository root unless stated otherwise. Weights stay local:
provide `config.json`, one `model.safetensors`, and `tokenizer.json` in
`models/Qwen2.5-0.5B-Instruct` or `models/SmolLM2-135M-Instruct`. Preserve upstream
model licenses and attribution. Forge does not download models automatically.
For interactive chat also include the upstream `tokenizer_config.json`.

## CPU

Use Rust stable, Go 1.22+, CMake 3.16+ and a C++17 compiler. Cargo compiles the CPU
kernels itself; CMake separately builds their unit tests. Default x86-64 builds
use AVX2/FMA. Other CPU architectures are not qualified.

On Windows, install Visual Studio C++ Build Tools and LLVM-MinGW (including
`clang++` and `mingw32-make`) on PATH. The existing script imports MSVC environment
variables from common Build Tools locations. For another installation, use a
Visual Studio developer PowerShell. It uses your selected Rust toolchain and
requires `stable-x86_64-pc-windows-msvc`; it does not change your global default.

```powershell
.\scripts\build.ps1
.\scripts\test.ps1
.\bin\forge-cli.exe -m models/Qwen2.5-0.5B-Instruct -c 1024 -n 128
$env:FORGE_RUNTIME_DIR = "$PWD\runtime\target\release"
.\bin\forge-server.exe -model models/SmolLM2-135M-Instruct -addr 127.0.0.1:8080
```

In a second PowerShell:

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8080/v1/completions -ContentType application/json -Body '{"prompt":"The capital of France is","max_tokens":16,"temperature":0,"stream":false}'
```

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
Windows bindings load `forge_runtime.dll`; set `FORGE_RUNTIME_DIR` when running
outside the root with `go run`. Built server binaries also search relative to
their executable. Model paths are relative to the current directory; use an
absolute `-model`/`-m` path from other directories. The CLI links the runtime
directly and needs no DLL search setting. Build it alone with:

```sh
cargo build --release --manifest-path runtime/Cargo.toml --bin forge-cli
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
# With a real CUDA build and local GPU; not executed during this cleanup
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
cargo test --release -j 1 --manifest-path runtime/Cargo.toml --test chat_templates -- --include-ignored --test-threads=1
```

Repeat with Qwen and `tests/fixtures/pretrained/qwen2`. Generated activation binaries
are ignored; the reference generator, case metadata and tensor manifests are kept.
The checkpoint example drops the original model before reloading and checks exact
logit preservation. Omit the final `8` for FP32, or use `4` for INT4.

Current cleanup validation is on Windows x86-64 CPU. Linux builds and CUDA
execution were not retested. Historical details are in [history/](history/).
