# Convenience targets; Cargo builds the CPU kernels used by the runtime.
.DEFAULT_GOAL := build
MODEL ?= models/Qwen2.5-0.5B-Instruct

.PHONY: all build test bench run help kernels runtime go-smoke cuda-stub-check fmt
all: build

help:
	@echo "make build / test / bench / run MODEL=path/to/model"
	@echo "make cuda-stub-check checks CUDA linkage without a GPU"

ifeq ($(OS),Windows_NT)
build:
	powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build.ps1
test:
	powershell -NoProfile -ExecutionPolicy Bypass -File scripts/test.ps1
bench:
	powershell -NoProfile -ExecutionPolicy Bypass -File scripts/bench.ps1
run: build
	powershell -NoProfile -Command "$$env:FORGE_RUNTIME_DIR='$(CURDIR)/runtime/target/release'; & './bin/forge-server.exe' -model '$(MODEL)' -addr '127.0.0.1:8080'"
else
build: runtime
	mkdir -p bin
	go -C api build -o ../bin/forge-server ./cmd/forge-server
test: build kernels
	ctest --test-dir kernels/build --output-on-failure -C Release
	cargo test --release --manifest-path runtime/Cargo.toml
	go -C bindings/go test -count=1 ./...
	go -C api test -count=1 ./...
bench: kernels
	mkdir -p benchmarks/build
	$(CXX) -std=c++17 -O3 -mavx2 -mfma -pthread -Ikernels/include benchmarks/bench_matmul.cpp kernels/build/libforge_kernels.a -o benchmarks/build/bench_matmul
	./benchmarks/build/bench_matmul
run: build
	./bin/forge-server -model "$(MODEL)" -addr 127.0.0.1:8080
endif

kernels:
	cmake -S kernels -B kernels/build -DCMAKE_BUILD_TYPE=Release
	cmake --build kernels/build --config Release
runtime:
	cargo build --release --manifest-path runtime/Cargo.toml
go-smoke: runtime
	go -C bindings/go test -count=1 ./...
cuda-stub-check:
	cargo test --release --manifest-path runtime/Cargo.toml --features cuda-stub
fmt:
	cargo fmt --manifest-path runtime/Cargo.toml
	go -C api fmt ./...
	go -C bindings/go fmt ./...
