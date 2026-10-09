# Forge top-level build. Works with GNU Make / mingw32-make.
# Phase 1: kernels + Rust runtime + Go FFI smoke test.

ROOT := $(CURDIR)
KERNELS_BUILD := $(ROOT)/kernels/build
RUNTIME_DIR := $(ROOT)/runtime
RUNTIME_TARGET := $(RUNTIME_DIR)/target/release
BINDINGS_GO := $(ROOT)/bindings/go
API_DIR := $(ROOT)/api

ifeq ($(OS),Windows_NT)
  EXE := .exe
  DLL := forge_runtime.dll
  PATHSEP := ;
  export CGO_ENABLED := 1
  # Ensure MinGW/LLVM tools win over MSVC for cgo.
  export CC ?= clang
  export CXX ?= clang++
else
  EXE :=
  DLL := libforge_runtime.so
  UNAME_S := $(shell uname -s)
  ifeq ($(UNAME_S),Darwin)
    DLL := libforge_runtime.dylib
  endif
  PATHSEP := :
  export CGO_ENABLED := 1
endif

.PHONY: all build test bench clean kernels kernels-test runtime runtime-test \
        go-smoke bench-kernels help run fmt cuda-stub-check

all: build test

help:
	@echo "Forge targets:"
	@echo "  make build         - build C++ kernels + Rust runtime"
	@echo "  make test          - run kernel, Rust, and Go FFI tests"
	@echo "  make bench         - synthetic matmul micro-benchmark"
	@echo "  make run           - Phase 1 placeholder server"
	@echo "  make clean         - remove build artifacts"
	@echo "  make cuda-stub-check - compile CUDA Rust path with stub (no GPU)"

build: kernels runtime

kernels:
	cmake -S $(ROOT)/kernels -B $(KERNELS_BUILD) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(KERNELS_BUILD) --config Release

kernels-test: kernels
	cd $(KERNELS_BUILD) && ctest --output-on-failure -C Release
	@# Also run the binary directly if ctest generator differs
	@if [ -f "$(KERNELS_BUILD)/forge_kernels_tests$(EXE)" ]; then \
		"$(KERNELS_BUILD)/forge_kernels_tests$(EXE)"; \
	elif [ -f "$(KERNELS_BUILD)/Release/forge_kernels_tests$(EXE)" ]; then \
		"$(KERNELS_BUILD)/Release/forge_kernels_tests$(EXE)"; \
	fi

runtime:
	cd $(RUNTIME_DIR) && cargo build --release

runtime-test: runtime
	cd $(RUNTIME_DIR) && cargo test --release

go-smoke: runtime
ifeq ($(OS),Windows_NT)
	@powershell -NoProfile -Command \
	  "$$env:FORGE_RUNTIME_DIR='$(RUNTIME_TARGET)'; \
	   Set-Location '$(BINDINGS_GO)'; \
	   go test -count=1 ./..."
else
	PATH="$(RUNTIME_TARGET)$(PATHSEP)$$PATH" \
	LD_LIBRARY_PATH="$(RUNTIME_TARGET)$(PATHSEP)$$LD_LIBRARY_PATH" \
	DYLD_LIBRARY_PATH="$(RUNTIME_TARGET)$(PATHSEP)$$DYLD_LIBRARY_PATH" \
	  go test -count=1 -C $(BINDINGS_GO) ./...
endif

test: kernels-test runtime-test go-smoke

bench: bench-kernels

bench-kernels: kernels
ifeq ($(OS),Windows_NT)
	@powershell -NoProfile -ExecutionPolicy Bypass -File $(ROOT)/scripts/bench.ps1
else
	@mkdir -p $(ROOT)/benchmarks/build
	$(CXX) -std=c++17 -O3 -mavx2 -mfma -I$(ROOT)/kernels/include \
	  $(ROOT)/benchmarks/bench_matmul.cpp \
	  $(KERNELS_BUILD)/libforge_kernels.a \
	  -lstdc++ -o $(ROOT)/benchmarks/build/bench_matmul$(EXE)
	$(ROOT)/benchmarks/build/bench_matmul$(EXE)
endif

run: runtime
ifeq ($(OS),Windows_NT)
	@powershell -NoProfile -Command \
	  "$$env:FORGE_RUNTIME_DIR='$(RUNTIME_TARGET)'; \
	   Set-Location '$(API_DIR)'; \
	   go run ./cmd/forge-server -model ../models/Qwen2.5-0.5B-Instruct -addr :8080"
else
	FORGE_RUNTIME_DIR=$(RUNTIME_TARGET) \
	  go run -C $(API_DIR) ./cmd/forge-server -model ../models/Qwen2.5-0.5B-Instruct -addr :8080
endif

# Compile/link CUDA Rust modules against forge_cuda_stub (no nvcc/GPU).
cuda-stub-check:
	cd $(RUNTIME_DIR) && cargo build --release --features cuda-stub
	cd $(RUNTIME_DIR) && cargo test --release --features cuda-stub --test cuda_compile_check -- --nocapture
	cd $(RUNTIME_DIR) && cargo test --release --features cuda-stub --test cuda_transformer_parity -- --nocapture

fmt:
	cd $(RUNTIME_DIR) && cargo fmt
	gofmt -w $(BINDINGS_GO) $(API_DIR)/cmd

clean:
	rm -rf $(KERNELS_BUILD) $(RUNTIME_DIR)/target $(ROOT)/benchmarks/build
	cd $(API_DIR) && go clean ./... || true
	cd $(BINDINGS_GO) && go clean ./... || true
