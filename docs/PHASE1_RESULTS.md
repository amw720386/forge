# Phase 1 Results

## What works

- Contiguous FP32 `Tensor` in Rust with shape/stride validation
- C++ `forge_kernels_matmul_f32` / `forge_kernels_matvec_f32` with AVX2/FMA when enabled
- Rust wrappers + public C ABI (`forge_matmul_f32`, `forge_matvec_f32`, `forge_version`)
- Go → Rust → C++ smoke tests (Windows: dynamic DLL load; Unix: cgo)
- CMake + Cargo + Makefile + `scripts/*.ps1` + GitHub Actions CI

## Commands

```bash
make build && make test && make bench
```

Windows:

```powershell
.\scripts\test.ps1
.\scripts\bench.ps1
```

## Test results (2026-10-08, Windows)

| Suite | Result |
|---|---|
| `forge_kernels_tests` | pass |
| `cargo test --release` (8 tests) | pass |
| `go test` bindings (4 tests) | pass |
| DLL exports | `forge_version`, `forge_matmul_f32`, `forge_matvec_f32` |

## Synthetic benchmark

| Host | Size | ms/iter | GFLOP/s | Notes |
|---|---|---|---|---|
| Windows x86_64, LLVM-MinGW clang++ -O3 -mavx2 -mfma | 256³ | 3.632 | 9.24 | synthetic only |

## Numerical tolerance

Relative error ≤ 1e-4 (+ small absolute tolerance) vs naive reference GEMM.
