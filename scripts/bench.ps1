# Synthetic matmul micro-benchmark.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
            [System.Environment]::GetEnvironmentVariable("Path", "User")

if (-not (Test-Path "$Root\kernels\build\libforge_kernels.a")) {
  & "$PSScriptRoot\build.ps1"
}

New-Item -ItemType Directory -Force -Path "$Root\benchmarks\build" | Out-Null
clang++ -std=c++17 -O3 -mavx2 -mfma `
  -I"$Root\kernels\include" `
  "$Root\benchmarks\bench_matmul.cpp" `
  "$Root\kernels\build\libforge_kernels.a" `
  -o "$Root\benchmarks\build\bench_matmul.exe"

& "$Root\benchmarks\build\bench_matmul.exe"
