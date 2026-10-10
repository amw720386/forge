# Synthetic matmul micro-benchmark.
param([ValidateRange(1, 64)][int]$Jobs = 1)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

& "$PSScriptRoot\build.ps1" -Jobs $Jobs
cmake --build "$Root\kernels\cmake-build-msvc" --config Release --target forge_matmul_bench --parallel $Jobs
if ($LASTEXITCODE -ne 0) { throw "Benchmark build failed ($LASTEXITCODE)" }

& "$Root\kernels\cmake-build-msvc\Release\forge_matmul_bench.exe"
if ($LASTEXITCODE -ne 0) { throw "Benchmark failed ($LASTEXITCODE)" }
