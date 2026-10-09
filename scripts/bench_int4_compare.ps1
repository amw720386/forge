# Safe INT4 vs INT8 vs FP32 comparison — one backend per process.
# Does not run unrestricted stress tests. Aborts if free RAM is low.
#
# Usage (from repo root):
#   .\scripts\bench_int4_compare.ps1
#   .\scripts\bench_int4_compare.ps1 -SkipFp32   # save ~2 GiB peak (INT8/INT4 only)

param(
    [string]$Model = "models/Qwen2.5-0.5B-Instruct",
    [int]$Group = 64,
    [int]$Prefill = 64,
    [int]$Decode = 32,
    [int]$Reps = 3,
    # Default 2048: enough headroom for INT8/INT4 single-backend; raise for FP32.
    [int]$MinAvailMib = 2048,
    [switch]$SkipFp32,
    [switch]$MicroOnly
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$Model = [System.IO.Path]::GetFullPath((Join-Path $root $Model))

$env:FORGE_BENCH_MEM_CAP_MIB = if ($env:FORGE_BENCH_MEM_CAP_MIB) { $env:FORGE_BENCH_MEM_CAP_MIB } else { "6144" }
$env:FORGE_BENCH_RAM_FRACTION = if ($env:FORGE_BENCH_RAM_FRACTION) { $env:FORGE_BENCH_RAM_FRACTION } else { "0.45" }

Write-Host "=== INT4 comparison (separate processes) ==="
Write-Host "model=$Model group=$Group prefill=$Prefill decode=$Decode reps=$Reps"
Write-Host "FORGE_NUM_THREADS=$($env:FORGE_NUM_THREADS) MEM_CAP=$($env:FORGE_BENCH_MEM_CAP_MIB)"

function Invoke-Diag([string]$Backend) {
    Write-Host ""
    Write-Host ">>>>>> PROCESS backend=$Backend <<<<<<"
    $args = @(
        "run", "--release", "-p", "forge-runtime", "--bin", "forge-int4-diag", "--",
        "--model", $Model,
        "--backend", $Backend,
        "--group", "$Group",
        "--prefill", "$Prefill",
        "--decode", "$Decode",
        "--reps", "$Reps",
        "--max-seq", "512",
        "--min-avail-mib", "$MinAvailMib"
    )
    if ($MicroOnly) {
        $args += "--skip-model"
    }
    # Microbench only once (on first backend) to avoid repeating synthetic work.
    if ($Backend -ne "int8" -and -not $MicroOnly) {
        $args += "--skip-micro"
    }
    & cargo @args
    if ($LASTEXITCODE -ne 0) {
        throw "forge-int4-diag --backend $Backend failed with exit $LASTEXITCODE"
    }
    # Give the OS a moment to reclaim working set before the next model load.
    Start-Sleep -Seconds 2
}

Push-Location runtime
try {
    if ($MicroOnly) {
        Invoke-Diag "int8"
        return
    }
    # INT8 first (smaller than FP32) — establishes non-paged baseline + microbench.
    Invoke-Diag "int8"
    Invoke-Diag "int4"
    if (-not $SkipFp32) {
        Invoke-Diag "fp32"
    }
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "Done. Compare SUMMARY lines and Δfaults_decode across backends."
Write-Host "If INT4/INT8 microbench ratio >> 1 and decode Δfaults are similar, the bottleneck is the kernel."
