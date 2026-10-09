# Build and run CPU kernels, Rust, Go bindings and API tests on Windows.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

& "$PSScriptRoot\build.ps1"

Write-Host "==> Kernel tests"
& "$Root\kernels\build\forge_kernels_tests.exe"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "==> Rust tests"
Push-Location "$Root\runtime"
cargo test --release
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Pop-Location

Write-Host "==> Go FFI smoke tests"
$env:FORGE_RUNTIME_DIR = "$Root\runtime\target\release"
Push-Location "$Root\bindings\go"
go test -count=1 ./...
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Pop-Location

Write-Host "==> Go API tests"
go -C "$Root\api" test -count=1 ./...
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "All CPU tests passed."
