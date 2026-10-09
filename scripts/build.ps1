# Build CPU kernels, Rust runtime, CLI and Go server on Windows.
param([ValidateRange(1, 64)][int]$Jobs = 1)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Import-VcVars {
  $candidates = @(
    "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
    "${env:ProgramFiles(x86)}\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
  )
  foreach ($bat in $candidates) {
    if (Test-Path $bat) {
      cmd /c "`"$bat`" && set" | ForEach-Object {
        if ($_ -match '^(.*?)=(.*)$') {
          [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
        }
      }
      if ($LASTEXITCODE -ne 0) { throw "MSVC environment setup failed ($LASTEXITCODE)" }
      return
    }
  }
  Write-Warning "vcvars64.bat not found; MSVC build may fail"
}

Import-VcVars

Write-Host "==> CMake kernels"
cmake -S "$Root\kernels" -B "$Root\kernels\build" -G "MinGW Makefiles" `
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed ($LASTEXITCODE)" }
cmake --build "$Root\kernels\build" --config Release --parallel $Jobs
if ($LASTEXITCODE -ne 0) { throw "Kernel build failed ($LASTEXITCODE)" }

Write-Host "==> Rust runtime"
cargo build --release -j $Jobs --manifest-path "$Root\runtime\Cargo.toml"
if ($LASTEXITCODE -ne 0) { throw "Rust build failed ($LASTEXITCODE)" }
New-Item -ItemType Directory -Force -Path "$Root\bin" | Out-Null
Copy-Item -LiteralPath "$Root\runtime\target\release\forge-cli.exe" -Destination "$Root\bin\forge-cli.exe"

Write-Host "==> Go server"
New-Item -ItemType Directory -Force -Path "$Root\bin" | Out-Null
go -C "$Root\api" build -o "$Root\bin\forge-server.exe" ./cmd/forge-server
if ($LASTEXITCODE -ne 0) { throw "Go build failed ($LASTEXITCODE)" }

Write-Host "Build complete."
