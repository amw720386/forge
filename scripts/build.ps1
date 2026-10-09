# Build Phase 1 artifacts on Windows (MSVC Rust + MinGW/LLVM C++ kernels optional).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
            [System.Environment]::GetEnvironmentVariable("Path", "User")

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
      return
    }
  }
  Write-Warning "vcvars64.bat not found; MSVC build may fail"
}

Import-VcVars
rustup default stable-x86_64-pc-windows-msvc | Out-Null

Write-Host "==> CMake kernels"
cmake -S "$Root\kernels" -B "$Root\kernels\build" -G "MinGW Makefiles" `
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++
cmake --build "$Root\kernels\build" --config Release

Write-Host "==> Rust runtime"
Push-Location "$Root\runtime"
cargo build --release
Pop-Location

Write-Host "Build complete."
