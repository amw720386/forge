# Refresh common toolchain paths for a PowerShell session.
$env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
            [System.Environment]::GetEnvironmentVariable("Path", "User")

# Prefer LLVM-MinGW if present (typical winget install location varies).
$mingwCandidates = @(
  "$env:LOCALAPPDATA\Microsoft\WinGet\Packages",
  "C:\llvm-mingw",
  "C:\Program Files\llvm-mingw"
)
Get-ChildItem -Path $mingwCandidates -Filter "clang.exe" -Recurse -ErrorAction SilentlyContinue |
  Select-Object -First 1 |
  ForEach-Object {
    $bin = $_.Directory.FullName
    if ($env:Path -notlike "*$bin*") {
      $env:Path = "$bin;$env:Path"
    }
  }

$env:CGO_ENABLED = "1"
Write-Host "CGO_ENABLED=$env:CGO_ENABLED"
Write-Host "clang: $(Get-Command clang -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source)"
Write-Host "go:    $(Get-Command go -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source)"
Write-Host "cargo: $(Get-Command cargo -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source)"
