param(
 [Parameter(Mandatory=$true)][DateTimeOffset]$Deadline,
 [Parameter(Mandatory=$true)][string]$RepoRoot
)
$deadline = $Deadline.ToUniversalTime()
while ([DateTimeOffset]::UtcNow -lt $deadline) { Start-Sleep -Seconds 10 }
Set-Location -LiteralPath $RepoRoot
for ($attempt=0; $attempt -lt 4; $attempt++) {
 & '.\bin\forge-cloud.exe' stop *> 'docs\_phase14_watchdog_stop.log'
 if ($LASTEXITCODE -eq 0) { exit 0 }
 Start-Sleep -Seconds 5
}
exit 1
