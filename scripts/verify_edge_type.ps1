param(
  [Parameter(Mandatory=$true)][string]$Tsp,
  [string]$Tour = "",
  [int]$Best = -1,
  [string]$BestCsv = ""
)
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = Join-Path $here "verify_edge_type.py"
if (!(Test-Path -LiteralPath $py)) {
  Write-Error "Missing $py. Place it in scripts/."
  exit 1
}

$cmd = @("python", $py, "--tsp", $Tsp)
if ($Tour -and $Tour.Trim().Length -gt 0) { $cmd += @("--tour", $Tour) }
if ($Best -ge 0) { $cmd += @("--best", $Best) }
if ($BestCsv -and $BestCsv.Trim().Length -gt 0) { $cmd += @("--bestcsv", $BestCsv) }
Write-Host "Running: $($cmd -join ' ')"
& $cmd
