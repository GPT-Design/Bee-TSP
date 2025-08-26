param(
    [string]$Results = "results",
    [string]$Algo = "bstsp",
    [string]$Best = "",
    [string]$Out = "results\summary\results_summary.csv"
)
# Project root is one level up from scripts/
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $ScriptDir
Set-Location $Root

if (-not (Test-Path "analyze_results.py")) {
    Write-Error "analyze_results.py not found in $Root. Place it at the project root."
    exit 1
}

if (-not (Test-Path $Results)) {
    Write-Error "Results directory not found: $Results"
    exit 1
}

# Ensure output folder exists
$OutDir = Split-Path $Out
if ($OutDir -and -not (Test-Path $OutDir)) { New-Item -ItemType Directory -Force -Path $OutDir | Out-Null }

# Build args
$argsList = @("--results", $Results, "--algo", $Algo, "--out", $Out)
if ($Best -and (Test-Path $Best)) {
    $argsList += @("--best", $Best)
} elseif ($Best) {
    Write-Warning "Best-known file not found: $Best (continuing without it)"
}

python analyze_results.py @argsList
