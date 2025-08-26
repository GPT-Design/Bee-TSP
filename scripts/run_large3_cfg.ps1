param(
  [string]$Config = "configs\params.yaml",
  [string]$TsplibDir = "",
  [string]$ResultsDir = "",
  [int]$Seeds = -1,
  [int]$WallSmall = -1,
  [int]$WallLarge = -1
)
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$runner = Join-Path $here "run_large3_cfg.py"
if (!(Test-Path $runner)) { Write-Error "Missing $runner"; exit 1 }

$cmd = @("python", $runner, "--config", $Config)
if ($TsplibDir -ne "") { $cmd += @("--tsplib_dir", $TsplibDir) }
if ($ResultsDir -ne "") { $cmd += @("--results_dir", $ResultsDir) }
if ($Seeds -ge 0) { $cmd += @("--seeds", $Seeds) }
if ($WallSmall -ge 0) { $cmd += @("--wall_small", $WallSmall) }
if ($WallLarge -ge 0) { $cmd += @("--wall_large", $WallLarge) }
Write-Host "Running: $($cmd -join ' ')"
& $cmd
