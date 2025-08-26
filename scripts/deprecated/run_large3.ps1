param(
  [string]$TsplibDir = "data\tsplib",
  [string]$ResultsDir = "results",
  [int]$Seeds = 5,
  [int]$WallSmall = 1800,
  [int]$WallLarge = 7200
)
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$runner = Join-Path $here "run_large3.py"
if (!(Test-Path $runner)) {
  Write-Error "run_large3.py not found in $here. Place this PS1 in the scripts\ folder."
  exit 1
}
python $runner --tsplib_dir $TsplibDir --results_dir $ResultsDir --seeds $Seeds --wall_small $WallSmall --wall_large $WallLarge
