param(
  [string]$Root = "results\bstsp_large3",
  [string]$BestCsv = "",
  [string]$OutCsv = "results\summary\large3_summary.csv"
)

# Resolve repo root based on where the script lives.
$Base = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
$RepoRoot = if ((Split-Path -Leaf $Base) -ieq 'scripts') { Split-Path -Parent $Base } else { $Base }

function Resolve-Rel([string]$p) {
  if ([string]::IsNullOrWhiteSpace($p)) { return $p }
  if ([System.IO.Path]::IsPathRooted($p)) { return $p }
  return (Join-Path $RepoRoot $p)
}

$RootPath   = Resolve-Rel $Root
$BestCsvPath= Resolve-Rel $BestCsv
$OutCsvPath = Resolve-Rel $OutCsv

function Load-BestKnown([string]$Path) {
  $map = @{}
  if ($Path -and (Test-Path -LiteralPath $Path)) {
    try {
      $rows = Import-Csv -LiteralPath $Path
      foreach ($r in $rows) {
        if ($r.instance -and $r.best_known) {
          $map[[string]$r.instance] = [double]$r.best_known
        }
      }
    } catch {
      Write-Warning ("Failed to read best-known CSV: " + $_.Exception.Message)
    }
  }
  return $map
}

function Read-Run([string]$File) {
  $anytime = New-Object System.Collections.Generic.List[object]
  $best = $null; $runtime = $null; $seed = $null
  try {
    foreach ($line in Get-Content -LiteralPath $File -ReadCount 1) {
      if ([string]::IsNullOrWhiteSpace($line)) { continue }
      try { $obj = $line | ConvertFrom-Json } catch { continue }
      if ($obj.event -eq "improve") {
        $t = [double]$obj.t; $b = [double]$obj.best
        $anytime.Add([pscustomobject]@{ t=$t; best=$b })
        if ($best -eq $null -or $b -lt $best) { $best = $b }
      } elseif ($obj.event -eq "summary") {
        if ($obj.seed -ne $null) { $seed = [int]$obj.seed }
        if ($obj.runtime -ne $null) { $runtime = [double]$obj.runtime }
        if ($obj.best -ne $null) {
          $b = [double]$obj.best
          if ($best -eq $null -or $b -lt $best) { $best = $b }
        }
      }
    }
  } catch {
    Write-Warning ("Failed reading ${File}: " + $_.Exception.Message)
  }
  $sorted = $anytime | Sort-Object -Property t
  return [pscustomobject]@{ seed=$seed; best=$best; runtime=$runtime; anytime=$sorted }
}

function Median([double[]]$Arr) {
  if (!$Arr -or $Arr.Count -eq 0) { return $null }
  $sorted = $Arr | Sort-Object
  $n = $sorted.Count
  if ($n -eq 1) { return [double]$sorted[0] }
  if ($n % 2 -eq 1) { return [double]$sorted[([int][math]::Floor($n/2))] }
  else {
    $a = [double]$sorted[($n/2)-1]; $b = [double]$sorted[$n/2]
    return ($a + $b) / 2.0
  }
}

function Percentile-NR([double[]]$Arr, [double]$P) {
  if (!$Arr -or $Arr.Count -eq 0) { return $null }
  $sorted = $Arr | Sort-Object
  $n = $sorted.Count
  if ($n -eq 1) { return [double]$sorted[0] }
  $rank = [math]::Ceiling($P * $n)
  if ($rank -lt 1) { $rank = 1 }
  if ($rank -gt $n) { $rank = $n }
  return [double]$sorted[$rank-1]
}

function TTT-Median($Runs, [double]$Ref, [double]$GapPct) {
  $target = $Ref * (1.0 + $GapPct / 100.0)
  $times = @()
  foreach ($r in $Runs) {
    $hit = $null
    foreach ($ev in $r.anytime) {
      if ([double]$ev.best -le $target) { $hit = [double]$ev.t; break }
    }
    if ($hit -ne $null) { $times += $hit }
  }
  if ($times.Count -eq 0) { return $null }
  return (Median $times)
}

# Ensure output folder exists
$OutDir = Split-Path -Parent $OutCsvPath
if ($OutDir -and -not (Test-Path -LiteralPath $OutDir)) {
  New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
}

if (-not (Test-Path -LiteralPath $RootPath)) {
  Write-Error "Results root not found: $RootPath"
  exit 1
}

$bestMap = Load-BestKnown $BestCsvPath
$rows = New-Object System.Collections.Generic.List[object]

Get-ChildItem -LiteralPath $RootPath -Directory | ForEach-Object {
  $inst = $_.Name
  $logs = Get-ChildItem -LiteralPath $_.FullName -Filter "seed_*.jsonl" | Sort-Object Name
  if ($logs.Count -eq 0) { return }

  $runs = @()
  foreach ($log in $logs) {
    $runs += (Read-Run $log.FullName)
  }

  $bests = @($runs | ForEach-Object { $_.best } | Where-Object { $_ -ne $null } | ForEach-Object {[double]$_})
  $rts   = @($runs | ForEach-Object { $_.runtime } | Where-Object { $_ -ne $null } | ForEach-Object {[double]$_})

  if ($bests.Count -eq 0) { return }

  $medianBest = Median $bests
  $p10Best = Percentile-NR $bests 0.10
  $p90Best = Percentile-NR $bests 0.90
  $medianRt = if ($rts.Count -gt 0) { Median $rts } else { $null }

  $obsBest = ($bests | Sort-Object | Select-Object -First 1)
  $ref = if ($bestMap.ContainsKey($inst)) { [double]$bestMap[$inst] } else { [double]$obsBest }

  $ttt1  = TTT-Median $runs $ref 1.0
  $ttt05 = TTT-Median $runs $ref 0.5
  $ttt01 = TTT-Median $runs $ref 0.1

  $medianGap = if ($ref -gt 0) { 100.0*($medianBest - $ref)/$ref } else { $null }
  $p10Gap = if ($ref -gt 0) { 100.0*($p10Best - $ref)/$ref } else { $null }
  $p90Gap = if ($ref -gt 0) { 100.0*($p90Best - $ref)/$ref } else { $null }

  $rows.Add([pscustomobject]@{
    instance = $inst
    num_runs = $runs.Count
    ref_best = $ref
    median_best = $medianBest
    p10_best = $p10Best
    p90_best = $p90Best
    median_runtime_s = $medianRt
    ttt_1pct_s = $ttt1
    ttt_0_5pct_s = $ttt05
    ttt_0_1pct_s = $ttt01
    median_gap_pct = $medianGap
    p10_gap_pct = $p10Gap
    p90_gap_pct = $p90Gap
  })
}

if ($rows.Count -eq 0) {
  Write-Warning "No runs found under $RootPath"
} else {
  $rows | Export-Csv -LiteralPath $OutCsvPath -NoTypeInformation
  Write-Host "Wrote $OutCsvPath with $($rows.Count) row(s)."
}
