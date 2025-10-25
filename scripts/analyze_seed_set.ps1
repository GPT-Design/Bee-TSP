param(
  [Parameter(Mandatory=$true)]
  [string[]]$Files,
  [string]$BestCsv = "",
  [double]$RefBest = [double]::NaN,
  [string]$OutCsv = ""
)

$Base = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
$RepoRoot = if ((Split-Path -Leaf $Base) -ieq 'scripts') { Split-Path -Parent $Base } else { $Base }

function Resolve-Rel([string]$p) {
  if ([string]::IsNullOrWhiteSpace($p)) { return $p }
  if ([System.IO.Path]::IsPathRooted($p)) { return $p }
  return (Join-Path $RepoRoot $p)
}

function Guess-Instance([string]$p) {
  try {
    $dir = Split-Path -Parent $p
    $name = Split-Path -Leaf $dir
    if ($name -and $name -match '^[A-Za-z0-9_\-]+$') { return $name }
  } catch { }
  return ""
}

$Files = $Files | ForEach-Object { Resolve-Rel $_ }

$BestMap = @{}
if ($BestCsv) {
  $BestCsvPath = Resolve-Rel $BestCsv
  if (Test-Path -LiteralPath $BestCsvPath) {
    try {
      $rows = Import-Csv -LiteralPath $BestCsvPath
      foreach ($r in $rows) {
        if ($r.instance -and $r.best_known) {
          $BestMap[[string]$r.instance] = [double]$r.best_known
        }
      }
    } catch {
      Write-Warning ("Failed to read best-known CSV: " + $_.Exception.Message)
    }
  } else {
    Write-Warning "BestCsv not found: $BestCsvPath"
  }
}

$rows = @()
$finals = @()
$instName = ""

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
  $improves = $sorted.Count
  return [pscustomobject]@{ seed=$seed; best=$best; runtime=$runtime; anytime=$sorted; improves=$improves }
}

foreach ($f in $Files) {
  if (!(Test-Path -LiteralPath $f)) {
    Write-Warning "Missing file: $f"
    continue
  }
  if (-not $instName) { $instName = Guess-Instance $f }
  $r = Read-Run $f
  if ($r.best -ne $null) {
    $rows += [pscustomobject]@{ file=$f; seed=$r.seed; best=$r.best; runtime=$r.runtime; improves=$r.improves; anytime=$r.anytime }
    $finals += [double]$r.best
  }
}

if ($rows.Count -eq 0) { Write-Error "No valid runs found."; exit 1 }

$refUsed = $null
if (-not [double]::IsNaN($RefBest)) {
  $refUsed = [double]$RefBest
} elseif ($instName -and $BestMap.ContainsKey($instName)) {
  $refUsed = [double]$BestMap[$instName]
} else {
  $refUsed = ($finals | Sort-Object | Select-Object -First 1)
  Write-Host "Reference not provided; using best observed across seeds: $refUsed"
}

function TTT([System.Collections.Generic.List[object]]$anytime, [double]$ref, [double]$gapPct) {
  $target = $ref * (1.0 + $gapPct/100.0)
  foreach ($ev in $anytime) {
    if ([double]$ev.best -le $target) { return [double]$ev.t }
  }
  return $null
}

$table = @()
foreach ($r in $rows) {
  $t1 = TTT $r.anytime $refUsed 1.0
  $t05 = TTT $r.anytime $refUsed 0.5
  $t01 = TTT $r.anytime $refUsed 0.1
  $gap = if ($refUsed -gt 0) { 100.0*([double]$r.best - $refUsed)/$refUsed } else { $null }
  $table += [pscustomobject]@{
    instance = $instName
    seed = $r.seed
    final_best = [double]$r.best
    gap_pct = $gap
    runtime_s = [double]$r.runtime
    improvements = [int]$r.improves
    ttt_1pct_s = $t1
    ttt_0_5pct_s = $t05
    ttt_0_1pct_s = $t01
    file = $r.file
  }
}

$sorted = $table | Sort-Object -Property final_best
$bestRow = $sorted | Select-Object -First 1
$medianBest = ($sorted | Select-Object -ExpandProperty final_best) | Measure-Object -Average | Select-Object -ExpandProperty Average
$medianGap = $null
$gaps = $sorted | Where-Object { $_.gap_pct -ne $null } | Select-Object -ExpandProperty gap_pct
if ($gaps.Count -gt 0) { $medianGap = ($gaps | Measure-Object -Average | Select-Object -ExpandProperty Average) }

Write-Host ""
Write-Host "Instance: $instName"
Write-Host "Reference best: $refUsed"
Write-Host ""
$table | Sort-Object seed | Format-Table -AutoSize
Write-Host ""
Write-Host ("Observed best: {0} (seed {1})" -f [int]$bestRow.final_best, $bestRow.seed)
if ($medianBest -ne $null) { Write-Host ("Median final best: {0}" -f [int]$medianBest) }
if ($medianGap -ne $null)  { Write-Host ("Median gap %: {0:N3}" -f $medianGap) }

if ($OutCsv -and $OutCsv.Trim().Length -gt 0) {
  $OutCsvPath = Resolve-Rel $OutCsv
  $outDir = Split-Path -Parent $OutCsvPath
  if ($outDir -and -not (Test-Path -LiteralPath $outDir)) {
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
  }
  $table | Export-Csv -LiteralPath $OutCsvPath -NoTypeInformation
  Write-Host "Saved per-seed summary: $OutCsvPath"
}
