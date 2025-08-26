param(
  [string]$ActiveRel  = "docs\Claude_Update_active.txt",
  [string]$ArchiveRel = "docs\update_history",
  [int]$StageId = 0,                # 0 = auto-next; else pin to a stage (e.g., 4)
  [string]$Suffix = "",             # e.g., "Decision" or "sklearn" (no leading underscore needed)
  [int]$Pad = 3
)

# Resolve paths
$activePath  = $ActiveRel
$archivePath = $ArchiveRel
$manifest    = Join-Path $archivePath "manifest.jsonl"

# Pre-flight
if (!(Test-Path $activePath)) { Write-Error "Missing $activePath"; exit 1 }
if (!(Test-Path $archivePath)) { New-Item -ItemType Directory -Path $archivePath | Out-Null }
if (!(Test-Path $manifest)) { New-Item -ItemType File -Path $manifest -Force | Out-Null }

# Determine stage id
if ($StageId -le 0) {
  $maxId = 0
  $existing = Get-ChildItem -Path $archivePath -Filter 'Claude_Update_*.txt' -ErrorAction SilentlyContinue
  foreach ($f in $existing) {
    if ($f.Name -match '^Claude_Update_(\d{3})') {
      $n = [int]$Matches[1]; if ($n -gt $maxId) { $maxId = $n }
    }
  }
  $id = $maxId + 1
} else {
  $id = $StageId
}

# Normalise suffix (optional)
if ($Suffix) {
  if (-not $Suffix.StartsWith('_')) { $Suffix = '_' + $Suffix }
  $Suffix = $Suffix -replace '\s+', '_'  # no spaces in filenames
}

# Archive filename
$destName = ('Claude_Update_{0:D' + $Pad + '}{1}.txt') -f $id, $Suffix
$dest = Join-Path $archivePath $destName
Copy-Item $activePath $dest -Force

# Manifest entry (JSONL, one line per archived file)
$now   = Get-Date -Format o
$bytes = (Get-Item $dest).Length
$sha   = (Get-FileHash -Algorithm SHA256 $dest).Hash.ToLower()
$entry = @{
  id     = $id
  date   = $now
  file   = $destName
  path   = (Resolve-Path $dest).Path
  bytes  = $bytes
  sha256 = $sha
} | ConvertTo-Json -Compress
Add-Content -Path $manifest -Value $entry

Write-Host "Archived  -> $dest"
Write-Host "Manifest  -> $manifest (appended)"
