param([switch]$WhatIf)

$ErrorActionPreference = "Stop"

function Ensure-Dir([string]$p) {
  if (!(Test-Path $p)) { New-Item -ItemType Directory -Path $p | Out-Null }
}

function Move-Rename([string]$src, [string]$dst) {
  if (!(Test-Path $src)) { return }
  $dstParent = Split-Path -Parent $dst
  Ensure-Dir $dstParent

  # If destination exists, stop (safer than merging silently)
  if (Test-Path $dst) {
    throw "Destination already exists: $dst (won't merge). Rename/remove it first."
  }

  Write-Host "Move: $src -> $dst"
  if ($WhatIf) {
    Move-Item -Path $src -Destination $dst -Force -WhatIf
  } else {
    Move-Item -Path $src -Destination $dst -Force
  }
}

# Run from repo root
if (!(Test-Path ".\README.md")) {
  throw "Run this from the repo root (folder that contains README.md)."
}

Write-Host "Reorganizing repo (v2)..." -ForegroundColor Cyan
if ($WhatIf) { Write-Host "WHATIF mode enabled (no changes will be made)." -ForegroundColor Yellow }

# Ensure high-level structure
Ensure-Dir ".\data"
Ensure-Dir ".\models"
Ensure-Dir ".\outputs"
Ensure-Dir ".\tools"

# 1) Data folders (move/rename)
Move-Rename ".\audio_raw"       ".\data\raw_audio"
Move-Rename ".\melspectrograms" ".\data\melspectrograms"

# 2) Models folders (move/rename)
# NOTE: We do NOT try to move .\models into .\models\base (impossible).
# If you have an existing root .\models you want to keep, we rename it first,
# but ONLY if it's not already the destination structure we just created.
# We'll move "legacy_models" contents into models\base instead (next block).

# Move the run-specific model dirs into models\
Move-Rename ".\models_top6" ".\models\top6"
Move-Rename ".\models_top7" ".\models\top7"

# 3) Legacy models handling (optional):
# If you previously had a folder named ".\models" containing artifacts (old runs),
# put those artifacts in ".\models\base" WITHOUT moving the whole ".\models" folder.
# We interpret "legacy" as any files directly in ".\models" root that are not top6/top7/base dirs.
Ensure-Dir ".\models\base"

# Move loose files from .\models\ into .\models\base (but keep folders top6/top7/base)
$legacyItems = Get-ChildItem ".\models" -Force |
  Where-Object {
    $_.Name -notin @("top6","top7","base") -and $_.FullName -notmatch "\\models\\(top6|top7|base)(\\|$)"
  }

foreach ($it in $legacyItems) {
  $dst = Join-Path ".\models\base" $it.Name
  if (Test-Path $dst) { throw "Legacy destination exists: $dst (won't overwrite)." }
  Write-Host "Move legacy: $($it.FullName) -> $dst"
  if ($WhatIf) {
    Move-Item -Path $it.FullName -Destination $dst -Force -WhatIf
  } else {
    Move-Item -Path $it.FullName -Destination $dst -Force
  }
}

# 4) Outputs (predictions)
Ensure-Dir ".\outputs\predictions"
Move-Rename ".\predictions_top6.csv" ".\outputs\predictions\predictions_top6.csv"
Move-Rename ".\predictions_top7.csv" ".\outputs\predictions\predictions_top7.csv"

# 5) Tools / helper lists
Move-Rename ".\filelist.txt" ".\tools\filelist.txt"
Move-Rename ".\filelist.bat" ".\tools\filelist.bat"

Write-Host "`nDone." -ForegroundColor Green
Write-Host "Next: update script paths to data/raw_audio, data/melspectrograms, models/top7, outputs/predictions."
Write-Host "Tip: run first with:  powershell -ExecutionPolicy Bypass -File .\scripts\repo_reorg_v2.ps1 -WhatIf"
