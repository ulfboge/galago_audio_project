param(
  [switch]$WhatIf
)

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

Write-Host "Repo root: $root"
Write-Host "Updating Python paths in scripts\*.py" -ForegroundColor Cyan
if ($WhatIf) { Write-Host "WHATIF mode enabled (no changes will be made)." -ForegroundColor Yellow }

$files = Get-ChildItem -Path (Join-Path $root "scripts") -Filter *.py

$replacements = @(
  # data folders
  @{ from = [regex]::Escape('Path(r"C:\Users\galag\GitHub\galago_audio_project\audio_raw")');         to = 'PROJECT_ROOT / "data" / "raw_audio"' },
  @{ from = [regex]::Escape('Path(r"C:\Users\galag\GitHub\galago_audio_project\melspectrograms")');  to = 'PROJECT_ROOT / "data" / "melspectrograms"' },

  # models (top6/top7)
  @{ from = [regex]::Escape('Path(r"C:\Users\galag\GitHub\galago_audio_project\models_top6\galago_cnn_top6_best.keras")');
     to   = 'PROJECT_ROOT / "models" / "top6" / "galago_cnn_top6_best.keras"' },

  @{ from = [regex]::Escape('Path(r"C:\Users\galag\GitHub\galago_audio_project\models_top7\galago_cnn_top7_best.keras")');
     to   = 'PROJECT_ROOT / "models" / "top7" / "galago_cnn_top7_best.keras"' },

  # any remaining relative models_top7 usage
  @{ from = [regex]::Escape('PROJECT_ROOT / "models_top7" / "galago_cnn_top7_best.keras"');
     to   = 'PROJECT_ROOT / "models" / "top7" / "galago_cnn_top7_best.keras"' },

  # training output folder for top7
  @{ from = [regex]::Escape('Path(r"C:\Users\galag\GitHub\galago_audio_project\models_top7")');
     to   = 'PROJECT_ROOT / "models" / "top7"' }
)

foreach ($f in $files) {
  $p = $f.FullName
  $text = Get-Content $p -Raw
  $orig = $text

  foreach ($r in $replacements) {
    $text = [regex]::Replace($text, $r.from, $r.to)
  }

  if ($text -ne $orig) {
    Write-Host "Update: $($f.Name)"
    if (-not $WhatIf) {
      Set-Content -Path $p -Value $text -Encoding UTF8
    }
  }
}

Write-Host "`nDone. Re-run your grep to confirm:" -ForegroundColor Green
Write-Host 'Select-String -Path scripts\*.py -Pattern "audio_raw|melspectrograms|models_top"' -ForegroundColor Gray
