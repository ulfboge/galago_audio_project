param(
  [Parameter(Mandatory = $true)]
  [string]$InputDir,

  [Parameter(Mandatory = $true)]
  [string]$OutputDir,

  [Parameter(Mandatory = $false)]
  [string]$TempDir = (Join-Path $env:TEMP "iucn_range_extract"),

  [Parameter(Mandatory = $false)]
  [string]$Ogr2OgrPath = "",

  [Parameter(Mandatory = $false)]
  [switch]$Overwrite
)

$ErrorActionPreference = "Stop"

function Ensure-Dir([string]$Path) {
  if (-not (Test-Path -LiteralPath $Path)) {
    New-Item -ItemType Directory -Path $Path | Out-Null
  }
}

function Resolve-Ogr2OgrPath([string]$ExplicitPath) {
  if ($ExplicitPath -and (Test-Path -LiteralPath $ExplicitPath)) {
    return $ExplicitPath
  }

  $cmd = Get-Command "ogr2ogr" -ErrorAction SilentlyContinue
  if ($cmd) {
    return $cmd.Source
  }

  # Try common install locations (OSGeo4W / QGIS)
  $candidates = @()

  if ($env:OSGEO4W_ROOT) {
    $candidates += (Join-Path $env:OSGEO4W_ROOT "bin\ogr2ogr.exe")
  }

  $candidates += "C:\OSGeo4W\bin\ogr2ogr.exe"
  $candidates += "C:\OSGeo4W64\bin\ogr2ogr.exe"

  foreach ($root in @("C:\Program Files", "C:\Program Files (x86)")) {
    if (-not (Test-Path -LiteralPath $root)) { continue }
    $qgisDirs = Get-ChildItem -LiteralPath $root -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -like "QGIS*" }
    foreach ($d in $qgisDirs) {
      $candidates += (Join-Path $d.FullName "bin\ogr2ogr.exe")
      $candidates += (Join-Path $d.FullName "apps\gdal\bin\ogr2ogr.exe")
    }
  }

  foreach ($p in $candidates) {
    if (Test-Path -LiteralPath $p) {
      return $p
    }
  }

  $msg = @(
    "Could not find ogr2ogr.",
    "Either add GDAL/ogr2ogr to PATH or pass -Ogr2OgrPath <full path to ogr2ogr.exe>.",
    "Examples:",
    "  -Ogr2OgrPath 'C:\OSGeo4W\bin\ogr2ogr.exe'",
    "  -Ogr2OgrPath 'C:\Program Files\QGIS 3.34.0\bin\ogr2ogr.exe'"
  ) -join "`n"

  throw $msg
}

$ogr2ogrExe = Resolve-Ogr2OgrPath $Ogr2OgrPath

function Resolve-ProjDataDir([string]$Ogr2OgrExe) {
  # Try to find a compatible proj.db shipped alongside the selected GDAL install,
  # to avoid conflicts with other PROJ installations (e.g. pyproj).
  $binDir = Split-Path -Parent $Ogr2OgrExe
  $candidates = @()

  # Walk up a few levels and check common layouts
  $p0 = Split-Path -Parent $binDir
  $p1 = Split-Path -Parent $p0
  $p2 = Split-Path -Parent $p1

  foreach ($base in @($p0, $p1, $p2)) {
    if (-not $base) { continue }
    $candidates += (Join-Path $base "share\proj")
    $candidates += (Join-Path $base "apps\proj\share\proj")
    $candidates += (Join-Path $base "Library\share\proj")
  }

  foreach ($d in $candidates) {
    if ((Test-Path -LiteralPath $d) -and (Test-Path -LiteralPath (Join-Path $d "proj.db"))) {
      return $d
    }
  }
  return ""
}

$projDataDir = Resolve-ProjDataDir $ogr2ogrExe
if ($projDataDir) {
  Write-Host "Using PROJ data dir: $projDataDir"
} else {
  Write-Host "WARN: Could not auto-detect PROJ data dir (proj.db). ogr2ogr may fail with EPSG transforms."
}
Ensure-Dir $InputDir
Ensure-Dir $OutputDir
Ensure-Dir $TempDir

$zipFiles = Get-ChildItem -LiteralPath $InputDir -File -Filter "*.zip"
if ($zipFiles.Count -eq 0) {
  Write-Host "No .zip files found in: $InputDir"
  exit 0
}

Write-Host "Converting IUCN range zips to GeoJSON"
Write-Host "Input:  $InputDir"
Write-Host "Output: $OutputDir"
Write-Host "Temp:   $TempDir"
Write-Host ""

foreach ($zip in $zipFiles) {
  $baseName = [IO.Path]::GetFileNameWithoutExtension($zip.Name)
  $outPath = Join-Path $OutputDir ($baseName + ".geojson")

  if ((Test-Path -LiteralPath $outPath) -and (-not $Overwrite)) {
    Write-Host "SKIP (exists): $($zip.Name)"
    continue
  }
  if ((Test-Path -LiteralPath $outPath) -and $Overwrite) {
    # GeoJSON driver won't overwrite an existing file; remove it first.
    Remove-Item -LiteralPath $outPath -Force
  }

  $extractDir = Join-Path $TempDir $baseName
  if (Test-Path -LiteralPath $extractDir) {
    Remove-Item -LiteralPath $extractDir -Recurse -Force
  }
  New-Item -ItemType Directory -Path $extractDir | Out-Null

  Write-Host "Extract: $($zip.Name)"
  Expand-Archive -LiteralPath $zip.FullName -DestinationPath $extractDir -Force

  $shps = Get-ChildItem -LiteralPath $extractDir -Recurse -File -Filter "*.shp"
  if ($shps.Count -eq 0) {
    Write-Host "WARN: No .shp found inside $($zip.Name). Skipping."
    continue
  }

  # If multiple SHPs exist, pick the largest (typical when zips include multiple layers).
  $shp = $shps | Sort-Object Length -Descending | Select-Object -First 1

  Write-Host "Convert: $($shp.Name) -> $([IO.Path]::GetFileName($outPath))"

  # Convert to WGS84 GeoJSON (RFC7946)
  # -t_srs EPSG:4326: ensure output CRS is WGS84 lat/lon (web map friendly)
  # -lco RFC7946=YES: strict GeoJSON spec suitable for web maps
  # -lco COORDINATE_PRECISION=6: reduce size without big geometry distortion
  # -skipfailures: continue even if a few geometries are invalid
  # Ensure ogr2ogr uses the PROJ database that matches the selected GDAL build.
  # This avoids common Windows conflicts where PROJ points at pyproj's proj.db.
  & $ogr2ogrExe `
    --config "PROJ_DATA" $projDataDir `
    --config "PROJ_LIB" $projDataDir `
    -overwrite `
    -f "GeoJSON" `
    -t_srs "EPSG:4326" `
    -lco "RFC7946=YES" `
    -lco "COORDINATE_PRECISION=6" `
    -skipfailures `
    $outPath `
    $shp.FullName

  if ($LASTEXITCODE -ne 0) {
    throw "ogr2ogr failed (exit code $LASTEXITCODE) for: $($zip.Name)"
  }

  # Clean up extraction folder to keep temp small
  Remove-Item -LiteralPath $extractDir -Recurse -Force
}

Write-Host ""
Write-Host "Done."

