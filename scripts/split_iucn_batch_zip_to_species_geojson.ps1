param(
  [Parameter(Mandatory = $true)]
  [string]$ZipPath,

  [Parameter(Mandatory = $true)]
  [string]$OutputDir,

  [Parameter(Mandatory = $false)]
  [string]$LayerName = "data_0",

  [Parameter(Mandatory = $false)]
  [string]$SpeciesField = "SCI_NAME",

  [Parameter(Mandatory = $false)]
  [int]$MaxSpecies = 0,

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

  $candidates = @()
  if ($env:OSGEO4W_ROOT) {
    $candidates += (Join-Path $env:OSGEO4W_ROOT "bin\ogr2ogr.exe")
  }
  $candidates += "C:\OSGeo4W\bin\ogr2ogr.exe"
  $candidates += "C:\OSGeo4W64\bin\ogr2ogr.exe"

  foreach ($p in $candidates) {
    if (Test-Path -LiteralPath $p) { return $p }
  }

  throw "Could not find ogr2ogr. Add it to PATH or pass -Ogr2OgrPath <full path>."
}

function Resolve-OgrInfoPath([string]$Ogr2OgrExe) {
  $binDir = Split-Path -Parent $Ogr2OgrExe
  $ogrinfo = Join-Path $binDir "ogrinfo.exe"
  if (Test-Path -LiteralPath $ogrinfo) { return $ogrinfo }
  throw "Could not find ogrinfo.exe next to ogr2ogr.exe at: $binDir"
}

function Resolve-ProjDataDir([string]$Ogr2OgrExe) {
  $binDir = Split-Path -Parent $Ogr2OgrExe
  $candidates = @()
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

function To-SafeFileBase([string]$SciName) {
  # "Paragalago rondoensis" -> "Paragalago_rondoensis"
  $s = $SciName.Trim()
  $s = $s -replace "\s+", "_"
  # Remove anything that's not alnum, underscore, or dash
  $s = $s -replace "[^A-Za-z0-9_-]", ""
  return $s
}

if (-not (Test-Path -LiteralPath $ZipPath)) {
  throw "Zip not found: $ZipPath"
}

Ensure-Dir $OutputDir

$ogr2ogrExe = Resolve-Ogr2OgrPath $Ogr2OgrPath
$ogrinfoExe = Resolve-OgrInfoPath $ogr2ogrExe
$projDataDir = Resolve-ProjDataDir $ogr2ogrExe

if ($projDataDir) {
  Write-Host "Using PROJ data dir: $projDataDir"
} else {
  Write-Host "WARN: Could not auto-detect PROJ data dir (proj.db)."
}

# GDAL virtual path to read shapefile(s) inside zip
$zipAbs = (Resolve-Path -LiteralPath $ZipPath).Path
$zipVsizip = "/vsizip/" + ($zipAbs -replace "\\", "/")

Write-Host "Splitting IUCN batch zip into per-species GeoJSON"
Write-Host "Zip:     $zipAbs"
Write-Host "Layer:   $LayerName"
Write-Host "Field:   $SpeciesField"
Write-Host "Output:  $OutputDir"
if ($MaxSpecies -gt 0) { Write-Host "Max:     $MaxSpecies (limit)" }
Write-Host ""

# Extract unique species names from ogrinfo output (feature count is usually manageable).
$lines = & $ogrinfoExe -ro -q -al -geom=NO `
  --config "PROJ_DATA" $projDataDir `
  --config "PROJ_LIB" $projDataDir `
  $zipVsizip 2>$null

$pattern = [regex]::Escape($SpeciesField) + " \(String\) = (.+)$"
$species = @()
foreach ($ln in $lines) {
  $m = [regex]::Match($ln, $pattern)
  if ($m.Success) {
    $species += $m.Groups[1].Value.Trim()
  }
}

$species = $species | Sort-Object -Unique
if ($species.Count -eq 0) {
  throw "No species values found. Check SpeciesField='$SpeciesField' and LayerName='$LayerName'."
}

Write-Host "Found $($species.Count) unique species names."

$index = @()
$i = 0
foreach ($sp in $species) {
  $i += 1
  if ($MaxSpecies -gt 0 -and $i -gt $MaxSpecies) { break }

  $base = To-SafeFileBase $sp
  $outPath = Join-Path $OutputDir ($base + ".geojson")

  if ((Test-Path -LiteralPath $outPath) -and (-not $Overwrite)) {
    Write-Host "SKIP (exists): $base"
    continue
  }
  if ((Test-Path -LiteralPath $outPath) -and $Overwrite) {
    Remove-Item -LiteralPath $outPath -Force
  }

  # Escape single quotes for OGR where clause
  $spEsc = $sp -replace "'", "''"
  $where = "$SpeciesField = '$spEsc'"

  Write-Host ("[{0}/{1}] {2}" -f $i, $species.Count, $sp)

  & $ogr2ogrExe `
    --config "PROJ_DATA" $projDataDir `
    --config "PROJ_LIB" $projDataDir `
    -overwrite `
    -f "GeoJSON" `
    -t_srs "EPSG:4326" `
    -lco "RFC7946=YES" `
    -lco "COORDINATE_PRECISION=6" `
    -skipfailures `
    -where $where `
    $outPath `
    $zipVsizip `
    $LayerName

  if ($LASTEXITCODE -ne 0) {
    throw "ogr2ogr failed (exit code $LASTEXITCODE) for species: $sp"
  }

  $index += [PSCustomObject]@{
    sci_name = $sp
    file = (Split-Path -Leaf $outPath)
  }
}

$indexPath = Join-Path $OutputDir "_index.csv"
$index | Export-Csv -NoTypeInformation -Encoding UTF8 $indexPath
Write-Host ""
Write-Host "Wrote index: $indexPath"
Write-Host "Done."

