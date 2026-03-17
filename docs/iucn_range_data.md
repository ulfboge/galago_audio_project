## IUCN range data storage + conversion

### Where to store the zipped shapefiles

Recommended layout (keep these **out of git**):

- `data/iucn/ranges_zips/`  
  Put the original IUCN `*.zip` downloads here (one zip per species or per batch, depending on what you downloaded).

- `data/iucn/ranges_geojson/`  
  Output GeoJSONs (WGS84/EPSG:4326, RFC7946) for use in the web app.

The zips can be large; consider keeping a backup copy outside the repo as well.

### Convert zips → GeoJSON (GDAL/ogr2ogr)

This repo includes a converter script:

- `scripts/convert_iucn_range_zips_to_geojson.ps1`

Example:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "c:\Users\galag\GitHub\galago_audio_project\scripts\convert_iucn_range_zips_to_geojson.ps1" `
  -InputDir "c:\Users\galag\GitHub\galago_audio_project\data\iucn\ranges_zips" `
  -OutputDir "c:\Users\galag\GitHub\galago_audio_project\data\iucn\ranges_geojson"
```

Notes:
- Requires GDAL on PATH (the `ogr2ogr` command must work).
- Outputs one `.geojson` per zip (name based on zip filename).

