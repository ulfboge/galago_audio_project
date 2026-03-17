# IUCN range data (Galagidae batch) — handover (2026-01-08)

## Goal
Take an IUCN “range data” **batch zip** (Galagidae), convert it to GeoJSON, and split it into **one GeoJSON per species** for use in a future web app (show predicted species distribution polygon on a map).

## Inputs / outputs (current workspace)
- **Input zip (batch)**:  
  `c:\Users\galag\GitHub\galago_audio_project\data\iucn\ranges_zips\redlist_species_data_88eab258-9991-40b4-ba6c-48d459c476ee.zip`

- **Batch GeoJSON (single file)** (from zip):  
  `c:\Users\galag\GitHub\galago_audio_project\data\iucn\ranges_geojson\redlist_species_data_88eab258-9991-40b4-ba6c-48d459c476ee.geojson`

- **Per-species GeoJSON directory** (split output):  
  `c:\Users\galag\GitHub\galago_audio_project\data\iucn\ranges_geojson\by_species\`

- **Per-species index CSV**:  
  `c:\Users\galag\GitHub\galago_audio_project\data\iucn\ranges_geojson\by_species\_index.csv`

> Note: `data/iucn/` is ignored by git (licensed/large).

## What was discovered about the batch
- The zip contains a single layer named **`data_0`**
- Species name field is **`SCI_NAME`** (e.g. `"Paragalago rondoensis"`)
- This Galagidae batch has **19 unique `SCI_NAME` values** (the splitter script prints them during export).

## GDAL/PROJ issues encountered (and how it was handled)
On Windows, GDAL can accidentally pick up **pyproj’s** `proj.db`, causing errors like:
- `DATABASE.LAYOUT.VERSION.MINOR = 4 whereas a number >= 6 is expected`

To avoid this, the scripts:
- auto-detect `ogr2ogr.exe` (OSGeo4W / QGIS installs)
- auto-detect a compatible PROJ data dir (`proj.db`) near that GDAL install
- run GDAL with:
  - `--config PROJ_DATA <detected_proj_dir>`
  - `--config PROJ_LIB <detected_proj_dir>`

## Scripts created/used today

### 1) Convert zips → GeoJSON (batch file)
- **Script**: `scripts/convert_iucn_range_zips_to_geojson.ps1`
- **Purpose**: For each `*.zip` in an input folder, extract the largest `.shp` and write a GeoJSON (RFC7946) to an output folder.

Example run (PowerShell):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "c:\Users\galag\GitHub\galago_audio_project\scripts\convert_iucn_range_zips_to_geojson.ps1" `
  -InputDir  "c:\Users\galag\GitHub\galago_audio_project\data\iucn\ranges_zips" `
  -OutputDir "c:\Users\galag\GitHub\galago_audio_project\data\iucn\ranges_geojson" `
  -Overwrite
```

If needed (PATH issues), pass an explicit GDAL path:

```powershell
  -Ogr2OgrPath "C:\OSGeo4W\bin\ogr2ogr.exe"
```

### 2) Split a batch zip → per-species GeoJSONs
- **Script**: `scripts/split_iucn_batch_zip_to_species_geojson.ps1`
- **Purpose**: Read the batch zip directly via GDAL `/vsizip/…`, discover unique `SCI_NAME` values, and export **one GeoJSON per species** plus `_index.csv`.
- **Defaults**:
  - `LayerName = "data_0"`
  - `SpeciesField = "SCI_NAME"`

The run that produced the 19 per-species GeoJSON files:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "c:\Users\galag\GitHub\galago_audio_project\scripts\split_iucn_batch_zip_to_species_geojson.ps1" `
  -ZipPath   "c:\Users\galag\GitHub\galago_audio_project\data\iucn\ranges_zips\redlist_species_data_88eab258-9991-40b4-ba6c-48d459c476ee.zip" `
  -OutputDir "c:\Users\galag\GitHub\galago_audio_project\data\iucn\ranges_geojson\by_species" `
  -Overwrite
```

Optional: limit during testing:

```powershell
  -MaxSpecies 5
```

### 3) Build a **model-label → GeoJSON** lookup (label alignment)
Why: your model labels use underscores (e.g. `Paragalago_rondoensis`) and there are a couple of naming mismatches vs IUCN attributes/files (e.g. `Galagoides_demidovii` vs IUCN `Galagoides demidoff`, and `Galagoides_sp_nov` vs `Galagoides kumbirensis`).

- **Script**: `scripts/build_iucn_geojson_index_by_label.py`
- **Inputs**:
  - `models/all_species/class_names_19.json`
  - `data/iucn/ranges_geojson/by_species/_index.csv`
- **Output (tracked, outside `data/iucn/`)**:
  - `data/iucn_geojson_index_by_label.json`

Run:

```powershell
python "c:\Users\galag\GitHub\galago_audio_project\scripts\build_iucn_geojson_index_by_label.py"
```

This writes a JSON mapping for *all* model labels and records any missing labels (currently `missing_count = 0`).

## Resulting per-species files (now present)
Directory:  
`data\iucn\ranges_geojson\by_species\`

Includes:
- `_index.csv`
- `Euoticus_elegantulus.geojson`
- `Euoticus_pallidus.geojson`
- `Galago_gallarum.geojson`
- `Galago_matschiei.geojson`
- `Galago_moholi.geojson`
- `Galago_senegalensis.geojson`
- `Galagoides_demidoff.geojson`
- `Galagoides_kumbirensis.geojson`
- `Galagoides_thomasi.geojson`
- `Otolemur_crassicaudatus.geojson`
- `Otolemur_garnettii.geojson`
- `Paragalago_cocos.geojson`
- `Paragalago_granti.geojson`
- `Paragalago_orinus.geojson`
- `Paragalago_rondoensis.geojson`
- `Paragalago_zanzibaricus.geojson`
- `Sciurocheirus_alleni.geojson`
- `Sciurocheirus_gabonensis.geojson`
- `Sciurocheirus_makandensis.geojson`

### 4) Polygon-based location priors (integrated)
- **Script**: `scripts/iucn_polygon_priors.py`
- **Purpose**: Pure-Python point-in-polygon engine (no shapely/geopandas) that checks if a lat/lon point is inside a species' IUCN range polygon.
- **Integration**: Used by `context_reranker.py` when `lat`/`lon` are provided.

### 5) Pipeline integration
- **Updated**: `scripts/predict_3stage_with_context.py`
- **New flags**: `--lat <float> --lon <float>`
- **Behavior**: When lat/lon provided, `get_location_prior()` uses polygon checks instead of country/region text matching.
  - Inside polygon → prior = 1.0
  - Outside polygon → prior = 0.1
  - No coordinates → prior = 0.5 (neutral)

### 6) Comparative evaluation
- **Script**: `scripts/evaluate_polygon_priors_impact.py`
- **Purpose**: Run predictions on a test set with/without polygon priors and compare metrics side-by-side.

Example:

```powershell
python "c:\Users\galag\GitHub\galago_audio_project\scripts\evaluate_polygon_priors_impact.py" `
  --filelist "c:\Users\galag\GitHub\galago_audio_project\data\splits\raw_audio_holdout_filelist.txt" `
  --tanzania-lat -6.8 `
  --tanzania-lon 39.28
```

Outputs:
- `outputs/evaluations/baseline_no_polygons.csv`
- `outputs/evaluations/with_polygon_priors.csv`
- `outputs/evaluations/polygon_priors_evaluation_summary.json`

## Notes for tomorrow (recommended next steps)
- **Label alignment**: confirm your classifier label naming matches IUCN names (spaces vs underscores, synonyms, “sp. nov.” cases). If needed, create a mapping table and/or rename the per-species GeoJSONs to exactly match model class names.
- **Web app query strategy** (fast):
  - Option A: load **one** combined GeoJSON with all species + build a client-side spatial index
  - Option B: load only the predicted species’ GeoJSON by filename using `_index.csv` (simple + small transfer)
- **Geometry simplification**: for web performance, consider simplifying polygons (e.g., `-simplify` in ogr2ogr or mapshaper) and/or converting to TopoJSON.

### Immediate (quantify polygon impact)
- **Run the evaluation**: Use `evaluate_polygon_priors_impact.py` on your holdout set to see if polygon priors improve accuracy/coverage.
- **Location mapping**: If you have explicit lat/lon for recordings, create a JSON mapping file (`filepath -> {lat, lon}`) for more accurate polygon tests.

### Iterative training (self-supervised learning)
- **High-confidence labeling**: Run pipeline on raw audio, extract clips where model is very confident (>90%), use `scripts/ingest_raw_audio_to_training_mels.py` to add them to training set.
- **Retrain**: This addresses domain shift (training PNGs vs inference WAVs) and can fix 0%-accuracy species.

### Iterative UX (Merlin-style)
- **Multi-pass recognition**: First guess (sound only) → user provides location → re-rank with polygon priors → if still uncertain, ask clarifying questions (habitat, tail appearance, etc.).

