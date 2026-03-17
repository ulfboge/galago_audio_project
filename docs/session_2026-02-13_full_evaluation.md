# Session Summary — 2026-02-13: Full 69-File Evaluation

## What was done

1. **Full 69-file 3-stage predictions**
   - **Baseline (17-class):**  
     `predict_3stage_with_context.py` (no `--classifier-model`) on all WAVs in `data/raw_audio`  
     → `outputs/predictions/predictions_3stage_baseline_full_temp0212.csv`
   - **Improved (19-class):**  
     Same, with `--classifier-model models/all_species/galago_cnn_all_19classes_improved_best.keras`  
     → `outputs/predictions/predictions_3stage_improved_full_temp0212.csv`
   - Settings: `--classifier-threshold 0.2 --temperature 0.212` for both.

2. **Accuracy analysis**
   - `analyze_prediction_accuracy.py` run on both CSVs (true labels from path/folder names).

---

## Results (69 files, raw_audio evaluation set)

| Metric        | Baseline (17-class) | Improved (19-class) |
|---------------|----------------------|----------------------|
| **Top-1 correct** | 14 (20.3%)        | **16 (23.2%)**       |
| **Top-3 accuracy** | 36.2%             | 36.2%                |
| Uncertain     | 0                   | 0                    |

### Per-species (top-1 → top-3)

| Species                   | Baseline (17)      | Improved (19)       |
|---------------------------|--------------------|---------------------|
| Galagoides_sp_nov         | 10/24 (41.7%) / 50% | 9/24 (37.5%) / 45.8% |
| Otolemur_crassicaudatus   | 0/8 (0%) / 12.5%   | **2/8 (25%) / 37.5%** |
| Otolemur_garnettii        | 1/8 (12.5%) / 25%  | 1/8 (12.5%) / 25%   |
| Paragalago_granti         | 1/3 (33.3%) / 33.3% | 1/3 (33.3%) / 33.3% |
| Paragalago_orinus         | 2/8 (25%) / 37.5%  | 0/8 (0%) / 25%      |
| Paragalago_rondoensis     | 0/11 (0%) / 45.5%  | **2/11 (18.2%) / 27.3%** |
| Paragalago_zanzibaricus   | 0/7 (0%) / 14.3%   | **1/7 (14.3%) / 42.9%** |

- **19-class gains:** +2 top-1 overall; better for Otolemur_crassicaudatus, Paragalago_rondoensis, Paragalago_zanzibaricus.
- **19-class regressions:** Galagoides_sp_nov (slightly lower top-1); Paragalago_orinus (worse top-1 and top-3).

---

## Boosted 19-class (2026-03-06)

After retraining with `SPECIES_OVERSAMPLE_BOOST` (Galagoides_sp_nov, Paragalago_orinus ×1.5), the same 69-file set was evaluated with the new best checkpoint.

| Metric           | Baseline (17) | Improved (19) | **Boosted (19)** |
|------------------|---------------|---------------|-------------------|
| **Top-1 correct**| 14 (20.3%)    | 16 (23.2%)    | **19 (27.5%)**    |
| **Top-3 accuracy**| 36.2%         | 36.2%         | 31.9%             |
| Uncertain        | 0             | 0             | 0                 |

### Per-species (boosted)

| Species                 | Boosted top-1      | Boosted top-3 |
|-------------------------|--------------------|---------------|
| Galagoides_sp_nov       | 10/24 (41.7%)      | 41.7%         |
| Otolemur_crassicaudatus | 0/8 (0%)           | 25.0%         |
| Otolemur_garnettii      | 1/8 (12.5%)        | 25.0%         |
| Paragalago_granti       | 1/3 (33.3%)        | 33.3%         |
| Paragalago_orinus       | 2/8 (25.0%)        | 25.0%         |
| Paragalago_rondoensis   | 1/11 (9.1%)        | 9.1%          |
| Paragalago_zanzibaricus | **4/7 (57.1%)**    | 57.1%         |

- **Boosted vs improved (19):** +3 top-1 (16→19). Paragalago_zanzibaricus improves (1/7→4/7); Paragalago_orinus recovers (0/8→2/8); Galagoides_sp_nov back to 10/24. Otolemur_crassicaudatus and Paragalago_rondoensis drop slightly vs unboosted 19-class.
- **Boosted vs baseline (17):** +5 top-1 (14→19). Best overall top-1 on this set.

---

## Recommendation

- **Default classifier:** For **best top-1 on the 69-file raw_audio set**, use the **boosted 19-class** model:  
  `--classifier-model models/all_species/galago_cnn_all_19classes_improved_best.keras --temperature 0.212`.  
  (The current best checkpoint is from the boosted run; same path.)
- **If you need higher top-3** on this set, the 17-class baseline (36.2% top-3) or unboosted 19-class may be preferable; boosted is 31.9% top-3.
- **17-class** remains available as the script default when `--classifier-model` is not set.

---

## Next steps completed (2026-02-13 follow-up)

1. **Full dataset + polygon priors**
   - Created `data/splits/raw_audio_full_filelist.txt` (69 files).
   - Ran `evaluate_polygon_priors_impact.py --filelist data/splits/raw_audio_full_filelist.txt --out-dir outputs/evaluations/polygon_priors_full`.
   - Result: With a single Tanzania point for all files, baseline vs polygon metrics were identical (20.3% top-1, 36.2% top-3). Location prior was uniform, so re-ranking did not change any prediction. To see an impact, use per-file locations (different sites) so species priors differ.
   - CSVs: `outputs/evaluations/polygon_priors_full/baseline_no_polygons.csv`, `with_polygon_priors.csv`.
2. **Targeted oversampling**
   - Added `SPECIES_OVERSAMPLE_BOOST` in `scripts/train_cnn_all_species_improved.py` (e.g. `Galagoides_sp_nov: 1.5`, `Paragalago_orinus: 1.5`). When retraining, those species get 50% more target samples. Set to `{}` to disable.
3. **Boosted retrain and evaluation (2026-03-06)**
   - Retrain completed; 69-file evaluation run with boosted 19-class → `outputs/predictions/predictions_3stage_improved_boosted_full_temp0212.csv`. Top-1: 19/69 (27.5%); see "Boosted 19-class" section above.

## Next steps completed (per-file locations + default model)

1. **Per-file location support**
   - **`predict_3stage_with_context.py`:** Added `--location-map-json <path>`. JSON keys = file paths, values = `{lat, lon}`. When set, each file uses its own lat/lon for context re-ranking.
   - **`data/recording_locations.json`:** Created from the 69-file list: Pugu/Rondo/Pande → Tanzania (-6.8, 39.28), others → Kenya (-1.3, 36.8).
   - **`evaluate_polygon_priors_impact.py`:** When `--location-map-json` is passed and `use_polygons` is True, the script passes `--location-map-json` to the predictor instead of a single `--lat`/`--lon`.
   - **Result:** Re-ran evaluation with per-file locations; top-1 accuracy stayed 14/69 (20.3%) for both baseline and with-polygons (17-class model). So on this set, polygon priors did not change predictions; pipeline and location map are in place for when location does help.
2. **Default classifier**
   - Default is now **19-class improved** when `galago_cnn_all_19classes_improved_best.keras` exists (`PREFER_19_IMPROVED = True`). Otherwise the script falls back to 17-class. Use `--classifier-model` to override.

## Next steps (for another session)

1. **Optional:** Retrain again with different `SPECIES_OVERSAMPLE_BOOST` or disable it and compare.
2. **Temperature scaling**
   - Already in pipeline (default T=0.212). Task "Implement temperature scaling" in Priority 6 can be marked done once confirmed in code.

---

## Paths

- **Prediction CSVs:**  
  `outputs/predictions/predictions_3stage_baseline_full_temp0212.csv`  
  `outputs/predictions/predictions_3stage_improved_full_temp0212.csv`  
  `outputs/predictions/predictions_3stage_improved_boosted_full_temp0212.csv` (boosted 19-class, best top-1)
- **Polygon priors evaluation (69-file):**  
  `outputs/evaluations/polygon_priors_full/baseline_no_polygons.csv`, `with_polygon_priors.csv`, `polygon_priors_evaluation_summary.json`
- **Full filelist:**  
  `data/splits/raw_audio_full_filelist.txt` (69 WAV paths)
- **Per-file locations:**  
  `data/recording_locations.json` (path → {lat, lon}); use with `--location-map-json`
- **Models:**  
  `models/all_species/galago_cnn_all_17classes_best.keras` (default)  
  `models/all_species/galago_cnn_all_19classes_improved_best.keras` (optional)
- **Pipeline:**  
  `scripts/predict_3stage_with_context.py` (supports `--classifier-model`, `--temperature`, `--classifier-threshold`)
- **Trainer (targeted oversampling):**  
  `scripts/train_cnn_all_species_improved.py` — set `SPECIES_OVERSAMPLE_BOOST` and retrain to test.
