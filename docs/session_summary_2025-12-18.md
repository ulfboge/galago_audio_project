# Session summary — 2025-12-18

## Goal (what we focused on today)

- Fix evaluation + pipeline issues blocking iteration.
- Reduce domain shift by aligning training data with the real evaluation WAVs.
- Make evaluation **leakage-free** using a proper holdout split by WAV file.
- Train a **17-class classifier with `not_galago` background** and evaluate on holdout.

---

## Key outcomes (high-signal)

- **Sanity overfit test now passes** (training pipeline can learn).
- **3-stage prediction CSV now retains raw top-1** even when output is `uncertain`.
- **Inference bug fixed**: classifier window aggregation was effectively applying softmax twice.
- Built and evaluated a **17-class model** (includes `not_galago`) on the 69 WAV dataset.
- Created a **leakage-free holdout split** (14 WAV holdout) and removed corresponding ingested PNG windows from training.
- **Leakage-free holdout improved** after retrain to **Top-3 = 50.0%** (from 35.7% baseline).

---

## Files changed / created today

### Edited

- `scripts/predict_3stage_with_context.py`
  - Added columns to CSV output:
    - `top1_species`, `top1_prob` (always filled when classifier runs).
  - Fixed classifier aggregation:
    - **Before**: averaged model outputs then applied `softmax` again.
    - **After**: average probabilities directly and renormalize once.
  - Added `--filelist <path>` option to run predictions on a specific set of WAVs.
  - Added safer model selection flags:
    - `USE_V2_CLASSIFIER = False` (kept v2 opt-in because it performed poorly)
    - `PREFER_17_CLASSIFIER = True`
  - Added class-name selection for 17-class:
    - Uses `models/all_species/class_names_17.json` when classifier is `galago_cnn_all_17classes_best.keras`.
  - Added background handling:
    - If top-1 is `not_galago`, output `species_result = uncertain`.

- `scripts/analyze_prediction_accuracy.py`
  - Updated to use `top1_species` when present (so “uncertain” doesn’t erase true top-1).
  - Expanded `LABEL_MAP` for raw_audio folder aliases (e.g. `G.granti`, `O.garnettii`, etc.).
  - (You later saved whitespace/newline formatting changes; logic stayed as above.)

- `scripts/analyze_top3_predictions.py`
  - Updated to use `top1_species` when present (not `species_result`).

- `scripts/sanity_overfit_tinyset.py`
  - Fixed double-normalization (removed extra `Rescaling(1/255)` because loader already divides by 255).
  - Replaced unicode PASS/FAIL markers with ASCII tags.

- `scripts/train_cnn_all_species_v2.py`
  - Removed `training=True` forced on augmentation so it respects Keras training/inference mode.

### Created

- `scripts/finetune_on_raw_audio.py`
  - Fine-tunes the existing 16-class model on windowed `data/raw_audio`.
  - Saved: `models/all_species/galago_cnn_all_16classes_finetuned_raw_audio.keras`

- `scripts/ingest_raw_audio_to_training_mels.py`
  - Converts `data/raw_audio/*.wav` into windowed PNGs under `data/melspectrograms/<species>/` using the same mel params.
  - Run result: **933** PNG windows added.

- `scripts/make_raw_audio_holdout_split.py`
  - Creates a leakage-free holdout split by WAV file.
  - Outputs:
    - `data/splits/raw_audio_holdout.json`
    - `data/splits/raw_audio_holdout_filelist.txt`

- `scripts/remove_ingested_holdout_pngs.py`
  - Removes leakage by moving ingested holdout PNGs out of training:
    - from `data/melspectrograms/*/rawaudio__<stem>__win*.png`
    - to `data/melspectrograms_holdout_removed/*/...`
  - Run result: **204** PNG windows moved.

### Added

- `models/all_species/class_names_17.json`
  - Class list for the 17-class model (includes `not_galago`).

---

## Commands run (high-level)

### Diagnostics

- `python scripts/sanity_overfit_tinyset.py`
  - **PASS** — reached ~100% training accuracy quickly.

### Predictions + evaluation (core)

- `python scripts/predict_3stage_with_context.py`
- `python scripts/predict_3stage_with_context.py Tanzania` (tested context rerank)
- `python scripts/analyze_prediction_accuracy.py`
- `python scripts/analyze_top3_predictions.py`

### Domain adaptation attempts

- `python scripts/finetune_on_raw_audio.py`
  - Built windows: train 690 / val 243
  - Result: **val accuracy ~0.193** (file-split)

### Ingest raw_audio to training mels

- `python scripts/ingest_raw_audio_to_training_mels.py`
  - Added: **933** PNG windows

### Leakage-free holdout setup

- `python scripts/make_raw_audio_holdout_split.py`
  - Total WAVs: 69
  - Holdout WAVs: 14

- `python scripts/remove_ingested_holdout_pngs.py`
  - Moved: **204** PNG windows

### Retraining

- `python scripts/train_cnn_all_species.py`
  - Trains a 17-class model (includes `not_galago`).
  - Final reported: **Validation accuracy (17 classes): 0.944**
  - Saved:
    - `models/all_species/galago_cnn_all_17classes_best.keras`
    - `models/all_species/galago_cnn_all_17classes_final.keras`

### Leakage-free holdout evaluation

- `python scripts/predict_3stage_with_context.py --filelist data/splits/raw_audio_holdout_filelist.txt`
- `python scripts/analyze_top3_predictions.py`
- `python scripts/analyze_prediction_accuracy.py`

---

## Metrics/results (what we observed)

### Baseline (pre 17-class + domain alignment)

- Earlier 16-class model on 69 WAVs: **Top-3 ~15.9%** (domain shift)

### 17-class model (after ingesting raw_audio windows; NOTE this initially included leakage)

- 69 WAVs:
  - **Top-3 ~40.6%**

### Leakage-free holdout (14 WAVs)

- Holdout baseline (before leakage-free retrain):
  - **Top-1 14.3%**, **Top-2 28.6%**, **Top-3 35.7%**

- After leakage-free retrain completed:
  - **Top-1 21.4%**, **Top-2 28.6%**, **Top-3 50.0%**

(Important: holdout is small N=14, so variance is high; still a strong directional improvement.)

---

## Current state of the repo / system

- `predict_3stage_with_context.py` is now the main evaluation entrypoint.
- We are currently preferring the **17-class model** (includes `not_galago`).
- `data/melspectrograms` now includes additional `rawaudio__...` windows **except** the holdout ones (moved out).
- Holdout split artifacts exist under `data/splits/`.

---

## Known issues / caveats

- 69-WAV evaluation is not a clean generalization test once we ingest raw_audio windows into training.
  - That’s why we created and used the holdout split.
- The holdout set is small; tomorrow we should evaluate on the remaining 55 WAVs (train split) as a second check.

---

## Tomorrow plan (pick up here)

1) **Evaluate the remaining 55 WAVs** (the “train split” WAVs) using `--filelist` (we’ll generate a train-split filelist from `data/splits/raw_audio_holdout.json`).
2) **Report combined results**:
   - holdout (14) + train-split (55)
   - per-species breakdown and biggest confusions.
3) Consider **threshold tuning** for the 17-class model:
   - separate `not_galago` reject threshold from species confidence threshold.
4) Only then consider further architecture/augmentation changes (or segment-based training).
