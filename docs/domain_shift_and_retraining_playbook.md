## Domain Shift & Retraining Playbook

**Goal**: Reduce domain shift (training PNGs vs WAVs, Oxford Brookes vs raw_audio) and class imbalance, building on the calibrated 3‑stage pipeline.

This document gives you **concrete commands** to:
- Inspect class balance (already run once)
- Ingest raw WAVs into `data/melspectrograms` for training
- Retrain with an improved, class-balanced architecture
- Re‑evaluate with the calibrated 3‑stage pipeline

---

### 1. Inspect Current Class Balance (PNG vs WAV)

You can re-run the balance report anytime:

```powershell
cd "C:\Users\galag\GitHub\galago_audio_project"
python scripts\report_data_balance.py
```

This shows, per species:
- **PNG (mels)**: how many training images you have now
- **WAV (raw)**: how many source clips you have
- **PNG ratio / WAV ratio** (relative to the max class)

Use this to identify:
- Underrepresented classes in PNGs (PNG ratio < 0.3)
- Species with many WAVs but relatively few PNGs (domain shift risk)

---

### 2. Ingest Raw Audio into Training Mels (Reduce Domain Shift)

The ingestion script windows WAVs and writes PNGs that match your training preprocessing.

**Recommended: use the train-split filelist to avoid leakage**.

Example (adapt the filelist path if needed):

```powershell
cd "C:\Users\galag\GitHub\galago_audio_project"

python scripts\ingest_raw_audio_to_training_mels.py `
  --filelist "data\splits\raw_audio_train_filelist.txt" `
  --out-dir "data\melspectrograms" `
  --max-windows 8 `
  --prefix "rawaudio__"
```

- `--filelist`: ensures only training WAVs are ingested.
- `--max-windows`: caps windows per file so a few long recordings don’t dominate.
- `--prefix`: makes the new PNGs easy to identify (`rawaudio__...png`).

After this, you can re-run:

```powershell
python scripts\report_data_balance.py
```

to confirm that PNG counts increased, especially for underrepresented species.

---

### 3. Retrain an Improved, Balanced Classifier

For a first improved model, you can use `train_cnn_all_species_improved.py` (balanced oversampling + better head).

Basic usage:

```powershell
cd "C:\Users\galag\GitHub\galago_audio_project"

python scripts\train_cnn_all_species_improved.py
```

This script:
- Scans `data\melspectrograms\*` for species folders
- Caps oversampling to avoid extreme repetition
- Trains a more expressive CNN head
- Saves the best checkpoint to:
  - `models\all_species\galago_cnn_all_19classes_improved_best.keras` (or similar)

If you want to **experiment** without overwriting your current model directory, you can:
- Copy the script into a variant (e.g. `train_cnn_all_species_improved_experiment.py`)
- Or change its `OUT_DIR` to a separate experiment folder.

---

### 4. Plug the New Classifier into the 3‑Stage Pipeline

Once you have a new classifier `.keras` file, you can point `predict_3stage_with_context.py` at it **without changing code** using `--classifier-model`.

Example:

```powershell
cd "C:\Users\galag\GitHub\galago_audio_project"

python scripts\predict_3stage_with_context.py `
  --filelist "data\splits\raw_audio_holdout_filelist.txt" `
  --classifier-model "models\all_species\galago_cnn_all_19classes_improved_best.keras" `
  --classifier-threshold 0.2 `
  --temperature 0.212 `
  --out-csv "outputs\predictions\predictions_3stage_context_improved_temp0212.csv"
```

Notes:
- `--temperature 0.212` keeps using the calibrated confidence scaling.
- `--classifier-threshold` can be tuned later (e.g. with `sweep_3stage_thresholds.py`).

---

### 5. Re‑Evaluate and Compare

Use existing analysis scripts to compare the **old** and **new** models:

**Accuracy / uncertainty**:

```powershell
python scripts\analyze_prediction_accuracy.py `
  --csv "outputs\predictions\predictions_3stage_context.csv"

python scripts\analyze_prediction_accuracy.py `
  --csv "outputs\predictions\predictions_3stage_context_improved_temp0212.csv"
```

**Threshold/coverage trade‑offs**:

```powershell
python scripts\sweep_3stage_thresholds.py `
  --csv "outputs\predictions\predictions_3stage_context_improved_temp0212.csv"
```

This will show how coverage / top‑1 / top‑3 behave across thresholds (0.2–0.6).

---

### 6. Suggested Iteration Order

1. **Run ingestion** on the train split (`ingest_raw_audio_to_training_mels.py`).
2. **Confirm balance** with `report_data_balance.py`.
3. **Retrain** with `train_cnn_all_species_improved.py` (or `train_cnn_all_species_v2.py` if you want the newer architecture).
4. **Swap in the new model** via `--classifier-model` and run the 3‑stage pipeline with temperature scaling.
5. **Analyze accuracy & thresholds** with:
   - `analyze_prediction_accuracy.py`
   - `sweep_3stage_thresholds.py`
6. If results are good, update your **default classifier path** in `predict_3stage_with_context.py` to the new model.

This gives you a clear, repeatable path from the current calibrated model to a domain‑adapted, better‑balanced classifier without touching your existing evaluation infrastructure.

