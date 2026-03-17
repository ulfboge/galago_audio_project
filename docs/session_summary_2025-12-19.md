# Session summary — 2025-12-19

## Goal (what we focused on today)

- Evaluate the remaining 55 WAVs from the train split (completing yesterday's plan)
- Report combined results (holdout + train split)
- Analyze per-species breakdown and biggest confusions

---

## Key outcomes (high-signal)

- **Completed evaluation on all 69 WAVs** (14 holdout + 55 train)
- **Created split analysis script** (`analyze_split_results.py`) to compare holdout vs train performance
- **Combined results show consistent performance** across splits (no major overfitting detected)
- **Top-3 accuracy: 44.9%** (better than top-1 at 17.4%, suggesting model learns patterns but struggles with exact classification)

---

## Files changed / created today

### Created

- `scripts/analyze_split_results.py`
  - Analyzes predictions split by holdout vs train
  - Shows combined results, per-split breakdown, per-species accuracy, and top confusions
  - Helps identify overfitting and performance differences

- `data/splits/raw_audio_train_filelist.txt`
  - Filelist for the 55 train-split WAVs

- `data/splits/raw_audio_all_filelist.txt`
  - Combined filelist with all 69 WAVs

---

## Commands run (high-level)

### Predictions

- `python scripts/predict_3stage_with_context.py --filelist data/splits/raw_audio_train_filelist.txt`
  - Evaluated 55 train-split WAVs

- `python scripts/predict_3stage_with_context.py --filelist data/splits/raw_audio_all_filelist.txt`
  - Evaluated all 69 WAVs (combined holdout + train)

### Analysis

- `python scripts/analyze_prediction_accuracy.py`
  - Overall accuracy analysis

- `python scripts/analyze_split_results.py`
  - Split-by-split analysis (holdout vs train)

---

## Metrics/results (what we observed)

### Combined Results (69 WAVs)

- **Top-1 accuracy: 17.4%** (12/69 correct)
- **Top-3 accuracy: 44.9%** (31/69 in top-3)
- **Uncertain predictions: 36.2%** (25/69)
- **Correct among uncertain: 48.0%** (12/25 uncertain were correct in top-3)

### Split Comparison

**Holdout Split (14 files):**
- Top-1: 14.3% (2/14)
- Top-3: 50.0% (7/14)
- Uncertain: 42.9% (6/14)

**Train Split (55 files):**
- Top-1: 18.2% (10/55)
- Top-3: 43.6% (24/55)
- Uncertain: 34.5% (19/55)

**Key observation:** Performance is similar between holdout and train splits, suggesting the model is not severely overfitting. However, overall accuracy is still low.

### Per-Species Accuracy

| Species | Top-1 Accuracy | Top-3 Accuracy | Total Files |
|---------|---------------|---------------|-------------|
| Galagoides_sp_nov | 37.5% (9/24) | 50.0% | 24 |
| Otolemur_garnettii | 12.5% (1/8) | **87.5%** | 8 |
| Paragalago_granti | 33.3% (1/3) | 33.3% | 3 |
| Paragalago_orinus | 12.5% (1/8) | 37.5% | 8 |
| Paragalago_rondoensis | 0.0% (0/11) | 45.5% | 11 |
| Otolemur_crassicaudatus | 0.0% (0/8) | 12.5% | 8 |
| Paragalago_zanzibaricus | 0.0% (0/7) | 28.6% | 7 |

**Key findings:**
- **Otolemur_garnettii** has excellent top-3 accuracy (87.5%) but poor top-1 (12.5%), suggesting it's often confused with similar species but usually appears in top-3
- **Three species have 0% top-1 accuracy**: Otolemur_crassicaudatus, Paragalago_rondoensis, Paragalago_zanzibaricus
- **Galagoides_sp_nov** performs best overall (37.5% top-1)

### Top Confusions

1. **Galagoides_sp_nov → Otolemur_crassicaudatus** (6 times)
2. **Paragalago_orinus → Otolemur_garnettii** (5 times)
3. **Otolemur_crassicaudatus → Paragalago_orinus** (3 times)
4. **Otolemur_garnettii → Paragalago_orinus** (3 times)
5. **Paragalago_rondoensis → Otolemur_garnettii** (3 times)
6. **Paragalago_rondoensis → Galagoides_demidovii** (3 times)

**Patterns:**
- Strong confusion between **Otolemur** and **Paragalago** genera
- **Otolemur_crassicaudatus** is frequently confused with other species
- **Paragalago_rondoensis** is confused with both Otolemur and Galagoides

---

## Current state of the repo / system

- All 69 WAVs have been evaluated with the 17-class model
- Split analysis script available for comparing holdout vs train performance
- Predictions saved in `outputs/predictions/predictions_3stage_context.csv`

---

## Known issues / caveats

1. **Low overall accuracy (17.4% top-1)** despite good validation accuracy (94.2%)
   - Suggests domain shift between training data and raw_audio WAVs
   - Top-3 accuracy (44.9%) is better, indicating model learns patterns but struggles with exact classification

2. **Three species with 0% accuracy**
   - Otolemur_crassicaudatus, Paragalago_rondoensis, Paragalago_zanzibaricus
   - May need more training data or better representation

3. **High confusion between genera**
   - Otolemur ↔ Paragalago confusion suggests similar acoustic features
   - May benefit from hierarchical classification (genus first, then species)

4. **Uncertain predictions are often correct**
   - 48% of uncertain predictions have true label in top-3
   - Suggests threshold (0.4) may be too conservative, or model needs better calibration

---

## Tomorrow plan (pick up here)

1. **Threshold tuning**:
   - Test different confidence thresholds (0.3, 0.35, 0.45, 0.5)
   - Separate threshold for `not_galago` rejection vs species confidence
   - Analyze trade-off between confident predictions and accuracy

2. **Investigate species with 0% accuracy**:
   - Check training data distribution for Otolemur_crassicaudatus, Paragalago_rondoensis, Paragalago_zanzibaricus
   - Review specific misclassifications to understand patterns
   - Consider if these species need more training data

3. **Address genus-level confusion**:
   - Consider hierarchical classification approach
   - Or add genus-level features/constraints to the model

4. **Domain adaptation**:
   - The gap between validation accuracy (94.2%) and test accuracy (17.4%) suggests domain shift
   - Consider:
     - More aggressive data augmentation during training
     - Fine-tuning on raw_audio windows (already attempted, but may need different approach)
     - Segment-based training (train on 1-3s windows explicitly)

5. **Use top-3 predictions**:
   - Since top-3 accuracy is much better (44.9%), consider using top-3 for downstream applications
   - Or implement a "soft" classification that returns top-3 with probabilities

---

## Notes

- The model shows promise (top-3 accuracy ~45%) but needs improvement for production use
- Similar performance on holdout vs train suggests the model generalizes, but to a different distribution than the raw_audio test set
- The 17-class model (with `not_galago`) is being used, which may help with false positives but doesn't seem to improve species classification accuracy
