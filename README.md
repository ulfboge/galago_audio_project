# Galago Acoustic Species Classifier

This repository contains a convolutional neural network (CNN) for identifying galago (family Galagidae) species from vocalizations using mel-spectrogram representations.

The classifier is designed for **species-level acoustic identification** with explicit handling of uncertainty and out-of-scope taxa.

---

## 🎯 Scope

The current model is trained to recognize **16 galago species**:

1. *Euoticus elegantulus*
2. *Euoticus pallidus*
3. *Galago gallarum*
4. *Galago matschiei*
5. *Galago moholi*
6. *Galago senegalensis*
7. *Galagoides demidovii*
8. *Galagoides* sp. nov. (pooled individuals)
9. *Galagoides thomasi*
10. *Otolemur crassicaudatus*
11. *Otolemur garnettii*
12. *Paragalago cocos*
13. *Paragalago granti* (formerly *Galago granti*)
14. *Paragalago orinus*
15. *Paragalago rondoensis*
16. *Paragalago zanzibaricus*

**Note**: A 7-class model is also available for the original subset of species. See model files in `models/` directory.

---

## 🧠 Method overview

- Audio recordings (WAV) are converted to **mel-spectrogram images** (128 × 128).
- A CNN is trained using folder-based species labels.
- Predictions return a probability score; low-confidence results are flagged as uncertain.

---

## ✅ Recommended usage

- **Operational confidence threshold**: `0.6`
- Predictions with `probability < 0.6` should be treated as **uncertain**.

---

## 📊 Performance summary

### 16-Class Model (Current)
- **Validation accuracy**: **94.2%**
- **Training samples**: 3,408 files across 16 species
- **Model file**: `models/all_species/galago_cnn_all_16classes_best.keras`

### 7-Class Model (Original)
- **Validation accuracy**: **~90%**
- On a mixed batch of 69 recordings:
  - Mean prediction confidence: **0.90**
  - Uncertain predictions (p < 0.6): **4.3%**

---

## ⚠️ Important notes

- The classifier does **not** identify individuals or call types.
- Results reflect **acoustic similarity**, not taxonomic proof.
- Predictions should be interpreted in ecological and geographic context.

---

## 📝 Recent Updates

**December 16, 2025**: Improved 16-class model with increased capacity and better data balance. Validation accuracy improved from 92.0% to 94.2%. See [`docs/session_summary_2025-12-16.md`](docs/session_summary_2025-12-16.md) for details.

**December 12, 2025**: Fixed windowed prediction pipeline collapse issue. See [`docs/windowed_prediction_fix_summary.md`](docs/windowed_prediction_fix_summary.md) for details.

---

## 📁 Repository structure

