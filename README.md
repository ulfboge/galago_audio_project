---
title: Galago Call Demo
emoji: 🦔
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "4.44.0"
app_file: demo/upload_predict_gradio.py
pinned: false
license: mit
---

*Hugging Face Spaces uses the YAML header above. Weights are not in git — set secret/variable `GALAGO_HF_MODEL_REPO` on the Space (see [`demo/README_spaces.md`](demo/README_spaces.md)).*

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
- **Raw-audio / folder-based eval** can disagree with curator labels; see [`docs/evaluation_caveats.md`](docs/evaluation_caveats.md). Geographic deployment assumptions: [`docs/deployment_geographic_assumptions.md`](docs/deployment_geographic_assumptions.md).

---

## 📝 Recent Updates

**March 2026**: Major inference pipeline fix + new Malawi field data.
- Fixed mel-spectrogram orientation and normalization bug in `predict_3stage_with_context.py` (vertical flip + `[0,1]` scale). Regional accuracy: Malawi 25% → 100%, Kenya 23.5% → 97%, Tanzania 36.4% → 82%.
- Added 406 new WAVs from Malawi field recordings: Mkwazi *G. sp. nov. 1* (68), TRFF *G. granti* (229), Mughese *G. sp. nov. 3* (109). Now 7,243 total training PNGs.
- Fine-tuned classifier (ft4, after full retrain) resolves *G. sp. nov.* vs *P. rondoensis* on Malawi evals; Mkwazi holdout 100% top-1. `malawi_balanced` auto-loads ft4; Kenya/Tanzania profiles keep the retrained `improved_best` base unless you override.
- Full retrain in progress with all new data. See [`docs/session_eval_2026-03_retrain_v2.md`](docs/session_eval_2026-03_retrain_v2.md).

**December 16, 2025**: Improved 16-class model with increased capacity and better data balance. Validation accuracy improved from 92.0% to 94.2%. See [`docs/session_summary_2025-12-16.md`](docs/session_summary_2025-12-16.md) for details.

**December 12, 2025**: Fixed windowed prediction pipeline collapse issue. See [`docs/windowed_prediction_fix_summary.md`](docs/windowed_prediction_fix_summary.md) for details.

---

## 🌐 Try it locally (upload a WAV)

Prototype UI (Gradio): install `gradio`, then run `python demo/upload_predict_gradio.py` and open http://127.0.0.1:7860 — see **`demo/README.md`**. Single-file support uses `scripts/predict_3stage_with_context.py --wav <path>`.

---

## 📁 Repository structure

