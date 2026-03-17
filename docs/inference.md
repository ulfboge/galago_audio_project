# Inference and interpretation

## Prediction outputs

For each recording, the model returns:

- `predicted_species`
- `predicted_prob` (softmax probability)

---

## Confidence thresholds

Recommended interpretation:

- **p ≥ 0.6** → accept species prediction
- **p < 0.6** → treat as uncertain

Thresholds can be adjusted depending on application (e.g. stricter for automated pipelines).

---

## Batch inference

Batch prediction scripts process all WAV files under `audio_raw/` and export results as CSV files.

Outputs can be summarized using confidence statistics to assess uncertainty and model behavior.

---

## Interpretation guidance

Predictions should be evaluated alongside:

- known species distributions
- call type
- recording quality

The classifier provides **acoustic evidence**, not definitive taxonomic identification.
