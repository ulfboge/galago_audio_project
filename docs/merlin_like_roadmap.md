# Merlin-like Roadmap for Galago ID

## Current State Analysis

### What We Have
- ✅ 16-class species classifier (94.2% validation accuracy)
- ✅ Mel-spectrogram pipeline (128×128 RGB)
- ✅ Windowing and logit averaging
- ✅ 3,408 training samples across 16 species
- ✅ Confidence thresholding (0.6)

### What We're Missing (Merlin-like Features)
- ❌ **2-stage detection**: No "galago vs not-galago" detector
- ❌ **Negative classes**: No training on non-galago audio
- ❌ **Context re-ranking**: No location/season/time filtering
- ❌ **Hard negatives**: No training on confusing sounds
- ❌ **Precision/recall metrics**: Only accuracy reported
- ❌ **False positive tracking**: No evaluation on non-galago audio

---

## Implementation Plan

### Phase 1: 2-Stage Detection (Priority 1)

**Goal**: Add a binary "galago detector" before species classification.

**Why**: Reduces false positives massively. Only classify species if we're confident it's a galago.

**Implementation**:
1. Train binary classifier: `galago` vs `not_galago`
2. Use existing galago data as positive class
3. Collect/create negative class data:
   - Insects (crickets, cicadas)
   - Frogs
   - Birds
   - Wind/rain
   - Human speech
   - Equipment noise
   - Other primates (if available)

**Model Architecture**:
- Same mel-spectrogram input (128×128)
- Smaller CNN (fewer parameters than species classifier)
- Binary output: `[galago_prob, not_galago_prob]`

**Threshold**: 
- Only proceed to species classifier if `galago_prob > 0.7`
- Otherwise return "Not a galago" or "Unknown"

---

### Phase 2: Context Re-ranking (Priority 2)

**Goal**: Use location/season/time to re-rank species predictions.

**Implementation**:
1. Create species range database (country/region per species)
2. Create seasonality database (active months per species)
3. Create time-of-night preferences (nocturnal patterns)
4. Apply Bayesian re-ranking:
   ```
   final_prob = model_prob * prior_prob / normalization
   ```

**Data Sources**:
- IUCN Red List (geographic ranges)
- Literature (seasonality, activity patterns)
- User-provided metadata (GPS, timestamp)

**Output**:
- "Most likely here" (high model prob + high prior)
- "Possible but uncommon" (high model prob + low prior)
- "Unlikely here" (low model prob + low prior)

---

### Phase 3: Improved Evaluation (Priority 3)

**Goal**: Track metrics that matter in the field.

**Metrics to Add**:
1. **Per-species precision/recall**
2. **False positive rate** on non-galago audio
3. **Performance by device type** (if metadata available)
4. **Performance by distance/noise level**
5. **Confusion matrix** (already have, but improve visualization)

**Implementation**:
- Create evaluation script with all metrics
- Generate per-species reports
- Track false positives separately

---

### Phase 4: Hard Negatives Training (Priority 4)

**Goal**: Improve model by training on sounds it confuses.

**Implementation**:
1. Run model on test set
2. Identify misclassifications
3. Add misclassified examples to training as "hard negatives"
4. Retrain with hard negatives included

**Example**:
- If model confuses `Otolemur_crassicaudatus` with `Paragalago_rondoensis`
- Add more examples of both to training
- Or create explicit "hard negative" pairs

---

### Phase 5: Data Flywheel (Priority 5)

**Goal**: Continuously improve with new labeled data.

**Implementation**:
1. User submission interface (even if just CSV)
2. Expert review queue
3. Periodic retraining
4. Version tracking

---

## Immediate Next Steps (Next 2 Weeks)

### Week 1: Detector + Negative Classes
1. **Day 1-2**: Collect/create negative class dataset
   - Download free audio samples (insects, frogs, birds)
   - Create noise samples (wind, rain, equipment)
   - Organize into `data/melspectrograms/not_galago/` folder

2. **Day 3-4**: Train binary detector
   - Create `scripts/train_galago_detector.py`
   - Train on galago (positive) vs not_galago (negative)
   - Evaluate false positive rate

3. **Day 5**: Integrate detector into prediction pipeline
   - Update `batch_predict_from_wav_all_species.py`
   - Add 2-stage logic: detector → classifier
   - Test on mixed audio (galago + non-galago)

### Week 2: Context + Evaluation
1. **Day 1-2**: Create species range database
   - Research geographic ranges for 16 species
   - Create `data/species_ranges.json`
   - Add seasonality data if available

2. **Day 3-4**: Implement context re-ranker
   - Create `scripts/context_reranker.py`
   - Apply Bayesian re-ranking
   - Test on predictions

3. **Day 5**: Improve evaluation metrics
   - Update `analyze_prediction_accuracy.py`
   - Add precision/recall per species
   - Add false positive rate tracking

---

## Architecture Overview

```
Audio Input
    ↓
[Stage 1: Detector]
    ├─→ "Not a galago" → Return "Not a galago"
    └─→ "Galago detected" → Continue
         ↓
[Stage 2: Species Classifier]
    ├─→ Get top-3 species + probabilities
    └─→ Continue
         ↓
[Stage 3: Context Re-ranker]
    ├─→ Apply location prior
    ├─→ Apply seasonality prior
    ├─→ Apply time-of-night prior
    └─→ Re-rank predictions
         ↓
[Output]
    ├─→ Top species (with confidence)
    ├─→ "Most likely here" / "Possible" / "Unlikely"
    └─→ Top-3 candidates
```

---

## Data Requirements

### Negative Classes (Minimum Viable)
- **Insects**: 100-200 clips (crickets, cicadas)
- **Frogs**: 100-200 clips
- **Birds**: 100-200 clips (night birds if possible)
- **Noise**: 50-100 clips (wind, rain, equipment)
- **Human speech**: 50-100 clips
- **Total**: ~500-800 negative clips

### Species Range Data
- Geographic ranges (country/region level)
- Seasonality (if available)
- Time-of-night preferences (if available)

---

## Success Metrics

### Detector
- **False positive rate**: < 5% on non-galago audio
- **True positive rate**: > 95% on galago audio

### Classifier (with detector)
- **Top-3 accuracy**: > 40% (currently 25%)
- **False positives**: < 10% on non-galago audio

### Context Re-ranker
- **Improvement**: +10-20% accuracy when context matches
- **User trust**: Higher confidence in "most likely here" predictions

---

## Questions to Answer

1. **How many negative clips can we collect?** (affects detector quality)
2. **Do we have GPS/metadata for recordings?** (affects context re-ranker)
3. **What's the target use case?** (field app vs research tool)
4. **On-device or server-side?** (affects model size/complexity)

---

## References

- Merlin Bird ID: https://merlin.allaboutbirds.org/
- BirdNET (similar approach): https://birdnet.cornell.edu/
- AudioSet (negative class source): https://research.google.com/audioset/

