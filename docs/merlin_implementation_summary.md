# Merlin-like Implementation Summary

## What We've Created

Based on Merlin Bird-ID's approach, we've implemented a **2-stage detection and classification system** for galago species identification.

---

## Files Created

### 1. Documentation
- **`docs/merlin_like_roadmap.md`**: Complete implementation roadmap with phases and priorities
- **`docs/collecting_negative_class_data.md`**: Guide for collecting non-galago audio data
- **`docs/merlin_implementation_summary.md`**: This file

### 2. Training Scripts
- **`scripts/train_galago_detector.py`**: Trains binary detector (galago vs not-galago)
- **`scripts/prepare_negative_class.py`**: Helper to convert negative class audio to mel-spectrograms

### 3. Prediction Scripts
- **`scripts/predict_2stage_merlin_like.py`**: 2-stage prediction pipeline (detector → classifier)

---

## Architecture

### Current System (Before)
```
Audio → Species Classifier → Prediction
```
- Single-stage: directly classifies species
- No filtering of non-galago audio
- High false positive rate on non-galago sounds

### New System (Merlin-like)
```
Audio → Detector → Classifier → Prediction
         ↓
    "Not galago" → Stop
```
- **Stage 1**: Binary detector filters out non-galago audio
- **Stage 2**: Species classifier only runs on detected galago audio
- **Result**: Lower false positive rate, better field performance

---

## Implementation Status

### ✅ Completed
1. **Roadmap and documentation** - Complete implementation plan
2. **Detector training script** - Ready to use once negative data is collected
3. **2-stage prediction pipeline** - Integrated detector + classifier
4. **Helper scripts** - Tools to prepare negative class data

### ⏳ Pending (Next Steps)
1. **Collect negative class data** - Need 500-800 clips of non-galago audio
2. **Train detector** - Run `train_galago_detector.py` after collecting data
3. **Test 2-stage pipeline** - Evaluate on mixed audio (galago + non-galago)
4. **Context re-ranker** - Add location/season/time filtering (Phase 2)
5. **Improved metrics** - Add precision/recall per species (Phase 3)

---

## Quick Start Guide

### Step 1: Collect Negative Class Data

**Minimum viable**: 50-100 clips each of:
- Insects (crickets, cicadas)
- Frogs
- Birds (night birds)
- Noise (wind, rain, equipment)

**Sources**:
- Freesound.org (free, CC0 license)
- Your own field recordings
- Background noise from galago sessions

**See**: `docs/collecting_negative_class_data.md` for detailed guide

### Step 2: Prepare Negative Class Data

```bash
# Process each category
python scripts/prepare_negative_class.py E:/Audio/Insects insects
python scripts/prepare_negative_class.py E:/Audio/Frogs frogs
python scripts/prepare_negative_class.py E:/Audio/Birds birds
python scripts/prepare_negative_class.py E:/Audio/Noise noise
```

This creates mel-spectrograms in `data/melspectrograms/not_galago/`

### Step 3: Train Detector

```bash
python scripts/train_galago_detector.py
```

This will:
- Combine all galago species as positive class
- Use `not_galago` folder as negative class
- Train binary classifier
- Save model to `models/detector/galago_detector_best.keras`

### Step 4: Test 2-Stage Pipeline

```bash
python scripts/predict_2stage_merlin_like.py
```

This will:
- Run detector on all audio files
- Only classify species if detector says "galago"
- Save results to `outputs/predictions/predictions_2stage.csv`

---

## Expected Improvements

### Detector Performance
- **False positive rate**: < 5% on non-galago audio (target)
- **True positive rate**: > 95% on galago audio (target)

### Overall System
- **Reduced false positives**: Non-galago sounds filtered out before classification
- **Better field performance**: More reliable in real-world conditions
- **User trust**: Clear "not a galago" vs "galago detected" feedback

---

## Comparison: Before vs After

### Before (Single-Stage)
```
Input: Cricket sound
→ Species Classifier
→ Prediction: "Paragalago_granti" (0.12 confidence)
→ Result: "uncertain" (but still a false positive)
```

### After (2-Stage)
```
Input: Cricket sound
→ Detector: "not_galago" (0.15 confidence)
→ Stop (no species classification)
→ Result: "not_galago" (correct!)
```

---

## Next Phases (Future Work)

### Phase 2: Context Re-ranking
- Add location-based filtering
- Add seasonality data
- Add time-of-night preferences
- Re-rank predictions using Bayesian priors

### Phase 3: Improved Evaluation
- Precision/recall per species
- False positive rate tracking
- Performance by device/distance/noise

### Phase 4: Hard Negatives
- Identify confusing sounds
- Add to training as hard negatives
- Retrain with improved data

### Phase 5: Data Flywheel
- User submission interface
- Expert review queue
- Periodic retraining

---

## Key Differences from Merlin

### What We're Copying
- ✅ 2-stage detection approach
- ✅ Binary detector before classifier
- ✅ Context-aware re-ranking (planned)
- ✅ Focus on field performance

### What's Different
- **Scale**: Merlin has 10,000+ species, we have 16
- **Data**: Merlin has millions of labeled clips, we have thousands
- **Infrastructure**: Merlin has app infrastructure, we're research-focused
- **Approach**: We're adapting the *approach*, not copying the scale

---

## Questions to Answer

Before proceeding, consider:

1. **How many negative clips can you collect?**
   - Minimum: 50-100 per category (500-800 total)
   - Ideal: 500+ per category (2000+ total)

2. **Do you have GPS/metadata for recordings?**
   - Needed for context re-ranker (Phase 2)
   - Can start without it, but limits context features

3. **What's the target use case?**
   - Field app (needs on-device inference)
   - Research tool (can use server-side)
   - Both (needs model optimization)

4. **On-device or server-side?**
   - On-device: Need smaller models, quantization
   - Server-side: Can use larger models, more features

---

## Resources

- **Merlin Bird ID**: https://merlin.allaboutbirds.org/
- **BirdNET** (similar approach): https://birdnet.cornell.edu/
- **AudioSet** (negative class source): https://research.google.com/audioset/
- **Freesound.org** (free audio): https://freesound.org/

---

## Support

If you need help:
1. Check `docs/merlin_like_roadmap.md` for detailed plan
2. Check `docs/collecting_negative_class_data.md` for data collection
3. Review training scripts for implementation details

---

**Status**: Phase 1 (2-stage detection) is ready to implement. Next step: collect negative class data.

