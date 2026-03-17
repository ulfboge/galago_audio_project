# Model and Preprocessing Improvement Analysis

**Date**: January 20, 2026  
**Task**: Model or preprocessing improvements (or recalibration) to address low confidence

---

## Executive Summary

This analysis identified **temperature scaling** as a highly effective solution for improving model confidence calibration, with an **87.2% improvement** in Expected Calibration Error (ECE).

### Key Finding: Temperature Scaling

- **Optimal Temperature**: 0.212 (model is overconfident)
- **Raw ECE**: 0.6868 (poor calibration)
- **Scaled ECE**: 0.0877 (excellent calibration)
- **Improvement**: 87.2% reduction in calibration error

This indicates the model is **overconfident** - it assigns higher probabilities than it should. Temperature scaling "cools down" the predictions to better match actual accuracy.

---

## Analysis Results

### 1. Preprocessing Verification ✅

**Status**: Preprocessing appears consistent

- Training PNG std: 0.230
- Inference WAV std: 0.260
- Difference: ~13% (within acceptable range)

**Conclusion**: No major preprocessing mismatch detected. The slight difference is likely due to different data sources (training PNGs vs inference WAVs), but preprocessing pipeline itself is consistent.

---

### 2. Temperature Scaling Calibration 🎯

**Optimal Temperature**: 0.212

This means:
- Model is **overconfident** (temperature < 1.0)
- Predictions need to be "cooled down" by dividing logits by 0.212
- After scaling: `scaled_probs = softmax(logits / 0.212)`

**Calibration Metrics**:
- **Raw ECE**: 0.6868 (very poor - predictions don't match accuracy)
- **Scaled ECE**: 0.0877 (excellent - well-calibrated)
- **Improvement**: 87.2% reduction in calibration error

**Impact on Confidence**:
- Raw mean confidence: ~0.43 (on validation set)
- Scaled mean confidence: Will be lower but more accurate
- Predictions remain the same (same argmax), but confidence scores are better calibrated

---

### 3. Recommendations

#### HIGH Priority

1. **Address Class Imbalance**
   - Some species have very few samples
   - **Actions**:
     - Use class weights in loss function
     - Collect more data for underrepresented species
     - Use focal loss to focus on hard examples

2. **Reduce Domain Shift**
   - Training on PNGs vs inference on WAVs may cause issues
   - **Actions**:
     - Ingest raw audio into training set (`ingest_raw_audio_to_training_mels.py`)
     - Train directly on WAV files with on-the-fly preprocessing

#### MEDIUM Priority

3. **Apply Temperature Scaling**
   - Optimal temperature: 0.212
   - **Action**: Implement temperature scaling in inference pipeline

4. **Improve Data Augmentation**
   - More robust augmentation can improve generalization
   - **Actions**:
     - Add SpecAugment (time/frequency masking)
     - Add time stretching and pitch shifting
     - Add noise injection

5. **Increase Model Capacity**
   - Current model may be underfitting for 16 classes
   - **Actions**:
     - Add more filters to convolutional layers (e.g., 128→256→512→1024)
     - Increase dense layer size to 1024 units
     - Add attention mechanisms (e.g., SE blocks)

#### LOW Priority

6. **Consider Transfer Learning**
   - Pre-trained audio models may improve performance
   - **Actions**:
     - Use YAMNet or VGGish as feature extractor
     - Fine-tune on galago data

---

## Implementation Guide

### Applying Temperature Scaling

**Option 1: Modify Inference Scripts**

Add temperature scaling to `predict_3stage_with_context.py`:

```python
# After getting logits from model
logits = model.predict(rgb, verbose=0)
TEMPERATURE = 0.212  # From calibration
scaled_probs = tf.nn.softmax(logits / TEMPERATURE).numpy()
```

**Option 2: Create Wrapper Model**

Use the `create_temperature_scaled_model_wrapper()` function from `improve_model_confidence.py`:

```python
from scripts.improve_model_confidence import create_temperature_scaled_model_wrapper

scaled_model = create_temperature_scaled_model_wrapper(model, temperature=0.212)
```

**Option 3: Post-Process Predictions**

If you already have predictions with logits/probabilities:

```python
import numpy as np
import tensorflow as tf

TEMPERATURE = 0.212
scaled_probs = tf.nn.softmax(logits / TEMPERATURE).numpy()
```

---

## Expected Impact

### On Oxford Brookes Data

- **Before**: Most predictions have confidence < 0.3 (all marked "uncertain")
- **After**: With temperature scaling, confidence scores will be:
  - More accurate (better reflect actual accuracy)
  - Potentially higher (if model is actually correct)
  - Better calibrated (confidence matches accuracy)

### On Raw Audio Data

- **Before**: Mean confidence ~0.43, but poorly calibrated
- **After**: 
  - Better calibration (confidence matches accuracy)
  - More reliable threshold decisions
  - Better uncertainty estimates

---

## Files Generated

1. **`scripts/improve_model_confidence.py`** - Main analysis script
2. **`outputs/evaluation/temperature_scaling_calibration.json`** - Calibration results
3. **`outputs/evaluation/temperature_scaling_calibration.png`** - Calibration plots
4. **`outputs/evaluation/model_improvement_recommendations.json`** - Detailed recommendations
5. **`outputs/evaluation/training_vs_inference_image_stats.json`** - Preprocessing comparison

---

## Next Steps

### Immediate (Quick Win)

1. **Apply temperature scaling** to inference pipeline
   - Modify `predict_3stage_with_context.py` to use T=0.212
   - Re-run predictions and compare results
   - Measure impact on confidence distribution

### Short-term (1-2 weeks)

2. **Address class imbalance**
   - Review species sample counts
   - Implement class weights in training
   - Collect more data for underrepresented species

3. **Reduce domain shift**
   - Run `ingest_raw_audio_to_training_mels.py` on training split
   - Retrain model with augmented dataset
   - Compare performance before/after

### Medium-term (1-2 months)

4. **Improve model architecture**
   - Increase model capacity
   - Add attention mechanisms
   - Test on validation set

5. **Enhance data augmentation**
   - Implement SpecAugment
   - Add time/frequency masking
   - Test impact on generalization

---

## Technical Details

### Temperature Scaling

Temperature scaling is a post-hoc calibration method that:
- Learns a single parameter (temperature T) on validation set
- Scales logits: `scaled_logits = logits / T`
- Applies softmax: `scaled_probs = softmax(scaled_logits)`
- Does NOT change predictions (same argmax)
- DOES improve confidence calibration

**Why T=0.212?**
- T < 1.0 means model is overconfident
- Dividing by smaller number makes softmax output more uniform
- This "cools down" overconfident predictions
- Result: confidence scores better match actual accuracy

### Expected Calibration Error (ECE)

ECE measures how well confidence scores match accuracy:
- **Perfect calibration**: ECE = 0 (confidence always equals accuracy)
- **Poor calibration**: ECE > 0.1 (confidence doesn't match accuracy)
- **Our result**: ECE improved from 0.6868 → 0.0877 (87.2% improvement)

---

## Conclusion

**Temperature scaling is the most impactful immediate improvement**, providing an 87.2% reduction in calibration error. This should be implemented first, as it requires no retraining and can be applied immediately to existing models.

The analysis also identified **class imbalance** and **domain shift** as high-priority issues that should be addressed through data collection and training improvements.

---

**Analysis completed**: January 20, 2026
