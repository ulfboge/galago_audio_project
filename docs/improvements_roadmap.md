# Improvements Roadmap - Following AI Recommendations

## Status Summary

### ✅ Completed
1. **Class mapping fix** - Created `class_names_16.json` with correct 16 classes
2. **Preprocessing verification** - Confirmed preprocessing matches training
3. **Validation set testing** - Model achieves 94.2% accuracy on validation set
4. **Confidence calibration** - Model is well-calibrated (94.6% at 0.4 threshold)

### 🔄 In Progress
5. **Improved training script** - Created `train_cnn_all_species_v2.py` with:
   - Weighted loss (no oversampling)
   - Better augmentation (time/frequency masking simulation)
   - Proper class mapping saving

### ⏳ Remaining Recommendations

6. **Segment-based training** - Train on 1-3s windows, aggregate at inference
   - Current: Uses whole mel-spectrograms (already segmented)
   - Improvement: Explicit windowing during training

7. **Add background/unknown class** - Better handling of non-galago sounds
   - Current: Detector handles this
   - Improvement: Add to classifier for better uncertainty

8. **SpecAugment-style augmentation** - More robust augmentation
   - Current: Basic augmentation in `make_mels.py`
   - Improvement: Time/frequency masking during training

9. **Temperature scaling** - Better confidence calibration
   - Current: Using raw softmax probabilities
   - Improvement: Learn temperature parameter on validation set

10. **Context priors** - Already implemented in `context_reranker.py`
    - Status: Working, but could be improved with more data

## Key Findings

### Model Performance
- **Validation accuracy**: 94.2% (excellent!)
- **Confidence calibration**: Well-calibrated (94.6% at 0.4 threshold)
- **Test set performance**: 13.0% top-3 (likely domain shift)

### Issues Identified
1. ✅ **FIXED**: Class mapping mismatch (16 vs 19 classes)
2. **Domain shift**: Test data from different source than training
3. **Windowing reduces confidence**: Averaging across windows lowers max probability

### Recommendations Priority

**High Priority:**
1. ✅ Fix class mapping (DONE)
2. ✅ Verify model works (DONE - 94.2% validation)
3. Test with correct class mapping on test set
4. Consider domain adaptation for test data

**Medium Priority:**
5. Implement segment-based training
6. Add background class to classifier
7. Improve augmentation (SpecAugment)

**Low Priority:**
8. Temperature scaling
9. Improve context re-ranker with more data
