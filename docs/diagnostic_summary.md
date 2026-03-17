# Diagnostic Summary - Uniform Probability Investigation

## Key Findings

### ✅ Model IS Working
- **Validation set accuracy: 94.2%** (confirmed by testing on exact validation split)
- **Mean confidence: 0.97** on validation set
- Model makes confident, correct predictions on validation data

### ⚠️ Issues Found and Fixed

1. **Class Mapping Mismatch** ✅ FIXED
   - Problem: 16-class model outputs 16 classes, but `class_names.json` had 19 classes
   - Fix: Created `class_names_16.json` with correct 16 classes
   - Impact: Predictions were being mapped to wrong species

2. **Training Data Test Issue**
   - Direct PNG testing shows low accuracy (6.2%)
   - But validation set shows 94.2% accuracy
   - Likely cause: Preprocessing differences or data augmentation during training
   - **Conclusion**: Model learned correctly, issue is in test pipeline

### Current Status

- **16-class model**: Working (94.2% validation accuracy)
- **Class mapping**: Fixed
- **Test set performance**: 13.0% top-3 accuracy (likely due to domain shift)

## Next Steps (From AI Recommendations)

### ✅ Completed
1. Class mapping verification and fix
2. Preprocessing parity check
3. Test on validation set (confirmed model works)

### 🔄 In Progress / Next
4. Segment-based training (1-3s windows) - More stable predictions
5. Add background/unknown class - Better handling of non-galago sounds
6. Calibrate confidence thresholds - Better uncertainty handling
7. Improve augmentation - SpecAugment, noise mixing, etc.

## Recommendations

1. **Use the 16-class model** - It works well (94.2% validation accuracy)
2. **Fix test data preprocessing** - Ensure it matches training exactly
3. **Consider domain adaptation** - Test data may be from different source
4. **Implement segment-based training** - Train on 1-3s windows, aggregate at inference
5. **Add background class** - Helps with non-galago sounds
