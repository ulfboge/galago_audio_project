# Detector Investigation Summary

**Date**: December 18, 2025  
**Issue**: Detector rejecting known galago calls, especially Otolemur species

---

## Problem Statement

After retraining the detector with 360 negative samples, the detector is filtering out many known galago calls:
- **71% rejection rate** at threshold 0.7
- **64% rejection rate** at threshold 0.5
- **55% rejection rate** at threshold 0.3
- **All Otolemur files rejected** at thresholds 0.7 and 0.5
- **Only 1 Otolemur file passed** at threshold 0.3

---

## Investigation Results

### 1. Detector Output Analysis

**Otolemur Performance:**
- `Otolemur_crassicaudatus`: 0% pass rate (8/8 rejected)
  - Mean `galago_prob`: 0.178
  - Range: 0.011 - 0.392
- `Otolemur_garnettii`: 0% pass rate (8/8 rejected)
  - Mean `galago_prob`: 0.188
  - Range: 0.085 - 0.307

**Other Species Performance (at threshold 0.5):**
- `Galagoides_sp_nov`: 62.5% pass rate (mean prob: 0.665)
- `Paragalago_zanzibaricus`: 57.1% pass rate (mean prob: 0.492)
- `Paragalago_rondoensis`: 36.4% pass rate (mean prob: 0.453)
- `Paragalago_granti`: 33.3% pass rate (mean prob: 0.514)
- `Paragalago_orinus`: 12.5% pass rate (mean prob: 0.294)

### 2. Training Data Distribution Analysis

**Severe Imbalance Identified:**

| Species | Samples | % of Total | Ratio vs Max |
|---------|---------|------------|--------------|
| Galago_senegalensis | 1008 | 23.5% | 1.00x |
| Otolemur_crassicaudatus | 30 | 0.7% | 0.03x |
| Otolemur_garnettii | 42 | 1.0% | 0.04x |

**Key Findings:**
- Otolemur species have **33.6x fewer samples** than the most represented species
- Total Otolemur samples: **72 (1.7% of galago data)**
- Detector trained on 4,148 galago samples total
- Class imbalance: 4,148 galago vs. 360 not_galago (11.5:1 ratio)

### 3. Root Cause Analysis

**Primary Issue**: The detector learned a "typical galago" pattern that doesn't match Otolemur calls.

**Contributing Factors:**
1. **Severe underrepresentation**: Otolemur has 33.6x fewer training samples
2. **Acoustic differences**: Otolemur calls may be acoustically distinct from other galago species
3. **Class imbalance**: The model may have learned to recognize the majority pattern (Galago_senegalensis-like calls)

---

## Recommendations & Results

### ✅ Recommendation 1: Lower Detector Threshold

**Action**: Lowered threshold from 0.7 → 0.5 → 0.3

**Results:**
- Threshold 0.7: 20 files passed (29%)
- Threshold 0.5: 25 files passed (36%)
- Threshold 0.3: 31 files passed (45%)
- **1 Otolemur file passed** at 0.3 (still 15/16 rejected)

**Conclusion**: Lowering threshold helps but doesn't solve the Otolemur problem.

### ✅ Recommendation 2: Analyze Training Data Distribution

**Action**: Created analysis script to identify underrepresented species

**Results:**
- Confirmed severe Otolemur underrepresentation (1.7% of data)
- Identified 9 species with < 50 samples
- Imbalance ratio: 33.6x between most and least represented

**Conclusion**: Data imbalance is a major contributing factor.

### 🔄 Recommendation 3: Retrain Detector with Balanced Sampling

**Status**: Ready to implement

**Proposed Actions:**
1. **Balanced sampling**: Sample equal numbers from each species (e.g., 100-200 per species)
2. **Data augmentation**: Augment underrepresented species (Otolemur, etc.)
3. **Weighted loss**: Use class weights to balance training
4. **Oversample Otolemur**: Include more Otolemur samples in training

**Expected Impact**: Should improve Otolemur detection significantly.

### 🔄 Recommendation 4: Hierarchical Approach

**Status**: Future consideration

**Proposed Actions:**
1. First detect genus (Otolemur vs. other galago)
2. Then classify species within genus
3. Could use separate detectors for different genera

**Expected Impact**: Better handling of acoustically distinct groups.

---

## Current System Status

### Detector Metrics (from training)
- **Accuracy**: 99.4%
- **Precision**: 91.5%
- **Recall**: 100.0%
- **F1-score**: 95.6%

**Note**: These metrics are on validation data, which may not include many Otolemur samples.

### Test Set Performance (69 files)
- **Threshold 0.3**: 31 files passed (45%)
- **Top-3 accuracy**: 12.9%
- **Otolemur pass rate**: 6.25% (1/16 files)

---

## Next Steps

### Immediate (Short-term)
1. ✅ Lower detector threshold to 0.3 (completed)
2. ✅ Analyze training data distribution (completed)
3. ⏳ Retrain detector with balanced sampling
4. ⏳ Test retrained detector on Otolemur files

### Medium-term
1. Collect more Otolemur training data (if available)
2. Implement data augmentation for underrepresented species
3. Consider weighted loss function in training

### Long-term
1. Evaluate hierarchical detection approach
2. Collect field recordings of Otolemur calls
3. Fine-tune detector specifically for Otolemur

---

## Files Created

1. `scripts/investigate_detector.py` - Detector output analysis
2. `scripts/analyze_training_data_distribution.py` - Training data analysis
3. `outputs/evaluation/detector_investigation.csv` - Detailed detector outputs
4. `outputs/evaluation/training_data_distribution.json` - Training data summary

---

## Key Insights

1. **Data imbalance is the root cause**: Otolemur has 33.6x fewer samples than dominant species
2. **Threshold adjustment helps but doesn't solve**: Even at 0.3, most Otolemur files are rejected
3. **Detector learned majority pattern**: Model optimized for Galago_senegalensis-like calls
4. **Retraining with balance is needed**: Equal representation will improve Otolemur detection

---

**Status**: Investigation complete. Ready for retraining with balanced sampling.
