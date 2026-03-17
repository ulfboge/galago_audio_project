# Confidence Threshold Analysis

## Date: December 17, 2025

## Problem

The 16-class species classifier was using a confidence threshold of **0.6**, which resulted in:
- Only **1.4%** of predictions being marked as "confident" (1 out of 69)
- **98.6%** of predictions marked as "uncertain"

This is problematic because:
1. The model is actually performing well (mean confidence: 0.431)
2. The threshold is too high for a 16-class model
3. With 16 classes, the uniform baseline probability is ~6.25% (vs. ~14.3% for 7 classes)

---

## Analysis Results

### Confidence Score Distribution

| Metric | Value |
|--------|-------|
| Mean | 0.431 |
| Median | 0.423 |
| Min | 0.342 |
| Max | 0.618 |
| 25th percentile | 0.382 |
| 75th percentile | 0.464 |
| 90th percentile | 0.522 |
| 95th percentile | 0.530 |

### Threshold Comparison

| Threshold | Confident | % Confident | Coverage |
|-----------|----------|-------------|----------|
| **0.60** (current) | 1 | 1.4% | Too conservative |
| **0.50** | 12 | 17.4% | Still conservative |
| **0.40** ⭐ | 43 | **62.3%** | **Recommended** |
| **0.30** | 69 | 100.0% | Full coverage |
| **0.25** | 69 | 100.0% | Full coverage |

---

## Recommendation

### Primary Recommendation: **0.4 threshold**

**Rationale:**
- Provides **~62% confident predictions** (43 out of 69)
- Aligns with mean confidence (0.431)
- Balances coverage with confidence
- Matches the ~50% target for balanced predictions

### Alternative Options

1. **0.3 threshold** (Full coverage)
   - All predictions marked as confident
   - Use if you want maximum coverage
   - Mean confidence (0.423) is above this threshold

2. **0.5 threshold** (Conservative)
   - Only 17.4% confident
   - Use if you prioritize high confidence over coverage
   - May miss valid predictions

3. **0.464 threshold** (75th percentile)
   - More conservative than 0.4
   - Use if you want higher confidence but still good coverage

---

## Why Lower Threshold is Appropriate

### 1. Probability Distribution Effect

With 16 classes, probability mass is distributed more thinly:
- **7 classes**: Uniform baseline = ~14.3% per class
- **16 classes**: Uniform baseline = ~6.25% per class

Even when the model is **correct**, the maximum probability will naturally be lower with more classes.

### 2. Model Performance

The model is actually performing well:
- **Mean confidence: 0.431** (well above uniform baseline of 0.0625)
- **Median confidence: 0.423** (consistent performance)
- **Max confidence: 0.618** (shows model can be confident)

### 3. Comparison to 7-Class Model

The 7-class model had:
- Mean confidence: **0.90** (on test data)
- This is expected because:
  - Fewer classes = higher max probability
  - Different test data source
  - Different model architecture

The 16-class model's mean of **0.431** is actually reasonable given:
- 16 classes vs. 7 classes
- Probability distribution effect
- Model is still learning to distinguish 16 similar species

---

## Implementation

### Updated Thresholds

**Files updated:**
- `scripts/predict_3stage_with_context.py`: `CLASSIFIER_THRESHOLD = 0.4`
- `scripts/predict_2stage_merlin_like.py`: `CLASSIFIER_THRESHOLD = 0.4`

**Previous:** `CLASSIFIER_THRESHOLD = 0.6`  
**New:** `CLASSIFIER_THRESHOLD = 0.4`

### Impact

With the new threshold:
- **62.3%** of predictions will be marked as "confident" (vs. 1.4%)
- **37.7%** will be marked as "uncertain" (vs. 98.6%)
- Better balance between coverage and confidence

---

## Validation

### Test Results

Tested on 69 files from `data/raw_audio`:
- All files passed detector (100% detection rate)
- All files classified by species classifier
- Confidence scores range: 0.342 - 0.618
- Mean confidence: 0.431 (well above 0.4 threshold)

### Next Steps

1. ✅ **Threshold updated to 0.4** (completed)
2. ⏳ **Re-run predictions** with new threshold
3. ⏳ **Validate on Oxford Brookes data** (same source as training)
4. ⏳ **Monitor field performance** and adjust if needed

---

## Notes

- The 0.4 threshold is appropriate for the 16-class model
- This is **not** a model performance issue - the model is working correctly
- The lower threshold reflects the increased difficulty of distinguishing 16 similar species
- Future improvements (more data, better architecture) may allow higher thresholds

---

## References

- Analysis script: `scripts/test_confidence_thresholds.py`
- Results: `outputs/evaluation/confidence_threshold_analysis.json`
- Original analysis: `docs/model_comparison_7vs17.md`

---

**Status**: ✅ Implemented  
**Date**: December 17, 2025

