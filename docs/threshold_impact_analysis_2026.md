# Confidence Threshold Impact Analysis

**Date**: January 8, 2026  
**Task**: Test confidence threshold adjustment (0.3, 0.4, 0.5) and measure impact

---

## Summary

This analysis tested three confidence thresholds (0.3, 0.4, 0.5) on the 3-stage prediction pipeline to measure their impact on accuracy, precision, recall, coverage, and F1 score.

**Dataset**: 69 audio files from `data/raw_audio`

---

## Overall Results

| Threshold | Coverage | Confident | Uncertain | Accuracy | Precision | Recall | F1 Score | Top-3 Accuracy |
|-----------|----------|-----------|-----------|----------|-----------|--------|----------|---------------|
| **0.3**   | 100.0%   | 69        | 0         | 21.7%    | 21.7%     | 21.7%  | **21.7%** | 44.9%         |
| **0.4**   | 87.0%    | 60        | 9         | 20.0%    | 20.0%     | 17.4%  | 18.6%     | 44.9%         |
| **0.5**   | 72.5%    | 50        | 19        | **24.0%** | **24.0%** | 17.4%  | 20.2%     | 44.9%         |

### Key Findings

1. **Coverage vs Accuracy Trade-off**
   - Lower threshold (0.3): 100% coverage but lower precision (21.7%)
   - Higher threshold (0.5): Better precision (24.0%) but only 72.5% coverage
   - Medium threshold (0.4): Balanced at 87% coverage with 20.0% precision

2. **Top-3 Accuracy**
   - Consistent across all thresholds: **44.9%**
   - This suggests the model often gets the correct species in the top-3 predictions, but not as top-1

3. **Best Metrics**
   - **Best F1 Score**: Threshold 0.3 (21.7%)
   - **Highest Precision**: Threshold 0.5 (24.0%)
   - **Highest Recall**: Threshold 0.3 (21.7%)

---

## Detailed Breakdown

### Threshold 0.3
- **Coverage**: 100.0% (all 69 predictions marked as confident)
- **Accuracy**: 21.7% (15 correct out of 69 confident)
- **Uncertain**: 0 predictions
- **Trade-off**: Maximum coverage, but includes lower-confidence predictions

### Threshold 0.4
- **Coverage**: 87.0% (60 confident, 9 uncertain)
- **Accuracy**: 20.0% (12 correct out of 60 confident)
- **Uncertain**: 9 predictions (5 correct = 55.6% of uncertain)
- **Trade-off**: Good balance between coverage and precision

### Threshold 0.5
- **Coverage**: 72.5% (50 confident, 19 uncertain)
- **Accuracy**: 24.0% (12 correct out of 50 confident)
- **Uncertain**: 19 predictions (10 correct = 52.6% of uncertain)
- **Trade-off**: Higher precision but lower coverage

---

## Per-Species Performance (Top Species)

### Galagoides_sp_nov
- **Precision**: 64.3% (at threshold 0.3)
- **Recall**: 37.5%
- **F1**: 47.4%
- **Performance**: Best performing species

### Paragalago_rondoensis
- **Precision**: 100.0% (at threshold 0.3)
- **Recall**: 9.1%
- **F1**: 16.7%
- **Performance**: High precision but low recall (few predictions)

### Paragalago_orinus
- **Precision**: 10.0%
- **Recall**: 12.5%
- **F1**: 11.1%
- **Performance**: Low performance, many false positives

### Otolemur_crassicaudatus
- **Precision**: 10.0%
- **Recall**: 12.5%
- **F1**: 11.1%
- **Performance**: Low performance, many false positives

### Otolemur_garnettii
- **Precision**: 9.1%
- **Recall**: 25.0%
- **F1**: 13.3%
- **Performance**: Low precision, many false positives (20 FP)

---

## Recommendations

### For Maximum Coverage
**Use threshold 0.3**
- 100% of predictions marked as confident
- Best F1 score (21.7%)
- Best recall (21.7%)
- Suitable when you want predictions for all files

### For Balanced Performance
**Use threshold 0.4**
- 87% coverage (good balance)
- 20.0% precision
- 55.6% of uncertain predictions are correct (suggesting top-3 might be useful)
- Recommended for general use

### For Higher Precision
**Use threshold 0.5**
- Highest precision (24.0%)
- 72.5% coverage
- 52.6% of uncertain predictions are correct
- Use when you prioritize accuracy over coverage

---

## Observations

1. **Low Overall Accuracy**: The top-1 accuracy is relatively low (20-24%) across all thresholds. This suggests:
   - The model may need improvement
   - The dataset may be challenging
   - There may be label mismatches or data quality issues

2. **Top-3 Accuracy is Better**: At 44.9%, the top-3 accuracy is nearly double the top-1 accuracy, suggesting:
   - The model is often close to the correct answer
   - Using top-3 predictions might be more useful in practice
   - The confidence threshold may not be the main issue

3. **Species-Specific Performance**: 
   - `Galagoides_sp_nov` performs best (64.3% precision)
   - Some species have very low precision (e.g., `Otolemur_garnettii` with 9.1%)
   - This suggests class imbalance or training data issues

4. **Uncertain Predictions**: 
   - At threshold 0.4, 55.6% of uncertain predictions are correct
   - At threshold 0.5, 52.6% of uncertain predictions are correct
   - This suggests that many "uncertain" predictions are actually correct, indicating the threshold might be too conservative

---

## Next Steps

1. **Investigate Low Accuracy**
   - Check for label mismatches between file paths and true labels
   - Review training data quality
   - Consider model improvements (architecture, training data, etc.)

2. **Consider Top-3 Predictions**
   - Since top-3 accuracy is 44.9%, consider using top-3 predictions in the UI
   - Allow users to see top-3 species with confidence scores

3. **Species-Specific Thresholds**
   - Consider different thresholds for different species
   - Some species (like `Galagoides_sp_nov`) perform well, others don't

4. **Model Improvements**
   - Address class imbalance
   - Improve training data for low-performing species
   - Consider data augmentation or transfer learning

5. **Validation on Oxford Brookes Data**
   - Test thresholds on Oxford Brookes data (same source as training)
   - This will help validate if the low accuracy is dataset-specific

---

## Files Generated

- **Script**: `scripts/test_threshold_impact.py`
- **Results JSON**: `outputs/evaluation/threshold_impact_analysis.json`
- **This Document**: `docs/threshold_impact_analysis_2026.md`

---

## Technical Details

### Methodology
- Re-interpreted existing predictions from `predictions_3stage_context.csv`
- Applied different thresholds to `top1_prob` values
- Calculated metrics: accuracy, precision, recall, F1, coverage
- Per-species metrics for top 10 species by count

### Metrics Definitions
- **Coverage**: % of predictions marked as confident (≥ threshold)
- **Accuracy/Precision**: % of confident predictions that are correct
- **Recall**: % of true positives / (true positives + false negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **Top-3 Accuracy**: % where true label is in top-3 predictions

---

**Analysis completed**: January 8, 2026
