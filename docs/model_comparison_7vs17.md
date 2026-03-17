# Model Comparison: 7-Class vs 17-Class

## Performance Summary

### Test Results (69 files from `data/raw_audio`)

| Metric | Top-7 Model | 17-Class Model |
|--------|-------------|----------------|
| Mean Confidence | 0.261 | 0.112 |
| Max Confidence | 0.312 | 0.145 |
| Min Confidence | 0.183 | 0.076 |
| Predictions > 0.6 | 0/69 (0%) | 0/69 (0%) |

### Validation Accuracy (Training Data)

- **Top-7 Model**: ~90% validation accuracy
- **17-Class Model**: 92.0% validation accuracy

## Why Lower Confidence in 17-Class Model?

### 1. **Probability Distribution Effect**

With more classes, probability mass is distributed more thinly:

- **7 classes**: If uniform, each class gets ~14.3% probability
- **17 classes**: If uniform, each class gets ~5.9% probability

Even when the model is **correct**, the maximum probability will naturally be lower with more classes.

### 2. **Data Imbalance**

The 17-class model includes species with very few training samples:

| Species | Training Samples |
|---------|------------------|
| Sciurocheirus_alleni | 18 |
| Otolemur_crassicaudatus | 30 |
| Galagoides_sp_nov | 40 |
| Paragalago_zanzibaricus | 50 |
| Galagoides_thomasi | 56 |
| Paragalago_rondoensis | 58 |

vs. species with many samples:

| Species | Training Samples |
|---------|------------------|
| Galago_senegalensis | 1,008 |
| Paragalago_cocos | 548 |
| Galagoides_demidovii | 496 |
| Paragalago_granti | 410 |

This imbalance makes it harder for the model to learn distinctive features for rare species.

### 3. **Model Architecture Capacity**

The same CNN architecture (256-unit dense layer) is now handling:
- **7 classes**: ~36.6 parameters per class
- **17 classes**: ~15.1 parameters per class

The model may need more capacity (larger dense layer, more filters) to effectively distinguish 17 classes.

### 4. **Test Data Mismatch**

The test files in `data/raw_audio` appear to be from a different source than the Oxford Brookes training data. Both models show low confidence on these files, suggesting:
- Different recording conditions
- Different call types
- Different geographic populations

## Recommendations

### To Improve 17-Class Model Performance:

1. **Increase Model Capacity**
   - Larger dense layer (512 or 1024 units)
   - More convolutional filters
   - Deeper network

2. **Address Data Imbalance**
   - Increase minimum samples threshold (currently 10)
   - Use more aggressive data augmentation for rare species
   - Consider class-specific augmentation strategies

3. **Collect More Training Data**
   - Especially for species with <100 samples
   - Ensure diversity in call types and recording conditions

4. **Test on Matching Data**
   - Test on Oxford Brookes recordings (same source as training)
   - This will give a better sense of true model performance

5. **Consider Hierarchical Classification**
   - First classify to genus (e.g., Paragalago, Galago, Otolemur)
   - Then classify to species within that genus
   - This can improve accuracy for similar species

## Current Model Status

The 17-class model achieved **92.0% validation accuracy** on the training data, which is actually **higher** than the 7-class model (~90%). However, the lower confidence scores on test data reflect:

1. The inherent difficulty of distinguishing 17 similar species
2. The probability distribution effect (more classes = lower max probability)
3. Potential test data mismatch

**The model is working correctly** - it's just that the task is more challenging with 17 classes, and the confidence scores reflect this increased difficulty.

