# Next Steps Roadmap

## Current Status ✅

**Completed:**
- ✅ 3-stage Merlin-like system (detector → classifier → context re-ranker)
- ✅ Binary detector (99.6% accuracy, 100% recall)
- ✅ Context re-ranking (location-based)
- ✅ Comprehensive evaluation metrics
- ✅ All core infrastructure

**Known Issues:**
- ⚠️ Low species confidence (most predictions are "uncertain")
- ⚠️ Limited negative data (only 100 synthetic samples)
- ⚠️ Context re-ranker incomplete (seasonality/time-of-night not implemented)

---

## Recommended Next Steps (Prioritized)

### 🔴 Priority 1: Address Low Confidence Issue

**Problem**: Most species predictions are below 0.6 confidence threshold, marked as "uncertain"

**Root Causes** (from analysis):
1. **Probability distribution effect**: 16 classes = lower max probability (~6.25% uniform baseline)
2. **Data imbalance**: Some species have <100 samples vs. 1,008 for others
3. **Test data mismatch**: Test files from different source than training data
4. **Model capacity**: May need more capacity for 16 classes

**Actions**:

#### Option A: Adjust Confidence Threshold (Quick Fix)
- Lower threshold from 0.6 to 0.3-0.4 for 16-class model
- Accept that with more classes, max probability will be lower
- **Pros**: Immediate improvement, no retraining
- **Cons**: May increase false positives

#### Option B: Improve Model Architecture (Better Solution)
- Increase model capacity further (1024-unit dense layer)
- Add attention mechanisms
- Use transfer learning (pre-trained audio models)
- **Pros**: Better discrimination, higher confidence
- **Cons**: Requires retraining, more compute

#### Option C: Hierarchical Classification (Best Long-term)
- First classify to genus (Paragalago, Galago, Otolemur, etc.)
- Then classify to species within genus
- **Pros**: Reduces confusion, higher confidence
- **Cons**: Requires restructuring, more complex

**Recommendation**: Start with **Option A** (quick validation), then **Option B** (if needed)

---

### 🟡 Priority 2: Expand Negative Data Collection

**Current State**: Only 100 synthetic noise samples

**Goal**: Collect diverse real-world negative examples

**Actions**:
1. **Add real field recordings**:
   - Insects (crickets, cicadas, katydids)
   - Frogs (various species)
   - Birds (night birds, similar frequency range)
   - Other primates (if available)
   - Human speech/activity
   - Equipment noise

2. **Sources**:
   - Freesound.org (free audio library)
   - Xeno-canto (bird sounds)
   - Field recordings from collaborators
   - Generate more synthetic samples

3. **Target**: 500-1000 negative samples (5-10x current)

**Impact**: 
- Better detector robustness
- Lower false positive rate
- More realistic field performance

---

### 🟡 Priority 3: Complete Context Re-ranker

**Current State**: Location working, seasonality/time-of-night not implemented

**Actions**:

1. **Add Seasonality Data**:
   - Research active months per species
   - Breeding seasons
   - Activity patterns by month
   - Update `data/species_ranges.json` with seasonality

2. **Add Time-of-Night Preferences**:
   - Research peak activity times per species
   - Early night vs. late night patterns
   - Moon phase effects (if available)
   - Update context re-ranker with species-specific patterns

3. **Fine-tune Context Weight**:
   - Test different `alpha` values (0.3, 0.5, 0.7)
   - Measure impact on accuracy
   - Optimize for field use

**Impact**: Better predictions in specific contexts

---

### 🟢 Priority 4: Field Testing & Validation

**Goal**: Validate system on real-world scenarios

**Actions**:

1. **Test on Diverse Datasets**:
   - Oxford Brookes recordings (same source as training)
   - Different geographic locations
   - Different recording conditions
   - Different call types

2. **Measure Real Performance**:
   - Precision/recall per species
   - False positive rate on non-galago audio
   - Context re-ranking impact
   - Detector filtering effectiveness

3. **User Feedback**:
   - Test with field researchers
   - Collect edge cases
   - Identify common failures

**Impact**: Understand real-world performance, identify improvements

---

### 🟢 Priority 5: Hard Negatives Training

**Goal**: Improve detector discrimination on confusing sounds

**Actions**:

1. **Collect Hard Negatives**:
   - Sounds similar to galago calls
   - Other small mammals
   - Similar frequency ranges
   - Ambiguous recordings

2. **Retrain Detector**:
   - Add hard negatives to training
   - Focus on boundary cases
   - Improve discrimination

**Impact**: Better detector, fewer false positives

---

### 🔵 Priority 6: Performance Optimization

**Goal**: Speed up inference for real-time use

**Actions**:

1. **Model Optimization**:
   - Quantization (INT8)
   - Pruning
   - Model distillation

2. **Inference Optimization**:
   - Batch processing
   - GPU acceleration
   - Mobile deployment (TensorFlow Lite)

**Impact**: Faster predictions, mobile-ready

---

## Recommended Implementation Order

### Phase 1: Quick Wins (1-2 days)
1. ✅ **Adjust confidence threshold** (Option A)
   - Test with 0.3, 0.4, 0.5 thresholds
   - Measure impact on predictions
   - Document findings

2. ✅ **Add more negative data** (synthetic + real)
   - Generate 200 more synthetic samples
   - Download 100-200 real samples from Freesound
   - Retrain detector

### Phase 2: Model Improvements (3-5 days)
3. ✅ **Improve model architecture** (Option B)
   - Increase capacity (1024-unit dense)
   - Add dropout/regularization
   - Retrain and compare

4. ✅ **Complete context re-ranker**
   - Add seasonality data
   - Add time-of-night preferences
   - Test and optimize

### Phase 3: Validation & Deployment (1-2 weeks)
5. ✅ **Field testing**
   - Test on diverse datasets
   - Measure real performance
   - Collect feedback

6. ✅ **Hard negatives training**
   - Collect confusing sounds
   - Retrain detector
   - Validate improvements

---

## Immediate Action Items

**This Week:**
1. [ ] Test confidence threshold adjustment (0.3, 0.4, 0.5)
2. [ ] Collect 200+ more negative samples
3. [ ] Retrain detector with expanded negative data
4. [ ] Test on Oxford Brookes data (same source as training)

**Next Week:**
1. [ ] Improve model architecture (if threshold adjustment insufficient)
2. [ ] Add seasonality data to context re-ranker
3. [ ] Field test on diverse datasets
4. [ ] Document performance improvements

---

## Success Metrics

**Short-term (1 month):**
- ✅ >50% predictions above confidence threshold
- ✅ 500+ negative samples
- ✅ Context re-ranker complete (location + season + time)
- ✅ Field-tested on 3+ datasets

**Long-term (3 months):**
- ✅ >80% predictions above confidence threshold
- ✅ 1000+ negative samples
- ✅ Hard negatives training complete
- ✅ Mobile-ready deployment
- ✅ Published performance metrics

---

## Questions to Answer

1. **What's the acceptable confidence threshold for 16 classes?**
   - Is 0.3-0.4 acceptable, or do we need 0.6+?
   - How does this compare to Merlin Bird-ID?

2. **How much negative data is enough?**
   - Current: 100 samples
   - Target: 500-1000?
   - What's the optimal ratio (galago:not_galago)?

3. **What's the priority: accuracy or confidence?**
   - High accuracy with lower confidence?
   - Or higher confidence with acceptable accuracy?

4. **What's the deployment target?**
   - Desktop application?
   - Mobile app?
   - Web service?
   - This affects optimization priorities

---

**Last Updated**: December 17, 2025
**Status**: Ready for implementation

