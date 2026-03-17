# Galago Audio Project - Task List for Microsoft To Do

**Last Updated**: January 8, 2026  
**Status**: Ready for implementation

This document contains a comprehensive list of tasks organized by priority for the Galago Acoustic Species Classifier project. Use this to populate your Microsoft To Do list and track progress.

---

## 🔴 Priority 1: Address Low Confidence Issue

### Quick Fixes (1-2 days)
- [x] Test confidence threshold adjustment (0.3, 0.4, 0.5) and measure impact
- [x] Document findings from threshold testing
- [x] Test on Oxford Brookes data (same source as training) to validate threshold changes

### Model Improvements (3-5 days)
- [ ] Improve model architecture — increase capacity (1024-unit dense layer)
- [ ] Add attention mechanisms to model
- [ ] Consider transfer learning with pre-trained audio models
- [ ] Retrain model with improved architecture and compare results

### Long-term
- [ ] Research hierarchical classification approach (genus → species)
- [ ] Evaluate feasibility of hierarchical classification

---

## 🟡 Priority 2: Expand Negative Data Collection

### This Week
- [ ] Collect 200+ more negative samples (synthetic + real)
- [ ] Generate 200 more synthetic noise samples
- [ ] Download 100-200 real samples from Freesound.org (insects, frogs, birds, etc.)
- [ ] Retrain detector with expanded negative data (target: 500-1000 samples)

### Sources to Explore
- [ ] Freesound.org (free audio library)
- [ ] Xeno-canto (bird sounds)
- [ ] Field recordings from collaborators
- [ ] Collect hard negatives (sounds similar to galago calls)

---

## 🟡 Priority 3: Complete Context Re-ranker

### Seasonality Data
- [ ] Research active months per species
- [ ] Research breeding seasons per species
- [ ] Research activity patterns by month
- [ ] Update `data/species_ranges.json` with seasonality data
- [ ] Implement seasonality priors in context re-ranker

### Time-of-Night Preferences
- [ ] Research peak activity times per species
- [ ] Research early night vs. late night patterns
- [ ] Research moon phase effects (if available)
- [ ] Update context re-ranker with species-specific time patterns

### Optimization
- [ ] Test different context weight `alpha` values (0.3, 0.5, 0.7)
- [ ] Measure impact on accuracy
- [ ] Optimize for field use

---

## 🟢 Priority 4: Evaluation and Validation

### Full Dataset Evaluation
- [x] Run full dataset evaluation with polygon priors (`evaluate_polygon_priors_impact.py`) — 69-file set; see session_2026-02-13_full_evaluation.md (single location → no accuracy change)
- [x] Run per-species accuracy analysis on baseline predictions (69-file set; see session_2026-02-13_full_evaluation.md)
- [x] Run per-species accuracy analysis on polygon-prior predictions (identical to baseline when location is uniform)
- [x] Compare which species benefit most (baseline 17-class vs improved 19-class; see session_2026-02-13_full_evaluation.md)

### Field Testing
- [ ] Test on diverse datasets (different locations, conditions, call types)
- [ ] Measure precision/recall per species
- [ ] Measure false positive rate on non-galago audio
- [ ] Evaluate context re-ranking impact
- [ ] Evaluate detector filtering effectiveness
- [ ] Collect user feedback from field researchers
- [ ] Document edge cases and common failures

### Location Data
- [x] Create `data/recording_locations.json` with explicit lat/lon for recordings (inferred: Pugu/Rondo/Pande → Tanzania, other → Kenya)
- [x] Test polygon priors with location mapping (per-file via `--location-map-json`; no accuracy change on 69-file set with 17-class)

---

## 🟢 Priority 5: Code Organization and Infrastructure

### Refactoring
- [ ] Define one canonical inference script
- [ ] Define one canonical evaluation workflow
- [ ] Consolidate duplicate logic (top-6 vs top-7, windowed vs non-windowed)
- [ ] Refactor scripts into clearer modules (data → inference → evaluation)
- [ ] Label which scripts are legacy vs active

### Documentation
- [ ] Add high-level README diagram explaining the flow
- [ ] Add README-level documentation for script responsibilities
- [ ] Document which script is "authoritative"
- [ ] Create configuration files instead of hard-coded paths

### Testing
- [ ] Add automated tests
- [ ] Set up model versioning/experiment tracking
- [ ] Standardize data schema/metadata handling

---

## 🔵 Priority 6: Advanced Improvements

### Training Enhancements
- [ ] Implement segment-based training (1-3s windows, aggregate at inference)
- [ ] Add SpecAugment-style augmentation (time/frequency masking)
- [x] Implement temperature scaling for better confidence calibration (T=0.212 in predict_3stage_with_context.py)
- [ ] Consider adding background/unknown class to classifier

### Iterative Training (Self-Supervised Learning)
- [ ] Run pipeline on all raw audio with high confidence threshold (0.9)
- [ ] Extract high-confidence predictions from CSV
- [ ] Ingest high-confidence raw audio into training set
- [ ] Retrain model with augmented dataset

---

## 🔵 Priority 7: Deployment Options

### Web App Prototype
- [ ] Build frontend: upload audio → click map for location → show top-3 predictions
- [ ] Build backend: run `predict_3stage_with_context.py` with user-provided lat/lon
- [ ] Implement map display: show predicted species' distribution polygon using GeoJSON
- [ ] Decide on GeoJSON loading strategy (all at once vs on-demand)

### Iterative UX (Merlin-Style)
- [ ] Design multi-pass recognition flow
- [ ] Implement Pass 1: sound-only prediction
- [ ] Implement Pass 2: location-based re-ranking
- [ ] Implement Pass 3: clarifying questions (habitat, tail, size)
- [ ] Extend `context_reranker.py` to accept additional context fields

### Performance Optimization
- [ ] Model quantization (INT8)
- [ ] Model pruning
- [ ] Model distillation
- [ ] Batch processing optimization
- [ ] GPU acceleration
- [ ] Mobile deployment (TensorFlow Lite)

### Geometry Optimization
- [ ] Simplify polygons for web performance
- [ ] Convert to TopoJSON for smaller file sizes
- [ ] Consider tile-based serving for large-scale deployment

---

## 🔵 Priority 8: Long-Term Features

### Active Learning
- [ ] Collect user corrections ("Was this correct?")
- [ ] Store corrections for model improvement
- [ ] Retrain periodically on new labeled data

### Real-Time Refinement
- [ ] Show predictions as audio plays
- [ ] Update predictions as more context arrives (location, time, etc.)

### Packaging
- [ ] Package as a library
- [ ] Prepare for publication/reproducible research

---

## 📊 Success Metrics to Track

### Short-term (1 month)
- [ ] >50% predictions above confidence threshold
- [ ] 500+ negative samples collected
- [ ] Context re-ranker complete (location + season + time)
- [ ] Field-tested on 3+ datasets

### Long-term (3 months)
- [ ] >80% predictions above confidence threshold
- [ ] 1000+ negative samples collected
- [ ] Hard negatives training complete
- [ ] Mobile-ready deployment
- [ ] Published performance metrics

---

## 📝 Notes

### Current System State
- ✅ 3-stage Merlin-like system (detector → classifier → context re-ranker)
- ✅ Binary detector (99.6% accuracy, 100% recall)
- ✅ Context re-ranking (location-based with polygon priors)
- ✅ Comprehensive evaluation metrics
- ✅ All core infrastructure

### Known Issues
- ⚠️ Low species confidence (most predictions are "uncertain")
- ⚠️ Limited negative data (only 100 synthetic samples)
- ⚠️ Context re-ranker incomplete (seasonality/time-of-night not implemented)

### Key Questions to Answer
1. What's the acceptable confidence threshold for 16 classes? (Is 0.3-0.4 acceptable, or do we need 0.6+?)
2. How much negative data is enough? (Current: 100 samples, Target: 500-1000?)
3. What's the priority: accuracy or confidence?
4. What's the deployment target? (Desktop app, mobile app, web service?)

---

## 🔗 Related Documents

- [Next Steps Roadmap](next_steps_roadmap.md) - Detailed roadmap with implementation phases
- [Session Summary 2026-01-08](2026-01-08_session_summary_and_handover.md) - Session notes
- [Session 2026-02-13: Full 69-file evaluation](session_2026-02-13_full_evaluation.md) - Baseline vs 19-class comparison, recommendation
- [Improvements Roadmap](improvements_roadmap.md) - AI recommendations and status
- [Merlin-like Roadmap](merlin_like_roadmap.md) - 2-stage detection implementation plan

---

**How to Use This Document:**
1. Copy tasks into Microsoft To Do as needed
2. Check off items as you complete them
3. Update this document periodically to reflect progress
4. Use the priority levels to guide your work order
