# Diagnostic Findings - Uniform Probability Issue

## Critical Issue Found: Class Mapping Mismatch

### Problem
The 16-class model outputs **16 classes**, but `class_names.json` has **19 classes**. This causes predictions to be mapped to the wrong species!

### Evidence
1. **Debug script shows model IS making confident predictions**:
   - Max probabilities: 0.97, 0.99, 0.43 (NOT uniform!)
   - Model outputs 16 probabilities, but we're trying to map to 19 classes

2. **Class mapping verification**:
   - 16-class model: 16 output classes
   - class_names.json: 19 classes
   - **Mismatch = wrong predictions**

### Root Cause
The 16-class model was trained with `MIN_SAMPLES_PER_CLASS = 30`, which excludes:
- `Sciurocheirus_alleni` (18 samples)
- `Sciurocheirus_gabonensis` (2 samples)  
- `Sciurocheirus_makandensis` (2 samples)

But `class_names.json` includes all 19 species, causing index misalignment.

### Fix Applied
1. Created `class_names_16.json` with the correct 16 classes
2. Updated `predict_3stage_with_context.py` to use model-specific class names
3. The 16 classes are (alphabetically sorted):
   - Euoticus_elegantulus
   - Euoticus_pallidus
   - Galago_gallarum
   - Galago_matschiei
   - Galago_moholi
   - Galago_senegalensis
   - Galagoides_demidovii
   - Galagoides_sp_nov
   - Galagoides_thomasi
   - Otolemur_crassicaudatus
   - Otolemur_garnettii
   - Paragalago_cocos
   - Paragalago_granti
   - Paragalago_orinus
   - Paragalago_rondoensis
   - Paragalago_zanzibaricus

### Next Steps
1. Re-run predictions with correct class mapping
2. Verify predictions are now correct
3. Test the 16-class model on your test set

### Sanity Check Results
The overfit test **failed** (stuck at 50% accuracy on 2 classes). This suggests there may be additional issues, but the class mapping fix should resolve the primary problem.
