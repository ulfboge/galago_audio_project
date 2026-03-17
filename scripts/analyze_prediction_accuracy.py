"""
Analyze prediction accuracy by comparing predicted species to true labels.
Even if confidence is low, we can check if the model is getting the right species.
Supports both batch prediction format and 3-stage pipeline format.
"""
import csv
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Try 3-stage CSV first, fallback to batch format
CSV_PATH_3STAGE = PROJECT_ROOT / "outputs" / "predictions" / "predictions_3stage_context.csv"
CSV_PATH_BATCH = PROJECT_ROOT / "outputs" / "predictions" / "predictions_all_species.csv"

# Label mapping (same as batch script)
LABEL_MAP = {
    "G.sp.nov.1": "Galagoides_sp_nov",
    "G.sp.nov.3": "Galagoides_sp_nov",
    "G.granti": "Paragalago_granti",
    "G.orinus": "Paragalago_orinus",
    "G.rondoensis": "Paragalago_rondoensis",
    "G.zanzibaricus": "Paragalago_zanzibaricus",
    "O.crassicaudatus": "Otolemur_crassicaudatus",
    "O.garnettii": "Otolemur_garnettii",
    # legacy/aliases
    "Galago_granti": "Paragalago_granti",
}

AUDIO_DIR = PROJECT_ROOT / "data" / "raw_audio"

def get_mapped_label_from_filepath(filepath: str) -> str:
    """Extract mapped folder label from filepath."""
    try:
        wav_path = Path(filepath)
        rel = wav_path.relative_to(AUDIO_DIR)
        src_folder = rel.parts[0] if len(rel.parts) > 1 else wav_path.parent.name
        return LABEL_MAP.get(src_folder, src_folder)
    except Exception:
        # Fallback: try to extract from path
        parts = Path(filepath).parts
        for part in parts:
            if part in LABEL_MAP:
                return LABEL_MAP[part]
        # Last resort: use parent folder name
        return Path(filepath).parent.name

def main():
    print("Analyzing prediction accuracy...")
    print("=" * 60)

    # Optional override: --csv <path>
    csv_override = None
    args = sys.argv[1:]
    if "--csv" in args:
        idx = args.index("--csv")
        if idx + 1 >= len(args):
            print("ERROR: --csv requires a path")
            return
        csv_override = Path(args[idx + 1])
    
    # Determine which CSV to use
    if csv_override is not None:
        CSV_PATH = csv_override
        if not CSV_PATH.exists():
            print(f"ERROR: CSV not found: {CSV_PATH}")
            return
        print(f"Using CSV: {CSV_PATH.name}")
        # Detect format from headers
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames or [])
        is_3stage = "top1_species" in fieldnames and "detector_result" in fieldnames
    else:
        if CSV_PATH_3STAGE.exists():
            CSV_PATH = CSV_PATH_3STAGE
            is_3stage = True
            print(f"Using 3-stage pipeline results: {CSV_PATH.name}")
        elif CSV_PATH_BATCH.exists():
            CSV_PATH = CSV_PATH_BATCH
            is_3stage = False
            print(f"Using batch prediction results: {CSV_PATH.name}")
        else:
            print(f"ERROR: No prediction CSV found!")
            print(f"  Expected: {CSV_PATH_3STAGE} or {CSV_PATH_BATCH}")
            return
    
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    correct = 0
    total = 0
    uncertain_count = 0
    correct_uncertain = 0  # Correct predictions that were marked uncertain
    filtered_by_detector = 0  # Files filtered out by detector (3-stage only)
    
    # Per-species accuracy
    species_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'top3_correct': 0, 'filtered': 0})
    
    # Confusion tracking
    confusion = defaultdict(lambda: defaultdict(int))
    
    for row in rows:
        # Extract true label
        if is_3stage:
            # 3-stage format: extract from filepath
            true_label = get_mapped_label_from_filepath(row['filepath'])
            # species_result may be "uncertain"; top1_species (if present) holds the raw best species
            predicted_output = row.get('species_result', 'N/A')
            top1 = row.get('top1_species', row.get('species_result', 'N/A'))
            top2 = row['top2_species']
            top3 = row['top3_species']
            detector_result = row.get('detector_result', 'galago')
            
            # Skip files filtered by detector
            if detector_result != 'galago':
                filtered_by_detector += 1
                species_stats[true_label]['total'] += 1
                species_stats[true_label]['filtered'] += 1
                continue
        else:
            # Batch format: has mapped_folder_label
            true_label = row['mapped_folder_label']
            predicted = row['predicted_species']
            top2 = row['top2_species']
            top3 = row['top3_species']
        
        total += 1
        species_stats[true_label]['total'] += 1
        
        # Count "uncertain" based on the output label (thresholded)
        if is_3stage and predicted_output == 'uncertain':
            uncertain_count += 1
            # For uncertain, check if true label is in raw top-3 (top1/top2/top3)
            if true_label in [top1, top2, top3]:
                correct_uncertain += 1
                species_stats[true_label]['top3_correct'] += 1
        else:
            # "Correct predictions" uses the output label (thresholded)
            if (predicted_output if is_3stage else predicted) == true_label:
                correct += 1
                species_stats[true_label]['correct'] += 1
            else:
                # Track confusion
                confusion[true_label][(predicted_output if is_3stage else predicted)] += 1
            
            # Check top 3
            if is_3stage:
                if true_label in [top1, top2, top3]:
                    species_stats[true_label]['top3_correct'] += 1
            else:
                if true_label in [predicted, top2, top3]:
                    species_stats[true_label]['top3_correct'] += 1
    
    # Overall accuracy
    accuracy = (correct / total) * 100 if total > 0 else 0
    top3_accuracy = sum(s['top3_correct'] for s in species_stats.values()) / total * 100 if total > 0 else 0
    
    print(f"\nOverall Results:")
    print(f"  Total files analyzed: {total}")
    if is_3stage and filtered_by_detector > 0:
        print(f"  Filtered by detector: {filtered_by_detector} (not classified)")
        print(f"  Passed detector: {total}")
    print(f"  Correct predictions: {correct} ({accuracy:.1f}%)")
    print(f"  Uncertain predictions: {uncertain_count} ({uncertain_count/total*100:.1f}%)")
    if uncertain_count > 0:
        print(f"  Correct among uncertain: {correct_uncertain} ({correct_uncertain/uncertain_count*100:.1f}% of uncertain)")
    print(f"  Top-3 accuracy: {top3_accuracy:.1f}%")
    
    # Per-species breakdown
    print(f"\nPer-Species Accuracy:")
    print("-" * 60)
    for species in sorted(species_stats.keys()):
        stats = species_stats[species]
        total_for_species = stats['total']
        if total_for_species == 0:
            continue
        acc = (stats['correct'] / total_for_species) * 100 if total_for_species > 0 else 0
        top3_acc = (stats['top3_correct'] / total_for_species) * 100 if total_for_species > 0 else 0
        filtered_str = f" (filtered: {stats['filtered']})" if stats['filtered'] > 0 else ""
        print(f"  {species:30s}: {stats['correct']:2d}/{total_for_species:2d} ({acc:5.1f}%) | Top-3: {top3_acc:5.1f}%{filtered_str}")
    
    # Top confusions
    print(f"\nTop Confusions (True -> Predicted):")
    print("-" * 60)
    confusions_sorted = []
    for true_label, preds in confusion.items():
        for pred, count in preds.items():
            confusions_sorted.append((true_label, pred, count))
    confusions_sorted.sort(key=lambda x: x[2], reverse=True)
    
    for true_label, pred, count in confusions_sorted[:10]:
        print(f"  {true_label:30s} -> {pred:30s}: {count} times")
    
    print("\n" + "=" * 60)
    print("\nKey Insights:")
    print(f"  - Even with low confidence, the model is {'accurate' if accuracy > 50 else 'struggling'}")
    print(f"  - Top-3 accuracy of {top3_accuracy:.1f}% suggests the model is learning patterns")
    print(f"  - Consider lowering confidence threshold or using top-3 predictions")

if __name__ == "__main__":
    main()

