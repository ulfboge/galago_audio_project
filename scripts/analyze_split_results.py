"""
Analyze prediction accuracy split by holdout vs train split.
Shows combined results and per-split breakdown.
"""
import csv
import json
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load split information
SPLIT_JSON = PROJECT_ROOT / "data" / "splits" / "raw_audio_holdout.json"
CSV_PATH = PROJECT_ROOT / "outputs" / "predictions" / "predictions_3stage_context.csv"

# Label mapping (same as analyze_prediction_accuracy.py)
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

def analyze_split(predictions, split_name, split_files):
    """Analyze predictions for a specific split."""
    correct = 0
    total = 0
    uncertain_count = 0
    correct_uncertain = 0
    filtered_by_detector = 0
    
    species_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'top3_correct': 0, 'filtered': 0})
    confusion = defaultdict(lambda: defaultdict(int))
    
    split_files_set = set(split_files)
    
    for row in predictions:
        filepath = row['filepath']
        if filepath not in split_files_set:
            continue
        
        true_label = get_mapped_label_from_filepath(filepath)
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
        
        total += 1
        species_stats[true_label]['total'] += 1
        
        # Count "uncertain" based on the output label (thresholded)
        if predicted_output == 'uncertain':
            uncertain_count += 1
            # For uncertain, check if true label is in raw top-3
            if true_label in [top1, top2, top3]:
                correct_uncertain += 1
                species_stats[true_label]['top3_correct'] += 1
        else:
            # "Correct predictions" uses the output label (thresholded)
            if predicted_output == true_label:
                correct += 1
                species_stats[true_label]['correct'] += 1
            else:
                # Track confusion
                confusion[true_label][predicted_output] += 1
            
            # Check top 3
            if true_label in [top1, top2, top3]:
                species_stats[true_label]['top3_correct'] += 1
    
    accuracy = (correct / total) * 100 if total > 0 else 0
    top3_accuracy = sum(s['top3_correct'] for s in species_stats.values()) / total * 100 if total > 0 else 0
    
    return {
        'name': split_name,
        'total': total,
        'correct': correct,
        'accuracy': accuracy,
        'uncertain_count': uncertain_count,
        'correct_uncertain': correct_uncertain,
        'top3_accuracy': top3_accuracy,
        'filtered_by_detector': filtered_by_detector,
        'species_stats': species_stats,
        'confusion': confusion
    }

def main():
    print("Analyzing prediction accuracy by split...")
    print("=" * 60)

    # Optional overrides:
    #   --csv <path>
    #   --split-json <path>
    args = sys.argv[1:]
    csv_path = CSV_PATH
    split_json = SPLIT_JSON

    if "--csv" in args:
        idx = args.index("--csv")
        if idx + 1 >= len(args):
            print("ERROR: --csv requires a path")
            return
        csv_path = Path(args[idx + 1])

    if "--split-json" in args:
        idx = args.index("--split-json")
        if idx + 1 >= len(args):
            print("ERROR: --split-json requires a path")
            return
        split_json = Path(args[idx + 1])
    
    # Load split information
    if not split_json.exists():
        print(f"ERROR: Split JSON not found: {split_json}")
        return
    
    with open(split_json, 'r') as f:
        split_data = json.load(f)
    
    holdout_files = set(split_data['holdout_files'])
    train_files = set(split_data['train_files'])
    
    # Load predictions
    if not csv_path.exists():
        print(f"ERROR: Predictions CSV not found: {csv_path}")
        return
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        predictions = list(reader)
    
    # Analyze each split
    holdout_results = analyze_split(predictions, "Holdout", holdout_files)
    train_results = analyze_split(predictions, "Train", train_files)
    
    # Combined results
    combined_total = holdout_results['total'] + train_results['total']
    combined_correct = holdout_results['correct'] + train_results['correct']
    combined_uncertain = holdout_results['uncertain_count'] + train_results['uncertain_count']
    combined_correct_uncertain = holdout_results['correct_uncertain'] + train_results['correct_uncertain']
    combined_top3_correct = sum(s['top3_correct'] for s in holdout_results['species_stats'].values()) + \
                           sum(s['top3_correct'] for s in train_results['species_stats'].values())
    combined_top3_accuracy = (combined_top3_correct / combined_total * 100) if combined_total > 0 else 0
    combined_accuracy = (combined_correct / combined_total * 100) if combined_total > 0 else 0
    
    # Print results
    print(f"\n{'='*60}")
    print("COMBINED RESULTS (Holdout + Train)")
    print(f"{'='*60}")
    print(f"  Total files analyzed: {combined_total}")
    print(f"  Correct predictions: {combined_correct} ({combined_accuracy:.1f}%)")
    print(f"  Uncertain predictions: {combined_uncertain} ({combined_uncertain/combined_total*100:.1f}%)")
    if combined_uncertain > 0:
        print(f"  Correct among uncertain: {combined_correct_uncertain} ({combined_correct_uncertain/combined_uncertain*100:.1f}% of uncertain)")
    print(f"  Top-3 accuracy: {combined_top3_accuracy:.1f}%")
    
    # Per-split breakdown
    for results in [holdout_results, train_results]:
        print(f"\n{'-'*60}")
        print(f"{results['name'].upper()} SPLIT ({results['total']} files)")
        print(f"{'-'*60}")
        print(f"  Correct predictions: {results['correct']} ({results['accuracy']:.1f}%)")
        if results['total'] > 0:
            print(f"  Uncertain predictions: {results['uncertain_count']} ({results['uncertain_count']/results['total']*100:.1f}%)")
        else:
            print(f"  Uncertain predictions: {results['uncertain_count']} (N/A)")
        if results['uncertain_count'] > 0:
            print(f"  Correct among uncertain: {results['correct_uncertain']} ({results['correct_uncertain']/results['uncertain_count']*100:.1f}% of uncertain)")
        print(f"  Top-3 accuracy: {results['top3_accuracy']:.1f}%")
        if results['filtered_by_detector'] > 0:
            print(f"  Filtered by detector: {results['filtered_by_detector']}")
    
    # Combined per-species breakdown
    print(f"\n{'-'*60}")
    print("COMBINED PER-SPECIES ACCURACY")
    print(f"{'-'*60}")
    
    # Merge species stats
    all_species = set(holdout_results['species_stats'].keys()) | set(train_results['species_stats'].keys())
    for species in sorted(all_species):
        holdout_stats = holdout_results['species_stats'].get(species, {'correct': 0, 'total': 0, 'top3_correct': 0})
        train_stats = train_results['species_stats'].get(species, {'correct': 0, 'total': 0, 'top3_correct': 0})
        
        total_for_species = holdout_stats['total'] + train_stats['total']
        if total_for_species == 0:
            continue
        
        correct_for_species = holdout_stats['correct'] + train_stats['correct']
        top3_correct_for_species = holdout_stats['top3_correct'] + train_stats['top3_correct']
        
        acc = (correct_for_species / total_for_species) * 100 if total_for_species > 0 else 0
        top3_acc = (top3_correct_for_species / total_for_species) * 100 if total_for_species > 0 else 0
        
        print(f"  {species:30s}: {correct_for_species:2d}/{total_for_species:2d} ({acc:5.1f}%) | Top-3: {top3_acc:5.1f}%")
    
    # Combined top confusions
    print(f"\n{'-'*60}")
    print("TOP CONFUSIONS (True -> Predicted)")
    print(f"{'-'*60}")
    
    # Merge confusions
    all_confusions = defaultdict(lambda: defaultdict(int))
    for true_label, preds in holdout_results['confusion'].items():
        for pred, count in preds.items():
            all_confusions[true_label][pred] += count
    for true_label, preds in train_results['confusion'].items():
        for pred, count in preds.items():
            all_confusions[true_label][pred] += count
    
    confusions_sorted = []
    for true_label, preds in all_confusions.items():
        for pred, count in preds.items():
            confusions_sorted.append((true_label, pred, count))
    confusions_sorted.sort(key=lambda x: x[2], reverse=True)
    
    for true_label, pred, count in confusions_sorted[:15]:
        print(f"  {true_label:30s} -> {pred:30s}: {count} times")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
