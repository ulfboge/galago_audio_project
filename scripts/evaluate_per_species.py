"""
Per-species evaluation for the 16-class model.
Shows detailed metrics for each species in the test set.
"""
import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "outputs" / "predictions" / "predictions_all_species.csv"

# Label mapping
LABEL_MAP = {
    "G.sp.nov.1": "Galagoides_sp_nov",
    "G.sp.nov.3": "Galagoides_sp_nov",
    "Galago_granti": "Paragalago_granti",
}

def load_predictions() -> List[Dict]:
    """Load predictions from CSV."""
    rows = []
    with CSV_PATH.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Map folder label
            src_folder = row['source_folder']
            mapped = LABEL_MAP.get(src_folder, src_folder)
            row['mapped_label'] = mapped
            rows.append(row)
    return rows

def evaluate_species(rows: List[Dict], species: str) -> Dict:
    """Evaluate predictions for a specific species."""
    species_rows = [r for r in rows if r['mapped_label'] == species]
    if not species_rows:
        return None
    
    total = len(species_rows)
    top1_correct = 0
    top3_correct = 0
    
    for row in species_rows:
        predicted = row['predicted_species']
        top2 = row['top2_species']
        top3 = row['top3_species']
        
        # Top-1 accuracy
        if predicted == species:
            top1_correct += 1
        
        # Top-3 accuracy
        if species in [predicted, top2, top3]:
            top3_correct += 1
    
    return {
        'species': species,
        'total': total,
        'top1_correct': top1_correct,
        'top1_accuracy': (top1_correct / total * 100) if total > 0 else 0,
        'top3_correct': top3_correct,
        'top3_accuracy': (top3_correct / total * 100) if total > 0 else 0,
    }

def main():
    print("Per-Species Evaluation for 16-Class Model")
    print("=" * 70)
    
    if not CSV_PATH.exists():
        print(f"ERROR: Predictions file not found: {CSV_PATH}")
        return
    
    rows = load_predictions()
    print(f"\nLoaded {len(rows)} predictions")
    
    # Get all species in test set
    species_set = {r['mapped_label'] for r in rows}
    
    # Evaluate each species
    results = []
    for species in sorted(species_set):
        result = evaluate_species(rows, species)
        if result:
            results.append(result)
    
    # Print table
    print(f"\n{'Species':<30} {'Total':>6} {'Top-1':>8} {'Top-1%':>8} {'Top-3':>8} {'Top-3%':>8}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['species']:<30} {r['total']:>6} {r['top1_correct']:>8} "
              f"{r['top1_accuracy']:>7.1f}% {r['top3_correct']:>8} {r['top3_accuracy']:>7.1f}%")
    
    # Summary
    total_files = sum(r['total'] for r in results)
    total_top1 = sum(r['top1_correct'] for r in results)
    total_top3 = sum(r['top3_correct'] for r in results)
    
    print("-" * 70)
    print(f"{'OVERALL':<30} {total_files:>6} {total_top1:>8} "
          f"{(total_top1/total_files*100):>7.1f}% {total_top3:>8} {(total_top3/total_files*100):>7.1f}%")
    
    # Species needing improvement
    print(f"\n{'='*70}")
    print("Species with Top-3 Accuracy < 50%:")
    print("-" * 70)
    for r in sorted(results, key=lambda x: x['top3_accuracy']):
        if r['top3_accuracy'] < 50:
            print(f"  {r['species']:<30}: {r['top3_accuracy']:>5.1f}% ({r['top3_correct']}/{r['total']})")

if __name__ == "__main__":
    main()
