"""
Analyze top-3 predictions to see actual model performance.
Even if marked "uncertain", the model might be correct in top-3.
"""
from pathlib import Path
import csv
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "outputs" / "predictions" / "predictions_3stage_context.csv"

# Label mapping (from analyze_prediction_accuracy.py)
LABEL_MAP = {
    "G.sp.nov.1": "Galagoides_sp_nov",
    "G.sp.nov.3": "Galagoides_sp_nov",
    "G.granti": "Paragalago_granti",
    "G.orinus": "Paragalago_orinus",
    "G.rondoensis": "Paragalago_rondoensis",
    "G.zanzibaricus": "Paragalago_zanzibaricus",
    "O.crassicaudatus": "Otolemur_crassicaudatus",
    "O.garnettii": "Otolemur_garnettii",
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
        parts = Path(filepath).parts
        for part in parts:
            if part in LABEL_MAP:
                return LABEL_MAP[part]
        return Path(filepath).parent.name

def main():
    print("=" * 70)
    print("Top-3 Prediction Analysis")
    print("=" * 70)
    
    if not CSV_PATH.exists():
        print(f"ERROR: CSV not found: {CSV_PATH}")
        return
    
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    total = 0
    top1_correct = 0
    top2_correct = 0
    top3_correct = 0
    
    species_stats = defaultdict(lambda: {'total': 0, 'top1': 0, 'top2': 0, 'top3': 0})
    
    for row in rows:
        true_label = get_mapped_label_from_filepath(row['filepath'])
        # species_result can be 'uncertain' when below threshold; use top1_species if present
        predicted = row.get('top1_species', row['species_result'])
        top2 = row.get('top2_species', 'N/A')
        top3 = row.get('top3_species', 'N/A')
        
        # Skip if filtered by detector
        if row.get('detector_result') != 'galago':
            continue
        
        total += 1
        species_stats[true_label]['total'] += 1
        
        # Check top-1
        if predicted == true_label:
            top1_correct += 1
            species_stats[true_label]['top1'] += 1
        
        # Check top-2
        if true_label in [predicted, top2]:
            top2_correct += 1
            species_stats[true_label]['top2'] += 1
        
        # Check top-3
        if true_label in [predicted, top2, top3]:
            top3_correct += 1
            species_stats[true_label]['top3'] += 1
    
    print(f"\nOverall Results:")
    print(f"  Total files: {total}")
    print(f"  Top-1 correct: {top1_correct} ({top1_correct/total*100:.1f}%)")
    print(f"  Top-2 correct: {top2_correct} ({top2_correct/total*100:.1f}%)")
    print(f"  Top-3 correct: {top3_correct} ({top3_correct/total*100:.1f}%)")
    
    print(f"\n{'='*70}")
    print("Per-Species Top-3 Analysis")
    print(f"{'='*70}")
    print(f"{'Species':<30} {'Total':<8} {'Top-1':<8} {'Top-2':<8} {'Top-3':<8}")
    print("-" * 70)
    
    for species in sorted(species_stats.keys()):
        stats = species_stats[species]
        total_s = stats['total']
        if total_s == 0:
            continue
        top1_pct = stats['top1'] / total_s * 100
        top2_pct = stats['top2'] / total_s * 100
        top3_pct = stats['top3'] / total_s * 100
        print(f"{species:<30} {total_s:<8} {top1_pct:>6.1f}%  {top2_pct:>6.1f}%  {top3_pct:>6.1f}%")
    
    # Show some example predictions
    print(f"\n{'='*70}")
    print("Sample Predictions (showing top-3)")
    print(f"{'='*70}")
    
    for i, row in enumerate(rows[:10]):
        true_label = get_mapped_label_from_filepath(row['filepath'])
        predicted = row.get('top1_species', row['species_result'])
        top2 = row.get('top2_species', 'N/A')
        top3 = row.get('top3_species', 'N/A')
        prob = row.get('species_prob', 'N/A')
        
        if row.get('detector_result') != 'galago':
            continue
        
        in_top3 = true_label in [predicted, top2, top3]
        marker = "[OK]" if in_top3 else "[X]"
        
        print(f"\n{marker} {Path(row['filepath']).name}")
        print(f"  True: {true_label}")
        print(f"  Top-3: {predicted}, {top2}, {top3}")
        print(f"  Prob: {prob}")

if __name__ == "__main__":
    main()
