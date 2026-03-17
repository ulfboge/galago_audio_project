"""
Test confidence threshold adjustment (0.3, 0.4, 0.5) and measure impact.

This script:
1. Loads existing predictions from predictions_3stage_context.csv
2. Re-interprets predictions at different thresholds (0.3, 0.4, 0.5)
3. Calculates comprehensive metrics: accuracy, precision, recall, F1, coverage
4. Compares results side-by-side
5. Generates a detailed report

Note: This does NOT re-run inference; it re-interprets existing probabilities.
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_CSV = PROJECT_ROOT / "outputs" / "predictions" / "predictions_3stage_context.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Thresholds to test
THRESHOLDS = [0.3, 0.4, 0.5]

# Label mapping (same as other scripts)
LABEL_MAP = {
    "G.sp.nov.1": "Galagoides_sp_nov",
    "G.sp.nov.3": "Galagoides_sp_nov",
    "G.granti": "Paragalago_granti",
    "G.orinus": "Paragalago_orinus",
    "G.rondoensis": "Paragalago_rondoensis",
    "G.zanzibaricus": "Paragalago_zanzibaricus",
    "O.crassicaudatus": "Otolemur_crassicaudatus",
    "O.garnettii": "Otolemur_garnettii",
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
        parts = Path(filepath).parts
        for part in parts:
            if part in LABEL_MAP:
                return LABEL_MAP[part]
        return Path(filepath).parent.name


def load_predictions(csv_path: Path) -> List[Dict]:
    """Load predictions from CSV."""
    if not csv_path.exists():
        print(f"ERROR: Predictions file not found: {csv_path}")
        print(f"\nPlease run predictions first:")
        print(f"  python scripts/predict_3stage_with_context.py")
        return []
    
    predictions = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            predictions.append(row)
    
    return predictions


def evaluate_threshold(predictions: List[Dict], threshold: float) -> Dict:
    """
    Evaluate predictions at a given threshold.
    
    Returns metrics including:
    - coverage: % of predictions marked as confident
    - accuracy: % of confident predictions that are correct
    - precision: same as accuracy (for confident predictions)
    - recall: % of true positives / (true positives + false negatives)
    - F1: harmonic mean of precision and recall
    - top3_accuracy: % where true label is in top-3
    """
    total = 0
    confident = 0
    uncertain = 0
    correct_confident = 0
    correct_uncertain = 0
    top3_correct = 0
    
    # Per-species metrics
    species_stats = defaultdict(lambda: {
        'true_positives': 0,
        'false_positives': 0,
        'false_negatives': 0,
        'true_negatives': 0,
        'total_true': 0,
        'confident_predictions': 0,
    })
    
    # Confusion matrix
    confusion = defaultdict(lambda: defaultdict(int))
    
    for row in predictions:
        # Skip errors
        if row.get('detector_result') == 'error' or row.get('species_result') == 'error':
            continue
        
        # Skip files filtered by detector
        if row.get('detector_result') != 'galago':
            continue
        
        # Extract true label
        true_label = get_mapped_label_from_filepath(row['filepath'])
        
        # Get prediction info
        top1_species = row.get('top1_species', '')
        top1_prob_str = row.get('top1_prob', 'N/A')
        top2_species = row.get('top2_species', '')
        top3_species = row.get('top3_species', '')
        
        if top1_prob_str == 'N/A':
            uncertain += 1
            total += 1
            species_stats[true_label]['total_true'] += 1
            species_stats[true_label]['false_negatives'] += 1
            continue
        
        try:
            top1_prob = float(top1_prob_str)
        except (ValueError, TypeError):
            uncertain += 1
            total += 1
            species_stats[true_label]['total_true'] += 1
            species_stats[true_label]['false_negatives'] += 1
            continue
        
        total += 1
        species_stats[true_label]['total_true'] += 1
        
        # Check if confident at this threshold
        is_confident = top1_prob >= threshold
        
        if is_confident:
            confident += 1
            predicted_label = top1_species
            
            # Check if correct
            if predicted_label == true_label:
                correct_confident += 1
                species_stats[true_label]['true_positives'] += 1
            else:
                # False positive for predicted species
                species_stats[predicted_label]['false_positives'] += 1
                # False negative for true species
                species_stats[true_label]['false_negatives'] += 1
                # Track confusion
                confusion[true_label][predicted_label] += 1
        else:
            uncertain += 1
            # For uncertain predictions, check if true label is in top-3
            if true_label in [top1_species, top2_species, top3_species]:
                correct_uncertain += 1
            # Count as false negative
            species_stats[true_label]['false_negatives'] += 1
        
        # Check top-3 accuracy (regardless of confidence)
        if true_label in [top1_species, top2_species, top3_species]:
            top3_correct += 1
    
    # Calculate overall metrics
    coverage = (confident / total * 100) if total > 0 else 0.0
    accuracy = (correct_confident / confident * 100) if confident > 0 else 0.0
    precision = accuracy  # For confident predictions, precision = accuracy
    top3_accuracy = (top3_correct / total * 100) if total > 0 else 0.0
    
    # Calculate recall (true positives / (true positives + false negatives))
    total_true_positives = sum(s['true_positives'] for s in species_stats.values())
    total_false_negatives = sum(s['false_negatives'] for s in species_stats.values())
    recall = (total_true_positives / (total_true_positives + total_false_negatives) * 100) if (total_true_positives + total_false_negatives) > 0 else 0.0
    
    # Calculate F1 score
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    # Per-species metrics
    per_species_metrics = {}
    for species, stats in species_stats.items():
        tp = stats['true_positives']
        fp = stats['false_positives']
        fn = stats['false_negatives']
        
        sp_precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
        sp_recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
        sp_f1 = (2 * sp_precision * sp_recall / (sp_precision + sp_recall)) if (sp_precision + sp_recall) > 0 else 0.0
        
        per_species_metrics[species] = {
            'precision': sp_precision,
            'recall': sp_recall,
            'f1': sp_f1,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'total_true': stats['total_true'],
        }
    
    return {
        'threshold': threshold,
        'total': total,
        'confident': confident,
        'uncertain': uncertain,
        'coverage': coverage,
        'correct_confident': correct_confident,
        'correct_uncertain': correct_uncertain,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'top3_accuracy': top3_accuracy,
        'per_species': per_species_metrics,
        'confusion': dict(confusion),
    }


def print_comparison_table(results: List[Dict]):
    """Print side-by-side comparison table."""
    print("\n" + "="*80)
    print("THRESHOLD COMPARISON - OVERALL METRICS")
    print("="*80)
    print(f"{'Metric':<25} {'0.3':>12} {'0.4':>12} {'0.5':>12}")
    print("-"*80)
    
    metrics = [
        ('Coverage (%)', 'coverage'),
        ('Confident Predictions', 'confident'),
        ('Uncertain Predictions', 'uncertain'),
        ('Accuracy (%)', 'accuracy'),
        ('Precision (%)', 'precision'),
        ('Recall (%)', 'recall'),
        ('F1 Score (%)', 'f1'),
        ('Top-3 Accuracy (%)', 'top3_accuracy'),
    ]
    
    for label, key in metrics:
        values = [f"{r[key]:.1f}" if isinstance(r[key], float) else f"{r[key]}" for r in results]
        print(f"{label:<25} {values[0]:>12} {values[1]:>12} {values[2]:>12}")
    
    print("\n" + "="*80)
    print("THRESHOLD COMPARISON - DETAILED BREAKDOWN")
    print("="*80)
    
    for r in results:
        print(f"\nThreshold: {r['threshold']:.1f}")
        print(f"  Total files: {r['total']}")
        print(f"  Confident: {r['confident']} ({r['coverage']:.1f}%)")
        print(f"  Uncertain: {r['uncertain']} ({100-r['coverage']:.1f}%)")
        print(f"  Correct (confident): {r['correct_confident']} ({r['accuracy']:.1f}%)")
        print(f"  Correct (uncertain): {r['correct_uncertain']} ({r['correct_uncertain']/r['uncertain']*100:.1f}% of uncertain)" if r['uncertain'] > 0 else "  Correct (uncertain): 0")
        print(f"  Precision: {r['precision']:.1f}%")
        print(f"  Recall: {r['recall']:.1f}%")
        print(f"  F1 Score: {r['f1']:.1f}%")
        print(f"  Top-3 Accuracy: {r['top3_accuracy']:.1f}%")


def print_per_species_comparison(results: List[Dict], top_n: int = 10):
    """Print per-species comparison for top N species by total count."""
    print("\n" + "="*80)
    print(f"PER-SPECIES METRICS (Top {top_n} by total count)")
    print("="*80)
    
    # Get all species and their total counts
    all_species = set()
    for r in results:
        all_species.update(r['per_species'].keys())
    
    # Get total counts across all thresholds
    species_totals = {}
    for species in all_species:
        for r in results:
            if species in r['per_species']:
                species_totals[species] = max(
                    species_totals.get(species, 0),
                    r['per_species'][species]['total_true']
                )
    
    # Sort by total count
    sorted_species = sorted(species_totals.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    for species, _ in sorted_species:
        print(f"\n{species}:")
        print(f"  {'Metric':<20} {'0.3':>10} {'0.4':>10} {'0.5':>10}")
        print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
        
        for r in results:
            if species in r['per_species']:
                sp_metrics = r['per_species'][species]
                print(f"  {'Precision (%)':<20} {sp_metrics['precision']:>10.1f}")
                print(f"  {'Recall (%)':<20} {sp_metrics['recall']:>10.1f}")
                print(f"  {'F1 (%)':<20} {sp_metrics['f1']:>10.1f}")
                print(f"  {'TP/FP/FN':<20} {sp_metrics['true_positives']}/{sp_metrics['false_positives']}/{sp_metrics['false_negatives']}")
                break  # Only print once per species (they should be similar across thresholds)


def save_results(results: List[Dict]):
    """Save results to JSON file."""
    output_file = OUTPUT_DIR / "threshold_impact_analysis.json"
    
    # Convert to JSON-serializable format
    json_results = []
    for r in results:
        json_r = {
            'threshold': float(r['threshold']),
            'total': r['total'],
            'confident': r['confident'],
            'uncertain': r['uncertain'],
            'coverage': float(r['coverage']),
            'correct_confident': r['correct_confident'],
            'correct_uncertain': r['correct_uncertain'],
            'accuracy': float(r['accuracy']),
            'precision': float(r['precision']),
            'recall': float(r['recall']),
            'f1': float(r['f1']),
            'top3_accuracy': float(r['top3_accuracy']),
            'per_species': {
                species: {
                    'precision': float(metrics['precision']),
                    'recall': float(metrics['recall']),
                    'f1': float(metrics['f1']),
                    'true_positives': metrics['true_positives'],
                    'false_positives': metrics['false_positives'],
                    'false_negatives': metrics['false_negatives'],
                    'total_true': metrics['total_true'],
                }
                for species, metrics in r['per_species'].items()
            },
        }
        json_results.append(json_r)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


def print_recommendations(results: List[Dict]):
    """Print recommendations based on results."""
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    # Find threshold with best F1 score
    best_f1 = max(results, key=lambda x: x['f1'])
    print(f"\nBest F1 Score: Threshold {best_f1['threshold']:.1f} (F1: {best_f1['f1']:.1f}%)")
    
    # Find threshold with best balance (coverage >= 50% and accuracy >= 70%)
    balanced = [r for r in results if r['coverage'] >= 50 and r['accuracy'] >= 70]
    if balanced:
        best_balanced = max(balanced, key=lambda x: x['f1'])
        print(f"\nBest Balanced (≥50% coverage, ≥70% accuracy): Threshold {best_balanced['threshold']:.1f}")
        print(f"  Coverage: {best_balanced['coverage']:.1f}%")
        print(f"  Accuracy: {best_balanced['accuracy']:.1f}%")
        print(f"  F1: {best_balanced['f1']:.1f}%")
    
    # Find threshold with highest precision
    best_precision = max(results, key=lambda x: x['precision'])
    print(f"\nHighest Precision: Threshold {best_precision['threshold']:.1f} (Precision: {best_precision['precision']:.1f}%)")
    
    # Find threshold with highest recall
    best_recall = max(results, key=lambda x: x['recall'])
    print(f"\nHighest Recall: Threshold {best_recall['threshold']:.1f} (Recall: {best_recall['recall']:.1f}%)")
    
    print("\n" + "-"*80)
    print("Trade-offs:")
    print("  - Lower threshold (0.3): More coverage, more predictions, but lower precision")
    print("  - Medium threshold (0.4): Balanced coverage and precision")
    print("  - Higher threshold (0.5): Higher precision, but less coverage")
    print("-"*80)


def main():
    print("="*80)
    print("CONFIDENCE THRESHOLD IMPACT ANALYSIS")
    print("="*80)
    print(f"\nTesting thresholds: {THRESHOLDS}")
    print(f"Using predictions from: {PREDICTIONS_CSV.name}")
    
    # Load predictions
    print("\nLoading predictions...")
    predictions = load_predictions(PREDICTIONS_CSV)
    
    if not predictions:
        return
    
    print(f"Loaded {len(predictions)} predictions")
    
    # Evaluate each threshold
    print("\nEvaluating thresholds...")
    results = []
    for threshold in THRESHOLDS:
        print(f"  Evaluating threshold {threshold:.1f}...")
        result = evaluate_threshold(predictions, threshold)
        results.append(result)
    
    # Print comparison
    print_comparison_table(results)
    print_per_species_comparison(results, top_n=10)
    print_recommendations(results)
    
    # Save results
    save_results(results)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Review the metrics above")
    print("2. Choose a threshold based on your priorities:")
    print("   - Need more predictions? Use lower threshold (0.3)")
    print("   - Need higher accuracy? Use higher threshold (0.5)")
    print("   - Want balance? Use medium threshold (0.4)")
    print("3. Update CLASSIFIER_THRESHOLD in prediction scripts if needed")
    print("4. Re-run predictions with chosen threshold")


if __name__ == "__main__":
    main()
