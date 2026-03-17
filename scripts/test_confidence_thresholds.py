"""
Test different confidence thresholds for the 16-class species classifier.

This script analyzes predictions at different thresholds to determine
the optimal balance between confidence and coverage.
"""
from pathlib import Path
import csv
import json
import numpy as np
import sys
from collections import defaultdict

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_CSV = PROJECT_ROOT / "outputs" / "predictions" / "predictions_3stage_context.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_predictions(csv_path: Path):
    """Load predictions from CSV."""
    if not csv_path.exists():
        print(f"ERROR: Predictions file not found: {csv_path}")
        print(f"\nPlease run predictions first:")
        print(f"  python scripts/predict_3stage_with_context.py")
        return []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)

def analyze_threshold(predictions: list, threshold: float):
    """Analyze predictions at a given confidence threshold."""
    total = 0
    confident = 0
    uncertain = 0
    not_classified = 0
    errors = 0
    
    confidence_scores = []
    confident_predictions = []
    uncertain_predictions = []
    
    species_counts = defaultdict(int)
    species_confident = defaultdict(int)
    
    for row in predictions:
        total += 1
        
        # Skip errors
        if row.get('detector_result') == 'error' or row.get('species_result') == 'error':
            errors += 1
            continue
        
        # Check if not classified by detector
        if row.get('detector_result') == 'not_galago':
            not_classified += 1
            continue
        
        # Get species result and probability
        species_result = row.get('species_result', '')
        species_prob_str = row.get('species_prob', 'N/A')
        
        if species_prob_str == 'N/A':
            uncertain += 1
            continue
        
        try:
            species_prob = float(species_prob_str)
            confidence_scores.append(species_prob)
            
            # Count by species
            if species_result and species_result != 'uncertain':
                species_counts[species_result] += 1
            
            # Check threshold
            if species_prob >= threshold:
                confident += 1
                confident_predictions.append({
                    'file': Path(row['filepath']).name,
                    'species': species_result,
                    'confidence': species_prob,
                })
                if species_result and species_result != 'uncertain':
                    species_confident[species_result] += 1
            else:
                uncertain += 1
                uncertain_predictions.append({
                    'file': Path(row['filepath']).name,
                    'species': species_result,
                    'confidence': species_prob,
                })
        except (ValueError, TypeError):
            uncertain += 1
    
    return {
        'threshold': threshold,
        'total': total,
        'confident': confident,
        'uncertain': uncertain,
        'not_classified': not_classified,
        'errors': errors,
        'confident_pct': (confident / total * 100) if total > 0 else 0,
        'confidence_scores': confidence_scores,
        'mean_confidence': np.mean(confidence_scores) if confidence_scores else 0,
        'median_confidence': np.median(confidence_scores) if confidence_scores else 0,
        'min_confidence': np.min(confidence_scores) if confidence_scores else 0,
        'max_confidence': np.max(confidence_scores) if confidence_scores else 0,
        'species_counts': dict(species_counts),
        'species_confident': dict(species_confident),
        'confident_predictions': confident_predictions[:10],  # Top 10
        'uncertain_predictions': uncertain_predictions[:10],  # Top 10
    }

def print_analysis(results: dict):
    """Print threshold analysis results."""
    print(f"\n{'='*70}")
    print(f"Confidence Threshold: {results['threshold']:.2f}")
    print(f"{'='*70}")
    
    print(f"\nOverall Statistics:")
    print(f"  Total files: {results['total']}")
    print(f"  Confident (≥{results['threshold']:.2f}): {results['confident']} ({results['confident_pct']:.1f}%)")
    print(f"  Uncertain (<{results['threshold']:.2f}): {results['uncertain']}")
    print(f"  Not classified (detector): {results['not_classified']}")
    print(f"  Errors: {results['errors']}")
    
    if results['confidence_scores']:
        print(f"\nConfidence Score Distribution:")
        print(f"  Mean: {results['mean_confidence']:.3f}")
        print(f"  Median: {results['median_confidence']:.3f}")
        print(f"  Min: {results['min_confidence']:.3f}")
        print(f"  Max: {results['max_confidence']:.3f}")
        
        # Percentiles
        scores = results['confidence_scores']
        p25 = np.percentile(scores, 25)
        p75 = np.percentile(scores, 75)
        p90 = np.percentile(scores, 90)
        p95 = np.percentile(scores, 95)
        print(f"  25th percentile: {p25:.3f}")
        print(f"  75th percentile: {p75:.3f}")
        print(f"  90th percentile: {p90:.3f}")
        print(f"  95th percentile: {p95:.3f}")
    
    if results['species_counts']:
        print(f"\nSpecies Distribution (Top 10):")
        sorted_species = sorted(results['species_counts'].items(), 
                               key=lambda x: x[1], reverse=True)
        print(f"  {'Species':<30} {'Total':<10} {'Confident':<10} {'% Confident':<12}")
        print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*12}")
        for species, count in sorted_species[:10]:
            confident = results['species_confident'].get(species, 0)
            pct = (confident / count * 100) if count > 0 else 0
            print(f"  {species[:29]:<30} {count:<10} {confident:<10} {pct:>11.1f}%")

def compare_thresholds(thresholds: list, predictions: list):
    """Compare multiple thresholds."""
    print("\n" + "="*70)
    print("CONFIDENCE THRESHOLD COMPARISON")
    print("="*70)
    
    results_list = []
    for threshold in thresholds:
        results = analyze_threshold(predictions, threshold)
        results_list.append(results)
        print_analysis(results)
    
    # Summary comparison
    print(f"\n{'='*70}")
    print("SUMMARY COMPARISON")
    print(f"{'='*70}")
    print(f"\n{'Threshold':<12} {'Confident':<12} {'% Confident':<15} {'Mean Conf':<12} {'Median Conf':<12}")
    print("-"*70)
    for results in results_list:
        print(f"{results['threshold']:<12.2f} {results['confident']:<12} "
              f"{results['confident_pct']:<15.1f} {results['mean_confidence']:<12.3f} "
              f"{results['median_confidence']:<12.3f}")
    
    # Recommendation
    print(f"\n{'='*70}")
    print("RECOMMENDATION")
    print(f"{'='*70}")
    
    # Find threshold that gives ~50% confident predictions
    target_pct = 50.0
    best_threshold = None
    best_diff = float('inf')
    
    for results in results_list:
        diff = abs(results['confident_pct'] - target_pct)
        if diff < best_diff:
            best_diff = diff
            best_threshold = results['threshold']
    
    print(f"\nFor ~50% confident predictions:")
    print(f"  Recommended threshold: {best_threshold:.2f}")
    
    # Find threshold at 25th percentile
    if results_list[0]['confidence_scores']:
        scores = results_list[0]['confidence_scores']
        p25_threshold = np.percentile(scores, 25)
        print(f"\nFor 25th percentile coverage:")
        print(f"  Threshold: {p25_threshold:.3f}")
    
    # Find threshold at median
    if results_list[0]['confidence_scores']:
        median_threshold = np.median(scores)
        print(f"\nFor median coverage:")
        print(f"  Threshold: {median_threshold:.3f}")
    
    # Conservative recommendation (75th percentile)
    if results_list[0]['confidence_scores']:
        p75_threshold = np.percentile(scores, 75)
        print(f"\nFor conservative (75th percentile) coverage:")
        print(f"  Threshold: {p75_threshold:.3f}")
    
    return results_list

def save_results(results_list: list):
    """Save analysis results to JSON."""
    output_file = OUTPUT_DIR / "confidence_threshold_analysis.json"
    
    # Convert to JSON-serializable format
    json_results = []
    for r in results_list:
        json_r = {
            'threshold': float(r['threshold']),
            'total': r['total'],
            'confident': r['confident'],
            'uncertain': r['uncertain'],
            'not_classified': r['not_classified'],
            'errors': r['errors'],
            'confident_pct': float(r['confident_pct']),
            'mean_confidence': float(r['mean_confidence']),
            'median_confidence': float(r['median_confidence']),
            'min_confidence': float(r['min_confidence']),
            'max_confidence': float(r['max_confidence']),
            'species_counts': r['species_counts'],
            'species_confident': r['species_confident'],
        }
        json_results.append(json_r)
    
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")

def main():
    print("Confidence Threshold Analysis")
    print("="*70)
    
    # Load predictions
    print("\nLoading predictions...")
    predictions = load_predictions(PREDICTIONS_CSV)
    
    if not predictions:
        return
    
    print(f"Loaded {len(predictions)} predictions")
    
    # Test different thresholds
    thresholds = [0.6, 0.5, 0.4, 0.3, 0.25, 0.2]
    
    print(f"\nTesting thresholds: {thresholds}")
    print(f"This will show how many predictions become 'confident' at each threshold.")
    
    # Compare thresholds
    results_list = compare_thresholds(thresholds, predictions)
    
    # Save results
    save_results(results_list)
    
    print(f"\n{'='*70}")
    print("NEXT STEPS:")
    print(f"{'='*70}")
    print("1. Review the threshold comparison above")
    print("2. Choose a threshold based on your needs:")
    print("   - Higher threshold (0.5-0.6): More conservative, fewer false positives")
    print("   - Lower threshold (0.3-0.4): More coverage, more predictions")
    print("3. Update CLASSIFIER_THRESHOLD in prediction scripts")
    print("4. Re-run predictions with new threshold")

if __name__ == "__main__":
    main()

