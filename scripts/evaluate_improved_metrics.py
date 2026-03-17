"""
Improved evaluation metrics for galago species classifier.

Tracks precision/recall per species, false positive rate, and other
metrics that matter in the field (following Merlin Bird-ID approach).
"""
from pathlib import Path
import csv
import json
from collections import defaultdict
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_CSV = PROJECT_ROOT / "outputs" / "predictions" / "predictions_2stage.csv"

def load_predictions(csv_path: Path):
    """Load predictions from CSV."""
    if not csv_path.exists():
        print(f"ERROR: Predictions file not found: {csv_path}")
        return []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)

def get_true_label(filepath: str) -> str:
    """Extract true label from filepath."""
    # Try to extract from folder structure
    path = Path(filepath)
    parts = path.parts
    
    # Look for species folder names
    for part in parts:
        if part in ["G.sp.nov.1", "G.sp.nov.3"]:
            return "Galagoides_sp_nov"
        # Check if part matches a species name pattern
        if any(sp in part for sp in ["granti", "rondoensis", "zanzibaricus", "orinus", 
                                     "crassicaudatus", "garnettii", "senegalensis", 
                                     "demidovii", "thomasi", "cocos"]):
            # Map to canonical name
            if "granti" in part:
                return "Paragalago_granti"
            elif "rondoensis" in part:
                return "Paragalago_rondoensis"
            elif "zanzibaricus" in part:
                return "Paragalago_zanzibaricus"
            elif "orinus" in part:
                return "Paragalago_orinus"
            elif "crassicaudatus" in part:
                return "Otolemur_crassicaudatus"
            elif "garnettii" in part:
                return "Otolemur_garnettii"
            elif "senegalensis" in part:
                return "Galago_senegalensis"
            elif "demidovii" in part:
                return "Galagoides_demidovii"
            elif "thomasi" in part:
                return "Galagoides_thomasi"
            elif "cocos" in part:
                return "Paragalago_cocos"
    
    # Fallback to folder name
    if len(parts) > 1:
        return parts[-2]  # Parent folder
    
    return "unknown"

def calculate_metrics(predictions: list):
    """Calculate comprehensive evaluation metrics."""
    # Per-species statistics
    species_stats = defaultdict(lambda: {
        'true_positives': 0,
        'false_positives': 0,
        'false_negatives': 0,
        'total_true': 0,
        'total_predicted': 0,
    })
    
    # Overall statistics
    total = 0
    correct = 0
    uncertain_count = 0
    not_classified = 0
    
    # Confusion tracking
    confusion = defaultdict(lambda: defaultdict(int))
    
    for row in predictions:
        total += 1
        
        # Skip errors
        if row.get('detector_result') == 'error' or row.get('species_result') == 'error':
            continue
        
        # Get true label
        true_label = get_true_label(row['filepath'])
        if true_label == "unknown":
            continue
        
        # Get predicted label
        predicted = row.get('species_result', '')
        detector_result = row.get('detector_result', '')
        
        # Track detector results
        if detector_result == 'not_galago':
            not_classified += 1
            # This is a false negative if it's actually a galago
            if true_label != 'unknown':
                species_stats[true_label]['false_negatives'] += 1
                species_stats[true_label]['total_true'] += 1
            continue
        
        if predicted == 'uncertain' or predicted == 'not_classified':
            uncertain_count += 1
            species_stats[true_label]['total_true'] += 1
            continue
        
        # Count predictions
        species_stats[predicted]['total_predicted'] += 1
        species_stats[true_label]['total_true'] += 1
        
        # Check if correct
        if predicted == true_label:
            correct += 1
            species_stats[true_label]['true_positives'] += 1
        else:
            # False positive for predicted species
            species_stats[predicted]['false_positives'] += 1
            # False negative for true species
            species_stats[true_label]['false_negatives'] += 1
            # Track confusion
            confusion[true_label][predicted] += 1
    
    # Calculate per-species metrics
    species_metrics = {}
    for species, stats in species_stats.items():
        tp = stats['true_positives']
        fp = stats['false_positives']
        fn = stats['false_negatives']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        species_metrics[species] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'total_true': stats['total_true'],
            'total_predicted': stats['total_predicted'],
        }
    
    # Overall accuracy
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        'overall': {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'uncertain': uncertain_count,
            'not_classified': not_classified,
        },
        'per_species': species_metrics,
        'confusion': dict(confusion),
    }

def print_metrics(metrics: dict):
    """Print evaluation metrics in a readable format."""
    print("Improved Evaluation Metrics")
    print("="*60)
    
    overall = metrics['overall']
    print(f"\nOverall Performance:")
    print(f"  Total files: {overall['total']}")
    print(f"  Correct: {overall['correct']} ({overall['accuracy']*100:.1f}%)")
    print(f"  Uncertain: {overall['uncertain']}")
    print(f"  Not classified (detector): {overall['not_classified']}")
    
    print(f"\nPer-Species Metrics:")
    print("-"*60)
    print(f"{'Species':<30} {'Precision':<12} {'Recall':<12} {'F1':<12} {'TP':<6} {'FP':<6} {'FN':<6}")
    print("-"*60)
    
    for species in sorted(metrics['per_species'].keys()):
        m = metrics['per_species'][species]
        print(f"{species[:29]:<30} {m['precision']:>11.3f} {m['recall']:>11.3f} {m['f1_score']:>11.3f} "
              f"{m['true_positives']:>5} {m['false_positives']:>5} {m['false_negatives']:>5}")
    
    # Top confusions
    confusion = metrics['confusion']
    if confusion:
        print(f"\nTop Confusions (True -> Predicted):")
        print("-"*60)
        confusions_list = []
        for true_label, preds in confusion.items():
            for pred, count in preds.items():
                confusions_list.append((true_label, pred, count))
        confusions_list.sort(key=lambda x: x[2], reverse=True)
        
        for true_label, pred, count in confusions_list[:10]:
            print(f"  {true_label[:28]:<30} -> {pred[:28]:<30}: {count}")

def main():
    print("Loading predictions...")
    predictions = load_predictions(PREDICTIONS_CSV)
    
    if not predictions:
        print("No predictions found.")
        return
    
    print(f"Calculating metrics for {len(predictions)} predictions...")
    metrics = calculate_metrics(predictions)
    
    print_metrics(metrics)
    
    # Save to JSON
    output_json = PROJECT_ROOT / "outputs" / "evaluation" / "metrics_improved.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to JSON-serializable format
    metrics_json = {
        'overall': metrics['overall'],
        'per_species': {k: {kk: float(vv) if isinstance(vv, float) else vv 
                           for kk, vv in v.items()} 
                       for k, v in metrics['per_species'].items()},
        'confusion': {k: {kk: int(vv) for kk, vv in v.items()} 
                     for k, v in metrics['confusion'].items()},
    }
    
    with open(output_json, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    print(f"\nMetrics saved to: {output_json}")

if __name__ == "__main__":
    main()

