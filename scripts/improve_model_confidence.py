"""
Comprehensive script to improve model confidence through:
1. Preprocessing verification and improvements
2. Temperature scaling calibration
3. Model architecture recommendations
4. Testing improvements

This addresses the low confidence issue observed on Oxford Brookes and raw_audio datasets.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model paths
MODEL_PATH = PROJECT_ROOT / "models" / "all_species" / "galago_cnn_all_16classes_best.keras"
CLASS_NAMES_PATH = PROJECT_ROOT / "models" / "all_species" / "class_names_16.json"
MELS_DIR = PROJECT_ROOT / "data" / "melspectrograms"

# Preprocessing params (must match training)
SR = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FMIN = 200
FMAX = 10000
TARGET_FRAMES = 128

IMG_HEIGHT = 128
IMG_WIDTH = 128
VAL_SPLIT = 0.2
SEED = 42


def load_model_and_classes():
    """Load model and class names."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(f"Class names not found: {CLASS_NAMES_PATH}")
    
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = json.load(f)
    
    return model, class_names


def verify_preprocessing():
    """
    Verify preprocessing matches between training and inference.
    Returns statistics comparing training PNGs vs inference-generated images.
    """
    print("\n" + "="*70)
    print("STEP 1: VERIFYING PREPROCESSING CONSISTENCY")
    print("="*70)
    
    # Run the existing comparison script
    import subprocess
    script = PROJECT_ROOT / "scripts" / "compare_training_vs_inference_image_stats.py"
    
    if not script.exists():
        print(f"WARNING: Preprocessing comparison script not found: {script}")
        return None
    
    print("Running preprocessing comparison...")
    try:
        result = subprocess.run(
            [sys.executable, str(script), "--n-train", "256", "--n-wav", "64"],
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode == 0:
            print("✓ Preprocessing comparison completed")
            print(result.stdout)
        else:
            print(f"⚠ Preprocessing comparison had issues: {result.stderr}")
    except Exception as e:
        print(f"⚠ Could not run preprocessing comparison: {e}")
    
    # Load results if available
    stats_file = OUTPUT_DIR / "training_vs_inference_image_stats.json"
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        return stats
    return None


def temperature_scaling_calibration(model, class_names):
    """
    Implement temperature scaling for better confidence calibration.
    Learns optimal temperature parameter on validation set.
    """
    print("\n" + "="*70)
    print("STEP 2: TEMPERATURE SCALING CALIBRATION")
    print("="*70)
    
    # Create validation dataset
    print("Loading validation set...")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        MELS_DIR,
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        batch_size=32,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        validation_split=VAL_SPLIT,
        subset="validation",
        seed=SEED,
        class_names=class_names,
    )
    
    # Collect logits and labels
    print("Collecting predictions...")
    all_logits = []
    all_labels = []
    
    for images, labels in val_ds:
        # Get logits (before softmax)
        logits = model(images, training=False)
        all_logits.append(logits.numpy())
        all_labels.append(labels.numpy())
    
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Get predictions
    all_probs = tf.nn.softmax(all_logits).numpy()
    all_preds = np.argmax(all_probs, axis=1)
    max_probs = all_probs.max(axis=1)
    
    # Find optimal temperature using validation set
    print("Finding optimal temperature...")
    
    # Temperature scaling: softmax(logits / T)
    # We optimize T to minimize negative log-likelihood on validation set
    temperatures = np.logspace(-1, 1, 50)  # 0.1 to 10
    nlls = []
    
    for T in temperatures:
        scaled_probs = tf.nn.softmax(all_logits / T).numpy()
        # Negative log-likelihood
        nll = -np.mean(np.log(scaled_probs[np.arange(len(all_labels)), all_labels] + 1e-10))
        nlls.append(nll)
    
    optimal_temp_idx = np.argmin(nlls)
    optimal_temp = temperatures[optimal_temp_idx]
    
    print(f"  Optimal temperature: {optimal_temp:.3f}")
    print(f"  NLL at optimal temp: {nlls[optimal_temp_idx]:.4f}")
    
    # Apply temperature scaling
    scaled_probs = tf.nn.softmax(all_logits / optimal_temp).numpy()
    scaled_max_probs = scaled_probs.max(axis=1)
    
    # Calculate calibration metrics
    def expected_calibration_error(y_true, y_pred, y_prob, n_bins=10):
        """Calculate Expected Calibration Error (ECE)."""
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n = len(y_true)
        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
            if not np.any(mask):
                continue
            acc = float(np.mean(y_pred[mask] == y_true[mask]))
            conf = float(np.mean(y_prob[mask]))
            frac = float(np.sum(mask)) / float(n)
            ece += abs(acc - conf) * frac
        return float(ece)
    
    ece_raw = expected_calibration_error(all_labels, all_preds, max_probs)
    ece_scaled = expected_calibration_error(all_labels, all_preds, scaled_max_probs)
    
    print(f"\nCalibration Metrics:")
    print(f"  Raw ECE: {ece_raw:.4f}")
    print(f"  Scaled ECE: {ece_scaled:.4f}")
    print(f"  Improvement: {((ece_raw - ece_scaled) / ece_raw * 100):.1f}%")
    
    # Save results
    results = {
        'optimal_temperature': float(optimal_temp),
        'raw_ece': float(ece_raw),
        'scaled_ece': float(ece_scaled),
        'raw_mean_confidence': float(max_probs.mean()),
        'scaled_mean_confidence': float(scaled_max_probs.mean()),
        'raw_accuracy': float((all_preds == all_labels).mean()),
        'scaled_accuracy': float((all_preds == all_labels).mean()),  # Same predictions
    }
    
    output_file = OUTPUT_DIR / "temperature_scaling_calibration.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {output_file}")
    
    # Plot calibration curves
    frac_pos_raw, mean_pred_raw = calibration_curve(
        (all_preds == all_labels).astype(int),
        max_probs,
        n_bins=10,
        strategy='uniform'
    )
    frac_pos_scaled, mean_pred_scaled = calibration_curve(
        (all_preds == all_labels).astype(int),
        scaled_max_probs,
        n_bins=10,
        strategy='uniform'
    )
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect')
    plt.plot(mean_pred_raw, frac_pos_raw, 'o-', label=f'Raw (ECE={ece_raw:.3f})')
    plt.plot(mean_pred_scaled, frac_pos_scaled, 'o-', label=f'Scaled (ECE={ece_scaled:.3f})')
    plt.xlabel('Mean Predicted Confidence')
    plt.ylabel('Empirical Accuracy')
    plt.title('Calibration Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(temperatures, nlls, 'b-')
    plt.axvline(optimal_temp, color='r', linestyle='--', label=f'Optimal T={optimal_temp:.3f}')
    plt.xlabel('Temperature')
    plt.ylabel('Negative Log-Likelihood')
    plt.title('Temperature Optimization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    
    plt.tight_layout()
    plot_file = OUTPUT_DIR / "temperature_scaling_calibration.png"
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"✓ Plot saved to: {plot_file}")
    
    return optimal_temp, results


def generate_recommendations(preprocessing_stats, temp_scaling_results):
    """
    Generate recommendations for model and preprocessing improvements.
    """
    print("\n" + "="*70)
    print("STEP 3: GENERATING RECOMMENDATIONS")
    print("="*70)
    
    recommendations = {
        'preprocessing': [],
        'model_architecture': [],
        'training': [],
        'calibration': [],
    }
    
    # Preprocessing recommendations
    if preprocessing_stats:
        train_stats = preprocessing_stats.get('train', {}).get('stats', {})
        inf_stats = preprocessing_stats.get('inference', {}).get('stats', {})
        
        train_std = train_stats.get('mean_std', 0)
        inf_std = inf_stats.get('mean_std', 0)
        
        if abs(train_std - inf_std) > 0.05:
            recommendations['preprocessing'].append({
                'issue': 'Preprocessing mismatch detected',
                'description': f'Training std ({train_std:.3f}) differs from inference std ({inf_std:.3f})',
                'action': 'Verify that inference preprocessing exactly matches training preprocessing',
                'priority': 'HIGH'
            })
        else:
            recommendations['preprocessing'].append({
                'status': 'OK',
                'description': 'Preprocessing appears consistent',
            })
    
    # Calibration recommendations
    if temp_scaling_results:
        temp = temp_scaling_results.get('optimal_temperature', 1.0)
        ece_improvement = temp_scaling_results.get('raw_ece', 0) - temp_scaling_results.get('scaled_ece', 0)
        
        if temp != 1.0:
            recommendations['calibration'].append({
                'action': 'Apply temperature scaling',
                'temperature': temp,
                'description': f'Optimal temperature is {temp:.3f} (not 1.0), indicating miscalibration',
                'ece_improvement': float(ece_improvement),
                'priority': 'MEDIUM'
            })
        else:
            recommendations['calibration'].append({
                'status': 'OK',
                'description': 'Model is well-calibrated (temperature ≈ 1.0)',
            })
    
    # Model architecture recommendations
    recommendations['model_architecture'].extend([
        {
            'action': 'Increase model capacity',
            'description': 'Current model may be underfitting for 16 classes',
            'suggestions': [
                'Add more filters to convolutional layers (e.g., 128→256→512→1024)',
                'Increase dense layer size to 1024 units',
                'Add attention mechanisms (e.g., SE blocks)',
            ],
            'priority': 'MEDIUM'
        },
        {
            'action': 'Consider transfer learning',
            'description': 'Pre-trained audio models may improve performance',
            'suggestions': [
                'Use YAMNet or VGGish as feature extractor',
                'Fine-tune on galago data',
            ],
            'priority': 'LOW'
        },
    ])
    
    # Training recommendations
    recommendations['training'].extend([
        {
            'action': 'Improve data augmentation',
            'description': 'More robust augmentation can improve generalization',
            'suggestions': [
                'Add SpecAugment (time/frequency masking)',
                'Add time stretching and pitch shifting',
                'Add noise injection',
            ],
            'priority': 'MEDIUM'
        },
        {
            'action': 'Address class imbalance',
            'description': 'Some species have very few samples',
            'suggestions': [
                'Use class weights in loss function',
                'Collect more data for underrepresented species',
                'Use focal loss to focus on hard examples',
            ],
            'priority': 'HIGH'
        },
        {
            'action': 'Reduce domain shift',
            'description': 'Training on PNGs vs inference on WAVs may cause issues',
            'suggestions': [
                'Ingest raw audio into training set (ingest_raw_audio_to_training_mels.py)',
                'Train directly on WAV files with on-the-fly preprocessing',
            ],
            'priority': 'HIGH'
        },
    ])
    
    # Print recommendations
    for category, items in recommendations.items():
        if items:
            print(f"\n{category.upper().replace('_', ' ')}:")
            for i, item in enumerate(items, 1):
                if 'status' in item:
                    print(f"  {i}. [{item['status']}] {item['description']}")
                else:
                    priority = item.get('priority', 'UNKNOWN')
                    print(f"  {i}. [{priority}] {item['action']}")
                    print(f"     {item['description']}")
                    if 'suggestions' in item:
                        for suggestion in item['suggestions']:
                            print(f"       - {suggestion}")
    
    # Save recommendations
    output_file = OUTPUT_DIR / "model_improvement_recommendations.json"
    with open(output_file, 'w') as f:
        json.dump(recommendations, f, indent=2)
    print(f"\n✓ Recommendations saved to: {output_file}")
    
    return recommendations


def create_temperature_scaled_model_wrapper(model, temperature):
    """
    Create a wrapper model that applies temperature scaling.
    """
    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    logits = model(inputs, training=False)
    scaled_probs = tf.nn.softmax(logits / temperature)
    
    return tf.keras.Model(inputs=inputs, outputs=scaled_probs, name=f"{model.name}_temp_scaled")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--skip-preprocessing',
        action='store_true',
        help='Skip preprocessing verification step'
    )
    parser.add_argument(
        '--skip-calibration',
        action='store_true',
        help='Skip temperature scaling calibration step'
    )
    args = parser.parse_args()
    
    print("="*70)
    print("MODEL CONFIDENCE IMPROVEMENT ANALYSIS")
    print("="*70)
    
    # Load model
    print("\nLoading model...")
    model, class_names = load_model_and_classes()
    print(f"  Model: {MODEL_PATH.name}")
    print(f"  Classes: {len(class_names)}")
    
    # Step 1: Verify preprocessing
    preprocessing_stats = None
    if not args.skip_preprocessing:
        preprocessing_stats = verify_preprocessing()
    else:
        print("\nSkipping preprocessing verification (--skip-preprocessing)")
    
    # Step 2: Temperature scaling
    temp_scaling_results = None
    optimal_temp = 1.0
    if not args.skip_calibration:
        optimal_temp, temp_scaling_results = temperature_scaling_calibration(model, class_names)
    else:
        print("\nSkipping temperature scaling (--skip-calibration)")
    
    # Step 3: Generate recommendations
    recommendations = generate_recommendations(preprocessing_stats, temp_scaling_results)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nNext steps:")
    print("1. Review recommendations in: outputs/evaluation/model_improvement_recommendations.json")
    
    if temp_scaling_results:
        print(f"\n2. Apply temperature scaling with T={optimal_temp:.3f}:")
        print("   - Use create_temperature_scaled_model_wrapper() function")
        print("   - Or manually scale logits: scaled_probs = softmax(logits / T)")
    
    print("\n3. Prioritize HIGH priority recommendations:")
    high_priority = []
    for category, items in recommendations.items():
        for item in items:
            if item.get('priority') == 'HIGH':
                high_priority.append(f"  - {category}: {item.get('action', 'N/A')}")
    
    if high_priority:
        for item in high_priority:
            print(item)
    else:
        print("  (No HIGH priority items)")
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()
