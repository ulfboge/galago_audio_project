"""
Calibrate confidence thresholds using temperature scaling.
This helps set proper "uncertain" thresholds based on validation data.
"""
from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "all_species" / "galago_cnn_all_16classes_best.keras"
CLASS_NAMES_PATH = PROJECT_ROOT / "models" / "all_species" / "class_names_16.json"
MELS_DIR = PROJECT_ROOT / "data" / "melspectrograms"
OUT_DIR = PROJECT_ROOT / "outputs" / "evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_HEIGHT = 128
IMG_WIDTH = 128
VAL_SPLIT = 0.2
SEED = 42

def main():
    print("=" * 70)
    print("Calibrating Confidence Thresholds")
    print("=" * 70)
    
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found: {MODEL_PATH}")
        return
    
    if not CLASS_NAMES_PATH.exists():
        print(f"ERROR: Class names not found: {CLASS_NAMES_PATH}")
        return
    
    # Load class names
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = json.load(f)
    
    # Load model
    print(f"\nLoading model: {MODEL_PATH.name}")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Create validation dataset (same as training)
    print(f"\nLoading validation set...")
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
    
    # Collect predictions and true labels
    print("Collecting predictions...")
    all_probs = []
    all_true = []
    all_pred = []
    
    for images, labels in val_ds:
        probs = model.predict(images, verbose=0)
        preds = np.argmax(probs, axis=1)
        max_probs = probs.max(axis=1)
        
        all_probs.extend(max_probs)
        all_true.extend(labels.numpy())
        all_pred.extend(preds)
    
    all_probs = np.array(all_probs)
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    
    # Calculate accuracy at different confidence thresholds
    print(f"\n{'='*70}")
    print("Accuracy vs Confidence Threshold")
    print(f"{'='*70}")
    print(f"{'Threshold':<12} {'Coverage':<12} {'Accuracy':<12} {'Precision':<12}")
    print("-" * 70)
    
    thresholds = np.arange(0.1, 1.0, 0.05)
    results = []
    
    for threshold in thresholds:
        mask = all_probs >= threshold
        if mask.sum() == 0:
            continue
        
        coverage = mask.sum() / len(all_probs)
        correct = (all_pred[mask] == all_true[mask]).sum()
        accuracy = correct / mask.sum() if mask.sum() > 0 else 0
        
        # Precision = accuracy when we're confident
        precision = accuracy
        
        results.append({
            'threshold': threshold,
            'coverage': coverage,
            'accuracy': accuracy,
            'precision': precision,
        })
        
        print(f"{threshold:>10.2f}  {coverage:>10.1%}  {accuracy:>10.1%}  {precision:>10.1%}")
    
    # Find optimal threshold (balance between coverage and accuracy)
    print(f"\n{'='*70}")
    print("Recommended Thresholds")
    print(f"{'='*70}")
    
    # Threshold for 80% precision
    for r in results:
        if r['precision'] >= 0.80:
            print(f"Threshold for 80% precision: {r['threshold']:.2f} (coverage: {r['coverage']:.1%})")
            break
    
    # Threshold for 90% precision
    for r in results:
        if r['precision'] >= 0.90:
            print(f"Threshold for 90% precision: {r['threshold']:.2f} (coverage: {r['coverage']:.1%})")
            break
    
    # Current threshold (0.4)
    current_thresh = 0.4
    current_mask = all_probs >= current_thresh
    if current_mask.sum() > 0:
        current_acc = (all_pred[current_mask] == all_true[current_mask]).sum() / current_mask.sum()
        current_cov = current_mask.sum() / len(all_probs)
        print(f"\nCurrent threshold (0.4):")
        print(f"  Coverage: {current_cov:.1%}")
        print(f"  Accuracy: {current_acc:.1%}")
    
    # Save results
    results_path = OUT_DIR / "confidence_calibration.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved calibration results to: {results_path}")
    
    # Plot calibration curve
    plt.figure(figsize=(10, 6))
    
    thresholds_plot = [r['threshold'] for r in results]
    accuracies_plot = [r['accuracy'] for r in results]
    coverages_plot = [r['coverage'] for r in results]
    
    plt.subplot(1, 2, 1)
    plt.plot(thresholds_plot, accuracies_plot, 'b-', label='Accuracy')
    plt.axhline(y=0.8, color='r', linestyle='--', label='80% target')
    plt.axhline(y=0.9, color='g', linestyle='--', label='90% target')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Confidence Threshold')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(thresholds_plot, coverages_plot, 'r-', label='Coverage')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Coverage (fraction of predictions)')
    plt.title('Coverage vs Confidence Threshold')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = OUT_DIR / "confidence_calibration.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved calibration plot to: {plot_path}")

if __name__ == "__main__":
    main()
