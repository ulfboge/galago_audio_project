"""
Test the 16-class model on the exact validation set used during training.
This uses the same split (20% validation, seed=42) as training.
"""
from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "all_species" / "galago_cnn_all_16classes_best.keras"
CLASS_NAMES_PATH = PROJECT_ROOT / "models" / "all_species" / "class_names_16.json"
MELS_DIR = PROJECT_ROOT / "data" / "melspectrograms"

IMG_HEIGHT = 128
IMG_WIDTH = 128
VAL_SPLIT = 0.2
SEED = 42

def main():
    print("=" * 70)
    print("Testing 16-Class Model on Validation Set (Same as Training)")
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
    
    print(f"\nModel classes ({len(class_names)}):")
    for i, name in enumerate(class_names):
        print(f"  {i:2d}: {name}")
    
    # Load model
    print(f"\nLoading model: {MODEL_PATH.name}")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Create validation dataset using EXACT same parameters as training
    print(f"\nCreating validation dataset (same split as training)...")
    print(f"  Validation split: {VAL_SPLIT}")
    print(f"  Seed: {SEED}")
    print(f"  Class names: {len(class_names)}")
    
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
        class_names=class_names,  # Use exact same class names
    )
    
    print(f"  Validation samples: {len(val_ds) * 32}")
    
    # Evaluate
    print(f"\nEvaluating model on validation set...")
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    
    print(f"\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation accuracy: {val_acc:.4f} ({val_acc*100:.1f}%)")
    
    # Detailed predictions
    print(f"\n" + "=" * 70)
    print("Detailed Predictions")
    print("=" * 70)
    
    all_true = []
    all_pred = []
    all_probs = []
    
    for images, labels in val_ds:
        probs = model.predict(images, verbose=0)
        preds = np.argmax(probs, axis=1)
        all_true.extend(labels.numpy())
        all_pred.extend(preds)
        all_probs.extend(probs.max(axis=1))
    
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    all_probs = np.array(all_probs)
    
    # Per-class accuracy
    print(f"\nPer-class accuracy on validation set:")
    print(f"{'Species':<30} {'Correct':<10} {'Total':<10} {'Accuracy':<10} {'Mean Prob':<12}")
    print("-" * 70)
    
    for i, species in enumerate(class_names):
        mask = all_true == i
        if mask.sum() == 0:
            continue
        correct = (all_pred[mask] == i).sum()
        total = mask.sum()
        acc = correct / total if total > 0 else 0
        mean_prob = all_probs[mask].mean() if mask.sum() > 0 else 0
        print(f"{species:<30} {correct:<10} {total:<10} {acc*100:>6.1f}%    {mean_prob:>10.4f}")
    
    # Confusion
    print(f"\n" + "=" * 70)
    print("Top Confusions (True -> Predicted)")
    print("=" * 70)
    
    confusion = Counter()
    for true_idx, pred_idx in zip(all_true, all_pred):
        if true_idx != pred_idx:
            true_species = class_names[true_idx]
            pred_species = class_names[pred_idx]
            confusion[(true_species, pred_species)] += 1
    
    if confusion:
        print("\nTop 10 confusions:")
        for (true, pred), count in confusion.most_common(10):
            print(f"  {true:<30} -> {pred:<30}: {count} times")
    else:
        print("\nNo confusions - all predictions correct!")
    
    # Probability distribution
    print(f"\n" + "=" * 70)
    print("Probability Distribution")
    print("=" * 70)
    print(f"Mean max probability: {all_probs.mean():.4f}")
    print(f"Median max probability: {np.median(all_probs):.4f}")
    print(f"Min max probability: {all_probs.min():.4f}")
    print(f"Max max probability: {all_probs.max():.4f}")
    
    # Check for uniform predictions
    pred_counts = Counter(all_pred)
    print(f"\nPrediction distribution:")
    for i, species in enumerate(class_names):
        count = pred_counts.get(i, 0)
        pct = (count / len(all_pred)) * 100
        print(f"  {i:2d} {species:<30}: {count:4d} ({pct:5.1f}%)")
    
    # Summary
    print(f"\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    if val_acc > 0.8:
        print(f"[OK] Model performs well on validation set ({val_acc*100:.1f}%)")
        print("  If training data test failed, check:")
        print("    - Preprocessing differences")
        print("    - Data augmentation during training")
    elif val_acc > 0.5:
        print(f"[WARNING] Model performs moderately on validation set ({val_acc*100:.1f}%)")
    else:
        print(f"[ERROR] Model performs poorly on validation set ({val_acc*100:.1f}%)")
        print("  This suggests the model did not learn effectively")

if __name__ == "__main__":
    main()
