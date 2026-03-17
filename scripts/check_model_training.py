"""
Check if the 19-class balanced model actually learned during training.
This script:
1. Loads the model and tests it on training data
2. Analyzes prediction distributions
3. Checks if the model is outputting uniform probabilities
"""
from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Try improved model first, then fallback to balanced, then original
MODEL_PATH_IMPROVED = PROJECT_ROOT / "models" / "all_species" / "galago_cnn_all_19classes_improved_best.keras"
MODEL_PATH_BALANCED = PROJECT_ROOT / "models" / "all_species" / "galago_cnn_all_19classes_best.keras"
MODEL_PATH = MODEL_PATH_IMPROVED if MODEL_PATH_IMPROVED.exists() else MODEL_PATH_BALANCED
CLASS_NAMES_PATH = PROJECT_ROOT / "models" / "all_species" / "class_names.json"
MELS_DIR = PROJECT_ROOT / "data" / "melspectrograms"

def load_image_and_preprocess(img_path: Path):
    """Load and preprocess a PNG image."""
    img = tf.io.read_file(str(img_path))
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [128, 128])
    img = tf.cast(img, tf.float32) / 255.0
    return img

def test_model_on_training_data():
    """Test the model on actual training data to see if it learned."""
    model_name = "Improved" if MODEL_PATH == MODEL_PATH_IMPROVED else "Balanced"
    print("=" * 70)
    print(f"Testing 19-Class {model_name} Model on Training Data")
    print("=" * 70)
    
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found: {MODEL_PATH}")
        return
    
    if not CLASS_NAMES_PATH.exists():
        print(f"ERROR: Class names not found: {CLASS_NAMES_PATH}")
        return
    
    # Load model
    print("\nLoading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"  Model loaded: {MODEL_PATH.name}")
    
    # Load class names
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = json.load(f)
    print(f"  Classes: {len(class_names)}")
    
    # Test on a sample of training data from each species
    print("\n" + "=" * 70)
    print("Testing on training data samples...")
    print("=" * 70)
    
    results_by_species = {}
    all_predictions = []
    all_true_labels = []
    
    for species_dir in sorted(MELS_DIR.iterdir()):
        if not species_dir.is_dir() or species_dir.name == "not_galago":
            continue
        
        if species_dir.name not in class_names:
            continue
        
        species = species_dir.name
        png_files = list(species_dir.glob("*.png"))
        
        if len(png_files) == 0:
            continue
        
        # Test on first 10 samples (or all if fewer)
        test_files = png_files[:min(10, len(png_files))]
        species_idx = class_names.index(species)
        
        correct = 0
        probs_list = []
        
        for png_file in test_files:
            try:
                img = load_image_and_preprocess(png_file)
                img_batch = tf.expand_dims(img, 0)
                
                pred = model.predict(img_batch, verbose=0)[0]
                predicted_idx = np.argmax(pred)
                predicted_prob = pred[predicted_idx]
                
                if predicted_idx == species_idx:
                    correct += 1
                
                probs_list.append(predicted_prob)
                all_predictions.append(predicted_idx)
                all_true_labels.append(species_idx)
            except Exception as e:
                print(f"  Error processing {png_file.name}: {e}")
                continue
        
        if len(probs_list) > 0:
            results_by_species[species] = {
                'correct': correct,
                'total': len(probs_list),
                'accuracy': correct / len(probs_list),
                'mean_prob': np.mean(probs_list),
                'max_prob': np.max(probs_list),
                'min_prob': np.min(probs_list),
            }
    
    # Print results
    print(f"\n{'Species':<30} {'Acc':<8} {'Mean Prob':<12} {'Max Prob':<12} {'Min Prob':<12}")
    print("-" * 70)
    
    total_correct = 0
    total_samples = 0
    
    for species in sorted(results_by_species.keys()):
        r = results_by_species[species]
        total_correct += r['correct']
        total_samples += r['total']
        print(f"{species:<30} {r['accuracy']*100:>6.1f}%  {r['mean_prob']:>10.4f}  {r['max_prob']:>10.4f}  {r['min_prob']:>10.4f}")
    
    overall_acc = total_correct / total_samples if total_samples > 0 else 0
    print("-" * 70)
    print(f"{'OVERALL':<30} {overall_acc*100:>6.1f}%  {'':>10}  {'':>10}  {'':>10}")
    
    # Analyze prediction distribution
    print("\n" + "=" * 70)
    print("Prediction Distribution Analysis")
    print("=" * 70)
    
    if len(all_predictions) > 0:
        pred_counts = Counter(all_predictions)
        print(f"\nTotal predictions: {len(all_predictions)}")
        print(f"\nPredictions per class:")
        for idx in range(len(class_names)):
            count = pred_counts.get(idx, 0)
            pct = (count / len(all_predictions)) * 100
            print(f"  {idx:2d} {class_names[idx]:<30}: {count:4d} ({pct:5.1f}%)")
        
        # Check if distribution is uniform
        expected_uniform = len(all_predictions) / len(class_names)
        uniform_score = sum(abs(pred_counts.get(i, 0) - expected_uniform) for i in range(len(class_names)))
        uniform_score_pct = (uniform_score / len(all_predictions)) * 100
        
        print(f"\nUniformity score: {uniform_score_pct:.1f}% (lower = more uniform)")
        if uniform_score_pct < 10:
            print("  WARNING: Predictions are very uniform - model may not have learned!")
        elif uniform_score_pct < 30:
            print("  CAUTION: Predictions are somewhat uniform - model may be struggling")
        else:
            print("  OK: Predictions show good variation")
    
    # Test with random input
    print("\n" + "=" * 70)
    print("Testing with Random Input (Baseline)")
    print("=" * 70)
    
    random_input = np.random.rand(1, 128, 128, 3).astype(np.float32)
    random_pred = model.predict(random_input, verbose=0)[0]
    
    print(f"Random input prediction:")
    print(f"  Max probability: {random_pred.max():.4f}")
    print(f"  Min probability: {random_pred.min():.4f}")
    print(f"  Mean probability: {random_pred.mean():.4f}")
    print(f"  Expected uniform: {1.0/len(class_names):.4f}")
    
    if abs(random_pred.mean() - 1.0/len(class_names)) < 0.01:
        print("  WARNING: Model outputs nearly uniform probabilities on random input!")
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Overall accuracy on training samples: {overall_acc*100:.1f}%")
    if overall_acc < 0.2:
        print("  [WARNING] VERY LOW - Model likely did not learn")
    elif overall_acc < 0.5:
        print("  [WARNING] LOW - Model may be struggling")
    elif overall_acc < 0.8:
        print("  [OK] MODERATE - Model learned but could be better")
    else:
        print("  [OK] GOOD - Model learned well")
    
    if uniform_score_pct < 10:
        print("  [WARNING] Predictions are too uniform - model may not have learned")

if __name__ == "__main__":
    test_model_on_training_data()
