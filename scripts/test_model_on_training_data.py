"""
Test the 16-class model on training data using the exact inference pipeline.
This verifies the model actually learned and that preprocessing matches.
"""
from pathlib import Path
import json
import numpy as np
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "all_species" / "galago_cnn_all_16classes_best.keras"
CLASS_NAMES_PATH = PROJECT_ROOT / "models" / "all_species" / "class_names_16.json"
MELS_DIR = PROJECT_ROOT / "data" / "melspectrograms"

# Preprocessing params (must match training exactly)
SR = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FMIN = 200
FMAX = 10000
TARGET_FRAMES = 128

def make_mel_spectrogram(y: np.ndarray):
    """Make mel spectrogram with exact same params as training."""
    S = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        fmin=FMIN,
        fmax=FMAX,
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db

def pad_or_crop(S: np.ndarray, target_frames: int) -> np.ndarray:
    """Pad or crop to target frames."""
    n_mels, T = S.shape
    if T == target_frames:
        return S
    if T > target_frames:
        start = (T - target_frames) // 2
        return S[:, start:start + target_frames]
    pad_total = target_frames - T
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    pad_value = S.min()
    return np.pad(S, ((0, 0), (pad_left, pad_right)),
                  mode="constant", constant_values=pad_value)

def png_to_rgb_fixed(png_path: Path) -> np.ndarray:
    """Load PNG and convert to RGB float32 (exact match to training)."""
    img = tf.io.read_file(str(png_path))
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [128, 128])
    img = tf.cast(img, tf.float32) / 255.0
    return img.numpy()[np.newaxis, :, :, :]

def test_on_training_data():
    """Test model on training mel-spectrograms."""
    print("=" * 70)
    print("Testing 16-Class Model on Training Data")
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
    print(f"  Model: {MODEL_PATH.name}")
    print(f"  Output shape: {model.output_shape}")
    
    # Load class names
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = json.load(f)
    print(f"  Classes: {len(class_names)}")
    
    # Test on training data
    print("\n" + "=" * 70)
    print("Testing on training mel-spectrograms...")
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
        
        # Test on first 20 samples (or all if fewer)
        test_files = png_files[:min(20, len(png_files))]
        species_idx = class_names.index(species)
        
        correct = 0
        probs_list = []
        
        for png_file in test_files:
            try:
                rgb = png_to_rgb_fixed(png_file)
                pred = model.predict(rgb, verbose=0)[0]
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
    
    # Confusion analysis
    print("\n" + "=" * 70)
    print("Confusion Analysis")
    print("=" * 70)
    
    if len(all_predictions) > 0:
        confusion = Counter()
        for true_idx, pred_idx in zip(all_true_labels, all_predictions):
            if true_idx != pred_idx:
                true_species = class_names[true_idx]
                pred_species = class_names[pred_idx]
                confusion[(true_species, pred_species)] += 1
        
        if confusion:
            print("\nTop confusions (True -> Predicted):")
            for (true, pred), count in confusion.most_common(10):
                print(f"  {true:<30} -> {pred:<30}: {count} times")
        else:
            print("\nNo confusions - all predictions correct!")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Overall accuracy on training samples: {overall_acc*100:.1f}%")
    print(f"Total samples tested: {total_samples}")
    
    if overall_acc > 0.8:
        print("\n[OK] GOOD: Model learned well on training data")
        print("  Low test accuracy is likely due to:")
        print("    - Domain shift (different recording conditions)")
        print("    - Windowing/averaging reducing confidence")
        print("    - Test data from different source")
    elif overall_acc > 0.5:
        print("\n[WARNING] MODERATE: Model learned but could be better")
    else:
        print("\n[ERROR] POOR: Model may not have learned effectively")
        print("  Check:")
        print("    - Training logs for validation accuracy")
        print("    - Model architecture")
        print("    - Data quality")
        if overall_acc < 0.1:
            print("    - Possible data leakage or validation set issues")
            print("    - Model may be predicting same class for everything")

if __name__ == "__main__":
    test_on_training_data()
