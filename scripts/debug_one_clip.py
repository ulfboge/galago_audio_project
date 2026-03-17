"""
Debug script to diagnose uniform probability issues.
Checks preprocessing, logits, and model outputs for a single clip.
"""
from pathlib import Path
import json
import numpy as np
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "all_species" / "galago_cnn_all_16classes_best.keras"
# Use model-specific class names
CLASS_NAMES_PATH_16 = PROJECT_ROOT / "models" / "all_species" / "class_names_16.json"
CLASS_NAMES_PATH_19 = PROJECT_ROOT / "models" / "all_species" / "class_names.json"
CLASS_NAMES_PATH = CLASS_NAMES_PATH_16 if CLASS_NAMES_PATH_16.exists() else CLASS_NAMES_PATH_19
AUDIO_DIR = PROJECT_ROOT / "data" / "raw_audio"

# Preprocessing params (must match training)
SR = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FMIN = 200
FMAX = 10000
TARGET_FRAMES = 128

def load_audio(audio_path: Path):
    """Load audio with exact same params as training."""
    try:
        y, sr = librosa.load(str(audio_path), sr=SR, mono=True)
        return y, sr
    except Exception as e:
        print(f"ERROR loading audio: {e}")
        return None, None

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

def wav_to_rgb_fixed(y: np.ndarray, sr: int) -> np.ndarray:
    """Convert audio to RGB image (must match training exactly)."""
    S_db = make_mel_spectrogram(y)
    S_padded = pad_or_crop(S_db, TARGET_FRAMES)
    S_min = S_padded.min()
    S_max = S_padded.max()
    if S_max - S_min < 1e-6:
        S_norm = np.zeros_like(S_padded)
    else:
        S_norm = (S_padded - S_min) / (S_max - S_min)
    cmap = plt.colormaps["magma"]
    S_rgb = cmap(S_norm)[:, :, :3]
    S_rgb_float = (S_rgb * 255.0).astype(np.float32)
    return S_rgb_float[np.newaxis, :, :, :]

def debug_clip(audio_path: Path, model, class_names: list):
    """Debug a single audio clip through the full pipeline."""
    print("=" * 70)
    print(f"Debugging: {audio_path.name}")
    print("=" * 70)
    
    # 1. Load audio
    print("\n1. Loading audio...")
    y, sr = load_audio(audio_path)
    if y is None:
        return
    
    print(f"   Audio length: {len(y) / sr:.2f} seconds")
    print(f"   Sample rate: {sr} Hz")
    print(f"   RMS energy: {np.sqrt(np.mean(y**2)):.4f}")
    print(f"   % near-zero (< 0.01): {(np.abs(y) < 0.01).sum() / len(y) * 100:.1f}%")
    
    # 2. Preprocessing
    print("\n2. Preprocessing...")
    print(f"   Preprocessing params:")
    print(f"     SR: {SR}")
    print(f"     N_MELS: {N_MELS}")
    print(f"     N_FFT: {N_FFT}")
    print(f"     HOP_LENGTH: {HOP_LENGTH}")
    print(f"     FMIN: {FMIN}, FMAX: {FMAX}")
    print(f"     TARGET_FRAMES: {TARGET_FRAMES}")
    
    rgb = wav_to_rgb_fixed(y, sr)
    print(f"\n   Feature tensor shape: {rgb.shape}")
    print(f"   Feature tensor range: [{rgb.min():.4f}, {rgb.max():.4f}]")
    print(f"   Feature tensor mean: {rgb.mean():.4f}, std: {rgb.std():.4f}")
    print(f"   Feature tensor dtype: {rgb.dtype}")
    
    # Check for problematic values
    if rgb.min() < 0 or rgb.max() > 255:
        print(f"   WARNING: Values outside [0, 255] range!")
    if rgb.std() < 1e-6:
        print(f"   WARNING: Very low variance - might be all zeros or constant!")
    
    # 3. Model inference
    print("\n3. Model inference...")
    
    # Get logits (before softmax)
    # Check if model has softmax in output
    model_output = model(rgb)
    if isinstance(model_output, list):
        logits = model_output[0]
    else:
        logits = model_output
    
    # If model already has softmax, we need to get logits differently
    # For now, assume model outputs probabilities
    probs = logits.numpy()[0]
    
    print(f"   Output shape: {probs.shape}")
    print(f"   Output range: [{probs.min():.4f}, {probs.max():.4f}]")
    print(f"   Output sum: {probs.sum():.4f} (should be ~1.0 for probabilities)")
    
    # Check if uniform
    expected_uniform = 1.0 / len(class_names)
    max_prob = probs.max()
    min_prob = probs.min()
    mean_prob = probs.mean()
    
    print(f"\n   Uniformity check:")
    print(f"     Expected uniform prob: {expected_uniform:.4f}")
    print(f"     Actual mean prob: {mean_prob:.4f}")
    print(f"     Max prob: {max_prob:.4f}")
    print(f"     Min prob: {min_prob:.4f}")
    
    if abs(mean_prob - expected_uniform) < 0.01:
        print(f"     WARNING: Probabilities are nearly uniform!")
    
    # 4. Top predictions
    print("\n4. Top predictions:")
    top_indices = np.argsort(probs)[-5:][::-1]
    for i, idx in enumerate(top_indices, 1):
        print(f"   {i}. {class_names[idx]:30s}: {probs[idx]:.4f}")
    
    # 5. Class mapping check
    print("\n5. Class mapping:")
    print(f"   Total classes: {len(class_names)}")
    print(f"   First 5 classes: {class_names[:5]}")
    print(f"   Last 5 classes: {class_names[-5:]}")
    
    return {
        'audio_path': str(audio_path),
        'feature_shape': rgb.shape,
        'feature_range': (float(rgb.min()), float(rgb.max())),
        'feature_mean': float(rgb.mean()),
        'feature_std': float(rgb.std()),
        'probs_range': (float(probs.min()), float(probs.max())),
        'probs_mean': float(probs.mean()),
        'max_prob': float(max_prob),
        'top_class': class_names[top_indices[0]],
        'top_prob': float(probs[top_indices[0]]),
    }

def main():
    print("=" * 70)
    print("Debug One Clip - Diagnosing Uniform Probability Issues")
    print("=" * 70)
    
    # Load model
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found: {MODEL_PATH}")
        return
    
    print(f"\nLoading model: {MODEL_PATH.name}")
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
    
    # Load class names
    if not CLASS_NAMES_PATH.exists():
        print(f"ERROR: Class names not found: {CLASS_NAMES_PATH}")
        return
    
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = json.load(f)
    print(f"Classes: {len(class_names)}")
    
    # Find a test audio file
    wav_files = list(AUDIO_DIR.rglob("*.wav"))
    if not wav_files:
        print(f"\nERROR: No WAV files found in {AUDIO_DIR}")
        return
    
    # Test first few files
    print(f"\nFound {len(wav_files)} WAV files")
    print(f"Testing first 3 files...\n")
    
    results = []
    for wav_file in wav_files[:3]:
        result = debug_clip(wav_file, model, class_names)
        if result:
            results.append(result)
        print()
    
    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    if results:
        print(f"\nTested {len(results)} files")
        print(f"Average max probability: {np.mean([r['max_prob'] for r in results]):.4f}")
        print(f"Average feature std: {np.mean([r['feature_std'] for r in results]):.4f}")
        
        uniform_threshold = 1.0 / len(class_names)
        if np.mean([r['max_prob'] for r in results]) < uniform_threshold * 2:
            print(f"\nWARNING: Max probabilities are very low (near uniform {uniform_threshold:.4f})")
            print("This suggests the model is not learning or there's a preprocessing mismatch.")

if __name__ == "__main__":
    main()
