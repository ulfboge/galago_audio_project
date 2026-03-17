from pathlib import Path
import json
import numpy as np
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt

# ----------------- CONFIG -----------------
SR = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FMIN = 200
FMAX = 10000
TARGET_FRAMES = 128   # time dimension of spectrogram

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "all_species" / "galago_cnn_all_17classes_best.keras"
CLASS_NAMES_PATH = PROJECT_ROOT / "models" / "all_species" / "class_names.json"

# Load class names from JSON
if not CLASS_NAMES_PATH.exists():
    raise FileNotFoundError(f"Class names file not found: {CLASS_NAMES_PATH}")

with open(CLASS_NAMES_PATH, "r") as f:
    CLASS_NAMES = json.load(f)

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print(f"Model loaded. {len(CLASS_NAMES)} species classes available.")


def pad_or_crop(S: np.ndarray, target_frames: int) -> np.ndarray:
    """
    Ensure spectrogram has shape (N_MELS, target_frames) by
    center-cropping or padding with the minimum value.
    """
    n_mels, T = S.shape
    if T == target_frames:
        return S

    if T > target_frames:
        start = (T - target_frames) // 2
        end = start + target_frames
        return S[:, start:end]

    # T < target_frames: pad on both sides
    pad_total = target_frames - T
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    pad_value = S.min()
    S_padded = np.pad(S, ((0, 0), (pad_left, pad_right)),
                      mode="constant", constant_values=pad_value)
    return S_padded


def wav_to_rgb_array(wav_path: Path) -> np.ndarray:
    """Load wav, compute log-mel, convert to magma RGB image array (H,W,3) in [0,255]."""
    y, sr = librosa.load(wav_path, sr=SR, mono=True)

    # mel spectrogram
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    # fix size to (128, 128)
    S_fixed = pad_or_crop(S_db, TARGET_FRAMES)

    # normalize per image 0-1 (like imshow does internally)
    S_min = S_fixed.min()
    S_max = S_fixed.max()
    S_norm = (S_fixed - S_min) / (S_max - S_min + 1e-6)

    # apply magma colormap to get RGB
    try:
        magma = plt.colormaps.get_cmap("magma")
    except AttributeError:
        # Fallback for older matplotlib versions
        magma = plt.cm.get_cmap("magma")
    rgb = magma(S_norm)[:, :, :3]  # (H, W, 3) in [0,1]

    # convert to [0, 255] float32 (what TensorFlow expects)
    rgb_float = (rgb * 255.0).astype(np.float32)

    return rgb_float


def predict_wav(wav_path: Path, top_k: int = 5):
    """Predict species from a WAV file."""
    if not wav_path.exists():
        print(f"File not found: {wav_path}")
        return

    print(f"\nProcessing: {wav_path.name}")
    
    rgb = wav_to_rgb_array(wav_path)
    x = np.expand_dims(rgb, axis=0)  # (1, 128, 128, 3)

    probs = model.predict(x, verbose=0)[0]
    
    # Get top k predictions
    top_indices = probs.argsort()[::-1][:top_k]
    
    print(f"\nTop {top_k} predictions:")
    print("-" * 60)
    for i, idx in enumerate(top_indices, 1):
        species = CLASS_NAMES[idx]
        prob = float(probs[idx])
        confidence = "HIGH" if prob >= 0.6 else "LOW"
        print(f"{i}. {species:30s} {prob:.3f} ({confidence})")
    
    best_species = CLASS_NAMES[top_indices[0]]
    best_prob = float(probs[top_indices[0]])
    
    if best_prob < 0.6:
        print(f"\nWarning: Low confidence prediction ({best_prob:.3f})")
        print("   Consider this prediction as uncertain.")
    else:
        print(f"\nPredicted species: {best_species} (confidence: {best_prob:.3f})")
    
    return best_species, best_prob, probs


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python predict_from_wav_all_species.py <path_to_wav_file>")
        print("\nExample:")
        print("  python predict_from_wav_all_species.py data/raw_audio/Galago_granti/Buzz.wav")
        sys.exit(1)
    
    wav_path = Path(sys.argv[1])
    predict_wav(wav_path)

