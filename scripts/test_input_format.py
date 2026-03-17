"""Test script to compare input formats and see what the model expects."""
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from pathlib import Path
import librosa
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "top7" / "galago_cnn_top7_best.keras"
MELS_DIR = PROJECT_ROOT / "data" / "melspectrograms"

# Find a sample PNG from training data
sample_pngs = list(MELS_DIR.rglob("*.png"))
if not sample_pngs:
    print("No PNG files found in melspectrograms directory")
    exit(1)

sample_png = sample_pngs[0]
print(f"Testing with: {sample_png}")

# Load model
model = tf.keras.models.load_model(MODEL_PATH)
print(f"\nModel input shape: {model.input_shape}")

# Method 1: Load PNG the way training does (via image_dataset_from_directory)
print("\n" + "="*60)
print("Method 1: Loading PNG file (training method)")
print("="*60)
img = load_img(sample_png, target_size=(128, 128))
x_png = img_to_array(img)  # uint8 [0, 255]
print(f"PNG loaded shape: {x_png.shape}")
print(f"PNG dtype: {x_png.dtype}")
print(f"PNG min/max: {x_png.min()}/{x_png.max()}")

x_png_batch = np.expand_dims(x_png, axis=0)
probs_png = model.predict(x_png_batch, verbose=0)[0]
print(f"\nPredictions from PNG:")
for i, prob in enumerate(probs_png):
    print(f"  Class {i}: {prob:.4f}")
print(f"  Max prob: {probs_png.max():.4f} at class {probs_png.argmax()}")

# Method 2: Generate RGB from audio (our method)
print("\n" + "="*60)
print("Method 2: Generate RGB from audio (inference method)")
print("="*60)

# Find corresponding audio file (if exists)
# For now, let's just test the RGB generation from a sample audio
sample_wavs = list((PROJECT_ROOT / "data" / "raw_audio").rglob("*.wav"))
if sample_wavs:
    sample_wav = sample_wavs[0]
    print(f"Testing with audio: {sample_wav.name}")
    
    # Our RGB generation
    SR = 22050
    N_MELS = 128
    N_FFT = 2048
    HOP_LENGTH = 512
    FMIN = 200
    FMAX = 10000
    TARGET_FRAMES = 128
    
    y, sr = librosa.load(sample_wav, sr=SR, mono=True)
    
    # Take a 2.5 second window
    win = int(2.5 * sr)
    seg = y[:win] if len(y) >= win else y
    
    S = librosa.feature.melspectrogram(
        y=seg, sr=sr,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
        power=2.0,
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    
    # Pad/crop to 128 frames
    n_mels, T = S_db.shape
    if T > TARGET_FRAMES:
        start = (T - TARGET_FRAMES) // 2
        S_fixed = S_db[:, start:start + TARGET_FRAMES]
    elif T < TARGET_FRAMES:
        pad_total = TARGET_FRAMES - T
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        pad_value = S_db.min()
        S_fixed = np.pad(S_db, ((0, 0), (pad_left, pad_right)),
                        mode="constant", constant_values=pad_value)
    else:
        S_fixed = S_db
    
    # Normalize
    S_min = S_fixed.min()
    S_max = S_fixed.max()
    if S_max - S_min < 1e-6:
        S_norm = np.zeros_like(S_fixed)
    else:
        S_norm = (S_fixed - S_min) / (S_max - S_min)
    
    # Apply colormap
    magma = plt.cm.get_cmap("magma")
    rgb = magma(S_norm)[:, :, :3]
    rgb_float = (rgb * 255.0).astype(np.float32)
    
    print(f"Generated RGB shape: {rgb_float.shape}")
    print(f"Generated RGB dtype: {rgb_float.dtype}")
    print(f"Generated RGB min/max: {rgb_float.min()}/{rgb_float.max()}")
    
    x_rgb_batch = np.expand_dims(rgb_float, axis=0)
    probs_rgb = model.predict(x_rgb_batch, verbose=0)[0]
    print(f"\nPredictions from generated RGB:")
    for i, prob in enumerate(probs_rgb):
        print(f"  Class {i}: {prob:.4f}")
    print(f"  Max prob: {probs_rgb.max():.4f} at class {probs_rgb.argmax()}")
    
    # Compare
    print("\n" + "="*60)
    print("Comparison:")
    print("="*60)
    print(f"PNG max prob: {probs_png.max():.4f}")
    print(f"RGB max prob: {probs_rgb.max():.4f}")
    print(f"Difference: {abs(probs_png.max() - probs_rgb.max()):.4f}")

