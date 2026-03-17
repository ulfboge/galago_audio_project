from pathlib import Path
import numpy as np
import tensorflow as tf
import librosa
from matplotlib

# ----------------- CONFIG -----------------
SR = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FMIN = 200
FMAX = 10000
TARGET_FRAMES = 128   # time dimension of spectrogram

MODEL_PATH = PROJECT_ROOT / "models" / "top6" / "galago_cnn_top6_best.keras"

CLASS_NAMES = [
    "Paragalago_granti",  # Updated from Galago_granti per IUCN Red List
    "Galagoides_sp_nov",
    "Paragalago_rondoensis",
    "Paragalago_orinus",
    "Otolemur_crassicaudatus",
    "Otolemur_garnettii",
]

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)


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

    # normalize per image 0â€“1 (like imshow does internally)
    S_min = S_fixed.min()
    S_max = S_fixed.max()
    S_norm = (S_fixed - S_min) / (S_max - S_min + 1e-6)

    # apply magma colormap to get RGB
    magma = matplotlib.colormaps.get_cmap("magma")
    rgb = magma(S_norm)[:, :, :3]  # drop alpha, keep RGB
    rgb = rgb * 255.0  # scale to 0â€“255
    rgb = rgb.astype("float32")

    return rgb


def predict_wav(wav_path: Path, top_k: int = 3):
    if not wav_path.exists():
        print("File not found:", wav_path)
        return

    print("\nProcessing:", wav_path)
    rgb = wav_to_rgb_array(wav_path)

    # model expects (1, H, W, 3), values 0â€“255 (Rescaling layer divides by 255)
    x = np.expand_dims(rgb, axis=0)

    probs = model.predict(x, verbose=0)[0]
    top_indices = probs.argsort()[::-1][:top_k]

    print("Top predictions:")
    for idx in top_indices:
        species = CLASS_NAMES[idx]
        p = float(probs[idx])
        print(f"{species:25s}  p = {p:.3f}")


if __name__ == "__main__":
    # TODO: change this to an actual WAV file you want to test
    example_wav = Path(r"C:\Users\galag\GitHub\galago_audio_project\audio_raw\Paragalago_rondoensis\Double Unit Rolling Call - G rondoensis - Pugu.wav")
    predict_wav(example_wav)

