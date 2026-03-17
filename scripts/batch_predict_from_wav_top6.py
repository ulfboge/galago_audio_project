from pathlib import Path
import csv
import numpy as np
import tensorflow as tf
import librosa
import matplotlib  # <- fixes the colormap warning

# ------------ CONFIG ------------
SR = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FMIN = 200
FMAX = 10000
TARGET_FRAMES = 128

MODEL_PATH = PROJECT_ROOT / "models" / "top6" / "galago_cnn_top6_best.keras"

# Root folder containing WAVs (will search recursively)
AUDIO_DIR = PROJECT_ROOT / "data" / "raw_audio"

# Output CSV
OUT_CSV = Path(r"C:\Users\galag\GitHub\galago_audio_project\predictions_top6.csv")

CLASS_NAMES = [
    "Paragalago_granti",  # Updated from Galago_granti per IUCN Red List
    "Galagoides_sp_nov",
    "Paragalago_rondoensis",
    "Paragalago_orinus",
    "Otolemur_crassicaudatus",
    "Otolemur_garnettii",
]

CONFIDENCE_THRESHOLD = 0.60  # if best prob < this -> "uncertain"


print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)


def pad_or_crop(S: np.ndarray, target_frames: int) -> np.ndarray:
    n_mels, T = S.shape
    if T == target_frames:
        return S
    if T > target_frames:
        start = (T - target_frames) // 2
        end = start + target_frames
        return S[:, start:end]
    pad_total = target_frames - T
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    pad_value = S.min()
    return np.pad(S, ((0, 0), (pad_left, pad_right)),
                  mode="constant", constant_values=pad_value)


def wav_to_rgb_array(wav_path: Path) -> np.ndarray:
    y, sr = librosa.load(wav_path, sr=SR, mono=True)

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
    S_fixed = pad_or_crop(S_db, TARGET_FRAMES)

    # Normalize 0â€“1 per file
    S_min = S_fixed.min()
    S_max = S_fixed.max()
    S_norm = (S_fixed - S_min) / (S_max - S_min + 1e-6)

    magma = matplotlib.colormaps.get_cmap("magma")
    rgb = magma(S_norm)[:, :, :3]          # drop alpha channel
    rgb = (rgb * 255.0).astype("float32")  # back to 0â€“255 (model rescales internally)
    return rgb


def predict_wav_top3(wav_path: Path):
    rgb = wav_to_rgb_array(wav_path)
    x = np.expand_dims(rgb, axis=0)  # (1, H, W, 3)
    probs = model.predict(x, verbose=0)[0]

    top_idx = probs.argsort()[::-1][:3]
    top3 = [(CLASS_NAMES[i], float(probs[i])) for i in top_idx]
    return top3


def main():
    wav_files = sorted(AUDIO_DIR.rglob("*.wav"))
    if not wav_files:
        print("No WAV files found under:", AUDIO_DIR)
        return

    print(f"Found {len(wav_files)} WAV files under {AUDIO_DIR}")

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "filepath",
            "predicted_species",
            "predicted_prob",
            "top2_species", "top2_prob",
            "top3_species", "top3_prob",
        ])

        for wav in wav_files:
            top3 = predict_wav_top3(wav)

            best_species, best_p = top3[0]
            if best_p < CONFIDENCE_THRESHOLD:
                best_species = "uncertain"

            w.writerow([
                str(wav),
                best_species, f"{best_p:.3f}",
                top3[1][0], f"{top3[1][1]:.3f}",
                top3[2][0], f"{top3[2][1]:.3f}",
            ])

            print(f"{wav.name} -> {best_species} ({best_p:.3f})")

    print("\nSaved predictions to:", OUT_CSV)


if __name__ == "__main__":
    main()

