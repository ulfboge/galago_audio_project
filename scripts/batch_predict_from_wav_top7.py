from pathlib import Path
import csv
import math
import numpy as np
import tensorflow as tf
import librosa
import matplotlib  # fixes the colormap warning

# ------------ CONFIG ------------
SR = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FMIN = 200
FMAX = 10000
TARGET_FRAMES = 128

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "top7" / "galago_cnn_top7_best.keras"

# Root folder containing WAVs (will search recursively)
AUDIO_DIR = PROJECT_ROOT / "data" / "raw_audio"

# Output CSV
OUT_CSV = PROJECT_ROOT / "outputs" / "predictions" / "predictions_top7_windowed.csv"

CLASS_NAMES = [
    "Paragalago_granti",  # Updated from Galago_granti per IUCN Red List
    "Galagoides_sp_nov",
    "Paragalago_rondoensis",
    "Paragalago_orinus",
    "Paragalago_zanzibaricus",
    "Otolemur_crassicaudatus",
    "Otolemur_garnettii",
]

CONFIDENCE_THRESHOLD = 0.60  # if best prob < this -> "uncertain"

# ------------ LABEL MAP (folder -> expected class) ------------
# If a folder isn't listed here, the expected label falls back to the folder name itself.
LABEL_MAP = {
    "G.sp.nov.1": "Galagoides_sp_nov",
    "G.sp.nov.3": "Galagoides_sp_nov",
    # Add more aliases if needed:
    # "G.sp.nov.2": "Galagoides_sp_nov",
}

# ------------ WINDOWING / AGGREGATION ------------
WINDOW_SEC = 1.0        # window length in seconds
WINDOW_HOP_SEC = 0.5    # hop in seconds (overlap = WINDOW_SEC - WINDOW_HOP_SEC)

KEEP_FRACTION = 0.30    # keep top 30% loudest windows
MIN_KEEP = 3            # keep at least 3 windows if possible

AGG_METHOD = "median"   # "median" or "trimmed_mean"
TRIM_FRACTION = 0.10    # for trimmed_mean: drop 10% low + 10% high per class

print("Model path:", MODEL_PATH)
print("Exists:", MODEL_PATH.exists())
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
    return np.pad(
        S,
        ((0, 0), (pad_left, pad_right)),
        mode="constant",
        constant_values=pad_value,
    )


def wav_to_rgb_array_from_audio(y: np.ndarray, sr: int) -> np.ndarray:
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

    # Normalize 0–1 per window
    S_min = S_fixed.min()
    S_max = S_fixed.max()
    S_norm = (S_fixed - S_min) / (S_max - S_min + 1e-6)

    magma = matplotlib.colormaps.get_cmap("magma")
    rgb = magma(S_norm)[:, :, :3]
    rgb = (rgb * 255.0).astype("float32")
    return rgb


def split_windows(y: np.ndarray, sr: int, win_sec: float, hop_sec: float):
    win = int(round(win_sec * sr))
    hop = int(round(hop_sec * sr))
    if win <= 0 or hop <= 0 or len(y) < win:
        return []
    windows = []
    for start in range(0, len(y) - win + 1, hop):
        windows.append(y[start:start + win])
    return windows


def rms_energy(y_win: np.ndarray) -> float:
    rms = librosa.feature.rms(y=y_win, frame_length=1024, hop_length=512)[0]
    return float(np.mean(rms)) if rms.size else 0.0


def aggregate_probs(probs_stack: np.ndarray) -> np.ndarray:
    if probs_stack.shape[0] == 1:
        return probs_stack[0]

    if AGG_METHOD == "median":
        return np.median(probs_stack, axis=0)

    if AGG_METHOD == "trimmed_mean":
        n = probs_stack.shape[0]
        k = int(math.floor(TRIM_FRACTION * n))
        if k == 0:
            return np.mean(probs_stack, axis=0)
        sorted_vals = np.sort(probs_stack, axis=0)
        trimmed = sorted_vals[k:n - k, :]
        return np.mean(trimmed, axis=0)

    return np.mean(probs_stack, axis=0)


def get_source_folder(wav_path: Path) -> str:
    """
    Example:
      .../data/raw_audio/G.sp.nov.1/foo.wav -> "G.sp.nov.1"
    """
    return wav_path.parent.name


def get_expected_label(source_folder: str) -> str:
    """
    Map folder name to expected model class label.
    If no mapping exists, we assume the folder name already equals a class label.
    """
    return LABEL_MAP.get(source_folder, source_folder)


def predict_wav_windowed_top3(wav_path: Path):
    y, sr = librosa.load(wav_path, sr=SR, mono=True)

    wins = split_windows(y, sr, WINDOW_SEC, WINDOW_HOP_SEC)
    if not wins:
        return 0, [("uncertain", 0.0), ("", 0.0), ("", 0.0)]

    energies = np.array([rms_energy(w) for w in wins], dtype=float)
    n_total = len(wins)
    n_keep = max(MIN_KEEP, int(math.ceil(KEEP_FRACTION * n_total)))
    n_keep = min(n_keep, n_total)

    keep_idx = np.argsort(energies)[::-1][:n_keep]
    keep_wins = [wins[i] for i in keep_idx]

    probs_list = []
    for w in keep_wins:
        rgb = wav_to_rgb_array_from_audio(w, sr)
        x = np.expand_dims(rgb, axis=0)
        p = model.predict(x, verbose=0)[0]
        probs_list.append(p)

    probs_stack = np.vstack(probs_list)
    agg = aggregate_probs(probs_stack)

    top_idx = agg.argsort()[::-1][:3]
    top3 = [(CLASS_NAMES[i], float(agg[i])) for i in top_idx]
    return n_keep, top3


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(AUDIO_DIR.rglob("*.wav"))
    if not wav_files:
        print("No WAV files found under:", AUDIO_DIR)
        return

    print(f"Found {len(wav_files)} WAV files under {AUDIO_DIR}")

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "filepath",
            "source_folder",
            "expected_label",
            "match_expected",
            "n_windows",
            "predicted_species",
            "predicted_prob",
            "top2_species", "top2_prob",
            "top3_species", "top3_prob",
        ])

        for wav in wav_files:
            source_folder = get_source_folder(wav)
            expected_label = get_expected_label(source_folder)

            n_used, top3 = predict_wav_windowed_top3(wav)
            best_species, best_p = top3[0]

            if best_p < CONFIDENCE_THRESHOLD:
                best_species = "uncertain"

            # match logic: treat "uncertain" as non-match (you can change this if you want)
            match_expected = (best_species == expected_label)

            w.writerow([
                str(wav),
                source_folder,
                expected_label,
                str(match_expected),
                n_used,
                best_species, f"{best_p:.3f}",
                top3[1][0], f"{top3[1][1]:.3f}",
                top3[2][0], f"{top3[2][1]:.3f}",
            ])

            # Print a quick warning for mismatches (excluding "uncertain")
            if (not match_expected) and (best_species != "uncertain"):
                print(f"⚠ mismatch: {wav.name} | folder={source_folder} expected={expected_label} "
                      f"pred={best_species} ({best_p:.3f})")

            print(f"{wav.name} -> {best_species} ({best_p:.3f}) [windows used: {n_used}]")

    print("\nSaved predictions to:", OUT_CSV)


if __name__ == "__main__":
    main()
