from pathlib import Path
import csv
import numpy as np
import tensorflow as tf
import librosa
import matplotlib  # for colormap access

# ---------------- CONFIG ----------------
SR = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FMIN = 200
FMAX = 10000
TARGET_FRAMES = 128  # model expects 128x128

# Windowing (seconds)
WINDOW_SEC = 2.5
HOP_SEC = 1.25
MIN_WINDOWS = 1  # if file is short, still do one centered window

CONFIDENCE_THRESHOLD = 0.60  # if best prob < this -> "uncertain"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "top7" / "galago_cnn_top7_best.keras"
AUDIO_DIR  = PROJECT_ROOT / "data" / "raw_audio"
OUT_CSV    = PROJECT_ROOT / "outputs" / "predictions" / "predictions_top7_windowed.csv"

CLASS_NAMES = [
    "Paragalago_granti",  # Updated from Galago_granti per IUCN Red List
    "Galagoides_sp_nov",
    "Paragalago_rondoensis",
    "Paragalago_orinus",
    "Paragalago_zanzibaricus",
    "Otolemur_crassicaudatus",
    "Otolemur_garnettii",
]

# Map folder names -> model labels (handles your sp. nov. dilemma as ONE class)
LABEL_MAP = {
    "G.sp.nov.1": "Galagoides_sp_nov",
    "G.sp.nov.3": "Galagoides_sp_nov",
    # add more folder aliases here if you want
    # "G sp nov 1": "Galagoides_sp_nov",
}

# ---------------- HELPERS ----------------
def pad_or_crop(S: np.ndarray, target_frames: int) -> np.ndarray:
    """Pad/crop time axis of (n_mels, T) to target_frames."""
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

def mel_to_rgb(S_db_fixed: np.ndarray) -> np.ndarray:
    """
    Convert (128,128) dB mel -> (128,128,3) RGB float32 in 0..255.
    
    This matches exactly how make_mels.py saves PNGs:
    - matplotlib's imshow normalizes to [0,1] based on array min/max
    - Applies magma colormap
    - Saves as PNG (uint8 [0,255])
    - TensorFlow loads as float32 [0,255] via load_img/img_to_array
    """
    # Normalize exactly like matplotlib's imshow does
    S_min = S_db_fixed.min()
    S_max = S_db_fixed.max()
    if S_max - S_min < 1e-6:
        S_norm = np.zeros_like(S_db_fixed)
    else:
        S_norm = (S_db_fixed - S_min) / (S_max - S_min)

    # Apply magma colormap (same as training)
    try:
        # New matplotlib API (3.7+)
        magma = matplotlib.colormaps.get_cmap("magma")
    except AttributeError:
        # Fallback for older matplotlib
        magma = matplotlib.cm.get_cmap("magma")
    
    rgb = magma(S_norm)[:, :, :3]  # drop alpha, shape (H, W, 3) in [0,1]
    rgb = (rgb * 255.0).astype("float32")  # convert to [0,255] float32 (matches load_img output)
    return rgb

def window_starts(n_samples: int, sr: int, win_sec: float, hop_sec: float) -> list[int]:
    win = int(win_sec * sr)
    hop = int(hop_sec * sr)
    if n_samples <= win:
        # single centered window
        return [0]
    starts = list(range(0, n_samples - win + 1, hop))
    if len(starts) < MIN_WINDOWS:
        starts = [max(0, (n_samples - win) // 2)]
    return starts

def wav_window_to_rgb(y: np.ndarray, sr: int) -> np.ndarray:
    S = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
        power=2.0,
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    S_fixed = pad_or_crop(S_db, TARGET_FRAMES)
    return mel_to_rgb(S_fixed)

def predict_wav_windowed(model, wav_path: Path) -> tuple[np.ndarray, int]:
    """
    Return (avg_probs, n_windows).
    
    Uses probability averaging (simple mean) which works well when windows
    are consistent. For better results with diverse windows, consider logit averaging.
    """
    y, sr = librosa.load(wav_path, sr=SR, mono=True)
    win = int(WINDOW_SEC * sr)

    starts = window_starts(len(y), sr, WINDOW_SEC, HOP_SEC)
    probs_list = []

    for s in starts:
        seg = y[s:s + win]
        if len(seg) < win:
            # pad short tail window
            seg = np.pad(seg, (0, win - len(seg)), mode="constant")
        rgb = wav_window_to_rgb(seg, sr)
        x = np.expand_dims(rgb, axis=0)  # (1,128,128,3)
        p = model.predict(x, verbose=0)[0]  # (7,)
        probs_list.append(p)

    probs_stack = np.stack(probs_list, axis=0)       # (n_windows, 7)
    avg_probs = probs_stack.mean(axis=0)             # (7,) - simple mean of probabilities
    return avg_probs, len(starts)

def topk(probs: np.ndarray, k: int = 3):
    idx = probs.argsort()[::-1][:k]
    return [(CLASS_NAMES[i], float(probs[i])) for i in idx]

def get_source_folder(wav_path: Path, audio_root: Path) -> str:
    """Best effort: folder directly under AUDIO_DIR (e.g., 'G.sp.nov.1')."""
    try:
        rel = wav_path.relative_to(audio_root)
        return rel.parts[0] if len(rel.parts) > 1 else ""
    except Exception:
        return wav_path.parent.name

# ---------------- MAIN ----------------
def main():
    print("Model path:", MODEL_PATH)
    print("Exists:", MODEL_PATH.exists())
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)

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
            "mapped_folder_label",
            "n_windows",
            "predicted_species",
            "predicted_prob",
            "top2_species", "top2_prob",
            "top3_species", "top3_prob",
        ])

        for wav in wav_files:
            probs, nwin = predict_wav_windowed(model, wav)
            t3 = topk(probs, 3)

            best_species, best_p = t3[0]
            if best_p < CONFIDENCE_THRESHOLD:
                best_species_out = "uncertain"
            else:
                best_species_out = best_species

            src_folder = get_source_folder(wav, AUDIO_DIR)
            mapped = LABEL_MAP.get(src_folder, src_folder)

            w.writerow([
                str(wav),
                src_folder,
                mapped,
                nwin,
                best_species_out, f"{best_p:.3f}",
                t3[1][0], f"{t3[1][1]:.3f}",
                t3[2][0], f"{t3[2][1]:.3f}",
            ])

            print(f"{wav.name} -> {best_species_out} ({best_p:.3f})  [windows={nwin}]")

    print("\nSaved predictions to:", OUT_CSV)

if __name__ == "__main__":
    main()
