"""
Alternative windowed prediction script using majority voting instead of averaging.

This version:
- Uses per-window predictions (argmax)
- Aggregates via majority vote
- Confidence = fraction of windows voting for winner
"""

from pathlib import Path
import csv
import numpy as np
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt
from collections import Counter

# ---------------- CONFIG ----------------
SR = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FMIN = 200
FMAX = 10000
TARGET_FRAMES = 128

WINDOW_SEC = 2.5
HOP_SEC = 1.25
MIN_WINDOWS = 1

CONFIDENCE_THRESHOLD = 0.60

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "top7" / "galago_cnn_top7_best.keras"
AUDIO_DIR  = PROJECT_ROOT / "data" / "raw_audio"
OUT_CSV    = PROJECT_ROOT / "outputs" / "predictions" / "predictions_top7_windowed_majority.csv"

CLASS_NAMES = [
    "Paragalago_granti",  # Updated from Galago_granti per IUCN Red List
    "Galagoides_sp_nov",
    "Paragalago_rondoensis",
    "Paragalago_orinus",
    "Paragalago_zanzibaricus",
    "Otolemur_crassicaudatus",
    "Otolemur_garnettii",
]

LABEL_MAP = {
    "G.sp.nov.1": "Galagoides_sp_nov",
    "G.sp.nov.3": "Galagoides_sp_nov",
}


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


def wav_window_to_rgb_fixed(y: np.ndarray, sr: int) -> np.ndarray:
    """Convert window to RGB using exact training pipeline."""
    S = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
        power=2.0,
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    S_fixed = pad_or_crop(S_db, TARGET_FRAMES)
    
    S_min = S_fixed.min()
    S_max = S_fixed.max()
    if S_max - S_min < 1e-6:
        S_norm = np.zeros_like(S_fixed)
    else:
        S_norm = (S_fixed - S_min) / (S_max - S_min)
    
    magma = plt.cm.get_cmap("magma")
    rgb = magma(S_norm)[:, :, :3]
    rgb_float = (rgb * 255.0).astype(np.float32)
    
    return rgb_float


def window_starts(n_samples: int, sr: int, win_sec: float, hop_sec: float) -> list[int]:
    win = int(win_sec * sr)
    hop = int(hop_sec * sr)
    if n_samples <= win:
        return [0]
    starts = list(range(0, n_samples - win + 1, hop))
    if len(starts) < MIN_WINDOWS:
        starts = [max(0, (n_samples - win) // 2)]
    return starts


def predict_wav_majority_vote(model, wav_path: Path) -> tuple[str, float, int, dict]:
    """
    Predict using majority voting over windows.
    
    Returns:
        - predicted_class: majority vote winner
        - confidence: fraction of windows voting for winner
        - n_windows: number of windows
        - vote_counts: dict of class -> vote count
    """
    y, sr = librosa.load(wav_path, sr=SR, mono=True)
    win = int(WINDOW_SEC * sr)
    
    starts = window_starts(len(y), sr, WINDOW_SEC, HOP_SEC)
    votes = []
    probs_list = []
    
    for s in starts:
        seg = y[s:s + win]
        if len(seg) < win:
            seg = np.pad(seg, (0, win - len(seg)), mode="constant")
        
        rgb = wav_window_to_rgb_fixed(seg, sr)
        x = np.expand_dims(rgb, axis=0)
        probs = model.predict(x, verbose=0)[0]
        probs_list.append(probs)
        
        # Vote: argmax class
        pred_class_idx = int(np.argmax(probs))
        votes.append(pred_class_idx)
    
    # Majority vote
    vote_counts = Counter(votes)
    winner_idx, winner_count = vote_counts.most_common(1)[0]
    confidence = winner_count / len(votes)
    
    predicted_class = CLASS_NAMES[winner_idx]
    
    # If confidence is low, mark as uncertain
    if confidence < CONFIDENCE_THRESHOLD:
        predicted_class = "uncertain"
    
    # Build vote count dict for reporting
    vote_dict = {CLASS_NAMES[i]: count for i, count in vote_counts.items()}
    
    return predicted_class, confidence, len(votes), vote_dict


def topk_from_votes(vote_counts: dict, probs_list: list, k: int = 3):
    """Get top k classes based on votes, with average probability as tiebreaker."""
    # Sort by vote count, then by average probability
    class_stats = []
    for class_name in CLASS_NAMES:
        votes = vote_counts.get(class_name, 0)
        if votes > 0:
            # Average probability across windows that voted for this class
            class_idx = CLASS_NAMES.index(class_name)
            avg_prob = np.mean([p[class_idx] for p in probs_list])
            class_stats.append((class_name, votes, avg_prob))
    
    class_stats.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return [(name, prob) for name, votes, prob in class_stats[:k]]


def get_source_folder(wav_path: Path, audio_root: Path) -> str:
    try:
        rel = wav_path.relative_to(audio_root)
        return rel.parts[0] if len(rel.parts) > 1 else ""
    except Exception:
        return wav_path.parent.name


def main():
    print("Model path:", MODEL_PATH)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded.")
    
    wav_files = sorted(AUDIO_DIR.rglob("*.wav"))
    if not wav_files:
        print("No WAV files found under:", AUDIO_DIR)
        return
    
    print(f"Found {len(wav_files)} WAV files")
    print("\nUsing MAJORITY VOTE aggregation\n")
    
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "filepath",
            "source_folder",
            "mapped_folder_label",
            "n_windows",
            "predicted_species",
            "confidence_fraction",
            "vote_counts",
        ])
        
        for wav in wav_files:
            pred_class, confidence, nwin, vote_counts = predict_wav_majority_vote(model, wav)
            
            src_folder = get_source_folder(wav, AUDIO_DIR)
            mapped = LABEL_MAP.get(src_folder, src_folder)
            
            vote_str = "; ".join([f"{k}:{v}" for k, v in sorted(vote_counts.items(), key=lambda x: x[1], reverse=True)])
            
            w.writerow([
                str(wav),
                src_folder,
                mapped,
                nwin,
                pred_class,
                f"{confidence:.3f}",
                vote_str,
            ])
            
            print(f"{wav.name} -> {pred_class} (conf={confidence:.3f}, votes={vote_counts}) [windows={nwin}]")
    
    print("\nSaved predictions to:", OUT_CSV)


if __name__ == "__main__":
    main()

