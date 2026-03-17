"""
Test the model on Oxford Brookes recordings (same source as training data).
This will help us understand if low confidence is due to data mismatch or model issues.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt
import csv
import sys

# Fix Unicode encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# ---------------- CONFIG ----------------
SR = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FMIN = 200
FMAX = 10000
TARGET_FRAMES = 128

# Windowing
WINDOW_SEC = 2.5
HOP_SEC = 1.25
MIN_WINDOWS = 1

CONFIDENCE_THRESHOLD_DEFAULT = 0.60

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Model paths
MODEL_PATH_16 = PROJECT_ROOT / "models" / "all_species" / "galago_cnn_all_16classes_best.keras"
MODEL_PATH_17 = PROJECT_ROOT / "models" / "all_species" / "galago_cnn_all_17classes_best.keras"
if MODEL_PATH_16.exists():
    MODEL_PATH = MODEL_PATH_16
    print(f"Using improved 16-class model")
else:
    MODEL_PATH = MODEL_PATH_17
    print(f"Using 17-class model")

CLASS_NAMES_PATH = PROJECT_ROOT / "models" / "all_species" / "class_names.json"

# Oxford Brookes data source
OXFORD_BROOKES_DIR = Path(r"E:\Galagidae")

# Output
OUT_CSV = PROJECT_ROOT / "outputs" / "predictions" / "predictions_oxford_brookes.csv"

# Audio extensions
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".dat"}

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


def window_starts(n_samples: int, sr: int, win_sec: float, hop_sec: float) -> list[int]:
    """Calculate window start positions in samples."""
    win = int(win_sec * sr)
    hop = int(hop_sec * sr)
    if n_samples <= win:
        return [0]
    starts = list(range(0, n_samples - win + 1, hop))
    if len(starts) < MIN_WINDOWS:
        starts = [max(0, (n_samples - win) // 2)]
    return starts


def wav_window_to_rgb_fixed(y: np.ndarray, sr: int) -> np.ndarray:
    """Convert audio window to RGB array matching training preprocessing."""
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT,
        hop_length=HOP_LENGTH, fmin=FMIN, fmax=FMAX
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    S_padded = pad_or_crop(S_db, TARGET_FRAMES)
    
    # Normalize to [0, 1]
    S_norm = (S_padded - S_padded.min()) / (S_padded.max() - S_padded.min() + 1e-8)
    
    # Convert to RGB using magma colormap
    cmap = plt.colormaps["magma"]
    S_rgb = cmap(S_norm)[:, :, :3]  # (H, W, 3)
    S_rgb = (S_rgb * 255).astype(np.uint8)
    
    # Reshape to (1, H, W, 3) for model
    return S_rgb[np.newaxis, :, :, :]


def predict_wav_windowed(model, wav_path: Path, class_names: list) -> tuple:
    """Predict on a WAV file using windowing and logit averaging."""
    try:
        y, sr = librosa.load(str(wav_path), sr=SR)
    except Exception as e:
        print(f"  Error loading {wav_path.name}: {e}")
        return None, 0, None
    
    starts = window_starts(len(y), sr, WINDOW_SEC, HOP_SEC)
    logits_list = []
    
    for start in starts:
        end = start + int(WINDOW_SEC * sr)
        if end > len(y):
            end = len(y)
        y_win = y[start:end]
        
        if len(y_win) < int(0.5 * sr):  # Skip very short windows
            continue
        
        rgb = wav_window_to_rgb_fixed(y_win, sr)
        logits = model.predict(rgb, verbose=0)
        logits_list.append(logits[0])
    
    if not logits_list:
        return None, 0, None
    
    # Average logits across windows
    avg_logits = np.mean(logits_list, axis=0)
    probs = tf.nn.softmax(avg_logits).numpy()
    
    return probs, len(logits_list), avg_logits


# Species patterns from make_mels.py (simplified version)
SPECIES_PATTERNS = {
    "O_crassicaudatus": "Otolemur_crassicaudatus",
    "O. crassicaudatus": "Otolemur_crassicaudatus",
    "Otolemur crassicaudatus": "Otolemur_crassicaudatus",
    "O_garnettii": "Otolemur_garnettii",
    "O. garnettii": "Otolemur_garnettii",
    "Otolemur garnettii": "Otolemur_garnettii",
    "G_sengalensis": "Galago_senegalensis",
    "G. sengalensis": "Galago_senegalensis",
    "Galago senegalensis": "Galago_senegalensis",
    "G_moholi": "Galago_moholi",
    "G. moholi": "Galago_moholi",
    "Galago moholi": "Galago_moholi",
    "G_gallarum": "Galago_gallarum",
    "G. gallarum": "Galago_gallarum",
    "Galago gallarum": "Galago_gallarum",
    "G_matschiei": "Galago_matschiei",
    "G. matschiei": "Galago_matschiei",
    "Galago matschiei": "Galago_matschiei",
    "G_granti": "Paragalago_granti",
    "G. granti": "Paragalago_granti",
    "Galago granti": "Paragalago_granti",
    "Galagoides granti": "Paragalago_granti",
    "Paragalago granti": "Paragalago_granti",
    "G_zanzibaricus": "Paragalago_zanzibaricus",
    "G. zanzibaricus": "Paragalago_zanzibaricus",
    "Paragalago zanzibaricus": "Paragalago_zanzibaricus",
    "G_cocos": "Paragalago_cocos",
    "G. cocos": "Paragalago_cocos",
    "Paragalago cocos": "Paragalago_cocos",
    "G_rondoensis": "Paragalago_rondoensis",
    "G. rondoensis": "Paragalago_rondoensis",
    "Paragalago rondoensis": "Paragalago_rondoensis",
    "G_orinus": "Paragalago_orinus",
    "G. orinus": "Paragalago_orinus",
    "Paragalago orinus": "Paragalago_orinus",
    "G_demidovii": "Galagoides_demidovii",
    "G. demidovii": "Galagoides_demidovii",
    "Galagoides demidovii": "Galagoides_demidovii",
    "G_thomasi": "Galagoides_thomasi",
    "G. thomasi": "Galagoides_thomasi",
    "Galagoides thomasi": "Galagoides_thomasi",
    "G_sp_nov": "Galagoides_sp_nov",
    "G.sp.nov": "Galagoides_sp_nov",
    "sp.nov": "Galagoides_sp_nov",
    "sp nov": "Galagoides_sp_nov",
    "Paragalago sp. nov": "Galagoides_sp_nov",
    "Paragalago sp. nov. 3": "Galagoides_sp_nov",
    "S_gabonensis": "Sciurocheirus_gabonensis",
    "S. gabonensis": "Sciurocheirus_gabonensis",
    "Sciurocheirus gabonensis": "Sciurocheirus_gabonensis",
    "S_alleni": "Sciurocheirus_alleni",
    "S. alleni": "Sciurocheirus_alleni",
    "Sciurocheirus alleni": "Sciurocheirus_alleni",
    "E_elegantulus": "Euoticus_elegantulus",
    "E. elegantulus": "Euoticus_elegantulus",
    "Euoticus elegantulus": "Euoticus_elegantulus",
    "E_pallidus": "Euoticus_pallidus",
    "E. pallidus": "Euoticus_pallidus",
    "Euoticus pallidus": "Euoticus_pallidus",
}

def find_species_label(path: Path) -> str | None:
    """Return canonical species label based on patterns in the full path."""
    s = str(path)
    for pattern, label in SPECIES_PATTERNS.items():
        if pattern in s:
            return label
    return None

def get_species_from_folder(folder_path: Path, oxford_root: Path) -> str:
    """Extract species name from filename using pattern matching."""
    species = find_species_label(folder_path)
    if species:
        return species
    # Fallback: try parent folder name
    return folder_path.parent.name


def topk(probs: np.ndarray, class_names: list, k: int) -> list:
    """Get top k predictions."""
    top_indices = np.argsort(probs)[-k:][::-1]
    return [(class_names[i], probs[i]) for i in top_indices]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__ or "")
    p.add_argument(
        "--threshold",
        type=float,
        default=CONFIDENCE_THRESHOLD_DEFAULT,
        help="Confidence threshold for marking predictions as 'uncertain' "
             f"(default: {CONFIDENCE_THRESHOLD_DEFAULT}).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of audio files to process (0 = no limit, default: 100).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    confidence_threshold = args.threshold
    file_limit = args.limit

    print("Testing model on Oxford Brookes recordings...")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    print(f"Oxford Brookes data: {OXFORD_BROOKES_DIR}")
    print(f"Confidence threshold: {confidence_threshold:.2f}")
    
    if not OXFORD_BROOKES_DIR.exists():
        print(f"\nERROR: Oxford Brookes directory not found: {OXFORD_BROOKES_DIR}")
        print("Please check if the path is correct or if the drive is accessible.")
        return
    
    # Load model
    print("\nLoading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Load class names
    with open(CLASS_NAMES_PATH, 'r') as f:
        CLASS_NAMES = json.load(f)
    
    print(f"  - {len(CLASS_NAMES)} species classes\n")
    
    # Find audio files
    print("Finding audio files...")
    wav_files = []
    for ext in AUDIO_EXTS:
        wav_files.extend(OXFORD_BROOKES_DIR.rglob(f"*{ext}"))
    
    # Filter out system files (starting with ._)
    wav_files = [f for f in wav_files if not f.name.startswith('._')]
    
    # Limit to a reasonable number for testing (e.g., 50 files per species)
    # Or test on a subset
    print(f"Found {len(wav_files)} audio files")

    # For testing, let's use a sample (configurable limit, 0 = no limit)
    if file_limit > 0 and len(wav_files) > file_limit:
        print(f"  Limiting to first {file_limit} files for testing...")
        wav_files = wav_files[:file_limit]
    
    print(f"Processing {len(wav_files)} files...\n")
    
    # Process files
    results = []
    for i, wav in enumerate(wav_files, 1):
        print(f"[{i}/{len(wav_files)}] {wav.name}...", end=" ", flush=True)
        
        probs, nwin, logits = predict_wav_windowed(model, wav, CLASS_NAMES)
        if probs is None:
            print("SKIP (error)")
            continue
        
        t3 = topk(probs, CLASS_NAMES, 3)
        best_species, best_p = t3[0]

        if best_p < confidence_threshold:
            best_species_out = "uncertain"
        else:
            best_species_out = best_species
        
        true_label = get_species_from_folder(wav, OXFORD_BROOKES_DIR)
        
        results.append({
            'filepath': str(wav),
            'true_label': true_label,
            'predicted_species': best_species_out,
            'predicted_prob': f"{best_p:.3f}",
            'top2_species': t3[1][0],
            'top2_prob': f"{t3[1][1]:.3f}",
            'top3_species': t3[2][0],
            'top3_prob': f"{t3[2][1]:.3f}",
            'n_windows': nwin,
        })
        
        correct = "OK" if best_species == true_label else "XX"
        print(f"{correct} -> {best_species_out} ({best_p:.3f})")
    
    # Save results
    print(f"\nSaving results to {OUT_CSV}...")
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys() if results else [])
        w.writeheader()
        w.writerows(results)
    
    # Calculate accuracy
    if results:
        correct = sum(1 for r in results if r['predicted_species'] != 'uncertain' and r['predicted_species'] == r['true_label'])
        total = len(results)
        accuracy = (correct / total) * 100 if total > 0 else 0
        
        # Top-3 accuracy
        top3_correct = sum(1 for r in results if r['true_label'] in [r['predicted_species'], r['top2_species'], r['top3_species']])
        top3_acc = (top3_correct / total) * 100 if total > 0 else 0
        
        # Confidence stats
        probs = [float(r['predicted_prob']) for r in results if r['predicted_prob'] != 'uncertain']
        mean_conf = np.mean(probs) if probs else 0
        max_conf = max(probs) if probs else 0
        
        print(f"\nResults Summary:")
        print(f"  Total files: {total}")
        print(f"  Correct predictions: {correct} ({accuracy:.1f}%)")
        print(f"  Top-3 accuracy: {top3_correct} ({top3_acc:.1f}%)")
        print(f"  Mean confidence: {mean_conf:.3f}")
        print(f"  Max confidence: {max_conf:.3f}")
        print(
            f"  Predictions >= {confidence_threshold:.2f}: "
            f"{sum(1 for r in results if float(r['predicted_prob']) >= confidence_threshold)}/{total}"
        )
    
    print(f"\nDone! Results saved to {OUT_CSV}")

if __name__ == "__main__":
    main()

