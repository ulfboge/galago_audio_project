"""
2-stage Merlin-like prediction pipeline:
1. Detector: Is this a galago? (binary)
2. Classifier: Which species? (only if detector says yes)

This reduces false positives by filtering out non-galago audio first.
"""
from pathlib import Path
import json
import numpy as np
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt
import csv
import sys

# Import context re-ranker
import sys
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from context_reranker import rerank_predictions, get_location_status

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

# Thresholds
DETECTOR_THRESHOLD = 0.7  # Only classify if galago_prob > this
CLASSIFIER_THRESHOLD = 0.4  # Updated for 16-class model (was 0.6)
# Note: With 16 classes, max probability is naturally lower (~6.25% uniform baseline)
# Analysis shows 0.4 threshold gives ~62% confident predictions (mean confidence: 0.431)
CONTEXT_ALPHA = 0.5  # Weight for context priors (0.0 = no context, 1.0 = full context)

# Context (can be provided via command line or set here)
LOCATION = None  # e.g., "Tanzania", "Kenya"
MONTH = None  # 1-12
HOUR = None  # 0-23

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Model paths
DETECTOR_PATH = PROJECT_ROOT / "models" / "detector" / "galago_detector_best.keras"
CLASSIFIER_PATH_16 = PROJECT_ROOT / "models" / "all_species" / "galago_cnn_all_16classes_best.keras"
CLASSIFIER_PATH_17 = PROJECT_ROOT / "models" / "all_species" / "galago_cnn_all_17classes_best.keras"

# Use 16-class model if available, else 17-class
if CLASSIFIER_PATH_16.exists():
    CLASSIFIER_PATH = CLASSIFIER_PATH_16
else:
    CLASSIFIER_PATH = CLASSIFIER_PATH_17

CLASS_NAMES_PATH = PROJECT_ROOT / "models" / "all_species" / "class_names.json"
AUDIO_DIR = PROJECT_ROOT / "data" / "raw_audio"
OUT_CSV = PROJECT_ROOT / "outputs" / "predictions" / "predictions_2stage.csv"

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
    
    # Normalize to [0, 1] (matches matplotlib imshow)
    S_min = S_padded.min()
    S_max = S_padded.max()
    if S_max - S_min < 1e-6:
        S_norm = np.zeros_like(S_padded)
    else:
        S_norm = (S_padded - S_min) / (S_max - S_min)
    
    # Convert to RGB using magma colormap
    cmap = plt.colormaps["magma"]
    S_rgb = cmap(S_norm)[:, :, :3]  # (H, W, 3) in [0, 1]
    
    # Convert to [0, 255] float32 (model's Rescaling layer will divide by 255)
    S_rgb_float = (S_rgb * 255.0).astype(np.float32)
    
    # Reshape to (1, H, W, 3) for model
    return S_rgb_float[np.newaxis, :, :, :]

def predict_detector(detector_model, wav_path: Path) -> tuple:
    """Run detector on audio file. Returns (galago_prob, n_windows)."""
    try:
        y, sr = librosa.load(str(wav_path), sr=SR)
    except Exception as e:
        print(f"  Error loading {wav_path.name}: {e}")
        return None, 0
    
    starts = window_starts(len(y), sr, WINDOW_SEC, HOP_SEC)
    probs_list = []
    
    for start in starts:
        end = start + int(WINDOW_SEC * sr)
        if end > len(y):
            end = len(y)
        y_win = y[start:end]
        
        if len(y_win) < int(0.5 * sr):
            continue
        
        rgb = wav_window_to_rgb_fixed(y_win, sr)
        prob = detector_model.predict(rgb, verbose=0)[0][0]  # Binary output
        probs_list.append(prob)
    
    if not probs_list:
        return None, 0
    
    # Average probabilities across windows
    avg_prob = np.mean(probs_list)
    return avg_prob, len(probs_list)

def predict_classifier(classifier_model, wav_path: Path, class_names: list) -> tuple:
    """Run species classifier on audio file. Returns (probs, n_windows, logits)."""
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
        
        if len(y_win) < int(0.5 * sr):
            continue
        
        rgb = wav_window_to_rgb_fixed(y_win, sr)
        logits = classifier_model.predict(rgb, verbose=0)
        logits_list.append(logits[0])
    
    if not logits_list:
        return None, 0, None
    
    # Average logits across windows
    avg_logits = np.mean(logits_list, axis=0)
    probs = tf.nn.softmax(avg_logits).numpy()
    
    return probs, len(logits_list), avg_logits

def topk(probs: np.ndarray, class_names: list, k: int) -> list:
    """Get top k predictions."""
    top_indices = np.argsort(probs)[-k:][::-1]
    return [(class_names[i], probs[i]) for i in top_indices]

def get_source_folder(wav_path: Path, audio_root: Path) -> str:
    """Extract source folder name."""
    try:
        rel = wav_path.relative_to(audio_root)
        return rel.parts[0] if len(rel.parts) > 1 else ""
    except Exception:
        return wav_path.parent.name

# ---------------- MAIN ----------------
def main():
    print("2-Stage Merlin-like Prediction Pipeline")
    print("=" * 60)
    
    # Check if detector exists
    if not DETECTOR_PATH.exists():
        print(f"\nERROR: Detector model not found: {DETECTOR_PATH}")
        print(f"\nPlease train the detector first:")
        print(f"  1. Collect negative class data (see docs/collecting_negative_class_data.md)")
        print(f"  2. Run: python scripts/train_galago_detector.py")
        return
    
    if not CLASSIFIER_PATH.exists():
        print(f"\nERROR: Classifier model not found: {CLASSIFIER_PATH}")
        return
    
    # Load models
    print(f"\nLoading models...")
    print(f"  Detector: {DETECTOR_PATH.name}")
    detector = tf.keras.models.load_model(DETECTOR_PATH)
    
    print(f"  Classifier: {CLASSIFIER_PATH.name}")
    classifier = tf.keras.models.load_model(CLASSIFIER_PATH)
    
    # Load class names
    with open(CLASS_NAMES_PATH, "r") as f:
        CLASS_NAMES = json.load(f)
    
    print(f"  Classes: {len(CLASS_NAMES)} species")
    print(f"\nThresholds:")
    print(f"  Detector: {DETECTOR_THRESHOLD} (galago_prob must be > this)")
    print(f"  Classifier: {CLASSIFIER_THRESHOLD} (species confidence threshold)")
    print(f"  Context alpha: {CONTEXT_ALPHA} (context re-ranking weight)")
    
    if LOCATION:
        print(f"\nContext:")
        print(f"  Location: {LOCATION}")
    if MONTH is not None:
        print(f"  Month: {MONTH}")
    if HOUR is not None:
        print(f"  Hour: {HOUR}")
    if not (LOCATION or MONTH is not None or HOUR is not None):
        print(f"\nContext: None (no re-ranking)")
    
    # Find audio files
    wav_files = sorted(AUDIO_DIR.rglob("*.wav"))
    if not wav_files:
        print(f"\nNo WAV files found under: {AUDIO_DIR}")
        return
    
    print(f"\nFound {len(wav_files)} WAV files")
    print(f"\nProcessing...\n")
    
    # Process files
    results = []
    detector_stats = {"galago": 0, "not_galago": 0, "error": 0}
    
    for i, wav in enumerate(wav_files, 1):
        print(f"[{i}/{len(wav_files)}] {wav.name}...", end=" ", flush=True)
        
        # Stage 1: Detector
        galago_prob, det_windows = predict_detector(detector, wav)
        
        if galago_prob is None:
            detector_stats["error"] += 1
            result = {
                "filepath": str(wav),
                "detector_result": "error",
                "detector_prob": "N/A",
                "species_result": "error",
                "species_prob": "N/A",
                "top2_species": "N/A",
                "top2_prob": "N/A",
                "top3_species": "N/A",
                "top3_prob": "N/A",
            }
            results.append(result)
            print("ERROR (detector)")
            continue
        
        # Interpret detector output
        # Model outputs: 0 = galago, 1 = not_galago (alphabetical order)
        # So prob close to 0 = galago, prob close to 1 = not_galago
        # We want galago_prob, so use (1 - prob)
        galago_prob_actual = 1.0 - galago_prob
        
        # Check detector threshold
        if galago_prob_actual < DETECTOR_THRESHOLD:
            detector_stats["not_galago"] += 1
            result = {
                "filepath": str(wav),
                "detector_result": "not_galago",
                "detector_prob": f"{galago_prob_actual:.3f}",
                "species_result": "not_classified",
                "species_prob": "N/A",
                "top2_species": "N/A",
                "top2_prob": "N/A",
                "top3_species": "N/A",
                "top3_prob": "N/A",
            }
            results.append(result)
            print(f"NOT GALAGO ({galago_prob_actual:.3f})")
            continue
        
        # Stage 2: Classifier (only if detector says galago)
        detector_stats["galago"] += 1
        probs, nwin, logits = predict_classifier(classifier, wav, CLASS_NAMES)
        
        if probs is None:
            result = {
                "filepath": str(wav),
                "detector_result": "galago",
                "detector_prob": f"{galago_prob_actual:.3f}",
                "species_result": "error",
                "species_prob": "N/A",
                "top2_species": "N/A",
                "top2_prob": "N/A",
                "top3_species": "N/A",
                "top3_prob": "N/A",
            }
            results.append(result)
            print("ERROR (classifier)")
            continue
        
        # Get top predictions
        t3 = topk(probs, CLASS_NAMES, 3)
        predictions_list = [(species, prob) for species, prob in t3]
        
        # Stage 3: Context re-ranking (if context provided)
        if LOCATION or MONTH is not None or HOUR is not None:
            reranked = rerank_predictions(
                predictions_list,
                location=LOCATION,
                month=MONTH,
                hour=HOUR,
                alpha=CONTEXT_ALPHA
            )
            best_species, best_p, meta = reranked[0]
            top2_species, top2_p, _ = reranked[1] if len(reranked) > 1 else ("N/A", "N/A", {})
            top3_species, top3_p, _ = reranked[2] if len(reranked) > 2 else ("N/A", "N/A", {})
            location_status = get_location_status(best_species, LOCATION)
        else:
            # No context re-ranking
            best_species, best_p = t3[0]
            top2_species, top2_p = t3[1] if len(t3) > 1 else ("N/A", 0.0)
            top3_species, top3_p = t3[2] if len(t3) > 2 else ("N/A", 0.0)
            location_status = "N/A"
            meta = {}
        
        if best_p < CLASSIFIER_THRESHOLD:
            species_out = "uncertain"
        else:
            species_out = best_species
        
        result = {
            "filepath": str(wav),
            "detector_result": "galago",
            "detector_prob": f"{galago_prob_actual:.3f}",
            "species_result": species_out,
            "species_prob": f"{best_p:.3f}",
            "top2_species": top2_species,
            "top2_prob": f"{top2_p:.3f}" if isinstance(top2_p, float) else top2_p,
            "top3_species": top3_species,
            "top3_prob": f"{top3_p:.3f}" if isinstance(top3_p, float) else top3_p,
            "location_status": location_status,
            "original_prob": f"{meta.get('original_prob', best_p):.3f}" if meta else f"{best_p:.3f}",
        }
        results.append(result)
        
        status_marker = f" [{location_status}]" if location_status != "N/A" else ""
        print(f"GALAGO -> {species_out} ({best_p:.3f}){status_marker}")
    
    # Save results
    print(f"\nSaving results to {OUT_CSV}...")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys() if results else [])
        w.writeheader()
        w.writerows(results)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total files: {len(wav_files)}")
    print(f"  Detector results:")
    print(f"    Galago detected: {detector_stats['galago']} ({detector_stats['galago']/len(wav_files)*100:.1f}%)")
    print(f"    Not galago: {detector_stats['not_galago']} ({detector_stats['not_galago']/len(wav_files)*100:.1f}%)")
    print(f"    Errors: {detector_stats['error']}")
    
    # Species classification stats
    classified = [r for r in results if r['species_result'] not in ['error', 'not_classified', 'N/A']]
    if classified:
        confident = sum(1 for r in classified if r['species_result'] != 'uncertain')
        print(f"\n  Species classification:")
        print(f"    Classified: {len(classified)}")
        print(f"    Confident (> {CLASSIFIER_THRESHOLD}): {confident} ({confident/len(classified)*100:.1f}%)")
        print(f"    Uncertain: {len(classified) - confident}")
    
    print(f"\nResults saved to: {OUT_CSV}")

if __name__ == "__main__":
    main()

