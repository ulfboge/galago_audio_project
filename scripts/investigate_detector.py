"""
Investigate why the detector is rejecting known galago calls, especially Otolemur.
This script will analyze detector outputs for different species.
"""
from pathlib import Path
import json
import numpy as np
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt
import csv
from collections import defaultdict

# Same config as predict_3stage_with_context.py
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DETECTOR_PATH = PROJECT_ROOT / "models" / "detector" / "galago_detector_best.keras"
AUDIO_DIR = PROJECT_ROOT / "data" / "raw_audio"

# Label mapping
LABEL_MAP = {
    "G.sp.nov.1": "Galagoides_sp_nov",
    "G.sp.nov.3": "Galagoides_sp_nov",
    "Galago_granti": "Paragalago_granti",
}

def pad_or_crop(S: np.ndarray, target_frames: int) -> np.ndarray:
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
    win = int(win_sec * sr)
    hop = int(hop_sec * sr)
    if n_samples <= win:
        return [0]
    starts = list(range(0, n_samples - win + 1, hop))
    if len(starts) < MIN_WINDOWS:
        starts = [max(0, (n_samples - win) // 2)]
    return starts

def wav_window_to_rgb_fixed(y: np.ndarray, sr: int) -> np.ndarray:
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT,
        hop_length=HOP_LENGTH, fmin=FMIN, fmax=FMAX
    )
    S_db = librosa.power_to_db(S, ref=np.max)
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

def get_mapped_label_from_filepath(filepath: str) -> str:
    """Extract mapped folder label from filepath."""
    try:
        wav_path = Path(filepath)
        rel = wav_path.relative_to(AUDIO_DIR)
        src_folder = rel.parts[0] if len(rel.parts) > 1 else wav_path.parent.name
        return LABEL_MAP.get(src_folder, src_folder)
    except Exception:
        parts = Path(filepath).parts
        for part in parts:
            if part in LABEL_MAP:
                return LABEL_MAP[part]
        return Path(filepath).parent.name

def analyze_detector_output(detector_model, wav_path: Path) -> dict:
    """Analyze detector output for a single file."""
    try:
        y, sr = librosa.load(str(wav_path), sr=SR)
    except Exception as e:
        return {"error": str(e)}
    
    starts = window_starts(len(y), sr, WINDOW_SEC, HOP_SEC)
    probs_list = []
    raw_outputs = []
    
    for start in starts:
        end = start + int(WINDOW_SEC * sr)
        if end > len(y):
            end = len(y)
        y_win = y[start:end]
        if len(y_win) < int(0.5 * sr):
            continue
        rgb = wav_window_to_rgb_fixed(y_win, sr)
        raw_output = detector_model.predict(rgb, verbose=0)
        raw_outputs.append(raw_output[0])
        prob = raw_output[0][0]  # First (and only) output value
        probs_list.append(prob)
    
    if not probs_list:
        return {"error": "No valid windows"}
    
    # Check what the model actually outputs
    # Binary model with sigmoid: output is probability of class 1 (not_galago)
    # So prob close to 0 = galago, prob close to 1 = not_galago
    avg_prob = np.mean(probs_list)
    galago_prob_actual = 1.0 - avg_prob
    
    return {
        "raw_avg": float(avg_prob),
        "galago_prob": float(galago_prob_actual),
        "n_windows": len(probs_list),
        "min_raw": float(np.min(probs_list)),
        "max_raw": float(np.max(probs_list)),
        "std_raw": float(np.std(probs_list)),
        "all_raw": [float(p) for p in probs_list],
    }

def main():
    print("Investigating Detector Behavior")
    print("=" * 60)
    
    if not DETECTOR_PATH.exists():
        print(f"ERROR: Detector model not found: {DETECTOR_PATH}")
        return
    
    # Load detector
    print(f"\nLoading detector: {DETECTOR_PATH.name}")
    detector = tf.keras.models.load_model(DETECTOR_PATH)
    
    # Check model output shape
    print(f"Model input shape: {detector.input_shape}")
    print(f"Model output shape: {detector.output_shape}")
    
    # Load metadata
    metadata_path = PROJECT_ROOT / "models" / "detector" / "detector_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"Training samples: {metadata['training_samples']}")
        print(f"Validation metrics: {metadata['validation_metrics']}")
    
    # Find all WAV files
    wav_files = sorted(AUDIO_DIR.rglob("*.wav"))
    print(f"\nFound {len(wav_files)} WAV files")
    
    # Analyze by species
    species_results = defaultdict(list)
    
    print(f"\nAnalyzing detector outputs...\n")
    
    for wav in wav_files:
        true_label = get_mapped_label_from_filepath(str(wav))
        result = analyze_detector_output(detector, wav)
        result['filepath'] = str(wav)
        result['filename'] = wav.name
        result['true_label'] = true_label
        species_results[true_label].append(result)
    
    # Summary by species
    print("\n" + "=" * 60)
    print("Detector Output Summary by Species")
    print("=" * 60)
    
    # Current threshold
    DETECTOR_THRESHOLD = 0.5
    
    for species in sorted(species_results.keys()):
        results = species_results[species]
        if not results or any('error' in r for r in results):
            continue
        
        galago_probs = [r['galago_prob'] for r in results if 'galago_prob' in r]
        if not galago_probs:
            continue
        
        passed = sum(1 for p in galago_probs if p >= DETECTOR_THRESHOLD)
        rejected = len(galago_probs) - passed
        
        print(f"\n{species}:")
        print(f"  Total files: {len(galago_probs)}")
        print(f"  Passed detector (>={DETECTOR_THRESHOLD}): {passed} ({passed/len(galago_probs)*100:.1f}%)")
        print(f"  Rejected: {rejected} ({rejected/len(galago_probs)*100:.1f}%)")
        print(f"  Mean galago_prob: {np.mean(galago_probs):.3f}")
        print(f"  Min galago_prob: {np.min(galago_probs):.3f}")
        print(f"  Max galago_prob: {np.max(galago_probs):.3f}")
        print(f"  Std galago_prob: {np.std(galago_probs):.3f}")
    
    # Detailed breakdown for Otolemur
    print("\n" + "=" * 60)
    print("Detailed Otolemur Analysis")
    print("=" * 60)
    
    otolemur_species = [s for s in species_results.keys() if 'Otolemur' in s]
    for species in otolemur_species:
        print(f"\n{species}:")
        results = species_results[species]
        for r in sorted(results, key=lambda x: x.get('galago_prob', 0)):
            if 'error' in r:
                print(f"  {r['filename']}: ERROR - {r['error']}")
            else:
                status = "PASS" if r['galago_prob'] >= DETECTOR_THRESHOLD else "REJECT"
                print(f"  {r['filename']}: {status} - galago_prob={r['galago_prob']:.3f} (raw={r['raw_avg']:.3f}, windows={r['n_windows']})")
    
    # Save detailed results
    output_csv = PROJECT_ROOT / "outputs" / "evaluation" / "detector_investigation.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['filename', 'true_label', 'raw_avg', 'galago_prob', 'n_windows', 
                     'min_raw', 'max_raw', 'std_raw', 'status']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for species in sorted(species_results.keys()):
            for r in species_results[species]:
                if 'error' in r:
                    continue
                status = "PASS" if r['galago_prob'] >= DETECTOR_THRESHOLD else "REJECT"
                writer.writerow({
                    'filename': r['filename'],
                    'true_label': r['true_label'],
                    'raw_avg': r['raw_avg'],
                    'galago_prob': r['galago_prob'],
                    'n_windows': r['n_windows'],
                    'min_raw': r['min_raw'],
                    'max_raw': r['max_raw'],
                    'std_raw': r['std_raw'],
                    'status': status,
                })
    
    print(f"\n\nDetailed results saved to: {output_csv}")

if __name__ == "__main__":
    main()
