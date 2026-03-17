"""
3-stage Merlin-like prediction pipeline with context re-ranking:
1. Detector: Is this a galago? (binary)
2. Classifier: Which species? (only if detector says yes)
3. Context Re-ranker: Re-rank using location/season/time

This is the complete Merlin-like system.
"""
from pathlib import Path
import json
import numpy as np
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt
import csv
import sys
import math
import re

# Import context re-ranker
sys.path.insert(0, str(Path(__file__).parent))
from context_reranker import rerank_predictions, get_location_status, get_location_status_point

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

# Window selection / pooling (reduce dilution from silence/noise windows)
# - We compute RMS per window and keep only "active" windows (relative to the max RMS in-file).
# - Then we pool probabilities across the top-K most confident windows.
RMS_GATE_REL = 0.20          # keep windows with rms >= max_rms * RMS_GATE_REL
RMS_GATE_ABS = 1e-4          # always drop near-silence windows
POOL_TOPK_WINDOWS = 3        # number of windows to pool across after gating

# Thresholds
DETECTOR_THRESHOLD = 0.3  # Lowered from 0.7 to 0.5 to 0.3 to allow Otolemur through
# NOTE: We tune this based on desired coverage on the raw_audio WAV evaluation.
# If you want the system to return a label most of the time, use ~0.20–0.30.
# (On the 69-WAV set, 0.20 yields ~94% coverage.)
CLASSIFIER_THRESHOLD = 0.2
# Note: With 16+ classes, max probability is naturally lower (~6.25% uniform baseline at 16 classes).
CONTEXT_ALPHA = 0.5  # Weight for context priors

# Temperature scaling (confidence calibration) for classifier probabilities.
# From `scripts/improve_model_confidence.py`:
#   - Optimal temperature on validation set: ~0.212
#   - Reduces ECE from ~0.69 -> ~0.09 (better calibrated confidences).
CLASSIFIER_TEMPERATURE = 0.212

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Model paths
DETECTOR_PATH = PROJECT_ROOT / "models" / "detector" / "galago_detector_best.keras"
CLASSIFIER_PATH_16_V2 = PROJECT_ROOT / "models" / "all_species" / "galago_cnn_all_16classes_v2_best.keras"
CLASSIFIER_PATH_16 = PROJECT_ROOT / "models" / "all_species" / "galago_cnn_all_16classes_best.keras"
CLASSIFIER_PATH_17 = PROJECT_ROOT / "models" / "all_species" / "galago_cnn_all_17classes_best.keras"
CLASSIFIER_PATH_19 = PROJECT_ROOT / "models" / "all_species" / "galago_cnn_all_19classes_best.keras"
CLASSIFIER_PATH_19_IMPROVED = PROJECT_ROOT / "models" / "all_species" / "galago_cnn_all_19classes_improved_best.keras"

# v2 currently performs poorly (very low val accuracy). Keep it opt-in.
USE_V2_CLASSIFIER = False
# Prefer 19-class improved (boosted) when available for best top-1 on raw_audio eval; else 17-class.
PREFER_19_IMPROVED = True

# Model selection. Default: 19-class improved (best top-1) -> 17 -> 16. Use v2 only if enabled.
if USE_V2_CLASSIFIER and CLASSIFIER_PATH_16_V2.exists():
    CLASSIFIER_PATH = CLASSIFIER_PATH_16_V2
elif PREFER_19_IMPROVED and CLASSIFIER_PATH_19_IMPROVED.exists():
    CLASSIFIER_PATH = CLASSIFIER_PATH_19_IMPROVED
elif CLASSIFIER_PATH_17.exists():
    CLASSIFIER_PATH = CLASSIFIER_PATH_17
elif CLASSIFIER_PATH_16.exists():
    CLASSIFIER_PATH = CLASSIFIER_PATH_16
elif CLASSIFIER_PATH_19.exists():
    CLASSIFIER_PATH = CLASSIFIER_PATH_19
else:
    CLASSIFIER_PATH = CLASSIFIER_PATH_19_IMPROVED

# Try to load model-specific class names, fallback to generic.
# NOTE: class_names.json is not model-specific in this repo; prefer versioned files.
CLASS_NAMES_PATH_16 = PROJECT_ROOT / "models" / "all_species" / "class_names_16.json"
CLASS_NAMES_PATH_17 = PROJECT_ROOT / "models" / "all_species" / "class_names_17.json"
CLASS_NAMES_PATH_19 = PROJECT_ROOT / "models" / "all_species" / "class_names_19.json"
CLASS_NAMES_PATH_FALLBACK = PROJECT_ROOT / "models" / "all_species" / "class_names.json"
CLASS_NAMES_PATH = CLASS_NAMES_PATH_FALLBACK
AUDIO_DIR = PROJECT_ROOT / "data" / "raw_audio"
OUT_CSV = PROJECT_ROOT / "outputs" / "predictions" / "predictions_3stage_context.csv"

# Context (set via command line args or modify here)
# These will be set in main() from command line args

# ---------------- HELPERS (same as 2-stage) ----------------

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

def rms_energy(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float32)
    return float(np.sqrt(np.mean(np.square(y))) + 1e-12)

def select_active_windows(starts: list[int], y: np.ndarray, sr: int) -> list[int]:
    """Select likely-active windows using an RMS gate (relative + absolute)."""
    if not starts:
        return []
    win = int(WINDOW_SEC * sr)
    rms_list = []
    for s in starts:
        e = min(len(y), s + win)
        y_win = y[s:e]
        if len(y_win) < int(0.5 * sr):
            rms_list.append(0.0)
            continue
        rms_list.append(rms_energy(y_win))
    max_rms = max(rms_list) if rms_list else 0.0
    keep = []
    for s, r in zip(starts, rms_list):
        if r < RMS_GATE_ABS:
            continue
        if max_rms > 0 and r < (max_rms * RMS_GATE_REL):
            continue
        keep.append(s)
    # Always keep at least one (highest-RMS) window
    if not keep and rms_list:
        best_idx = int(np.argmax(np.asarray(rms_list)))
        keep = [starts[best_idx]]
    return keep

def predict_detector(detector_model, wav_path: Path) -> tuple:
    try:
        y, sr = librosa.load(str(wav_path), sr=SR)
    except Exception as e:
        return None, 0
    
    starts = window_starts(len(y), sr, WINDOW_SEC, HOP_SEC)
    starts = select_active_windows(starts, y, sr)
    probs_list = []
    
    for start in starts:
        end = start + int(WINDOW_SEC * sr)
        if end > len(y):
            end = len(y)
        y_win = y[start:end]
        if len(y_win) < int(0.5 * sr):
            continue
        rgb = wav_window_to_rgb_fixed(y_win, sr)
        # Detector is trained with label_mode='binary' on folders ['galago','not_galago'] (alphabetical),
        # so label 1 corresponds to 'not_galago'. With sigmoid output, predict() returns P(not_galago).
        not_galago_prob = float(detector_model.predict(rgb, verbose=0).reshape(-1)[0])
        probs_list.append(not_galago_prob)
    
    if not probs_list:
        return None, 0
    
    avg_not_galago_prob = float(np.mean(probs_list))
    return avg_not_galago_prob, len(probs_list)

def predict_classifier(classifier_model, wav_path: Path, class_names: list) -> tuple:
    try:
        y, sr = librosa.load(str(wav_path), sr=SR)
    except Exception as e:
        return None, 0, None
    
    starts = window_starts(len(y), sr, WINDOW_SEC, HOP_SEC)
    starts_all = starts
    starts = select_active_windows(starts, y, sr)
    probs_list = []
    conf_list = []
    
    for start in starts:
        end = start + int(WINDOW_SEC * sr)
        if end > len(y):
            end = len(y)
        y_win = y[start:end]
        if len(y_win) < int(0.5 * sr):
            continue
        rgb = wav_window_to_rgb_fixed(y_win, sr)
        # Model outputs softmax probabilities (Dense(..., activation="softmax"))
        probs = classifier_model.predict(rgb, verbose=0)[0]
        probs_list.append(probs)
        conf_list.append(float(np.max(probs)))
    
    if not probs_list:
        return None, 0, None
    
    # Pool only the top-K most confident windows (avoid diluting with weak/noisy windows)
    k = min(POOL_TOPK_WINDOWS, len(probs_list))
    if k <= 0:
        return None, 0, None
    top_idx = np.argsort(np.asarray(conf_list))[-k:]
    selected = [probs_list[i] for i in top_idx]
    avg_probs = np.mean(selected, axis=0)
    # Safety: normalize (should already sum to 1.0)
    s = float(np.sum(avg_probs))
    if s > 0:
        avg_probs = avg_probs / s

    # Apply temperature scaling in probability space (approximate logits/T softmax).
    # This improves calibration without changing the argmax.
    T = float(globals().get("CLASSIFIER_TEMPERATURE", 1.0))
    if T > 0 and abs(T - 1.0) > 1e-6:
        # Avoid log(0); tiny epsilon is fine for calibration.
        scaled = np.power(np.clip(avg_probs, 1e-12, 1.0), 1.0 / T)
        s2 = float(np.sum(scaled))
        if s2 > 0:
            avg_probs = scaled / s2

    return avg_probs, len(starts_all), avg_probs

def topk(probs: np.ndarray, class_names: list, k: int) -> list:
    top_indices = np.argsort(probs)[-k:][::-1]
    return [(class_names[i], probs[i]) for i in top_indices]

# ---------------- MAIN ----------------
def main():
    print("3-Stage Merlin-like Prediction Pipeline with Context")
    print("="*60)
    
    # Parse optional args.
    # Backwards compatible with:
    #   python predict_3stage_with_context.py [location] [month] [hour]
    # Flags:
    #   --filelist <path>
    #   --out-csv <path>
    #   --classifier-threshold <float>
    #   --detector-threshold <float>
    #   --pool-topk <int>
    #   --rms-gate-rel <float>
    #   --rms-gate-abs <float>
    #   --platt-json <path>          (optional: map top1_prob -> P(correct))
    #   --threshold-on <raw|platt>   (default: raw; 'platt' requires --platt-json)
    #   --infer-location-from-filename   (optional: infer location string from filename tokens)
    #   --lat <float> --lon <float>  (optional: use polygon-based location priors)
    #   --location-map-json <path>   (optional: per-file lat/lon; keys = filepath, values = {lat, lon})
    location = None
    month = None
    hour = None
    lat = None
    lon = None
    location_map_json_path = None
    filelist_path = None
    out_csv_path = None
    classifier_model_path_override = None
    detector_threshold = DETECTOR_THRESHOLD
    classifier_threshold = CLASSIFIER_THRESHOLD
    pool_topk = POOL_TOPK_WINDOWS
    rms_gate_rel = RMS_GATE_REL
    rms_gate_abs = RMS_GATE_ABS
    platt_json_path = None
    threshold_on = "raw"
    infer_location_from_filename = False
    classifier_temperature = CLASSIFIER_TEMPERATURE

    args = sys.argv[1:]
    i = 0
    positionals = []
    while i < len(args):
        a = args[i]
        if a == "--filelist":
            if i + 1 >= len(args):
                print("\nERROR: --filelist requires a path")
                return
            filelist_path = args[i + 1]
            i += 2
            continue
        if a == "--out-csv":
            if i + 1 >= len(args):
                print("\nERROR: --out-csv requires a path")
                return
            out_csv_path = args[i + 1]
            i += 2
            continue
        if a == "--classifier-model":
            if i + 1 >= len(args):
                print("\nERROR: --classifier-model requires a path")
                return
            classifier_model_path_override = args[i + 1]
            i += 2
            continue
        if a == "--detector-threshold":
            if i + 1 >= len(args):
                print("\nERROR: --detector-threshold requires a float")
                return
            detector_threshold = float(args[i + 1])
            i += 2
            continue
        if a == "--classifier-threshold":
            if i + 1 >= len(args):
                print("\nERROR: --classifier-threshold requires a float")
                return
            classifier_threshold = float(args[i + 1])
            i += 2
            continue
        if a == "--pool-topk":
            if i + 1 >= len(args):
                print("\nERROR: --pool-topk requires an int")
                return
            pool_topk = int(args[i + 1])
            i += 2
            continue
        if a == "--rms-gate-rel":
            if i + 1 >= len(args):
                print("\nERROR: --rms-gate-rel requires a float")
                return
            rms_gate_rel = float(args[i + 1])
            i += 2
            continue
        if a == "--rms-gate-abs":
            if i + 1 >= len(args):
                print("\nERROR: --rms-gate-abs requires a float")
                return
            rms_gate_abs = float(args[i + 1])
            i += 2
            continue
        if a == "--platt-json":
            if i + 1 >= len(args):
                print("\nERROR: --platt-json requires a path")
                return
            platt_json_path = args[i + 1]
            i += 2
            continue
        if a == "--threshold-on":
            if i + 1 >= len(args):
                print("\nERROR: --threshold-on requires 'raw' or 'platt'")
                return
            threshold_on = str(args[i + 1]).strip().lower()
            i += 2
            continue
        if a == "--infer-location-from-filename":
            infer_location_from_filename = True
            i += 1
            continue
        if a == "--lat":
            if i + 1 >= len(args):
                print("\nERROR: --lat requires a float")
                return
            lat = float(args[i + 1])
            i += 2
            continue
        if a == "--lon":
            if i + 1 >= len(args):
                print("\nERROR: --lon requires a float")
                return
            lon = float(args[i + 1])
            i += 2
            continue
        if a == "--location-map-json":
            if i + 1 >= len(args):
                print("\nERROR: --location-map-json requires a path")
                return
            location_map_json_path = args[i + 1]
            i += 2
            continue
        if a == "--temperature":
            if i + 1 >= len(args):
                print("\nERROR: --temperature requires a float")
                return
            classifier_temperature = float(args[i + 1])
            i += 2
            continue
        positionals.append(a)
        i += 1

    # Apply runtime overrides for window selection/pooling.
    # These are module-level values used by select_active_windows() and predict_classifier().
    globals()["POOL_TOPK_WINDOWS"] = pool_topk
    globals()["RMS_GATE_REL"] = rms_gate_rel
    globals()["RMS_GATE_ABS"] = rms_gate_abs
    globals()["CLASSIFIER_TEMPERATURE"] = classifier_temperature

    # Optional Platt scaling (calibrate top1_prob -> P(correct)).
    platt_coef = None
    platt_intercept = None
    if platt_json_path:
        pth = Path(platt_json_path)
        if not pth.exists():
            print(f"\nERROR: Platt JSON not found: {pth}")
            return
        try:
            data = json.loads(pth.read_text(encoding="utf-8"))
            platt = data.get("platt", {})
            platt_coef = float(platt.get("coef"))
            platt_intercept = float(platt.get("intercept"))
        except Exception as e:
            print(f"\nERROR: Failed to load Platt JSON: {e}")
            return

    if threshold_on not in {"raw", "platt"}:
        print("\nERROR: --threshold-on must be 'raw' or 'platt'")
        return
    if threshold_on == "platt" and (platt_coef is None or platt_intercept is None):
        print("\nERROR: --threshold-on platt requires --platt-json with coef/intercept")
        return

    # Optional per-file location map (filepath -> {lat, lon})
    location_map = None
    if location_map_json_path:
        pth = Path(location_map_json_path)
        if not pth.exists():
            print(f"\nERROR: Location map JSON not found: {pth}")
            return
        try:
            data = json.loads(pth.read_text(encoding="utf-8"))
            location_map = {}
            for k, v in data.items():
                if isinstance(v, dict) and "lat" in v and "lon" in v:
                    location_map[k.strip()] = {"lat": float(v["lat"]), "lon": float(v["lon"])}
        except Exception as e:
            print(f"\nERROR: Failed to load location map JSON: {e}")
            return

    if positionals:
        location = positionals[0]
    if len(positionals) > 1:
        month = int(positionals[1])
    if len(positionals) > 2:
        hour = int(positionals[2])

    out_csv = Path(out_csv_path) if out_csv_path else OUT_CSV

    def infer_location_for_file(wav_path: Path) -> tuple[str | None, str]:
        """
        Best-effort location inference for this repo.

        IMPORTANT: To avoid label leakage on this dataset, we only use *place tokens*
        that appear in filenames (e.g. 'Pugu', 'Rondo'), and we do NOT infer location
        from species-name substrings.
        """
        s = wav_path.name
        if re.search(r"\bPugu\b", s, flags=re.IGNORECASE):
            return "Tanzania", "filename:Pugu"
        if re.search(r"\bRondo\b", s, flags=re.IGNORECASE):
            return "Tanzania", "filename:Rondo"
        return None, "none"
    
    # Check models
    if not DETECTOR_PATH.exists():
        print(f"\nERROR: Detector model not found: {DETECTOR_PATH}")
        return
    
    classifier_path = Path(classifier_model_path_override) if classifier_model_path_override else CLASSIFIER_PATH
    if not classifier_path.exists():
        print(f"\nERROR: Classifier model not found: {classifier_path}")
        return
    
    # Load models
    print(f"\nLoading models...")
    detector = tf.keras.models.load_model(DETECTOR_PATH)
    classifier = tf.keras.models.load_model(classifier_path)

    # Pick class-name file based on classifier output size (avoid mismatches that look like nonsense/uniform).
    n_out = int(classifier.output_shape[1])
    if n_out == 16 and CLASS_NAMES_PATH_16.exists():
        names_path = CLASS_NAMES_PATH_16
    elif n_out == 17 and CLASS_NAMES_PATH_17.exists():
        names_path = CLASS_NAMES_PATH_17
    elif n_out == 19 and CLASS_NAMES_PATH_19.exists():
        names_path = CLASS_NAMES_PATH_19
    else:
        names_path = CLASS_NAMES_PATH_FALLBACK

    with open(names_path, "r") as f:
        CLASS_NAMES = json.load(f)
    
    print(f"  Detector: {DETECTOR_PATH.name}")
    print(f"  Classifier: {classifier_path.name}")
    print(f"  Classes: {len(CLASS_NAMES)} species")
    print(f"  Class names: {names_path.name}")
    print(f"\nThresholds:")
    print(f"  Detector: {detector_threshold}")
    print(f"  Classifier: {classifier_threshold}")
    print(f"\nWindow selection / pooling:")
    print(f"  RMS_GATE_REL: {RMS_GATE_REL}")
    print(f"  RMS_GATE_ABS: {RMS_GATE_ABS}")
    print(f"  POOL_TOPK_WINDOWS: {POOL_TOPK_WINDOWS}")
    if platt_coef is not None and platt_intercept is not None:
        print(f"\nConfidence calibration:")
        print(f"  Platt: coef={platt_coef:.3f} intercept={platt_intercept:.3f}")
        print(f"  Thresholding on: {threshold_on}")
    print(f"  Classifier temperature: {classifier_temperature}")
    print(f"  Context alpha: {CONTEXT_ALPHA}")
    if infer_location_from_filename:
        print("\nLocation inference: enabled (filename tokens)")
    if lat is not None and lon is not None:
        print("\nPolygon location: enabled (lat/lon)")
        print(f"  lat={lat}, lon={lon}")
    if location_map is not None:
        print("\nPer-file location map: enabled")
        print(f"  {len(location_map)} entries")
    
    if location:
        print(f"\nContext:")
        print(f"  Location: {location}")
    if month is not None:
        print(f"  Month: {month}")
    if hour is not None:
        print(f"  Hour: {hour}")
    if not (location or month is not None or hour is not None):
        print(f"\nContext: None (no re-ranking)")
    
    # Find audio files
    if filelist_path:
        filelist = Path(filelist_path)
        if not filelist.exists():
            print(f"\nERROR: Filelist not found: {filelist}")
            return
        wav_files = []
        for line in filelist.read_text(encoding="utf-8").splitlines():
            line = line.strip().strip('"')
            if not line:
                continue
            p = Path(line)
            if p.exists() and p.suffix.lower() == ".wav":
                wav_files.append(p)
        wav_files = sorted(wav_files)
    else:
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

        # If user didn't provide a location, optionally infer it from filename.
        location_used = location
        location_source = "cli" if location else "none"
        if (location_used is None) and infer_location_from_filename:
            inferred, src = infer_location_for_file(wav)
            if inferred:
                location_used = inferred
                location_source = src

        # Lat/lon for polygon priors: per-file from map, else global
        lat_used = lat
        lon_used = lon
        if location_map is not None:
            wav_str = str(wav)
            wav_resolved = str(Path(wav).resolve())
            entry = location_map.get(wav_str) or location_map.get(wav_resolved)
            if entry is not None:
                lat_used = entry["lat"]
                lon_used = entry["lon"]
        
        # Stage 1: Detector
        # predict_detector() returns P(not_galago) (sigmoid output for label 1).
        not_galago_prob, det_windows = predict_detector(detector, wav)
        
        if not_galago_prob is None:
            detector_stats["error"] += 1
            results.append({
                "filepath": str(wav),
                "detector_threshold": f"{detector_threshold:.3f}",
                "classifier_threshold": f"{classifier_threshold:.3f}",
                "threshold_on": threshold_on,
                "platt_top1_prob": "N/A",
                "location_used": location_used or "N/A",
                "location_source": location_source,
                "lat": f"{lat_used:.6f}" if lat_used is not None else "N/A",
                "lon": f"{lon_used:.6f}" if lon_used is not None else "N/A",
                "rms_gate_rel": f"{RMS_GATE_REL:.3f}",
                "rms_gate_abs": f"{RMS_GATE_ABS:.6f}",
                "pool_topk_windows": str(POOL_TOPK_WINDOWS),
                "detector_result": "error",
                "detector_prob": "N/A",
                "species_result": "error",
                "species_prob": "N/A",
                "top1_species": "N/A",
                "top1_prob": "N/A",
                "top2_species": "N/A", "top2_prob": "N/A",
                "top3_species": "N/A", "top3_prob": "N/A",
                "location_status": "N/A",
                "original_prob": "N/A",
            })
            print("ERROR")
            continue
        
        galago_prob = 1.0 - float(not_galago_prob)
        
        if galago_prob < detector_threshold:
            detector_stats["not_galago"] += 1
            results.append({
                "filepath": str(wav),
                "detector_threshold": f"{detector_threshold:.3f}",
                "classifier_threshold": f"{classifier_threshold:.3f}",
                "threshold_on": threshold_on,
                "platt_top1_prob": "N/A",
                "location_used": location_used or "N/A",
                "location_source": location_source,
                "lat": f"{lat_used:.6f}" if lat_used is not None else "N/A",
                "lon": f"{lon_used:.6f}" if lon_used is not None else "N/A",
                "rms_gate_rel": f"{RMS_GATE_REL:.3f}",
                "rms_gate_abs": f"{RMS_GATE_ABS:.6f}",
                "pool_topk_windows": str(POOL_TOPK_WINDOWS),
                "detector_result": "not_galago",
                "detector_prob": f"{galago_prob:.3f}",
                "species_result": "not_classified",
                "species_prob": "N/A",
                "top1_species": "N/A",
                "top1_prob": "N/A",
                "top2_species": "N/A", "top2_prob": "N/A",
                "top3_species": "N/A", "top3_prob": "N/A",
                "location_status": "N/A",
                "original_prob": "N/A",
            })
            print(f"NOT GALAGO ({galago_prob:.3f})")
            continue
        
        # Stage 2: Classifier
        detector_stats["galago"] += 1
        probs, nwin, logits = predict_classifier(classifier, wav, CLASS_NAMES)
        
        if probs is None:
            results.append({
                "filepath": str(wav),
                "detector_threshold": f"{detector_threshold:.3f}",
                "classifier_threshold": f"{classifier_threshold:.3f}",
                "threshold_on": threshold_on,
                "platt_top1_prob": "N/A",
                "location_used": location_used or "N/A",
                "location_source": location_source,
                "lat": f"{lat_used:.6f}" if lat_used is not None else "N/A",
                "lon": f"{lon_used:.6f}" if lon_used is not None else "N/A",
                "rms_gate_rel": f"{RMS_GATE_REL:.3f}",
                "rms_gate_abs": f"{RMS_GATE_ABS:.6f}",
                "pool_topk_windows": str(POOL_TOPK_WINDOWS),
                "detector_result": "galago",
                "detector_prob": f"{galago_prob:.3f}",
                "species_result": "error",
                "species_prob": "N/A",
                "top1_species": "N/A",
                "top1_prob": "N/A",
                "top2_species": "N/A", "top2_prob": "N/A",
                "top3_species": "N/A", "top3_prob": "N/A",
                "location_status": "N/A",
                "original_prob": "N/A",
            })
            print("ERROR")
            continue
        
        t3 = topk(probs, CLASS_NAMES, 3)
        predictions_list = [(species, prob) for species, prob in t3]
        
        # Stage 3: Context Re-ranking
        if location_used or (lat_used is not None and lon_used is not None) or month is not None or hour is not None:
            reranked = rerank_predictions(
                predictions_list,
                location=location_used,
                lat=lat_used,
                lon=lon_used,
                month=month,
                hour=hour,
                alpha=CONTEXT_ALPHA
            )
            best_species, best_p, meta = reranked[0]
            top2_species, top2_p, _ = reranked[1] if len(reranked) > 1 else ("N/A", 0.0, {})
            top3_species, top3_p, _ = reranked[2] if len(reranked) > 2 else ("N/A", 0.0, {})
            if lat_used is not None and lon_used is not None:
                location_status = get_location_status_point(best_species, lat=lat_used, lon=lon_used)
            else:
                location_status = get_location_status(best_species, location_used)
            original_prob = meta.get('original_prob', best_p)
        else:
            best_species, best_p = t3[0]
            top2_species, top2_p = t3[1] if len(t3) > 1 else ("N/A", 0.0)
            top3_species, top3_p = t3[2] if len(t3) > 2 else ("N/A", 0.0)
            location_status = "N/A"
            original_prob = best_p
        
        # Optional Platt scaling for top-1 confidence (map raw prob -> estimated P(correct))
        platt_prob = None
        if platt_coef is not None and platt_intercept is not None:
            z = platt_coef * float(best_p) + platt_intercept
            # stable sigmoid
            if z >= 0:
                ez = math.exp(-z)
                platt_prob = 1.0 / (1.0 + ez)
            else:
                ez = math.exp(z)
                platt_prob = ez / (1.0 + ez)

        threshold_score = float(best_p) if threshold_on == "raw" else float(platt_prob)

        # Special-case: classifier may include a background class
        if best_species == "not_galago":
            species_out = "uncertain"
            location_status = "N/A"
        elif threshold_score < classifier_threshold:
            species_out = "uncertain"
        else:
            species_out = best_species
        
        # Always store the model's raw top-1 species even if we output "uncertain"
        results.append({
            "filepath": str(wav),
            "detector_threshold": f"{detector_threshold:.3f}",
            "classifier_threshold": f"{classifier_threshold:.3f}",
            "threshold_on": threshold_on,
            "platt_top1_prob": f"{platt_prob:.3f}" if platt_prob is not None else "N/A",
            "location_used": location_used or "N/A",
            "location_source": location_source,
            "lat": f"{lat_used:.6f}" if lat_used is not None else "N/A",
            "lon": f"{lon_used:.6f}" if lon_used is not None else "N/A",
            "rms_gate_rel": f"{RMS_GATE_REL:.3f}",
            "rms_gate_abs": f"{RMS_GATE_ABS:.6f}",
            "pool_topk_windows": str(POOL_TOPK_WINDOWS),
            "detector_result": "galago",
            "detector_prob": f"{galago_prob:.3f}",
            "species_result": species_out,
            "species_prob": f"{best_p:.3f}",
            "top1_species": best_species,
            "top1_prob": f"{best_p:.3f}",
            "top2_species": top2_species,
            "top2_prob": f"{top2_p:.3f}" if isinstance(top2_p, float) else str(top2_p),
            "top3_species": top3_species,
            "top3_prob": f"{top3_p:.3f}" if isinstance(top3_p, float) else str(top3_p),
            "location_status": location_status,
            "original_prob": f"{original_prob:.3f}",
        })
        
        status_marker = f" [{location_status}]" if location_status != "N/A" else ""
        print(f"GALAGO -> {species_out} ({best_p:.3f}){status_marker}")
    
    # Save results
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys() if results else [])
        w.writeheader()
        w.writerows(results)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total files: {len(wav_files)}")
    print(f"  Detector: {detector_stats['galago']} galago, {detector_stats['not_galago']} not_galago")
    
    classified = [r for r in results if r['species_result'] not in ['error', 'not_classified', 'N/A']]
    if classified:
        confident = sum(1 for r in classified if r['species_result'] != 'uncertain')
        print(f"  Species: {len(classified)} classified, {confident} confident")
    
    print(f"\nResults saved to: {out_csv}")

if __name__ == "__main__":
    main()

