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

# Simple temporal consistency rule (optional):
# require the same top-1 class to appear in at least N of the pooled windows
# before returning a confident label. Set to 0 to disable.
CONSENSUS_MIN_COUNT = 0

# Thresholds
DETECTOR_THRESHOLD = 0.3  # Lowered from 0.7 to 0.5 to 0.3 to allow Otolemur through
# NOTE: We tune this based on desired coverage on the raw_audio WAV evaluation.
# If you want the system to return a label most of the time, use ~0.20–0.30.
# (On the 69-WAV set, 0.20 yields ~94% coverage.)
# Default confidence threshold for emitting a species label.
# On the 69-file raw_audio eval with the boosted 19-class model:
# - 0.20–0.30 yields ~100% coverage but lower reliability per prediction
# - 0.35 yields ~80% coverage with higher accuracy among covered predictions
CLASSIFIER_THRESHOLD = 0.35
# Note: With 16+ classes, max probability is naturally lower (~6.25% uniform baseline at 16 classes).
CONTEXT_ALPHA = 0.5  # Weight for context priors

# Temperature scaling (confidence calibration) for classifier probabilities.
# From `scripts/improve_model_confidence.py`:
#   - Optimal temperature on validation set: ~0.212
#   - Reduces ECE from ~0.69 -> ~0.09 (better calibrated confidences).
CLASSIFIER_TEMPERATURE = 0.212

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILES_JSON = PROJECT_ROOT / "configs" / "deployment_profiles.json"

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
    # Flip vertically to match training PNGs (generated with plt.imshow origin='lower',
    # which stores high-frequency bands at the top of the PNG file).
    # Also normalize to [0, 1] as training code does tf.cast / 255.0.
    S_rgb_float = S_rgb[::-1].astype(np.float32)
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
    top1_idx_list = []
    
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
        top1_idx_list.append(int(np.argmax(probs)))
    
    if not probs_list:
        return None, 0, None
    
    # Pool only the top-K most confident windows (avoid diluting with weak/noisy windows)
    k = min(POOL_TOPK_WINDOWS, len(probs_list))
    if k <= 0:
        return None, 0, None
    top_idx = np.argsort(np.asarray(conf_list))[-k:]
    selected = [probs_list[i] for i in top_idx]
    selected_top1_idx = [top1_idx_list[i] for i in top_idx]
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

    return avg_probs, len(starts_all), avg_probs, selected_top1_idx, k

def topk(probs: np.ndarray, class_names: list, k: int) -> list:
    top_indices = np.argsort(probs)[-k:][::-1]
    return [(class_names[i], probs[i]) for i in top_indices]


def infer_location_for_file(wav_path: Path) -> tuple[str | None, str]:
    """Infer Tanzania context from place tokens in filename (Pugu / Rondo)."""
    s = wav_path.name
    if re.search(r"\bPugu\b", s, flags=re.IGNORECASE):
        return "Tanzania", "filename:Pugu"
    if re.search(r"\bRondo\b", s, flags=re.IGNORECASE):
        return "Tanzania", "filename:Rondo"
    return None, "none"


def resolve_class_names_for_classifier(classifier_model) -> tuple[list, Path]:
    """Pick class_names JSON to match classifier output dimension."""
    n_out = int(classifier_model.output_shape[1])
    if n_out == 16 and CLASS_NAMES_PATH_16.exists():
        names_path = CLASS_NAMES_PATH_16
    elif n_out == 17 and CLASS_NAMES_PATH_17.exists():
        names_path = CLASS_NAMES_PATH_17
    elif n_out == 19 and CLASS_NAMES_PATH_19.exists():
        names_path = CLASS_NAMES_PATH_19
    else:
        names_path = CLASS_NAMES_PATH_FALLBACK
    with open(names_path, "r", encoding="utf-8") as f:
        names = json.load(f)
    if not isinstance(names, list):
        raise ValueError(f"{names_path.name} must be a JSON list of class name strings.")
    if len(names) != n_out:
        raise ValueError(
            f"Classifier output dim is {n_out} but {names_path.name} lists {len(names)} labels. "
            f"Add or fix models/all_species/class_names_{n_out}.json for this model."
        )
    return names, names_path


def run_single_wav(
    wav: Path,
    *,
    detector,
    classifier,
    class_names: list,
    location: str | None,
    month: int | None,
    hour: int | None,
    lat: float | None,
    lon: float | None,
    location_map: dict | None,
    infer_location_from_filename: bool,
    detector_threshold: float,
    classifier_threshold: float,
    pool_topk: int,
    rms_gate_rel: float,
    rms_gate_abs: float,
    classifier_temperature: float,
    consensus_min_count: int,
    context_alpha: float,
    threshold_on: str,
    platt_coef: float | None,
    platt_intercept: float | None,
    postprocess_mode: str,
) -> dict:
    """
    Run detector + classifier + context + gates for one WAV.
    Mutates module-level pool/RMS/temperature/consensus globals used by predict_classifier.
    """
    globals()["POOL_TOPK_WINDOWS"] = pool_topk
    globals()["RMS_GATE_REL"] = rms_gate_rel
    globals()["RMS_GATE_ABS"] = rms_gate_abs
    globals()["CLASSIFIER_TEMPERATURE"] = classifier_temperature
    globals()["CONSENSUS_MIN_COUNT"] = consensus_min_count
    globals()["CONTEXT_ALPHA"] = context_alpha

    def _base_row(**kwargs) -> dict:
        base = {
            "filepath": str(wav),
            "detector_threshold": f"{detector_threshold:.3f}",
            "classifier_threshold": f"{classifier_threshold:.3f}",
            "threshold_on": threshold_on,
            "platt_top1_prob": "N/A",
            "location_used": "N/A",
            "location_source": "none",
            "lat": "N/A",
            "lon": "N/A",
            "rms_gate_rel": f"{rms_gate_rel:.3f}",
            "rms_gate_abs": f"{rms_gate_abs:.6f}",
            "pool_topk_windows": str(pool_topk),
            "consensus_min_count": str(consensus_min_count),
            "consensus_pooled_k": "N/A",
            "consensus_best_count": "N/A",
            "postprocess_mode": str(postprocess_mode),
            "postprocess_action": "none",
            "detector_result": "error",
            "detector_prob": "N/A",
            "species_result": "error",
            "species_prob": "N/A",
            "top1_species": "N/A",
            "top1_prob": "N/A",
            "top2_species": "N/A",
            "top2_prob": "N/A",
            "top3_species": "N/A",
            "top3_prob": "N/A",
            "location_status": "N/A",
            "original_prob": "N/A",
            "acoustic_top10": "N/A",
        }
        base.update(kwargs)
        return base

    location_used = location
    location_source = "cli" if location else "none"
    if (location_used is None) and infer_location_from_filename:
        inferred, src = infer_location_for_file(wav)
        if inferred:
            location_used = inferred
            location_source = src

    lat_used = lat
    lon_used = lon
    if location_map is not None:
        wav_str = str(wav)
        wav_resolved = str(Path(wav).resolve())
        entry = location_map.get(wav_str) or location_map.get(wav_resolved)
        if entry is not None:
            lat_used = entry["lat"]
            lon_used = entry["lon"]

    not_galago_prob, det_windows = predict_detector(detector, wav)
    if not_galago_prob is None:
        return _base_row(
            location_used=location_used or "N/A",
            location_source=location_source,
            lat=f"{lat_used:.6f}" if lat_used is not None else "N/A",
            lon=f"{lon_used:.6f}" if lon_used is not None else "N/A",
            detector_result="error",
        )

    galago_prob = 1.0 - float(not_galago_prob)
    if galago_prob < detector_threshold:
        return _base_row(
            location_used=location_used or "N/A",
            location_source=location_source,
            lat=f"{lat_used:.6f}" if lat_used is not None else "N/A",
            lon=f"{lon_used:.6f}" if lon_used is not None else "N/A",
            postprocess_action="none",
            detector_result="not_galago",
            detector_prob=f"{galago_prob:.3f}",
            species_result="not_classified",
        )

    probs, nwin, logits, selected_top1_idx, pooled_k = predict_classifier(classifier, wav, class_names)
    if probs is None:
        return _base_row(
            location_used=location_used or "N/A",
            location_source=location_source,
            lat=f"{lat_used:.6f}" if lat_used is not None else "N/A",
            lon=f"{lon_used:.6f}" if lon_used is not None else "N/A",
            detector_result="galago",
            detector_prob=f"{galago_prob:.3f}",
            consensus_pooled_k="N/A",
        )

    t3 = topk(probs, class_names, 3)
    k_acoustic = min(10, len(class_names))
    t10 = topk(probs, class_names, k_acoustic)
    acoustic_top10 = " · ".join(f"{s} {float(p):.3f}" for s, p in t10)
    predictions_list = [(species, prob) for species, prob in t3]

    if location_used or (lat_used is not None and lon_used is not None) or month is not None or hour is not None:
        reranked = rerank_predictions(
            predictions_list,
            location=location_used,
            lat=lat_used,
            lon=lon_used,
            month=month,
            hour=hour,
            alpha=context_alpha,
        )
        best_species, best_p, meta = reranked[0]
        top2_species, top2_p, _ = reranked[1] if len(reranked) > 1 else ("N/A", 0.0, {})
        top3_species, top3_p, _ = reranked[2] if len(reranked) > 2 else ("N/A", 0.0, {})
        if lat_used is not None and lon_used is not None:
            location_status = get_location_status_point(best_species, lat=lat_used, lon=lon_used)
        else:
            location_status = get_location_status(best_species, location_used)
        original_prob = meta.get("original_prob", best_p)
    else:
        best_species, best_p = t3[0]
        top2_species, top2_p = t3[1] if len(t3) > 1 else ("N/A", 0.0)
        top3_species, top3_p = t3[2] if len(t3) > 2 else ("N/A", 0.0)
        location_status = "N/A"
        original_prob = best_p

    consensus_best_count = "N/A"
    if isinstance(selected_top1_idx, list) and pooled_k:
        try:
            consensus_best_count = str(
                sum(1 for idx in selected_top1_idx if class_names[int(idx)] == best_species)
            )
        except Exception:
            consensus_best_count = "N/A"

    platt_prob = None
    if platt_coef is not None and platt_intercept is not None:
        z = platt_coef * float(best_p) + platt_intercept
        if z >= 0:
            ez = math.exp(-z)
            platt_prob = 1.0 / (1.0 + ez)
        else:
            ez = math.exp(z)
            platt_prob = ez / (1.0 + ez)

    threshold_score = float(best_p) if threshold_on == "raw" else float(platt_prob)
    species_out = best_species

    if best_species == "not_galago":
        species_out = "uncertain"
        location_status = "N/A"
    elif consensus_min_count and consensus_min_count > 0 and consensus_best_count != "N/A":
        if int(consensus_best_count) < int(consensus_min_count):
            species_out = "uncertain"
    elif threshold_score < classifier_threshold:
        species_out = "uncertain"

    postprocess_action = "none"
    if postprocess_mode == "tanzania_rondoensis_guard":
        location_l = (location_used or "").strip().lower()
        wav_name = wav.name.lower()
        is_tanzania = (
            ("tanzania" in location_l)
            or ("filename:pugu" in location_source.lower())
            or ("filename:rondo" in location_source.lower())
            or ("pugu" in wav_name)
            or ("rondo" in wav_name)
            or ("pande" in wav_name)
        )
        if is_tanzania and species_out == "Galagoides_sp_nov":
            if (location_status == "out_of_range") and (top2_species == "Paragalago_rondoensis"):
                species_out = "Paragalago_rondoensis"
                postprocess_action = "spnov_to_rondoensis"
            else:
                species_out = "uncertain"
                postprocess_action = "spnov_to_uncertain"
    elif postprocess_mode == "tanzania_spnov_to_rondoensis":
        wav_name = wav.name.lower()
        location_l = (location_used or "").strip().lower()
        is_tanzania = (
            ("tanzania" in location_l)
            or ("filename:pugu" in location_source.lower())
            or ("filename:rondo" in location_source.lower())
            or ("pugu" in wav_name)
            or ("rondo" in wav_name)
            or ("pande" in wav_name)
        )
        if is_tanzania and species_out == "Galagoides_sp_nov":
            species_out = "Paragalago_rondoensis"
            postprocess_action = "spnov_to_rondoensis_aggressive"

    return {
        "filepath": str(wav),
        "detector_threshold": f"{detector_threshold:.3f}",
        "classifier_threshold": f"{classifier_threshold:.3f}",
        "threshold_on": threshold_on,
        "platt_top1_prob": f"{platt_prob:.3f}" if platt_prob is not None else "N/A",
        "location_used": location_used or "N/A",
        "location_source": location_source,
        "lat": f"{lat_used:.6f}" if lat_used is not None else "N/A",
        "lon": f"{lon_used:.6f}" if lon_used is not None else "N/A",
        "rms_gate_rel": f"{rms_gate_rel:.3f}",
        "rms_gate_abs": f"{rms_gate_abs:.6f}",
        "pool_topk_windows": str(pool_topk),
        "consensus_min_count": str(consensus_min_count),
        "consensus_pooled_k": str(pooled_k) if pooled_k is not None else "N/A",
        "consensus_best_count": consensus_best_count,
        "postprocess_mode": str(postprocess_mode),
        "postprocess_action": postprocess_action,
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
        "acoustic_top10": acoustic_top10,
    }


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
    #   --wav <path>                 (optional: single WAV file; skips filelist / scanning data/raw_audio)
    #   --location-map-json <path>   (optional: per-file lat/lon; keys = filepath, values = {lat, lon})
    #   --consensus-min-count <int>  (optional: require best class in >=N pooled windows; 0 disables)
    #   --profiles-json <path>       (optional: JSON with named parameter profiles)
    #   --profile <name>             (optional: apply a profile; explicit flags override)
    location = None
    month = None
    hour = None
    lat = None
    lon = None
    location_map_json_path = None
    filelist_path = None
    wav_single_path = None
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
    consensus_min_count = CONSENSUS_MIN_COUNT
    postprocess_mode = "none"
    profiles_json_path = str(DEFAULT_PROFILES_JSON) if DEFAULT_PROFILES_JSON.exists() else None
    profile_name = None

    args = sys.argv[1:]
    i = 0
    positionals = []

    def apply_profile(name: str) -> None:
        nonlocal classifier_threshold, detector_threshold, pool_topk, rms_gate_rel, rms_gate_abs
        nonlocal consensus_min_count, classifier_temperature, postprocess_mode
        nonlocal classifier_model_path_override
        try:
            if not profiles_json_path:
                print("\nERROR: --profile used but no profiles JSON is available")
                raise SystemExit(2)
            pth = Path(profiles_json_path)
            if not pth.exists():
                print(f"\nERROR: Profiles JSON not found: {pth}")
                raise SystemExit(2)
            data = json.loads(pth.read_text(encoding="utf-8"))
            prof = data.get(name)
            if not isinstance(prof, dict):
                print(f"\nERROR: Unknown profile: {name}")
                print(f"Available: {', '.join(sorted(data.keys()))}")
                raise SystemExit(2)
            classifier_threshold = float(prof.get("classifier_threshold", classifier_threshold))
            detector_threshold = float(prof.get("detector_threshold", detector_threshold))
            pool_topk = int(prof.get("pool_topk_windows", pool_topk))
            rms_gate_rel = float(prof.get("rms_gate_rel", rms_gate_rel))
            rms_gate_abs = float(prof.get("rms_gate_abs", rms_gate_abs))
            consensus_min_count = int(prof.get("consensus_min_count", consensus_min_count))
            classifier_temperature = float(prof.get("temperature", classifier_temperature))
            postprocess_mode = str(prof.get("postprocess_mode", postprocess_mode))
            # Context alpha is global; set it if present
            if "context_alpha" in prof:
                globals()["CONTEXT_ALPHA"] = float(prof["context_alpha"])
            # Optional profile-level classifier model override (only if not set via --classifier-model)
            if "classifier_model" in prof and classifier_model_path_override is None:
                model_val = prof["classifier_model"]
                # Support relative paths (relative to PROJECT_ROOT)
                m_path = Path(model_val)
                if not m_path.is_absolute():
                    m_path = PROJECT_ROOT / m_path
                classifier_model_path_override = str(m_path)
        except SystemExit:
            raise
        except Exception as e:
            print(f"\nERROR: Failed to apply profile '{name}': {e}")
            raise SystemExit(2)
    while i < len(args):
        a = args[i]
        if a == "--profiles-json":
            if i + 1 >= len(args):
                print("\nERROR: --profiles-json requires a path")
                return
            profiles_json_path = args[i + 1]
            i += 2
            continue
        if a == "--profile":
            if i + 1 >= len(args):
                print("\nERROR: --profile requires a name")
                return
            profile_name = args[i + 1]
            apply_profile(profile_name)
            i += 2
            continue
        if a == "--filelist":
            if i + 1 >= len(args):
                print("\nERROR: --filelist requires a path")
                return
            filelist_path = args[i + 1]
            i += 2
            continue
        if a == "--wav":
            if i + 1 >= len(args):
                print("\nERROR: --wav requires a path")
                return
            wav_single_path = args[i + 1]
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
        if a == "--consensus-min-count":
            if i + 1 >= len(args):
                print("\nERROR: --consensus-min-count requires an int")
                return
            consensus_min_count = int(args[i + 1])
            i += 2
            continue
        if a == "--postprocess-mode":
            if i + 1 >= len(args):
                print("\nERROR: --postprocess-mode requires a value")
                return
            postprocess_mode = str(args[i + 1]).strip().lower()
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
    globals()["CONSENSUS_MIN_COUNT"] = consensus_min_count

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

    CLASS_NAMES, names_path = resolve_class_names_for_classifier(classifier)

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
    print(f"  Consensus min-count: {CONSENSUS_MIN_COUNT}")
    if profile_name:
        print(f"\nProfile: {profile_name}")
    if platt_coef is not None and platt_intercept is not None:
        print(f"\nConfidence calibration:")
        print(f"  Platt: coef={platt_coef:.3f} intercept={platt_intercept:.3f}")
        print(f"  Thresholding on: {threshold_on}")
    print(f"  Classifier temperature: {classifier_temperature}")
    print(f"  Context alpha: {CONTEXT_ALPHA}")
    print(f"  Postprocess mode: {postprocess_mode}")
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
    if wav_single_path:
        wone = Path(wav_single_path)
        if not wone.exists():
            print(f"\nERROR: WAV not found: {wone}")
            return
        if wone.suffix.lower() != ".wav":
            print(f"\nERROR: --wav must point to a .wav file: {wone}")
            return
        wav_files = [wone.resolve()]
    elif filelist_path:
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
        row = run_single_wav(
            wav,
            detector=detector,
            classifier=classifier,
            class_names=CLASS_NAMES,
            location=location,
            month=month,
            hour=hour,
            lat=lat,
            lon=lon,
            location_map=location_map,
            infer_location_from_filename=infer_location_from_filename,
            detector_threshold=detector_threshold,
            classifier_threshold=classifier_threshold,
            pool_topk=pool_topk,
            rms_gate_rel=rms_gate_rel,
            rms_gate_abs=rms_gate_abs,
            classifier_temperature=classifier_temperature,
            consensus_min_count=consensus_min_count,
            context_alpha=float(CONTEXT_ALPHA),
            threshold_on=threshold_on,
            platt_coef=platt_coef,
            platt_intercept=platt_intercept,
            postprocess_mode=postprocess_mode,
        )
        results.append(row)
        dr = row["detector_result"]
        if dr == "error":
            detector_stats["error"] += 1
            print("ERROR")
        elif dr == "not_galago":
            detector_stats["not_galago"] += 1
            print(f"NOT GALAGO ({row['detector_prob']})")
        else:
            detector_stats["galago"] += 1
            if row["species_result"] == "error":
                print("ERROR")
            else:
                loc = row["location_status"]
                sm = f" [{loc}]" if loc != "N/A" else ""
                print(f"GALAGO -> {row['species_result']} ({row['species_prob']}){sm}")
    
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

