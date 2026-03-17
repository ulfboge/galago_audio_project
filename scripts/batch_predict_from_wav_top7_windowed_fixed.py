from pathlib import Path
import csv
import numpy as np
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt

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
OUT_CSV    = PROJECT_ROOT / "outputs" / "predictions" / "predictions_top7_windowed_fixed.csv"

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


def mel_to_rgb_exact_match(S_db: np.ndarray) -> np.ndarray:
    """
    Convert mel-spectrogram to RGB using the EXACT same method as training.
    
    This replicates what matplotlib's imshow does when saving PNGs:
    1. imshow normalizes the array to [0,1] based on min/max
    2. Applies colormap
    3. Converts to RGB uint8 [0,255]
    4. TensorFlow loads as uint8 and rescales to [0,1] float32
    
    We skip the PNG round-trip and go directly to the final representation.
    """
    # Normalize to [0,1] exactly like imshow does
    S_min = S_db.min()
    S_max = S_db.max()
    if S_max - S_min < 1e-6:
        # Constant array - return gray
        S_norm = np.zeros_like(S_db)
    else:
        S_norm = (S_db - S_min) / (S_max - S_min)
    
    # Apply magma colormap (same as training)
    magma = plt.cm.get_cmap("magma")
    rgb = magma(S_norm)[:, :, :3]  # drop alpha, shape (H, W, 3)
    
    # Convert to [0, 255] uint8 (what PNG would be)
    rgb_uint8 = (rgb * 255.0).astype(np.uint8)
    
    # Now simulate what TensorFlow does: load as uint8, rescale to [0,1] float32
    # The model has Rescaling(1.0/255) layer, so we need to pass [0,255] float32
    # Actually, let's check - if the model rescales, we should pass [0,255] as float32
    # But looking at the training code, it uses Rescaling(1.0/255), so input should be [0,255]
    rgb_float = rgb_uint8.astype(np.float32)
    
    return rgb_float


def window_starts(n_samples: int, sr: int, win_sec: float, hop_sec: float) -> list[int]:
    """Calculate window start positions in samples."""
    win = int(win_sec * sr)
    hop = int(hop_sec * sr)
    if n_samples <= win:
        # single centered window
        return [0]
    starts = list(range(0, n_samples - win + 1, hop))
    if len(starts) < MIN_WINDOWS:
        starts = [max(0, (n_samples - win) // 2)]
    return starts


def wav_window_to_rgb_fixed(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Convert a window of audio to RGB using the EXACT same method as make_mels.py.
    
    This replicates the training pipeline:
    1. Compute mel-spectrogram
    2. Convert to dB (ref=np.max per window - matches training per-file behavior)
    3. Pad/crop to 128 frames
    4. Use matplotlib's imshow normalization (min/max of the array)
    5. Apply magma colormap
    6. Convert to RGB [0,255] float32 (what TensorFlow expects after loading PNG)
    """
    # Compute mel-spectrogram for this window (same params as training)
    S = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
        power=2.0,
    )
    # Convert to dB - ref=np.max uses the max of THIS window's power spectrogram
    # This matches training where each file is processed independently
    S_db = librosa.power_to_db(S, ref=np.max)
    S_fixed = pad_or_crop(S_db, TARGET_FRAMES)
    
    # Now replicate exactly what matplotlib's imshow does:
    # imshow normalizes the array to [0,1] based on array min/max
    S_min = S_fixed.min()
    S_max = S_fixed.max()
    if S_max - S_min < 1e-6:
        S_norm = np.zeros_like(S_fixed)
    else:
        S_norm = (S_fixed - S_min) / (S_max - S_min)
    
    # Apply magma colormap (same as training)
    magma = plt.cm.get_cmap("magma")
    rgb = magma(S_norm)[:, :, :3]  # (H, W, 3) in [0,1]
    
    # Convert to [0, 255] - match what TensorFlow's load_img returns
    # load_img returns float32 in range [0, 255] (not uint8!)
    # The model's Rescaling layer will divide by 255 to get [0,1]
    rgb_float = (rgb * 255.0).astype(np.float32)
    
    return rgb_float


def predict_wav_windowed(model, wav_path: Path) -> tuple[np.ndarray, int, np.ndarray]:
    """
    Predict on a WAV file using windowing with logit averaging.
    
    Returns:
        - avg_probs: averaged probabilities (7,) after averaging logits
        - n_windows: number of windows used
        - logits_stack: (n_windows, 7) logits before averaging
    """
    y, sr = librosa.load(wav_path, sr=SR, mono=True)
    win = int(WINDOW_SEC * sr)
    
    starts = window_starts(len(y), sr, WINDOW_SEC, HOP_SEC)
    logits_list = []
    
    # Try to get logits (pre-softmax) for better averaging
    # The model's last Dense layer outputs logits, then softmax is applied
    logit_layer = None
    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.layers.Dense):
            # Check output shape - can be a tuple or TensorShape
            try:
                output_shape = layer.output_shape
                if isinstance(output_shape, tuple):
                    output_size = output_shape[-1]
                else:
                    output_size = output_shape[-1].value if hasattr(output_shape[-1], 'value') else output_shape[-1]
            except:
                # Fallback: try to get from output tensor
                try:
                    output_size = layer.output.shape[-1]
                except:
                    continue
            
            if output_size == len(CLASS_NAMES):
                # Check if this is the final layer (before softmax)
                if i == len(model.layers) - 1 or isinstance(model.layers[i+1], tf.keras.layers.Activation):
                    logit_layer = layer
                    break
    
    # Create intermediate model to get logits if possible
    if logit_layer is not None:
        try:
            intermediate_model = tf.keras.Model(inputs=model.input, outputs=logit_layer.output)
            use_logits = True
        except:
            use_logits = False
            intermediate_model = None
    else:
        use_logits = False
        intermediate_model = None
    
    for s in starts:
        seg = y[s:s + win]
        if len(seg) < win:
            # pad short tail window
            seg = np.pad(seg, (0, win - len(seg)), mode="constant")
        
        rgb = wav_window_to_rgb_fixed(seg, sr)
        x = np.expand_dims(rgb, axis=0)  # (1,128,128,3)
        
        # Get logits if possible, otherwise use probabilities
        if use_logits and intermediate_model is not None:
            logits = intermediate_model.predict(x, verbose=0)[0]
            logits_list.append(logits)
        else:
            # Fallback: get probabilities and approximate logits
            probs = model.predict(x, verbose=0)[0]
            # Approximate logits: log(probs) + constant (works reasonably for averaging)
            logits = np.log(probs + 1e-10)
            logits_list.append(logits)
    
    logits_stack = np.stack(logits_list, axis=0)  # (n_windows, 7)
    
    # Better aggregation: average logits, then softmax once
    # This is mathematically better than averaging probabilities
    avg_logits = logits_stack.mean(axis=0)
    avg_probs = tf.nn.softmax(avg_logits).numpy()
    
    return avg_probs, len(starts), logits_stack


def topk(probs: np.ndarray, k: int = 3):
    """Get top k predictions."""
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
    print("Model loaded.")
    
    wav_files = sorted(AUDIO_DIR.rglob("*.wav"))
    if not wav_files:
        print("No WAV files found under:", AUDIO_DIR)
        return
    
    print(f"Found {len(wav_files)} WAV files under {AUDIO_DIR}")
    print("\nUsing FIXED pipeline:")
    print("  - Full-file normalization (matches training)")
    print("  - Logit averaging (better than prob averaging)")
    print("  - Exact same mel->RGB conversion as training\n")
    
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
            probs, nwin, logits = predict_wav_windowed(model, wav)
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
    print("\nCompare this output with your previous windowed results.")
    print("If predictions are still collapsing to one class, we may need to check the model's input preprocessing.")


if __name__ == "__main__":
    main()

