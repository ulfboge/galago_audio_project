"""
Test detector in detail to see actual probabilities, not just threshold decisions.
"""
from pathlib import Path
import json
import numpy as np
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Config
SR = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FMIN = 200
FMAX = 10000
TARGET_FRAMES = 128
WINDOW_SEC = 2.5
HOP_SEC = 1.25

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DETECTOR_PATH = PROJECT_ROOT / "models" / "detector" / "galago_detector_best.keras"
AUDIO_DIR = PROJECT_ROOT / "data" / "raw_audio"

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
    if len(starts) < 1:
        starts = [max(0, (n_samples - win) // 2)]
    return starts

def wav_window_to_rgb_fixed(y: np.ndarray, sr: int) -> np.ndarray:
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT,
        hop_length=HOP_LENGTH, fmin=FMIN, fmax=FMAX
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    S_padded = pad_or_crop(S_db, TARGET_FRAMES)
    S_norm = (S_padded - S_padded.min()) / (S_padded.max() - S_padded.min() + 1e-8)
    cmap = plt.colormaps["magma"]
    S_rgb = cmap(S_norm)[:, :, :3]
    S_rgb = (S_rgb * 255).astype(np.uint8)
    return S_rgb[np.newaxis, :, :, :]

def predict_detector(detector, wav_path: Path):
    try:
        y, sr = librosa.load(str(wav_path), sr=SR)
    except Exception as e:
        return None, 0, []
    
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
        prob = detector.predict(rgb, verbose=0)[0][0]
        probs_list.append(prob)
    
    if not probs_list:
        return None, 0, []
    
    avg_prob = np.mean(probs_list)
    return avg_prob, len(probs_list), probs_list

def main():
    print("Detailed Detector Test")
    print("="*60)
    
    detector = tf.keras.models.load_model(DETECTOR_PATH)
    
    wav_files = sorted(AUDIO_DIR.rglob("*.wav"))[:10]  # Test first 10
    
    print(f"\nTesting {len(wav_files)} files...")
    print(f"\n{'File':<40} {'Avg Prob':<10} {'Windows':<10} {'Min':<10} {'Max':<10}")
    print("-"*60)
    
    for wav in wav_files:
        prob, nwin, probs = predict_detector(detector, wav)
        if prob is not None:
            min_p = min(probs) if probs else 0
            max_p = max(probs) if probs else 0
            print(f"{wav.name[:39]:<40} {prob:.4f}     {nwin:<10} {min_p:.4f}     {max_p:.4f}")
        else:
            print(f"{wav.name[:39]:<40} ERROR")

if __name__ == "__main__":
    main()

