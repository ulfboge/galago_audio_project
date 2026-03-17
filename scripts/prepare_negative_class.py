"""
Helper script to prepare negative class data for the galago detector.

This script helps organize non-galago audio into mel-spectrograms
for training the binary detector (galago vs not-galago).

Usage:
    python scripts/prepare_negative_class.py <source_dir> <category>
    
Example:
    python scripts/prepare_negative_class.py E:/Audio/Insects insects
    python scripts/prepare_negative_class.py E:/Audio/Frogs frogs
"""
from pathlib import Path
import sys
import librosa
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MELS_DIR = PROJECT_ROOT / "data" / "melspectrograms"
NEGATIVE_DIR = MELS_DIR / "not_galago"

# Audio parameters (must match training)
SR = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FMIN = 200
FMAX = 10000
TARGET_FRAMES = 128

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}

def make_mel_spectrogram(y: np.ndarray):
    """Compute log-mel spectrogram."""
    S = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db

def pad_or_crop(S: np.ndarray, target_frames: int) -> np.ndarray:
    """Pad/crop time axis to target_frames."""
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
    S_padded = np.pad(S, ((0, 0), (pad_left, pad_right)),
                      mode="constant", constant_values=pad_value)
    return S_padded

def save_spectrogram_png(S: np.ndarray, out_png: Path):
    """Save spectrogram as PNG."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(3, 3), dpi=100)
    plt.imshow(S, origin="lower", aspect="auto", cmap="magma")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0)
    plt.close()

def process_audio_file(audio_path: Path, category: str, output_dir: Path):
    """Process one audio file and save mel-spectrogram."""
    try:
        y, sr = librosa.load(str(audio_path), sr=SR, mono=True)
    except Exception as e:
        print(f"  [ERROR] Could not load {audio_path.name}: {e}")
        return False
    
    # Make spectrogram
    S_db = make_mel_spectrogram(y)
    S_fixed = pad_or_crop(S_db, TARGET_FRAMES)
    
    # Save PNG
    base_name = f"{category}_{audio_path.stem}.png"
    out_png = output_dir / base_name
    save_spectrogram_png(S_fixed, out_png)
    
    return True

def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/prepare_negative_class.py <source_dir> <category>")
        print("\nExample:")
        print("  python scripts/prepare_negative_class.py E:/Audio/Insects insects")
        print("  python scripts/prepare_negative_class.py E:/Audio/Frogs frogs")
        print("\nCategories: insects, frogs, birds, noise, human_speech, etc.")
        sys.exit(1)
    
    source_dir = Path(sys.argv[1])
    category = sys.argv[2]
    
    if not source_dir.exists():
        print(f"Error: Source directory not found: {source_dir}")
        sys.exit(1)
    
    # Create output directory
    NEGATIVE_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Preparing negative class data...")
    print(f"  Source: {source_dir}")
    print(f"  Category: {category}")
    print(f"  Output: {NEGATIVE_DIR}")
    print()
    
    # Find audio files
    audio_files = []
    for ext in AUDIO_EXTS:
        audio_files.extend(source_dir.rglob(f"*{ext}"))
        audio_files.extend(source_dir.rglob(f"*{ext.upper()}"))
    
    if len(audio_files) == 0:
        print(f"  No audio files found in {source_dir}")
        print(f"  Supported formats: {AUDIO_EXTS}")
        sys.exit(1)
    
    print(f"  Found {len(audio_files)} audio files")
    print(f"  Processing...")
    
    # Process files
    success = 0
    failed = 0
    
    for i, audio_file in enumerate(audio_files, 1):
        if process_audio_file(audio_file, category, NEGATIVE_DIR):
            success += 1
            if i % 10 == 0:
                print(f"    Processed {i}/{len(audio_files)}...", end="\r")
        else:
            failed += 1
    
    print(f"\n  Complete!")
    print(f"    Success: {success}")
    print(f"    Failed: {failed}")
    print(f"\n  Mel-spectrograms saved to: {NEGATIVE_DIR}")
    print(f"\n  Next step: Run 'python scripts/train_galago_detector.py'")

if __name__ == "__main__":
    main()

