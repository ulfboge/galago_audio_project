"""
Process additional audio files for underrepresented species.
This script processes unprocessed audio files to expand training data.
"""
from pathlib import Path
import sys
import librosa
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = Path(r"E:\Galagidae")
MELS_DIR = PROJECT_ROOT / "data" / "melspectrograms"

# Import from make_mels.py
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from make_mels import (
    SPECIES_PATTERNS, find_species_label, load_audio,
    make_mel_spectrogram, pad_or_crop, save_spectrogram_png,
    SR, N_MELS, N_FFT, HOP_LENGTH, FMIN, FMAX, TARGET_FRAMES,
    AUDIO_EXTS, USE_AUGMENTATION, AUG_PER_FILE, augment_audio
)

# Underrepresented species to prioritize
PRIORITY_SPECIES = [
    "Otolemur_crassicaudatus",
    "Otolemur_garnettii",
    "Galagoides_sp_nov",
    "Paragalago_rondoensis",
    "Paragalago_zanzibaricus",
    "Galagoides_thomasi",
    "Sciurocheirus_alleni",
]

def count_processed(species: str) -> int:
    """Count processed mel-spectrograms for a species."""
    species_dir = MELS_DIR / species
    if not species_dir.exists():
        return 0
    # Count non-augmented files (exclude _aug files)
    png_files = [f for f in species_dir.glob("*.png") if "_aug" not in f.name]
    return len(png_files)

def process_audio_file(audio_path: Path, species: str, idx_suffix: str = ""):
    """Process one audio file and save mel-spectrogram."""
    try:
        y, _ = load_audio(audio_path)
    except Exception as e:
        print(f"  [ERROR] Could not load {audio_path.name}: {e}")
        return False

    S_db = make_mel_spectrogram(y)
    S_fixed = pad_or_crop(S_db, TARGET_FRAMES)

    out_dir = MELS_DIR / species
    out_dir.mkdir(parents=True, exist_ok=True)
    
    base_name = audio_path.stem + idx_suffix + ".png"
    out_png = out_dir / base_name
    
    # Skip if already exists
    if out_png.exists():
        return False
    
    save_spectrogram_png(S_fixed, out_png)
    return True

def main():
    print("Processing Additional Audio Files for Underrepresented Species")
    print("=" * 70)
    
    if not BASE_DIR.exists():
        print(f"ERROR: Base directory not found: {BASE_DIR}")
        return
    
    # Find all audio files
    audio_files = []
    for ext in AUDIO_EXTS:
        audio_files.extend(BASE_DIR.rglob(f"*{ext}"))
        audio_files.extend(BASE_DIR.rglob(f"*{ext.upper()}"))
    
    print(f"Found {len(audio_files)} total audio files")
    
    # Group by species (skip macOS metadata files)
    species_audio = {}
    for audio_file in audio_files:
        # Skip macOS resource fork files
        if audio_file.name.startswith("._"):
            continue
        species = find_species_label(audio_file)
        if species and species in PRIORITY_SPECIES:
            if species not in species_audio:
                species_audio[species] = []
            species_audio[species].append(audio_file)
    
    # Process files for each priority species
    total_processed = 0
    
    for species in PRIORITY_SPECIES:
        if species not in species_audio:
            continue
        
        raw_files = species_audio[species]
        processed_count = count_processed(species)
        
        print(f"\n{species}:")
        print(f"  Raw audio files: {len(raw_files)}")
        print(f"  Already processed: {processed_count}")
        print(f"  Processing additional files...")
        
        species_processed = 0
        for audio_file in raw_files:
            # Process original
            if process_audio_file(audio_file, species):
                species_processed += 1
                total_processed += 1
                print(f"    [OK] {audio_file.name}")
            
            # Process augmented versions
            if USE_AUGMENTATION:
                try:
                    y, _ = load_audio(audio_file)
                    aug_list = augment_audio(y)[:AUG_PER_FILE]
                    
                    for i, y_aug in enumerate(aug_list, start=1):
                        S_db_aug = make_mel_spectrogram(y_aug)
                        S_fixed_aug = pad_or_crop(S_db_aug, TARGET_FRAMES)
                        
                        out_dir = MELS_DIR / species
                        idx_suffix = f"_aug{i}"
                        out_png = out_dir / (audio_file.stem + idx_suffix + ".png")
                        
                        if not out_png.exists():
                            save_spectrogram_png(S_fixed_aug, out_png)
                            print(f"    [AUG] {audio_file.name} -> aug{i}")
                except Exception as e:
                    print(f"    [ERROR] Augmentation failed for {audio_file.name}: {e}")
        
        new_count = count_processed(species)
        print(f"  New total: {new_count} processed files (+{new_count - processed_count})")
    
    print(f"\n{'='*70}")
    print(f"Total new files processed: {total_processed}")
    print(f"\nNext step: Retrain models with expanded dataset")

if __name__ == "__main__":
    main()
