"""
Process missing audio files more aggressively.
This version will attempt to process all files, even if some fail.
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

def get_processed_filenames(species: str) -> set:
    """Get set of processed filenames (without _aug) for a species."""
    species_dir = MELS_DIR / species
    if not species_dir.exists():
        return set()
    # Get base names (without _aug suffix)
    processed = set()
    for png_file in species_dir.glob("*.png"):
        name = png_file.stem
        if "_aug" in name:
            name = name.split("_aug")[0]
        processed.add(name)
    return processed

def process_file_safe(audio_path: Path, species: str) -> bool:
    """Safely process one audio file, return True if successful."""
    # Skip macOS metadata files
    if audio_path.name.startswith("._"):
        return False
    
    try:
        y, _ = load_audio(audio_path)
    except Exception as e:
        # Silently skip files that can't be loaded
        return False

    try:
        S_db = make_mel_spectrogram(y)
        S_fixed = pad_or_crop(S_db, TARGET_FRAMES)

        out_dir = MELS_DIR / species
        out_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = audio_path.stem + ".png"
        out_png = out_dir / base_name
        
        # Skip if already exists
        if out_png.exists():
            return False
        
        save_spectrogram_png(S_fixed, out_png)
        return True
    except Exception:
        return False

def main():
    print("Processing Missing Audio Files for Underrepresented Species")
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
    
    # Group by species (skip macOS metadata)
    species_audio = {}
    skipped_metadata = 0
    for audio_file in audio_files:
        if audio_file.name.startswith("._"):
            skipped_metadata += 1
            continue
        species = find_species_label(audio_file)
        if species and species in PRIORITY_SPECIES:
            if species not in species_audio:
                species_audio[species] = []
            species_audio[species].append(audio_file)
    
    print(f"Skipped {skipped_metadata} macOS metadata files")
    
    # Process files for each priority species
    total_new = 0
    
    for species in PRIORITY_SPECIES:
        if species not in species_audio:
            continue
        
        raw_files = species_audio[species]
        processed_names = get_processed_filenames(species)
        
        print(f"\n{species}:")
        print(f"  Raw audio files: {len(raw_files)}")
        print(f"  Already processed: {len(processed_names)}")
        
        # Find unprocessed files
        unprocessed = []
        for audio_file in raw_files:
            if audio_file.stem not in processed_names:
                unprocessed.append(audio_file)
        
        print(f"  Unprocessed: {len(unprocessed)}")
        
        if len(unprocessed) == 0:
            print(f"  All files already processed!")
            continue
        
        print(f"  Processing {len(unprocessed)} new files...")
        
        species_new = 0
        errors = 0
        
        for audio_file in unprocessed:
            if process_file_safe(audio_file, species):
                species_new += 1
                total_new += 1
                if species_new % 10 == 0:
                    print(f"    Processed {species_new}/{len(unprocessed)}...")
            else:
                errors += 1
        
        print(f"  Successfully processed: {species_new} files")
        if errors > 0:
            print(f"  Failed to process: {errors} files (likely corrupted or unsupported format)")
        
        # Process augmented versions for successfully processed files
        if USE_AUGMENTATION and species_new > 0:
            print(f"  Generating augmented versions...")
            aug_count = 0
            for audio_file in unprocessed:
                if audio_file.stem in get_processed_filenames(species):
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
                                aug_count += 1
                    except Exception:
                        pass
            
            if aug_count > 0:
                print(f"  Generated {aug_count} augmented versions")
        
        new_total = len(get_processed_filenames(species))
        print(f"  New total: {new_total} processed files (+{new_total - len(processed_names)})")
    
    print(f"\n{'='*70}")
    print(f"Total new files processed: {total_new}")
    if total_new > 0:
        print(f"\nNext step: Retrain models with expanded dataset")
    else:
        print(f"\nAll files already processed. Ready to retrain with current dataset.")

if __name__ == "__main__":
    main()
