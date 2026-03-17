"""
Check which raw audio files haven't been processed into mel-spectrograms yet.
Helps identify opportunities to expand training data for underrepresented species.
"""
from pathlib import Path
from collections import defaultdict
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = Path(r"E:\Galagidae")
MELS_DIR = PROJECT_ROOT / "data" / "melspectrograms"

# Audio extensions
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".dat"}

# Import species patterns from make_mels.py
import sys
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
try:
    from make_mels import SPECIES_PATTERNS, find_species_label as make_mels_find_species
    find_species_label = make_mels_find_species
except ImportError:
    # Fallback if import fails
    def find_species_label(path: Path) -> str:
        """Try to identify species from file path."""
        path_str = str(path)
        # Simple pattern matching
        patterns = {
            "rondoensis": "Paragalago_rondoensis",
            "rondo": "Paragalago_rondoensis",
            "G_rondoensis": "Paragalago_rondoensis",
            "Paragalago rondoensis": "Paragalago_rondoensis",
            "Galagoides rondoensis": "Paragalago_rondoensis",
        }
        for pattern, species in patterns.items():
            if pattern in path_str:
                return species
        return None

def count_processed_files(species: str) -> int:
    """Count mel-spectrograms for a species."""
    species_dir = MELS_DIR / species
    if not species_dir.exists():
        return 0
    return len(list(species_dir.glob("*.png")))

def main():
    print("Checking Unprocessed Audio Files")
    print("=" * 70)
    
    if not BASE_DIR.exists():
        print(f"ERROR: Base directory not found: {BASE_DIR}")
        return
    
    # Find all audio files
    audio_files = []
    for ext in AUDIO_EXTS:
        audio_files.extend(BASE_DIR.rglob(f"*{ext}"))
        audio_files.extend(BASE_DIR.rglob(f"*{ext.upper()}"))
    
    print(f"\nFound {len(audio_files)} audio files in {BASE_DIR}")
    
    # Group by species
    species_audio = defaultdict(list)
    for audio_file in audio_files:
        species = find_species_label(audio_file)
        if species:
            species_audio[species].append(audio_file)
    
    # Compare with processed files
    print(f"\n{'Species':<30} {'Raw Audio':>12} {'Processed':>12} {'Missing':>12} {'% Processed':>12}")
    print("-" * 70)
    
    for species in sorted(species_audio.keys()):
        raw_count = len(species_audio[species])
        processed_count = count_processed_files(species)
        missing = raw_count - processed_count
        pct = (processed_count / raw_count * 100) if raw_count > 0 else 0
        
        print(f"{species:<30} {raw_count:>12} {processed_count:>12} {missing:>12} {pct:>11.1f}%")
    
    # Focus on underrepresented species
    print(f"\n{'='*70}")
    print("Underrepresented Species (from training data analysis):")
    print("-" * 70)
    
    underrepresented = [
        "Otolemur_crassicaudatus",
        "Otolemur_garnettii",
        "Galagoides_sp_nov",
        "Paragalago_rondoensis",
        "Paragalago_zanzibaricus",
    ]
    
    for species in underrepresented:
        if species in species_audio:
            raw_count = len(species_audio[species])
            processed_count = count_processed_files(species)
            missing = raw_count - processed_count
            
            if missing > 0:
                print(f"\n{species}:")
                print(f"  Raw audio files: {raw_count}")
                print(f"  Processed mels: {processed_count}")
                print(f"  Unprocessed: {missing} ({missing/raw_count*100:.1f}% of raw files)")
                print(f"  Recommendation: Process {missing} more files to expand training data")

if __name__ == "__main__":
    main()
