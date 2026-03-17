from pathlib import Path
from collections import Counter

# Adjust if needed
MELS_DIR = PROJECT_ROOT / "data" / "melspectrograms"

def main():
    counts = Counter()

    if not MELS_DIR.exists():
        print(f"Directory not found: {MELS_DIR}")
        return

    for species_dir in MELS_DIR.iterdir():
        if not species_dir.is_dir():
            continue

        n_files = sum(1 for f in species_dir.iterdir() if f.suffix.lower() == ".png")
        counts[species_dir.name] = n_files

    print("PNG files per species:\n")
    for species, n in sorted(counts.items(), key=lambda x: x[0].lower()):
        print(f"{species:30s} {n:4d}")

if __name__ == "__main__":
    main()

