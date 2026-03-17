"""
Analyze training data distribution to identify underrepresented species.
This helps understand why the detector might struggle with certain species.
"""
from pathlib import Path
from collections import defaultdict
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MELS_DIR = PROJECT_ROOT / "data" / "melspectrograms"

def main():
    print("Analyzing Training Data Distribution")
    print("=" * 60)
    
    # Count samples per species
    species_counts = {}
    total_galago = 0
    
    for species_dir in sorted(MELS_DIR.iterdir()):
        if not species_dir.is_dir():
            continue
        if species_dir.name == "not_galago":
            continue
        
        png_files = list(species_dir.glob("*.png"))
        count = len(png_files)
        species_counts[species_dir.name] = count
        total_galago += count
    
    # Negative class
    not_galago_dir = MELS_DIR / "not_galago"
    not_galago_count = len(list(not_galago_dir.glob("*.png"))) if not_galago_dir.exists() else 0
    
    # Sort by count
    sorted_species = sorted(species_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nSpecies Distribution (sorted by sample count):")
    print("-" * 60)
    print(f"{'Species':<35} {'Samples':>8} {'% of Total':>12} {'Ratio vs Max':>15}")
    print("-" * 60)
    
    max_count = max(species_counts.values()) if species_counts else 1
    
    for species, count in sorted_species:
        pct = (count / total_galago * 100) if total_galago > 0 else 0
        ratio = count / max_count if max_count > 0 else 0
        print(f"{species:<35} {count:>8} {pct:>11.1f}% {ratio:>14.2f}x")
    
    print("-" * 60)
    print(f"{'TOTAL (galago)':<35} {total_galago:>8}")
    print(f"{'not_galago':<35} {not_galago_count:>8}")
    print(f"{'TOTAL (all)':<35} {total_galago + not_galago_count:>8}")
    
    # Identify underrepresented species
    print(f"\n{'='*60}")
    print("Underrepresented Species Analysis")
    print("=" * 60)
    
    # Species with < 100 samples
    underrepresented = [(s, c) for s, c in sorted_species if c < 100]
    if underrepresented:
        print(f"\nSpecies with < 100 samples ({len(underrepresented)} species):")
        for species, count in underrepresented:
            print(f"  {species}: {count} samples")
    
    # Species with < 50 samples
    very_underrepresented = [(s, c) for s, c in sorted_species if c < 50]
    if very_underrepresented:
        print(f"\nSpecies with < 50 samples ({len(very_underrepresented)} species):")
        for species, count in very_underrepresented:
            print(f"  {species}: {count} samples")
    
    # Calculate imbalance ratio
    if species_counts:
        min_count = min(species_counts.values())
        max_count = max(species_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        print(f"\nImbalance Ratio: {imbalance_ratio:.1f}x (max/min)")
        print(f"  Most represented: {max(species_counts.items(), key=lambda x: x[1])[0]} ({max_count} samples)")
        print(f"  Least represented: {min(species_counts.items(), key=lambda x: x[1])[0]} ({min_count} samples)")
    
    # Otolemur specific analysis
    print(f"\n{'='*60}")
    print("Otolemur Analysis")
    print("=" * 60)
    
    otolemur_species = [s for s in species_counts.keys() if 'Otolemur' in s]
    if otolemur_species:
        otolemur_total = sum(species_counts[s] for s in otolemur_species)
        otolemur_pct = (otolemur_total / total_galago * 100) if total_galago > 0 else 0
        print(f"\nOtolemur species in training:")
        for species in otolemur_species:
            count = species_counts[species]
            pct = (count / total_galago * 100) if total_galago > 0 else 0
            print(f"  {species}: {count} samples ({pct:.1f}% of galago data)")
        print(f"  Total Otolemur: {otolemur_total} samples ({otolemur_pct:.1f}% of galago data)")
        
        # Compare to most represented
        if max_count > 0:
            ratio = otolemur_total / max_count
            print(f"  Ratio vs most represented species: {ratio:.2f}x")
    
    # Save summary
    summary = {
        "total_galago_samples": total_galago,
        "total_not_galago_samples": not_galago_count,
        "species_counts": species_counts,
        "imbalance_ratio": imbalance_ratio if species_counts else None,
        "underrepresented_species": {s: c for s, c in underrepresented} if underrepresented else {},
        "very_underrepresented_species": {s: c for s, c in very_underrepresented} if very_underrepresented else {},
    }
    
    output_path = PROJECT_ROOT / "outputs" / "evaluation" / "training_data_distribution.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n\nSummary saved to: {output_path}")

if __name__ == "__main__":
    main()
