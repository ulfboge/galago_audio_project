"""
Comprehensive negative class data collection script.

This script provides multiple methods to collect negative class audio:
1. Freesound API (if API key available)
2. Manual download links
3. Synthetic noise generation
4. Process existing files
"""
from pathlib import Path
import sys
import json
import subprocess
import numpy as np

# Fix Unicode encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False
    print("Warning: soundfile not installed. Install with: pip install soundfile")

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("Warning: requests not installed. Install with: pip install requests")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NEGATIVE_DATA_DIR = PROJECT_ROOT / "data" / "negative_audio_raw"
NEGATIVE_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Target counts per category
TARGET_COUNTS = {
    "insects": 100,
    "frogs": 100,
    "birds": 100,
    "noise": 50,
    "human_speech": 50,
}

def generate_synthetic_noise(category: str, num_samples: int, sr: int = 22050):
    """Generate synthetic noise samples for testing."""
    if not HAS_SOUNDFILE:
        print("  [SKIP] soundfile not installed. Install with: pip install soundfile")
        return 0
    
    category_dir = NEGATIVE_DATA_DIR / category
    category_dir.mkdir(parents=True, exist_ok=True)
    
    duration = 3.0  # 3 seconds
    
    print(f"  Generating {num_samples} synthetic {category} samples...")
    
    for i in range(num_samples):
        t = np.linspace(0, duration, int(sr * duration))
        
        if category == "wind":
            # Wind: low frequency noise with envelope
            noise = np.random.randn(len(t)) * 0.15
            # Low-pass effect
            noise = np.convolve(noise, np.ones(200)/200, mode='same')
            # Varying amplitude
            envelope = 0.4 + 0.3 * np.sin(2 * np.pi * 0.3 * t)
            audio = noise * envelope
            
        elif category == "rain":
            # Rain: high frequency bursts
            noise = np.random.randn(len(t)) * 0.08
            # Add high frequency components
            for freq in [2000, 4000, 6000, 8000]:
                phase = np.random.rand() * 2 * np.pi
                noise += 0.03 * np.sin(2 * np.pi * freq * t + phase)
            audio = noise
            
        elif category == "equipment":
            # Equipment: clicks, pops, hum
            audio = np.random.randn(len(t)) * 0.03
            # Add 60Hz hum
            audio += 0.05 * np.sin(2 * np.pi * 60 * t)
            # Add occasional clicks
            for _ in range(np.random.randint(3, 8)):
                click_pos = int(np.random.rand() * len(t))
                click_duration = int(0.01 * sr)  # 10ms
                if click_pos + click_duration < len(t):
                    audio[click_pos:click_pos+click_duration] += np.random.randn(click_duration) * 0.15
        
        elif category == "background":
            # Generic background noise
            audio = np.random.randn(len(t)) * 0.1
            # Add some low frequency rumble
            audio += 0.05 * np.sin(2 * np.pi * 50 * t)
        
        else:
            # Generic white noise
            audio = np.random.randn(len(t)) * 0.1
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.8
        
        # Save
        filename = f"{category}_synthetic_{i+1:03d}.wav"
        dest_path = category_dir / filename
        sf.write(str(dest_path), audio, sr)
    
    print(f"    Generated {num_samples} files in {category_dir}")
    return num_samples

def print_freesound_instructions():
    """Print instructions for Freesound downloads."""
    print("\n" + "="*60)
    print("FREESOUND.ORG DOWNLOAD INSTRUCTIONS")
    print("="*60)
    print("\n1. Go to: https://freesound.org/")
    print("2. Create a free account")
    print("3. Search for audio (use these terms):")
    print("\n   INSECTS:")
    print("     - cricket")
    print("     - cicada")
    print("     - katydid")
    print("     - grasshopper")
    print("\n   FROGS:")
    print("     - frog")
    print("     - toad")
    print("     - amphibian")
    print("\n   BIRDS:")
    print("     - owl")
    print("     - nightjar")
    print("     - nocturnal bird")
    print("\n   NOISE:")
    print("     - wind")
    print("     - rain")
    print("     - background noise")
    print("\n4. Filter by license: CC0 or CC-BY (free to use)")
    print("5. Download files (WAV or MP3)")
    print("6. Save to: data/negative_audio_raw/<category>/")
    print("\nTarget: 50-100 files per category")

def print_organization_guide():
    """Print guide for organizing files."""
    print("\n" + "="*60)
    print("FILE ORGANIZATION")
    print("="*60)
    print(f"\nOrganize your downloaded files like this:")
    print(f"\n{NEGATIVE_DATA_DIR}/")
    print("  insects/")
    print("    cricket_001.wav")
    print("    cicada_002.wav")
    print("    ...")
    print("  frogs/")
    print("    frog_001.wav")
    print("    ...")
    print("  birds/")
    print("    owl_001.wav")
    print("    ...")
    print("  noise/")
    print("    wind_001.wav")
    print("    rain_002.wav")
    print("    ...")
    print("  human_speech/")
    print("    speech_001.wav")
    print("    ...")

def check_existing_files():
    """Check what negative data already exists."""
    print("\nChecking existing negative class data...")
    
    total = 0
    for category in TARGET_COUNTS.keys():
        category_dir = NEGATIVE_DATA_DIR / category
        if category_dir.exists():
            files = list(category_dir.glob("*.wav")) + list(category_dir.glob("*.mp3"))
            count = len(files)
            total += count
            target = TARGET_COUNTS[category]
            status = "OK" if count >= target else "LOW"
            print(f"  {status:4s} {category:15s}: {count:3d}/{target:3d} files")
        else:
            print(f"  MISS {category:15s}: 0/{TARGET_COUNTS[category]:3d} files (folder missing)")
    
    print(f"\n  Total: {total} files")
    
    if total >= 400:  # Minimum viable
        print("  Status: OK - Ready to process!")
    elif total >= 200:
        print("  Status: LOW - Minimum viable (200+), but more is better")
    else:
        print("  Status: NEED MORE - Target: 400+ files")

def main():
    print("Comprehensive Negative Class Data Collection")
    print("="*60)
    print(f"\nOutput directory: {NEGATIVE_DATA_DIR}")
    
    # Check existing files
    check_existing_files()
    
    print("\n" + "="*60)
    print("COLLECTION METHODS")
    print("="*60)
    print("\n1. Generate synthetic noise (for testing)")
    print("2. Freesound.org download guide")
    print("3. File organization guide")
    print("4. Process existing files to mel-spectrograms")
    print("5. Check what's needed")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == "1":
        print("\nGenerating synthetic noise samples...")
        print("(These are for testing - real field recordings are better)")
        
        for category in ["wind", "rain", "equipment", "background"]:
            generate_synthetic_noise(category, num_samples=20)
        
        print("\n✓ Synthetic noise generated!")
        print(f"  Location: {NEGATIVE_DATA_DIR}")
        print("\nNext: Add real field recordings for better performance")
    
    elif choice == "2":
        print_freesound_instructions()
    
    elif choice == "3":
        print_organization_guide()
    
    elif choice == "4":
        print("\nTo process existing files to mel-spectrograms:")
        print("\nFor each category:")
        print("  python scripts/prepare_negative_class.py <category_dir> <category>")
        print("\nExample:")
        print("  python scripts/prepare_negative_class.py data/negative_audio_raw/insects insects")
        print("  python scripts/prepare_negative_class.py data/negative_audio_raw/frogs frogs")
        print("  python scripts/prepare_negative_class.py data/negative_audio_raw/birds birds")
        print("  python scripts/prepare_negative_class.py data/negative_audio_raw/noise noise")
    
    elif choice == "5":
        print("\n" + "="*60)
        print("WHAT'S NEEDED")
        print("="*60)
        print("\nMinimum viable dataset:")
        print("  - 50-100 files per category")
        print("  - Total: ~400-500 files")
        print("\nIdeal dataset:")
        print("  - 200+ files per category")
        print("  - Total: ~1000+ files")
        print("\nCategories needed:")
        for category, target in TARGET_COUNTS.items():
            print(f"  - {category:15s}: {target:3d} files")
        print("\nSources:")
        print("  1. Freesound.org (free, CC0/CC-BY)")
        print("  2. Your own field recordings")
        print("  3. Background noise from galago sessions")
        print("  4. Synthetic noise (for testing)")
    
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()

