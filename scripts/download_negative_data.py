"""
Download negative class audio data from free sources.

This script helps collect non-galago audio for training the detector.
Supports multiple sources: Freesound API, manual download links, and synthetic noise.
"""
from pathlib import Path
import sys
import json
import requests
import time
from urllib.parse import urlparse
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NEGATIVE_DATA_DIR = PROJECT_ROOT / "data" / "negative_audio_raw"
NEGATIVE_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Categories and search terms
CATEGORIES = {
    "insects": ["cricket", "cicada", "katydid", "grasshopper", "insect"],
    "frogs": ["frog", "toad", "amphibian"],
    "birds": ["owl", "nightjar", "night bird", "nocturnal bird"],
    "noise": ["wind", "rain", "equipment noise", "background noise"],
    "human_speech": ["human speech", "talking", "voice"],
}

# Freesound API (optional - requires API key)
FREESOUND_API_KEY = None  # Set this if you have an API key
FREESOUND_API_URL = "https://freesound.org/apiv2/search/text/"

def download_file(url: str, dest_path: Path, max_retries: int = 3) -> bool:
    """Download a file from URL."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30, stream=True)
            response.raise_for_status()
            
            with open(dest_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            print(f"    Error downloading {url}: {e}")
            return False
    return False

def search_freesound(query: str, category: str, max_results: int = 20) -> list:
    """Search Freesound API for audio files."""
    if not FREESOUND_API_KEY:
        print(f"  [SKIP] Freesound API key not set. Skipping API search.")
        print(f"  You can get a free API key at: https://freesound.org/apiv2/apply/")
        return []
    
    params = {
        "query": query,
        "filter": "duration:[0.5 TO 10]",  # 0.5 to 10 seconds
        "fields": "id,name,download,previews",
        "page_size": max_results,
    }
    
    headers = {"Authorization": f"Token {FREESOUND_API_KEY}"}
    
    try:
        response = requests.get(FREESOUND_API_URL, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get("results", [])
    except Exception as e:
        print(f"  [ERROR] Freesound API error: {e}")
        return []

def download_from_freesound(category: str, max_per_category: int = 50):
    """Download audio from Freesound for a category."""
    if not FREESOUND_API_KEY:
        print(f"\n[SKIP] Freesound API not configured.")
        print(f"To use Freesound API:")
        print(f"  1. Get free API key: https://freesound.org/apiv2/apply/")
        print(f"  2. Edit this script and set FREESOUND_API_KEY")
        return 0
    
    category_dir = NEGATIVE_DATA_DIR / category
    category_dir.mkdir(parents=True, exist_ok=True)
    
    queries = CATEGORIES.get(category, [category])
    downloaded = 0
    
    print(f"\nDownloading {category} from Freesound...")
    
    for query in queries:
        if downloaded >= max_per_category:
            break
        
        print(f"  Searching: '{query}'...")
        results = search_freesound(query, category, max_results=max_per_category)
        
        for result in results:
            if downloaded >= max_per_category:
                break
            
            # Try to get download URL (may require authentication)
            download_url = result.get("download")
            if not download_url:
                # Fall back to preview
                previews = result.get("previews", {})
                download_url = previews.get("preview-hq-mp3") or previews.get("preview-lq-mp3")
            
            if not download_url:
                continue
            
            # Generate filename
            file_id = result.get("id")
            name = result.get("name", "audio")
            # Clean filename
            name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).strip()
            ext = ".mp3" if "mp3" in download_url else ".wav"
            filename = f"{category}_{file_id}_{name}{ext}"
            dest_path = category_dir / filename
            
            if dest_path.exists():
                print(f"    [SKIP] {filename} (already exists)")
                continue
            
            print(f"    Downloading: {name[:50]}...", end=" ", flush=True)
            
            if download_file(download_url, dest_path):
                downloaded += 1
                print(f"OK ({downloaded}/{max_per_category})")
                time.sleep(1)  # Be nice to the API
            else:
                print("FAILED")
    
    return downloaded

def generate_synthetic_noise(category: str, num_samples: int = 20):
    """Generate synthetic noise samples."""
    import numpy as np
    import soundfile as sf
    
    category_dir = NEGATIVE_DATA_DIR / category
    category_dir.mkdir(parents=True, exist_ok=True)
    
    sr = 22050
    duration = 3.0  # 3 seconds
    
    print(f"\nGenerating synthetic {category} noise...")
    
    for i in range(num_samples):
        if category == "wind":
            # Wind noise: low frequency noise with varying amplitude
            t = np.linspace(0, duration, int(sr * duration))
            noise = np.random.randn(len(t)) * 0.1
            # Low-pass filter effect (simplified)
            noise = np.convolve(noise, np.ones(100)/100, mode='same')
            # Varying amplitude
            envelope = 0.3 + 0.2 * np.sin(2 * np.pi * 0.5 * t)
            audio = noise * envelope
            
        elif category == "rain":
            # Rain noise: high frequency noise bursts
            t = np.linspace(0, duration, int(sr * duration))
            noise = np.random.randn(len(t)) * 0.05
            # High frequency emphasis
            for freq in [2000, 4000, 6000]:
                noise += 0.02 * np.sin(2 * np.pi * freq * t + np.random.rand() * 2 * np.pi)
            audio = noise
            
        elif category == "equipment":
            # Equipment noise: clicks, pops, hum
            t = np.linspace(0, duration, int(sr * duration))
            audio = np.random.randn(len(t)) * 0.02
            # Add occasional clicks
            for _ in range(5):
                click_pos = int(np.random.rand() * len(t))
                click_duration = int(0.01 * sr)  # 10ms
                audio[click_pos:click_pos+click_duration] += np.random.randn(click_duration) * 0.1
            
        else:
            # Generic noise
            t = np.linspace(0, duration, int(sr * duration))
            audio = np.random.randn(len(t)) * 0.1
        
        # Normalize
        audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.8
        
        # Save
        filename = f"{category}_synthetic_{i+1:03d}.wav"
        dest_path = category_dir / filename
        sf.write(str(dest_path), audio, sr)
        print(f"  Generated: {filename}")
    
    return num_samples

def print_manual_download_guide():
    """Print guide for manual downloads."""
    print("\n" + "="*60)
    print("MANUAL DOWNLOAD GUIDE")
    print("="*60)
    print("\nIf automated download doesn't work, you can download manually:")
    print("\n1. Freesound.org (Free, CC0/CC-BY licenses):")
    print("   https://freesound.org/")
    print("   - Search for: cricket, cicada, frog, owl, wind, rain")
    print("   - Filter by: CC0 or CC-BY license")
    print("   - Download as WAV or MP3")
    print("   - Save to: data/negative_audio_raw/<category>/")
    
    print("\n2. Xeno-canto (Bird sounds):")
    print("   https://xeno-canto.org/")
    print("   - Search for night birds in Africa")
    print("   - Download recordings")
    
    print("\n3. Your own recordings:")
    print("   - Background noise from field recordings")
    print("   - Equipment noise")
    print("   - Wind/rain recordings")
    
    print("\n4. Organize files:")
    print("   data/negative_audio_raw/")
    print("     insects/")
    print("     frogs/")
    print("     birds/")
    print("     noise/")
    print("     human_speech/")
    
    print("\n5. Process with:")
    print("   python scripts/prepare_negative_class.py <category_dir> <category>")

def main():
    print("Negative Class Data Collection")
    print("="*60)
    print(f"\nOutput directory: {NEGATIVE_DATA_DIR}")
    
    # Check for API key
    if FREESOUND_API_KEY:
        print(f"\nFreesound API: Configured")
    else:
        print(f"\nFreesound API: Not configured (optional)")
    
    print("\nOptions:")
    print("  1. Download from Freesound (requires API key)")
    print("  2. Generate synthetic noise samples")
    print("  3. Show manual download guide")
    print("  4. Process existing files")
    
    choice = input("\nEnter choice (1-4, or 'all' for 1+2): ").strip().lower()
    
    total_downloaded = 0
    
    if choice in ['1', 'all']:
        # Download from Freesound
        for category in CATEGORIES.keys():
            downloaded = download_from_freesound(category, max_per_category=50)
            total_downloaded += downloaded
            if downloaded > 0:
                time.sleep(2)  # Be nice to API
    
    if choice in ['2', 'all']:
        # Generate synthetic noise
        for category in ['wind', 'rain', 'equipment']:
            generated = generate_synthetic_noise(category, num_samples=20)
            total_downloaded += generated
    
    if choice == '3':
        print_manual_download_guide()
        return
    
    if choice == '4':
        print("\nTo process existing files, use:")
        print("  python scripts/prepare_negative_class.py <source_dir> <category>")
        return
    
    print(f"\n{'='*60}")
    print(f"Download complete!")
    print(f"  Total files: {total_downloaded}")
    print(f"  Location: {NEGATIVE_DATA_DIR}")
    
    if total_downloaded > 0:
        print(f"\nNext step: Process audio files to mel-spectrograms")
        print(f"  python scripts/prepare_negative_class.py <category_dir> <category>")
    else:
        print_manual_download_guide()

if __name__ == "__main__":
    main()

