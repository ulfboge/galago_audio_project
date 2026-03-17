"""
Batch process all negative audio samples to mel-spectrograms.

This processes all categories of negative samples and adds them to the
not_galago class for detector training.
"""
from pathlib import Path
import sys
import subprocess

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NEGATIVE_DATA_DIR = PROJECT_ROOT / "data" / "negative_audio_raw"
PREPARE_SCRIPT = PROJECT_ROOT / "scripts" / "prepare_negative_class.py"

def main():
    print("Processing All Negative Samples to Mel-Spectrograms")
    print("="*60)
    
    if not NEGATIVE_DATA_DIR.exists():
        print(f"ERROR: Negative data directory not found: {NEGATIVE_DATA_DIR}")
        print(f"\nPlease generate synthetic samples first:")
        print(f"  python scripts/generate_synthetic_noise.py")
        return
    
    # Find all category directories
    categories = [d.name for d in NEGATIVE_DATA_DIR.iterdir() 
                  if d.is_dir() and not d.name.startswith('.')]
    
    if not categories:
        print(f"No category directories found in {NEGATIVE_DATA_DIR}")
        return
    
    print(f"\nFound {len(categories)} categories: {', '.join(categories)}")
    print(f"\nProcessing each category...")
    print("-"*60)
    
    total_processed = 0
    
    for category in sorted(categories):
        category_dir = NEGATIVE_DATA_DIR / category
        
        # Count audio files
        audio_files = list(category_dir.glob("*.wav")) + \
                     list(category_dir.glob("*.mp3")) + \
                     list(category_dir.glob("*.flac"))
        
        if not audio_files:
            print(f"\n[{category}] No audio files found, skipping...")
            continue
        
        print(f"\n[{category}] Processing {len(audio_files)} files...")
        
        # Run prepare_negative_class.py for this category
        # Note: The script expects a category name, we'll use "noise" as the generic category
        # or the actual category name
        cmd = [
            sys.executable,
            str(PREPARE_SCRIPT),
            str(category_dir),
            category  # Use category name as the label
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8' if sys.platform == 'win32' else None
            )
            
            if result.returncode == 0:
                # Count processed files from output
                output_lines = result.stdout.split('\n')
                processed_count = len(audio_files)  # Assume all processed if no error
                total_processed += processed_count
                print(f"  ✓ Processed {processed_count} files")
            else:
                print(f"  ✗ Error processing {category}:")
                print(f"    {result.stderr}")
        except Exception as e:
            print(f"  ✗ Error running script: {e}")
    
    print(f"\n{'='*60}")
    print(f"Complete!")
    print(f"  Total categories processed: {len(categories)}")
    print(f"  Total files processed: {total_processed}")
    
    # Count final mel-spectrograms
    mels_dir = PROJECT_ROOT / "data" / "melspectrograms" / "not_galago"
    if mels_dir.exists():
        final_count = len(list(mels_dir.glob("*.png")))
        print(f"  Total mel-spectrograms in not_galago: {final_count}")
    
    print(f"\nNext step: Train detector with expanded dataset")
    print(f"  python scripts/train_galago_detector.py")

if __name__ == "__main__":
    main()

