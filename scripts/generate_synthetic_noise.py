"""
Generate synthetic noise samples for negative class training.

This creates basic noise samples (wind, rain, equipment) that can be used
for initial detector training. Real field recordings are better, but these
provide a starting point.
"""
from pathlib import Path
import sys
import numpy as np

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    print("ERROR: soundfile not installed.")
    print("Install with: pip install soundfile")
    sys.exit(1)

# Fix Unicode encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NEGATIVE_DATA_DIR = PROJECT_ROOT / "data" / "negative_audio_raw"

# Audio parameters
SR = 22050
DURATION = 3.0  # 3 seconds per sample

# Number of samples to generate per category
# Expanded for better detector training
SAMPLES_PER_CATEGORY = {
    "wind": 50,        # Increased from 30
    "rain": 50,        # Increased from 30
    "equipment": 30,    # Increased from 20
    "background": 30,  # Increased from 20
    "insects": 50,     # NEW: Simulated insect sounds
    "frogs": 50,       # NEW: Simulated frog sounds
}

def generate_wind_noise(duration: float, sr: int) -> np.ndarray:
    """Generate wind-like noise."""
    t = np.linspace(0, duration, int(sr * duration))
    
    # Base noise
    noise = np.random.randn(len(t)) * 0.15
    
    # Low-pass filter effect (wind is low frequency)
    # Simple moving average
    window_size = 200
    noise_filtered = np.convolve(noise, np.ones(window_size)/window_size, mode='same')
    
    # Varying amplitude envelope (wind gusts)
    envelope = 0.4 + 0.3 * np.sin(2 * np.pi * 0.3 * t)
    envelope += 0.2 * np.sin(2 * np.pi * 0.7 * t + np.pi/4)
    
    audio = noise_filtered * envelope
    return audio

def generate_rain_noise(duration: float, sr: int) -> np.ndarray:
    """Generate rain-like noise."""
    t = np.linspace(0, duration, int(sr * duration))
    
    # Base high-frequency noise
    noise = np.random.randn(len(t)) * 0.08
    
    # Add high frequency components (raindrops)
    for freq in [2000, 4000, 6000, 8000]:
        phase = np.random.rand() * 2 * np.pi
        amplitude = 0.03 + np.random.rand() * 0.02
        noise += amplitude * np.sin(2 * np.pi * freq * t + phase)
    
    # Add occasional "drops" (impulses)
    for _ in range(np.random.randint(10, 20)):
        drop_pos = int(np.random.rand() * len(t))
        drop_duration = int(0.005 * sr)  # 5ms
        if drop_pos + drop_duration < len(t):
            noise[drop_pos:drop_pos+drop_duration] += np.random.randn(drop_duration) * 0.1
    
    return noise

def generate_equipment_noise(duration: float, sr: int) -> np.ndarray:
    """Generate equipment noise (clicks, pops, hum)."""
    t = np.linspace(0, duration, int(sr * duration))
    
    # Base noise
    audio = np.random.randn(len(t)) * 0.03
    
    # Add 60Hz hum (power line interference)
    audio += 0.05 * np.sin(2 * np.pi * 60 * t)
    audio += 0.02 * np.sin(2 * np.pi * 120 * t)  # Harmonic
    
    # Add occasional clicks/pops
    num_clicks = np.random.randint(3, 8)
    for _ in range(num_clicks):
        click_pos = int(np.random.rand() * len(t))
        click_duration = int(0.01 * sr)  # 10ms
        if click_pos + click_duration < len(t):
            click_shape = np.exp(-np.linspace(0, 5, click_duration))
            audio[click_pos:click_pos+click_duration] += np.random.randn(click_duration) * 0.15 * click_shape
    
    return audio

def generate_background_noise(duration: float, sr: int) -> np.ndarray:
    """Generate generic background noise."""
    t = np.linspace(0, duration, int(sr * duration))
    
    # White noise
    audio = np.random.randn(len(t)) * 0.1
    
    # Add some low frequency rumble
    audio += 0.05 * np.sin(2 * np.pi * 50 * t)
    audio += 0.03 * np.sin(2 * np.pi * 100 * t)
    
    return audio

def generate_insect_noise(duration: float, sr: int) -> np.ndarray:
    """Generate insect-like noise (crickets, cicadas)."""
    t = np.linspace(0, duration, int(sr * duration))
    
    # Base high-frequency noise (insect stridulation)
    audio = np.random.randn(len(t)) * 0.05
    
    # Add chirping patterns (crickets: 2-8 kHz)
    for freq in [3000, 4000, 5000, 6000, 7000]:
        # Chirp pattern: on/off rhythm
        chirp_rate = 2.0 + np.random.rand() * 3.0  # 2-5 Hz
        chirp_pattern = (np.sin(2 * np.pi * chirp_rate * t) > 0).astype(float)
        phase = np.random.rand() * 2 * np.pi
        amplitude = 0.03 + np.random.rand() * 0.02
        audio += amplitude * chirp_pattern * np.sin(2 * np.pi * freq * t + phase)
    
    # Add occasional trills (cicadas: continuous)
    if np.random.rand() > 0.5:
        trill_freq = 4000 + np.random.rand() * 2000  # 4-6 kHz
        trill_amplitude = 0.02 + np.random.rand() * 0.02
        trill_modulation = 10 + np.random.rand() * 20  # 10-30 Hz modulation
        audio += trill_amplitude * np.sin(2 * np.pi * trill_freq * t) * \
                 (1 + 0.3 * np.sin(2 * np.pi * trill_modulation * t))
    
    return audio

def generate_frog_noise(duration: float, sr: int) -> np.ndarray:
    """Generate frog-like noise (croaks, calls)."""
    t = np.linspace(0, duration, int(sr * duration))
    
    # Base audio
    audio = np.zeros(len(t))
    
    # Add croaking patterns (frogs: 500-3000 Hz)
    num_croaks = np.random.randint(3, 8)
    for _ in range(num_croaks):
        # Random position
        croak_pos = int(np.random.rand() * (len(t) - int(0.5 * sr)))
        croak_duration = int((0.1 + np.random.rand() * 0.3) * sr)  # 0.1-0.4s
        
        if croak_pos + croak_duration < len(t):
            # Croak frequency sweep (typical of frog calls)
            croak_freq_start = 500 + np.random.rand() * 1000  # 500-1500 Hz
            croak_freq_end = croak_freq_start + np.random.rand() * 1000  # Up to 2500 Hz
            
            croak_t = np.linspace(0, croak_duration / sr, croak_duration)
            freq_sweep = np.linspace(croak_freq_start, croak_freq_end, croak_duration)
            
            # Generate croak with frequency sweep
            croak_audio = np.sin(2 * np.pi * freq_sweep * croak_t)
            
            # Envelope (attack-decay)
            envelope = np.exp(-croak_t * 5)  # Exponential decay
            envelope *= (1 - np.exp(-croak_t * 20))  # Attack
            
            # Add harmonics
            croak_audio += 0.3 * np.sin(2 * np.pi * freq_sweep * 2 * croak_t)
            croak_audio += 0.2 * np.sin(2 * np.pi * freq_sweep * 3 * croak_t)
            
            audio[croak_pos:croak_pos+croak_duration] += croak_audio * envelope * 0.3
    
    # Add some background noise
    audio += np.random.randn(len(t)) * 0.02
    
    return audio

def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Normalize audio to prevent clipping."""
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.8  # Leave some headroom
    return audio

def main():
    print("Generating Synthetic Noise Samples")
    print("="*60)
    print(f"\nOutput directory: {NEGATIVE_DATA_DIR}")
    print(f"Sample rate: {SR} Hz")
    print(f"Duration per sample: {DURATION} seconds")
    
    generators = {
        "wind": generate_wind_noise,
        "rain": generate_rain_noise,
        "equipment": generate_equipment_noise,
        "background": generate_background_noise,
        "insects": generate_insect_noise,
        "frogs": generate_frog_noise,
    }
    
    total_generated = 0
    
    for category, num_samples in SAMPLES_PER_CATEGORY.items():
        category_dir = NEGATIVE_DATA_DIR / category
        category_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nGenerating {num_samples} {category} samples...")
        generator = generators[category]
        
        for i in range(num_samples):
            # Generate audio
            audio = generator(DURATION, SR)
            audio = normalize_audio(audio)
            
            # Save
            filename = f"{category}_synthetic_{i+1:03d}.wav"
            dest_path = category_dir / filename
            sf.write(str(dest_path), audio, SR)
            
            if (i + 1) % 10 == 0:
                print(f"  Generated {i+1}/{num_samples}...", end="\r")
        
        print(f"  Generated {num_samples}/{num_samples} files")
        total_generated += num_samples
    
    print(f"\n{'='*60}")
    print(f"Complete!")
    print(f"  Total files generated: {total_generated}")
    print(f"  Location: {NEGATIVE_DATA_DIR}")
    print(f"\nNext steps:")
    print(f"  1. Add real field recordings for better performance")
    print(f"  2. Process to mel-spectrograms:")
    print(f"     python scripts/prepare_negative_class.py {NEGATIVE_DATA_DIR}/wind noise")
    print(f"     python scripts/prepare_negative_class.py {NEGATIVE_DATA_DIR}/rain noise")
    print(f"     python scripts/prepare_negative_class.py {NEGATIVE_DATA_DIR}/equipment noise")
    print(f"     python scripts/prepare_negative_class.py {NEGATIVE_DATA_DIR}/background noise")
    print(f"  3. Train detector:")
    print(f"     python scripts/train_galago_detector.py")

if __name__ == "__main__":
    main()

