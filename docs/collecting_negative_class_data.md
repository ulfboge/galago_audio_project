# Collecting Negative Class Data

## Overview

To train a robust galago detector, we need negative class data (non-galago audio). This guide helps you collect and prepare this data.

---

## Required Negative Classes

### Minimum Viable Dataset
- **Insects**: 100-200 clips (crickets, cicadas, katydids)
- **Frogs**: 100-200 clips (especially night-active species)
- **Birds**: 100-200 clips (night birds: owls, nightjars)
- **Noise**: 50-100 clips (wind, rain, equipment noise)
- **Human speech**: 50-100 clips
- **Total**: ~500-800 negative clips

### Ideal Dataset
- **Insects**: 500+ clips
- **Frogs**: 500+ clips
- **Birds**: 500+ clips
- **Noise**: 200+ clips
- **Human speech**: 200+ clips
- **Other primates**: 200+ clips (if available)
- **Total**: 2000+ negative clips

---

## Data Sources

### Free Audio Databases

1. **Freesound.org**
   - Search: "cricket", "cicada", "frog", "owl", "nightjar"
   - Filter: CC0 or CC-BY licenses
   - Download: WAV or MP3 format

2. **Xeno-canto**
   - Bird sounds database
   - https://xeno-canto.org/
   - Search for night birds in Africa

3. **Macaulay Library**
   - Cornell Lab of Ornithology
   - https://www.macaulaylibrary.org/
   - Requires account (free)

4. **AudioSet**
   - Google's audio dataset
   - https://research.google.com/audioset/
   - Large-scale but may need filtering

5. **Your own recordings**
   - Field recordings that you know are NOT galagos
   - Background noise from galago recording sessions
   - Equipment noise, wind, rain

### Creating Noise Samples

1. **Wind noise**: Record wind in trees/grass
2. **Rain**: Record rain on leaves/ground
3. **Equipment**: Record microphone handling, cable noise
4. **Background**: Extract quiet sections from field recordings

---

## Organization

### Folder Structure

Organize your negative class data by category:

```
E:/Audio/NegativeClasses/
  insects/
    cricket_001.wav
    cicada_002.wav
    ...
  frogs/
    frog_001.wav
    ...
  birds/
    owl_001.wav
    nightjar_002.wav
    ...
  noise/
    wind_001.wav
    rain_002.wav
    ...
  human_speech/
    speech_001.wav
    ...
```

### File Naming

Use descriptive names:
- `insects_cricket_001.wav`
- `frogs_night_001.wav`
- `birds_owl_001.wav`
- `noise_wind_001.wav`

---

## Processing

### Using the Helper Script

Once you have audio files organized, use the helper script to convert them to mel-spectrograms:

```bash
# Process insects
python scripts/prepare_negative_class.py E:/Audio/NegativeClasses/insects insects

# Process frogs
python scripts/prepare_negative_class.py E:/Audio/NegativeClasses/frogs frogs

# Process birds
python scripts/prepare_negative_class.py E:/Audio/NegativeClasses/birds birds

# Process noise
python scripts/prepare_negative_class.py E:/Audio/NegativeClasses/noise noise

# Process human speech
python scripts/prepare_negative_class.py E:/Audio/NegativeClasses/human_speech human_speech
```

The script will:
1. Load each audio file
2. Convert to mel-spectrogram (128×128)
3. Save as PNG in `data/melspectrograms/not_galago/`

### Manual Processing

If you prefer to process manually, use the same pipeline as galago data:
1. Run `scripts/make_mels.py` but point to negative class folders
2. Manually move PNGs to `data/melspectrograms/not_galago/`

---

## Quality Guidelines

### Audio Requirements
- **Duration**: 2-5 seconds per clip (will be windowed)
- **Sample rate**: Any (will be resampled to 22050 Hz)
- **Format**: WAV, MP3, FLAC, M4A, OGG
- **Quality**: Clear audio (avoid heavy distortion)

### What to Include
- ✅ Clear, distinct sounds
- ✅ Various recording conditions
- ✅ Different distances/angles
- ✅ Different times of night
- ✅ Various habitats

### What to Avoid
- ❌ Very short clips (< 0.5 seconds)
- ❌ Heavily distorted audio
- ❌ Mixed sounds (galago + other)
- ❌ Unclear/unidentifiable sounds

---

## Quick Start (Minimal Dataset)

If you need to get started quickly:

1. **Download 50-100 clips from Freesound.org**
   - Search: "cricket", "frog", "owl"
   - Filter: CC0 license
   - Download as WAV

2. **Organize into folders**
   ```
   E:/Audio/QuickNegative/
     insects/
     frogs/
     birds/
   ```

3. **Process with script**
   ```bash
   python scripts/prepare_negative_class.py E:/Audio/QuickNegative/insects insects
   python scripts/prepare_negative_class.py E:/Audio/QuickNegative/frogs frogs
   python scripts/prepare_negative_class.py E:/Audio/QuickNegative/birds birds
   ```

4. **Train detector**
   ```bash
   python scripts/train_galago_detector.py
   ```

---

## Validation

After processing, check your negative class data:

```bash
# Count mel-spectrograms
ls data/melspectrograms/not_galago/*.png | wc -l

# Should have at least 50-100 PNGs for minimal dataset
# Ideally 500+ for good performance
```

---

## Tips

1. **Start small**: 50-100 negative clips is enough to test the pipeline
2. **Iterate**: Add more negative data as you find false positives
3. **Balance**: Try to match the number of negative samples to positive samples
4. **Diversity**: Include various types of non-galago sounds
5. **Hard negatives**: After training, identify sounds the model confuses and add more of those

---

## Next Steps

Once you have negative class data:

1. ✅ Verify mel-spectrograms are in `data/melspectrograms/not_galago/`
2. ✅ Run `python scripts/train_galago_detector.py`
3. ✅ Evaluate detector on test set
4. ✅ Integrate detector into prediction pipeline

See `docs/merlin_like_roadmap.md` for the full implementation plan.

