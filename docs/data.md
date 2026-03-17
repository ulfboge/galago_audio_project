# Data and preprocessing

## Audio recordings

The dataset consists of galago vocalizations collected from multiple sources and individuals. Recordings include a wide range of call types (e.g. buzzes, screeches, whistles, yaps, rolling calls).

Audio files are stored as WAV files in species-specific folders.

---

## Species pooling

Recordings from multiple individuals of *Galagoides* sp. nov. (including “sp. nov. 1” and “sp. nov. 3”) were **pooled into a single class** to capture intra-specific acoustic variation.

This approach avoids overfitting to individuals and reflects the intended species-level use of the classifier.

---

## Spectrogram generation

Each audio file is converted to a mel-spectrogram with the following properties:

- Fixed image size: **128 × 128**
- Frequency represented on the mel scale
- Saved as RGB images for CNN compatibility

Spectrograms are stored in `melspectrograms/`, with one folder per species.

---

## Inclusion criteria

- Clear vocalizations with sufficient signal-to-noise ratio
- Confirmed or strongly supported species attribution

## Exclusion criteria

- Extremely noisy recordings
- Uncertain species identity
- Fossil taxa (no extant vocalizations)
