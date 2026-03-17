from pathlib import Path
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# CONFIG â€" adjust these to your actual locations / preferences
# ---------------------------------------------------------------------

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Folder that contains all your call recordings
# Updated to use Oxford Brookes University recordings
BASE_DIR = Path(r"E:\Galagidae")

# Where to save mel-spectrogram images
OUT_DIR = PROJECT_ROOT / "data" / "melspectrograms"

# Audio & spectrogram parameters
SR = 22050            # sample rate
N_MELS = 128          # mel bands (height)
N_FFT = 2048
HOP_LENGTH = 512
FMIN = 200
FMAX = 10000

# Fixed time dimension (width) in spectrogram frames
TARGET_FRAMES = 128   # â‰ˆ few seconds at this SR & hop

# Augmentation settings
USE_AUGMENTATION = True
AUG_PER_FILE = 2      # number of augmented versions per original

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".dat"}  # extend if needed (DAT files from Oxford Brookes)


# ---------------------------------------------------------------------
# SPECIES PATTERNS  ->  CANONICAL LABELS
# Now includes G. kumbirensis & S. makandensis
# ---------------------------------------------------------------------

SPECIES_PATTERNS = {
    # ---------------- OTOLEMUR ----------------
    "O_crassicaudatus":         "Otolemur_crassicaudatus",
    "O. crassicaudatus":        "Otolemur_crassicaudatus",
    "Otolemur crassicaudatus":  "Otolemur_crassicaudatus",

    "O_garnettii":              "Otolemur_garnettii",
    "O. garnettii":             "Otolemur_garnettii",
    "Otolemur garnettii":       "Otolemur_garnettii",

    # ---------------- GALAGO ----------------
    "G_sengalensis":            "Galago_senegalensis",  # spelling as in your files
    "G. sengalensis":           "Galago_senegalensis",
    "Galago senegalensis":      "Galago_senegalensis",

    "G_moholi":                 "Galago_moholi",
    "G. moholi":                "Galago_moholi",
    "Galago moholi":            "Galago_moholi",

    "G_gallarum":               "Galago_gallarum",
    "G. gallarum":              "Galago_gallarum",
    "Galago gallarum":          "Galago_gallarum",

    "G_matschiei":              "Galago_matschiei",
    "G. matschiei":             "Galago_matschiei",
    "G_matschei":               "Galago_matschiei",
    "Galago matschiei":         "Galago_matschiei",

    "G_granti":                 "Paragalago_granti",
    "G. granti":                "Paragalago_granti",
    "Galago granti":            "Paragalago_granti",  # Updated to Paragalago per IUCN Red List
    "Galagoides granti":        "Paragalago_granti",
    "Paragalago granti":        "Paragalago_granti",  # Correct nomenclature per IUCN Red List

    # ---------------- PARAGALAGO (East African dwarf) ----------------
    "G_zanzibaricus":           "Paragalago_zanzibaricus",
    "G. zanzibaricus":          "Paragalago_zanzibaricus",
    "Paragalago zanzibaricus":  "Paragalago_zanzibaricus",

    "G_cocos":                  "Paragalago_cocos",
    "G. cocos":                 "Paragalago_cocos",
    "Paragalago cocos":         "Paragalago_cocos",

    "G_rondoensis":             "Paragalago_rondoensis",
    "G. rondoensis":            "Paragalago_rondoensis",
    "Paragalago rondoensis":    "Paragalago_rondoensis",

    "G_orinus":                 "Paragalago_orinus",
    "G. orinus":                "Paragalago_orinus",
    "Paragalago orinus":        "Paragalago_orinus",

    "G_arthuri":                "Paragalago_arthuri",
    "G. arthuri":               "Paragalago_arthuri",
    "Paragalago arthuri":       "Paragalago_arthuri",

    # ---------------- GALAGOIDES (western dwarf) ----------------
    "G_demidovii":              "Galagoides_demidovii",
    "G. demidovii":             "Galagoides_demidovii",
    "Galagoides demidovii":     "Galagoides_demidovii",

    "G_thomasi":                "Galagoides_thomasi",
    "G. thomasi":               "Galagoides_thomasi",
    "Galagoides thomasi":       "Galagoides_thomasi",

    "G_kumbakumba":             "Galagoides_kumbakumba",
    "G. kumbakumba":            "Galagoides_kumbakumba",
    "Galagoides kumbakumba":    "Galagoides_kumbakumba",

    "G_phasma":                 "Galagoides_phasma",
    "G. phasma":                "Galagoides_phasma",
    "Galagoides phasma":        "Galagoides_phasma",

    # NEW: Angolan dwarf galago (extant)
    "G_kumbirensis":            "Galagoides_kumbirensis",
    "G. kumbirensis":           "Galagoides_kumbirensis",
    "Galagoides kumbirensis":   "Galagoides_kumbirensis",
    "Angolan dwarf galago":     "Galagoides_kumbirensis",

    # ---------------- sp. nov. (extra class) ----------------
    "G_sp_nov":                 "Galagoides_sp_nov",
    "G.sp.nov":                 "Galagoides_sp_nov",
    "sp.nov":                   "Galagoides_sp_nov",
    "sp nov":                   "Galagoides_sp_nov",
    "Paragalago sp. nov":       "Galagoides_sp_nov",  # Oxford Brookes variant
    "Paragalago sp. nov. 3":    "Galagoides_sp_nov",  # Oxford Brookes variant

    # ---------------- SCIUROCHEIRUS ----------------
    "S_gabonensis":             "Sciurocheirus_gabonensis",
    "S. gabonensis":            "Sciurocheirus_gabonensis",
    "Sciurocheirus gabonensis": "Sciurocheirus_gabonensis",

    "S_alleni":                 "Sciurocheirus_alleni",
    "S. alleni":                "Sciurocheirus_alleni",
    "Sciurocheirus alleni":     "Sciurocheirus_alleni",

    "S_cameronensis":               "Sciurocheirus_cameronensis",
    "S. cameronensis":              "Sciurocheirus_cameronensis",
    "Sciurocheirus cameronensis":   "Sciurocheirus_cameronensis",

    # NEW: MakandÃ© squirrel galago (extant)
    "S_makandensis":                "Sciurocheirus_makandensis",
    "S. makandensis":               "Sciurocheirus_makandensis",
    "Sciurocheirus makandensis":    "Sciurocheirus_makandensis",
    "Makande squirrel galago":      "Sciurocheirus_makandensis",
    "MakandÃ© squirrel galago":      "Sciurocheirus_makandensis",

    # ---------------- EUOTICUS ----------------
    "E_elegantulus":            "Euoticus_elegantulus",
    "E. elegantulus":           "Euoticus_elegantulus",
    "Euoticus elegantulus":     "Euoticus_elegantulus",

    "E_pallidus":               "Euoticus_pallidus",
    "E. pallidus":              "Euoticus_pallidus",
    "Euoticus pallidus":        "Euoticus_pallidus",
}


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def find_species_label(path: Path) -> str | None:
    """Return canonical species label based on patterns in the full path."""
    s = str(path)
    for pattern, label in SPECIES_PATTERNS.items():
        if pattern in s:
            return label
    return None


def clean_output_dir(out_dir: Path):
    """Remove all existing PNG files under OUT_DIR."""
    if not out_dir.exists():
        return

    removed = 0
    for png in out_dir.rglob("*.png"):
        try:
            png.unlink()
            removed += 1
        except Exception as e:
            print(f"[WARN] Could not remove {png}: {e}")

    print(f"Cleaned output directory. Removed {removed} PNG files.")


def load_audio(audio_path: Path):
    """Load audio as mono, fixed sample rate."""
    return librosa.load(audio_path, sr=SR, mono=True)


def make_mel_spectrogram(y: np.ndarray):
    """Compute log-mel spectrogram (n_mels x T)."""
    S = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db


def pad_or_crop(S: np.ndarray, target_frames: int) -> np.ndarray:
    """
    Ensure spectrogram has shape (N_MELS, target_frames) by
    center-cropping or padding with the minimum value.
    """
    n_mels, T = S.shape
    if T == target_frames:
        return S

    if T > target_frames:
        start = (T - target_frames) // 2
        end = start + target_frames
        return S[:, start:end]

    pad_total = target_frames - T
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    pad_value = S.min()
    S_padded = np.pad(S, ((0, 0), (pad_left, pad_right)),
                      mode="constant", constant_values=pad_value)
    return S_padded


def save_spectrogram_png(S: np.ndarray, out_png: Path):
    """Save spectrogram matrix as a PNG (no axes, fixed size)."""
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(3, 3), dpi=100)
    plt.imshow(S, origin="lower", aspect="auto", cmap="magma")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0)
    plt.close()


def augment_audio(y: np.ndarray) -> list:
    """
    Simple augmentations:
    - small Gaussian noise
    - slight time-stretch
    Returns a list of augmented audio arrays.
    """
    augmented = []

    # 1) Add light noise
    noise_scale = 0.01 * (np.max(np.abs(y)) + 1e-6)
    noise = np.random.randn(len(y)) * noise_scale
    augmented.append(y + noise)

    # 2) Slight time-stretch
    rate = np.random.uniform(0.9, 1.1)
    try:
        y_stretch = librosa.effects.time_stretch(y, rate)
        augmented.append(y_stretch)
    except Exception:
        pass

    return augmented


def process_audio_file(audio_path: Path, species: str, idx_suffix: str = ""):
    """Compute mel spectrogram for one audio signal and save PNG."""
    try:
        y, _ = load_audio(audio_path)
    except Exception as e:
        print(f"[ERROR] Could not load {audio_path}: {e}")
        return

    S_db = make_mel_spectrogram(y)
    S_fixed = pad_or_crop(S_db, TARGET_FRAMES)

    out_dir = OUT_DIR / species
    base_name = audio_path.stem + idx_suffix + ".png"
    out_png = out_dir / base_name
    save_spectrogram_png(S_fixed, out_png)

    print(f"[OK] {audio_path} -> {out_png}")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    # 1) Clean old PNGs
    clean_output_dir(OUT_DIR)

    processed = 0

    for audio_path in BASE_DIR.rglob("*"):
        if audio_path.suffix.lower() not in AUDIO_EXTS:
            continue

        species = find_species_label(audio_path)
        if species is None:
            continue

        # Original
        process_audio_file(audio_path, species)
        processed += 1

        # Augmented versions
        if USE_AUGMENTATION:
            try:
                y, _ = load_audio(audio_path)
            except Exception as e:
                print(f"[ERROR] Could not load for augmentation {audio_path}: {e}")
                continue

            aug_list = augment_audio(y)[:AUG_PER_FILE]

            for i, y_aug in enumerate(aug_list, start=1):
                S_db_aug = make_mel_spectrogram(y_aug)
                S_fixed_aug = pad_or_crop(S_db_aug, TARGET_FRAMES)

                out_dir = OUT_DIR / species
                idx_suffix = f"_aug{i}"
                out_png = out_dir / (audio_path.stem + idx_suffix + ".png")
                save_spectrogram_png(S_fixed_aug, out_png)

                print(f"[AUG] {audio_path} -> {out_png}")

    print(f"Done. Processed {processed} original files.")


if __name__ == "__main__":
    main()

