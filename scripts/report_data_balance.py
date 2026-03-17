"""
Quick report of class balance and domain coverage.

Outputs:
- Per-species PNG counts in data/melspectrograms
- Per-species WAV counts in data/raw_audio (based on folder / LABEL_MAP)
- Simple imbalance ratios to guide retraining / ingestion
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys


if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MELS_DIR = PROJECT_ROOT / "data" / "melspectrograms"
RAW_DIR = PROJECT_ROOT / "data" / "raw_audio"

# Same mapping used in several scripts (ingest_raw_audio_to_training_mels, analyze_prediction_accuracy, etc.)
LABEL_MAP = {
    "G.sp.nov.1": "Galagoides_sp_nov",
    "G.sp.nov.3": "Galagoides_sp_nov",
    "G.granti": "Paragalago_granti",
    "G.orinus": "Paragalago_orinus",
    "G.rondoensis": "Paragalago_rondoensis",
    "G.zanzibaricus": "Paragalago_zanzibaricus",
    "O.crassicaudatus": "Otolemur_crassicaudatus",
    "O.garnettii": "Otolemur_garnettii",
    # legacy/aliases
    "Galago_granti": "Paragalago_granti",
}


def label_from_raw_path(p: Path) -> str:
    """Infer canonical label from raw_audio path using LABEL_MAP + folder name."""
    try:
        rel = p.relative_to(RAW_DIR)
        src_folder = rel.parts[0] if len(rel.parts) > 1 else p.parent.name
        return LABEL_MAP.get(src_folder, src_folder)
    except Exception:
        for part in p.parts:
            if part in LABEL_MAP:
                return LABEL_MAP[part]
        return p.parent.name


def report_mels() -> Counter:
    counts: Counter[str] = Counter()
    if not MELS_DIR.exists():
        print(f"[mels] WARNING: directory not found: {MELS_DIR}")
        return counts

    for species_dir in sorted(MELS_DIR.iterdir()):
        if not species_dir.is_dir():
            continue
        if species_dir.name == "not_galago":
            continue
        n_png = sum(1 for _ in species_dir.glob("*.png"))
        counts[species_dir.name] = n_png
    return counts


def report_raw() -> Counter:
    counts: Counter[str] = Counter()
    if not RAW_DIR.exists():
        print(f"[raw] WARNING: directory not found: {RAW_DIR}")
        return counts

    wav_exts = {".wav", ".WAV"}
    for p in RAW_DIR.rglob("*"):
        if not p.is_file() or p.suffix not in wav_exts:
            continue
        lbl = label_from_raw_path(p)
        counts[lbl] += 1
    return counts


def main() -> None:
    print("=" * 72)
    print("DATA BALANCE REPORT")
    print("=" * 72)

    mels_counts = report_mels()
    raw_counts = report_raw()

    all_labels = sorted(set(mels_counts.keys()) | set(raw_counts.keys()))

    if not all_labels:
        print("No species found in mels or raw_audio.")
        return

    max_mels = max(mels_counts.values() or [0])
    max_raw = max(raw_counts.values() or [0])

    print("\nPer-species counts:")
    print("-" * 72)
    print(f"{'Species':30} {'PNG (mels)':>12} {'WAV (raw)':>12} {'PNG ratio':>10} {'WAV ratio':>10}")
    print("-" * 72)
    for lbl in all_labels:
        n_png = mels_counts.get(lbl, 0)
        n_wav = raw_counts.get(lbl, 0)
        png_ratio = (n_png / max_mels) if max_mels > 0 else 0.0
        wav_ratio = (n_wav / max_raw) if max_raw > 0 else 0.0
        print(
            f"{lbl[:29]:30} "
            f"{n_png:12d} {n_wav:12d} "
            f"{png_ratio:10.2f} {wav_ratio:10.2f}"
        )

    print("\nImbalance hints:")
    print("- Species with PNG ratio < 0.3 are strong candidates for augmentation or extra data.")
    print("- Species with WAV >> PNG likely suffer from domain shift (underrepresented in training PNGs).")
    print("- Species with WAV ~0 but PNG > 0 may be overfitted to training distribution.")


if __name__ == "__main__":
    main()

