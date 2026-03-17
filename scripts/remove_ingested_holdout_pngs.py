"""\
Remove leakage: move `rawaudio__...` PNGs generated from holdout WAV files
out of `data/melspectrograms/`.

- Reads `data/splits/raw_audio_holdout.json`
- For each holdout WAV stem, moves matching files:
    data/melspectrograms/*/rawaudio__<stem>__win*.png
  into:
    data/melspectrograms_holdout_removed/*/

This is reversible (we move, not delete).
"""

from __future__ import annotations

from pathlib import Path
import json
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_JSON = PROJECT_ROOT / "data" / "splits" / "raw_audio_holdout.json"
MELS_DIR = PROJECT_ROOT / "data" / "melspectrograms"
REMOVED_DIR = PROJECT_ROOT / "data" / "melspectrograms_holdout_removed"


def main() -> None:
    if not SPLIT_JSON.exists():
        raise FileNotFoundError(f"Missing split file: {SPLIT_JSON}")

    payload = json.loads(SPLIT_JSON.read_text(encoding="utf-8"))
    holdout_files = [Path(p) for p in payload.get("holdout_files", [])]
    if not holdout_files:
        raise RuntimeError("Split JSON has no holdout_files")

    stems = sorted({p.stem for p in holdout_files})

    moved = 0
    missing = 0

    for stem in stems:
        pattern = f"rawaudio__{stem}__win*.png"
        matches = list(MELS_DIR.glob(f"*/{pattern}"))
        if not matches:
            missing += 1
            continue

        for src in matches:
            dest_dir = REMOVED_DIR / src.parent.name
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / src.name
            if dest.exists():
                # already moved
                continue
            shutil.move(str(src), str(dest))
            moved += 1

    print(f"Holdout WAV stems: {len(stems)}")
    print(f"Moved PNGs: {moved}")
    print(f"Stems with zero matches: {missing}")
    print(f"Removed PNGs stored under: {REMOVED_DIR}")


if __name__ == "__main__":
    main()
