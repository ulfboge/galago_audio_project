"""\
Create a leakage-free holdout split of `data/raw_audio/*.wav`.

- Splits *by WAV file* stratified by the top-level folder under raw_audio
- Writes:
  - `data/splits/raw_audio_holdout.json`
  - `data/splits/raw_audio_holdout_filelist.txt` (absolute paths, one per line)

These holdout WAVs should NOT contribute any `rawaudio__...` PNGs to
`data/melspectrograms/` during training.
"""

from __future__ import annotations

from pathlib import Path
import json
import random

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = PROJECT_ROOT / "data" / "raw_audio"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
HOLDOUT_FRAC = 0.2
MIN_PER_GROUP = 1


def group_key(wav_path: Path) -> str:
    try:
        rel = wav_path.relative_to(AUDIO_DIR)
        return rel.parts[0]
    except Exception:
        return wav_path.parent.name


def main() -> None:
    wavs = sorted(AUDIO_DIR.rglob("*.wav"))
    if not wavs:
        raise RuntimeError(f"No WAV files found under {AUDIO_DIR}")

    groups: dict[str, list[Path]] = {}
    for w in wavs:
        groups.setdefault(group_key(w), []).append(w)

    random.seed(SEED)
    holdout: list[Path] = []
    train: list[Path] = []

    for g, files in sorted(groups.items()):
        files_shuf = files[:]
        random.shuffle(files_shuf)
        n = len(files_shuf)
        k = max(MIN_PER_GROUP, int(round(n * HOLDOUT_FRAC)))
        k = min(k, n)  # safety
        hold = files_shuf[:k]
        tr = files_shuf[k:]
        holdout.extend(hold)
        train.extend(tr)

    out_json = SPLITS_DIR / "raw_audio_holdout.json"
    out_txt = SPLITS_DIR / "raw_audio_holdout_filelist.txt"

    payload = {
        "seed": SEED,
        "holdout_frac": HOLDOUT_FRAC,
        "min_per_group": MIN_PER_GROUP,
        "audio_dir": str(AUDIO_DIR),
        "n_total": len(wavs),
        "n_holdout": len(holdout),
        "n_train": len(train),
        "holdout_files": [str(p.resolve()) for p in sorted(holdout)],
        "train_files": [str(p.resolve()) for p in sorted(train)],
        "by_group": {
            g: {
                "n": len(files),
                "n_holdout": len([p for p in holdout if group_key(p) == g]),
            }
            for g, files in sorted(groups.items())
        },
    }

    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    out_txt.write_text("\n".join(payload["holdout_files"]) + "\n", encoding="utf-8")

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_txt}")
    print(f"Total WAVs: {len(wavs)}")
    print(f"Holdout WAVs: {len(holdout)}")


if __name__ == "__main__":
    main()
