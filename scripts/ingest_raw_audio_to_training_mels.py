"""\
Ingest `data/raw_audio` WAVs into `data/melspectrograms/<species>` as additional
training samples to reduce domain shift.

- Uses the same mel params as existing pipeline
- Windows each WAV (WINDOW_SEC/HOP_SEC) to create multiple training PNGs
- Saves PNGs with unique names prefixed `rawaudio__...`

Safe to re-run: it skips files that already exist.

New (leakage-safe):
- You can pass `--filelist` to ingest only the train split WAVs (and avoid holdout leakage).
- You can pass `--out-dir` to write to a separate mel folder (recommended for experiments).
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import librosa

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = PROJECT_ROOT / "data" / "raw_audio"
MELS_DIR = PROJECT_ROOT / "data" / "melspectrograms"

# Match predict_3stage_with_context
SR = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FMIN = 200
FMAX = 10000
TARGET_FRAMES = 128
WINDOW_SEC = 2.5
HOP_SEC = 1.25
MIN_WINDOWS = 1

# Folder -> species label mapping
LABEL_MAP = {
    "G.sp.nov.1": "Galagoides_sp_nov",
    "G.sp.nov.3": "Galagoides_sp_nov",
    "G.granti": "Paragalago_granti",
    "G.orinus": "Paragalago_orinus",
    "G.rondoensis": "Paragalago_rondoensis",
    "G.zanzibaricus": "Paragalago_zanzibaricus",
    "O.crassicaudatus": "Otolemur_crassicaudatus",
    "O.garnettii": "Otolemur_garnettii",
}

# Reuse the existing PNG writer from make_mels for consistency
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from make_mels import save_spectrogram_png  # type: ignore


def parse_args() -> dict[str, object]:
    """
    Options:
      --filelist <path>   Only ingest WAVs listed (absolute paths recommended)
      --out-dir <path>    Output mel root (default: data/melspectrograms)
      --max-windows N     Limit windows per WAV (0 = no limit)
      --prefix STR        Output filename prefix (default: 'rawaudio__')
    """
    args = sys.argv[1:]
    out: dict[str, object] = {
        "filelist": None,
        "out_dir": MELS_DIR,
        "max_windows": 0,
        "prefix": "rawaudio__",
    }
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--filelist":
            out["filelist"] = Path(args[i + 1])
            i += 2
            continue
        if a == "--out-dir":
            out["out_dir"] = Path(args[i + 1])
            i += 2
            continue
        if a == "--max-windows":
            out["max_windows"] = int(args[i + 1])
            i += 2
            continue
        if a == "--prefix":
            out["prefix"] = str(args[i + 1])
            i += 2
            continue
        raise SystemExit(f"Unknown arg: {a}")
    return out


def pad_or_crop(S: np.ndarray, target_frames: int) -> np.ndarray:
    n_mels, T = S.shape
    if T == target_frames:
        return S
    if T > target_frames:
        start = (T - target_frames) // 2
        return S[:, start : start + target_frames]
    pad_total = target_frames - T
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    pad_value = S.min()
    return np.pad(S, ((0, 0), (pad_left, pad_right)), mode="constant", constant_values=pad_value)


def window_starts(n_samples: int, sr: int, win_sec: float, hop_sec: float) -> list[int]:
    win = int(win_sec * sr)
    hop = int(hop_sec * sr)
    if n_samples <= win:
        return [0]
    starts = list(range(0, n_samples - win + 1, hop))
    if len(starts) < MIN_WINDOWS:
        starts = [max(0, (n_samples - win) // 2)]
    return starts


def mel_db_window(y_win: np.ndarray, sr: int) -> np.ndarray:
    S = librosa.feature.melspectrogram(
        y=y_win,
        sr=sr,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    return pad_or_crop(S_db, TARGET_FRAMES)


def label_from_path(wav_path: Path) -> str | None:
    try:
        rel = wav_path.relative_to(AUDIO_DIR)
        folder = rel.parts[0]
    except Exception:
        folder = wav_path.parent.name
    return LABEL_MAP.get(folder, folder)


def main():
    opts = parse_args()
    out_root: Path = opts["out_dir"]  # type: ignore[assignment]
    filelist: Path | None = opts["filelist"]  # type: ignore[assignment]
    max_windows: int = int(opts["max_windows"])
    prefix: str = str(opts["prefix"])

    print("Ingesting raw_audio into training mels...")
    print(f"  From: {AUDIO_DIR}")
    print(f"  To:   {out_root}")
    if filelist:
        print(f"  Filelist: {filelist}")
    if max_windows:
        print(f"  Max windows per WAV: {max_windows}")
    print(f"  Prefix: {prefix}")

    if filelist:
        if not filelist.exists():
            print(f"ERROR: filelist not found: {filelist}")
            return
        wavs = []
        for line in filelist.read_text(encoding="utf-8").splitlines():
            s = line.strip().strip('"')
            if not s:
                continue
            p = Path(s)
            if p.exists() and p.suffix.lower() == ".wav":
                wavs.append(p)
        wavs = sorted(wavs)
    else:
        wavs = sorted(AUDIO_DIR.rglob("*.wav"))
    if not wavs:
        print("No WAV files found.")
        return

    added = 0
    skipped = 0
    errors = 0

    for wav in wavs:
        species = label_from_path(wav)
        if not species:
            skipped += 1
            continue

        out_dir = out_root / species
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            y, sr = librosa.load(str(wav), sr=SR, mono=True)
        except Exception:
            errors += 1
            continue

        starts = window_starts(len(y), sr, WINDOW_SEC, HOP_SEC)
        for wi, start in enumerate(starts):
            if max_windows and wi >= max_windows:
                break
            end = min(len(y), start + int(WINDOW_SEC * sr))
            y_win = y[start:end]
            if len(y_win) < int(0.5 * sr):
                continue

            out_name = f"{prefix}{wav.stem}__win{wi}.png"
            out_png = out_dir / out_name
            if out_png.exists():
                skipped += 1
                continue

            try:
                S = mel_db_window(y_win, sr)
                save_spectrogram_png(S, out_png)
                added += 1
            except Exception:
                errors += 1

    print("\nDone")
    print(f"  Added PNGs:   {added}")
    print(f"  Skipped PNGs: {skipped}")
    print(f"  Errors:       {errors}")


if __name__ == "__main__":
    main()
