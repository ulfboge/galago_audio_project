"""\
Compare training PNG image stats vs inference-generated images from raw_audio.

Goal: detect preprocessing / rendering domain shift.

We compute simple per-image statistics (min/max/mean/std) on arrays in [0,1]:
- training: load existing PNGs from data/melspectrograms/**.png
- inference: generate images from raw WAVs using the same mel params + magma colormap
  (matching scripts/predict_3stage_with_context.py preprocessing)

Outputs:
- outputs/evaluation/training_vs_inference_image_stats.json
- outputs/evaluation/training_vs_inference_image_stats.png
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MELS_DIR = PROJECT_ROOT / "data" / "melspectrograms"
RAW_AUDIO_DIR = PROJECT_ROOT / "data" / "raw_audio"
OUT_JSON = PROJECT_ROOT / "outputs" / "evaluation" / "training_vs_inference_image_stats.json"
OUT_PNG = PROJECT_ROOT / "outputs" / "evaluation" / "training_vs_inference_image_stats.png"

# Must match predict_3stage_with_context.py
SR = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FMIN = 200
FMAX = 10000
TARGET_FRAMES = 128

WINDOW_SEC = 2.5


@dataclass(frozen=True)
class Args:
    n_train: int
    n_wav: int
    seed: int


def parse_args() -> Args:
    args = sys.argv[1:]
    n_train = 256
    n_wav = 64
    seed = 0

    i = 0
    while i < len(args):
        a = args[i]
        if a == "--n-train":
            n_train = int(args[i + 1]); i += 2; continue
        if a == "--n-wav":
            n_wav = int(args[i + 1]); i += 2; continue
        if a == "--seed":
            seed = int(args[i + 1]); i += 2; continue
        raise SystemExit(f"Unknown arg: {a}")

    return Args(n_train=n_train, n_wav=n_wav, seed=seed)


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
    pad_value = float(S.min()) if S.size else 0.0
    return np.pad(S, ((0, 0), (pad_left, pad_right)), mode="constant", constant_values=pad_value)


def wav_to_rgb01(y: np.ndarray, sr: int) -> np.ndarray:
    """Return float32 RGB in [0,1], shape (H,W,3) where H=N_MELS, W=TARGET_FRAMES."""
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH, fmin=FMIN, fmax=FMAX
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    S_fixed = pad_or_crop(S_db, TARGET_FRAMES)
    mn = float(S_fixed.min())
    mx = float(S_fixed.max())
    if mx - mn < 1e-6:
        S_norm = np.zeros_like(S_fixed, dtype=np.float32)
    else:
        S_norm = ((S_fixed - mn) / (mx - mn)).astype(np.float32)
    cmap = plt.colormaps["magma"]
    rgb = cmap(S_norm)[:, :, :3].astype(np.float32)  # already [0,1]
    return rgb


def load_png01(path: Path) -> np.ndarray:
    b = tf.io.read_file(str(path))
    img = tf.image.decode_png(b, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # -> [0,1]
    return img.numpy()


def stats(images: list[np.ndarray]) -> dict[str, float]:
    if not images:
        return {"n": 0, "mean_std": 0.0, "p50_std": 0.0, "p10_std": 0.0, "p90_std": 0.0, "mean_min": 0.0, "mean_max": 0.0}
    stds = np.asarray([float(im.std()) for im in images], dtype=np.float32)
    mins = np.asarray([float(im.min()) for im in images], dtype=np.float32)
    maxs = np.asarray([float(im.max()) for im in images], dtype=np.float32)
    return {
        "n": int(len(images)),
        "mean_std": float(stds.mean()),
        "p10_std": float(np.percentile(stds, 10)),
        "p50_std": float(np.percentile(stds, 50)),
        "p90_std": float(np.percentile(stds, 90)),
        "mean_min": float(mins.mean()),
        "mean_max": float(maxs.mean()),
    }


def main() -> None:
    a = parse_args()
    rng = np.random.default_rng(a.seed)

    # Training PNG sample
    if not MELS_DIR.exists():
        raise SystemExit(f"Missing training PNG dir: {MELS_DIR}")
    train_pngs = [p for p in MELS_DIR.rglob("*.png")]
    if not train_pngs:
        raise SystemExit(f"No PNGs found under: {MELS_DIR}")
    n_train = min(a.n_train, len(train_pngs))
    train_sel = [train_pngs[int(i)] for i in rng.choice(len(train_pngs), size=n_train, replace=False)]
    train_imgs = [load_png01(p) for p in train_sel]

    # Raw audio sample -> inference images
    if not RAW_AUDIO_DIR.exists():
        raise SystemExit(f"Missing raw audio dir: {RAW_AUDIO_DIR}")
    wavs = sorted(RAW_AUDIO_DIR.rglob("*.wav"))
    if not wavs:
        raise SystemExit(f"No WAVs found under: {RAW_AUDIO_DIR}")
    n_wav = min(a.n_wav, len(wavs))
    wav_sel = [wavs[int(i)] for i in rng.choice(len(wavs), size=n_wav, replace=False)]

    inf_imgs: list[np.ndarray] = []
    inf_names: list[str] = []
    win = int(WINDOW_SEC * SR)
    for p in wav_sel:
        y, sr = librosa.load(str(p), sr=SR, mono=True)
        if len(y) <= win:
            y_win = y
        else:
            start = (len(y) - win) // 2
            y_win = y[start : start + win]
        rgb = wav_to_rgb01(y_win, sr)
        inf_imgs.append(rgb)
        inf_names.append(p.name)

    # Summaries
    train_s = stats(train_imgs)
    inf_s = stats(inf_imgs)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": a.seed,
                "train": {
                    "n_requested": a.n_train,
                    "n_used": train_s["n"],
                    "stats": train_s,
                    "example_paths": [str(p) for p in train_sel[:25]],
                },
                "inference": {
                    "n_requested": a.n_wav,
                    "n_used": inf_s["n"],
                    "window_sec": WINDOW_SEC,
                    "stats": inf_s,
                    "example_files": inf_names[:25],
                },
                "params": {
                    "SR": SR,
                    "N_MELS": N_MELS,
                    "N_FFT": N_FFT,
                    "HOP_LENGTH": HOP_LENGTH,
                    "FMIN": FMIN,
                    "FMAX": FMAX,
                    "TARGET_FRAMES": TARGET_FRAMES,
                },
            },
            f,
            indent=2,
        )

    # Plot std distributions
    plt.figure(figsize=(10, 4))
    train_stds = [float(im.std()) for im in train_imgs]
    inf_stds = [float(im.std()) for im in inf_imgs]
    bins = np.linspace(0.0, max(max(train_stds, default=0.0), max(inf_stds, default=0.0), 1e-6), 30)
    plt.hist(train_stds, bins=bins, alpha=0.6, label=f"training PNGs (n={len(train_stds)})")
    plt.hist(inf_stds, bins=bins, alpha=0.6, label=f"inference windows (n={len(inf_stds)})")
    plt.xlabel("Per-image std (RGB in [0,1])")
    plt.ylabel("Count")
    plt.title("Training vs inference image std distribution")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    plt.close()

    print("Training vs inference image stats")
    print("=" * 72)
    print(f"Saved: {OUT_JSON}")
    print(f"Saved: {OUT_PNG}")
    print("\nTrain stats:", train_s)
    print("Infer stats:", inf_s)


if __name__ == "__main__":
    main()

