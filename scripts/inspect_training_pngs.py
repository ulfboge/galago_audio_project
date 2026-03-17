"""\
Sanity-check training spectrogram PNGs.

This script:
- samples PNGs from data/melspectrograms/<class>/*.png
- computes basic image statistics (min/max/mean/std)
- flags near-constant/low-variance images
- saves a montage image for visual inspection

Usage:
  python scripts/inspect_training_pngs.py
  python scripts/inspect_training_pngs.py --n 64 --seed 42
  python scripts/inspect_training_pngs.py --out outputs/evaluation/training_png_sanity.png
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MELS_DIR = PROJECT_ROOT / "data" / "melspectrograms"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "evaluation" / "training_png_sanity.png"


@dataclass(frozen=True)
class Args:
    n: int
    seed: int
    out: Path
    per_class: int


def parse_args() -> Args:
    args = sys.argv[1:]
    n = 64
    seed = 0
    out = DEFAULT_OUT
    per_class = 0

    i = 0
    while i < len(args):
        a = args[i]
        if a == "--n":
            n = int(args[i + 1]); i += 2; continue
        if a == "--seed":
            seed = int(args[i + 1]); i += 2; continue
        if a == "--out":
            out = Path(args[i + 1]); i += 2; continue
        if a == "--per-class":
            per_class = int(args[i + 1]); i += 2; continue
        raise SystemExit(f"Unknown arg: {a}")

    return Args(n=n, seed=seed, out=out, per_class=per_class)


def list_pngs() -> dict[str, list[Path]]:
    if not MELS_DIR.exists():
        raise FileNotFoundError(f"Missing mels directory: {MELS_DIR}")

    by_class: dict[str, list[Path]] = {}
    for d in sorted(MELS_DIR.iterdir()):
        if not d.is_dir():
            continue
        files = sorted(d.glob("*.png"))
        if files:
            by_class[d.name] = files
    return by_class


def load_png(path: Path) -> np.ndarray:
    """Load PNG to float32 array in [0,1], shape (H,W,3)."""
    b = tf.io.read_file(str(path))
    img = tf.image.decode_png(b, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # -> [0,1]
    return img.numpy()


def montage(images: list[np.ndarray], labels: list[str], out: Path, cols: int = 8) -> None:
    if not images:
        return
    rows = int(np.ceil(len(images) / cols))
    plt.figure(figsize=(cols * 2, rows * 2))
    for i, (img, lab) in enumerate(zip(images, labels), 1):
        ax = plt.subplot(rows, cols, i)
        ax.imshow(img)
        ax.set_title(lab[:24], fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()


def main() -> None:
    a = parse_args()
    rng = np.random.default_rng(a.seed)

    by_class = list_pngs()
    classes = list(by_class.keys())

    total_pngs = sum(len(v) for v in by_class.values())
    print("Training PNG sanity check")
    print("=" * 70)
    print(f"MELS_DIR: {MELS_DIR}")
    print(f"Classes found: {len(classes)}")
    print(f"Total PNGs: {total_pngs}")

    # Choose sample set
    selected: list[Path] = []
    if a.per_class > 0:
        for cls in classes:
            files = by_class[cls]
            k = min(a.per_class, len(files))
            idx = rng.choice(len(files), size=k, replace=False)
            selected.extend([files[int(i)] for i in idx])
        rng.shuffle(selected)
        selected = selected[: a.n]
    else:
        all_files = [p for v in by_class.values() for p in v]
        k = min(a.n, len(all_files))
        idx = rng.choice(len(all_files), size=k, replace=False)
        selected = [all_files[int(i)] for i in idx]

    print(f"\nSampled: {len(selected)} images (seed={a.seed})")

    # Stats
    lows = []
    imgs = []
    labels = []

    per_img_std = []
    per_img_min = []
    per_img_max = []

    for p in selected:
        img = load_png(p)
        mn = float(img.min())
        mx = float(img.max())
        mu = float(img.mean())
        sd = float(img.std())
        per_img_min.append(mn)
        per_img_max.append(mx)
        per_img_std.append(sd)

        cls = p.parent.name
        imgs.append(img)
        labels.append(cls)

        # Heuristics for "blank" images
        if sd < 0.01 or (mx - mn) < 0.05:
            lows.append((p, sd, mn, mx, mu))

    per_img_std_arr = np.asarray(per_img_std)
    print("\nPer-image stats (on [0,1] arrays):")
    print(f"  std:  mean={per_img_std_arr.mean():.4f}  p10={np.percentile(per_img_std_arr,10):.4f}  p50={np.percentile(per_img_std_arr,50):.4f}  p90={np.percentile(per_img_std_arr,90):.4f}")
    print(f"  min:  mean={np.mean(per_img_min):.4f}  min={np.min(per_img_min):.4f}  max={np.max(per_img_min):.4f}")
    print(f"  max:  mean={np.mean(per_img_max):.4f}  min={np.min(per_img_max):.4f}  max={np.max(per_img_max):.4f}")

    print(f"\nLow-variance flags: {len(lows)}/{len(selected)}")
    for p, sd, mn, mx, mu in lows[:15]:
        print(f"  - {p.parent.name}/{p.name}: std={sd:.4f} min={mn:.3f} max={mx:.3f} mean={mu:.3f}")
    if len(lows) > 15:
        print(f"  ... ({len(lows)-15} more)")

    # Per-class summary (std mean)
    cls_to_stds: dict[str, list[float]] = {}
    for p, sd in zip(selected, per_img_std):
        cls_to_stds.setdefault(p.parent.name, []).append(sd)

    print("\nPer-class mean std (sampled):")
    for cls in sorted(cls_to_stds.keys()):
        arr = np.asarray(cls_to_stds[cls])
        print(f"  {cls:28s} n={len(arr):3d}  mean_std={arr.mean():.4f}")

    # Save montage
    montage(imgs, labels, a.out)
    print(f"\nSaved montage: {a.out}")


if __name__ == "__main__":
    main()
