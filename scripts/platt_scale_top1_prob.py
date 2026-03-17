"""\
Fit a simple Platt-scaling calibrator on top1_prob -> P(correct).

This is a *binary* calibration model:
  y = 1 if (top1_species == true_label) else 0
  x = top1_prob (raw model confidence)

It helps quantify miscalibration and provides a simple post-hoc mapping that can
be used to interpret confidence thresholds more meaningfully on the same domain.

NOTE: With only ~69 examples, this is mainly diagnostic and can overfit.

Usage:
  python scripts/platt_scale_top1_prob.py
  python scripts/platt_scale_top1_prob.py --csv outputs/predictions/predictions_3stage_context_thr0.20.csv
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression


if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_ROOT / "outputs" / "predictions" / "predictions_3stage_context_thr0.20.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AUDIO_DIR = PROJECT_ROOT / "data" / "raw_audio"

LABEL_MAP = {
    "G.sp.nov.1": "Galagoides_sp_nov",
    "G.sp.nov.3": "Galagoides_sp_nov",
    "G.granti": "Paragalago_granti",
    "G.orinus": "Paragalago_orinus",
    "G.rondoensis": "Paragalago_rondoensis",
    "G.zanzibaricus": "Paragalago_zanzibaricus",
    "O.crassicaudatus": "Otolemur_crassicaudatus",
    "O.garnettii": "Otolemur_garnettii",
    "Galago_granti": "Paragalago_granti",
}


def get_mapped_label_from_filepath(filepath: str) -> str:
    try:
        wav_path = Path(filepath)
        rel = wav_path.relative_to(AUDIO_DIR)
        src_folder = rel.parts[0] if len(rel.parts) > 1 else wav_path.parent.name
        return LABEL_MAP.get(src_folder, src_folder)
    except Exception:
        parts = Path(filepath).parts
        for part in parts:
            if part in LABEL_MAP:
                return LABEL_MAP[part]
        return Path(filepath).parent.name


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if not np.any(mask):
            continue
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(y_prob[mask]))
        frac = float(np.sum(mask)) / float(n)
        ece += abs(acc - conf) * frac
    return float(ece)


@dataclass(frozen=True)
class Row:
    top1_prob: float
    correct: int


def load_rows(path: Path) -> list[Row]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    rows: list[Row] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"filepath", "detector_result", "top1_species", "top1_prob"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing columns: {sorted(missing)}")

        for r in reader:
            if r.get("detector_result", "") != "galago":
                continue
            fp = r.get("filepath", "")
            if not fp:
                continue
            true_label = get_mapped_label_from_filepath(fp)
            top1_species = r.get("top1_species", "")
            try:
                p = float(r.get("top1_prob", "0") or 0.0)
            except ValueError:
                p = 0.0
            rows.append(Row(top1_prob=p, correct=1 if top1_species == true_label else 0))
    return rows


def main() -> None:
    args = sys.argv[1:]
    csv_path = DEFAULT_CSV
    n_bins = 10

    i = 0
    while i < len(args):
        a = args[i]
        if a == "--csv":
            csv_path = Path(args[i + 1]); i += 2; continue
        if a == "--bins":
            n_bins = int(args[i + 1]); i += 2; continue
        raise SystemExit(f"Unknown arg: {a}")

    rows = load_rows(csv_path)
    if not rows:
        raise SystemExit("No usable rows found (detector_result=='galago').")

    x = np.asarray([r.top1_prob for r in rows], dtype=np.float32).reshape(-1, 1)
    y = np.asarray([r.correct for r in rows], dtype=np.int32)

    # Baseline metrics
    ece_raw = expected_calibration_error(y, x.reshape(-1), n_bins=n_bins)
    brier_raw = float(np.mean((x.reshape(-1) - y) ** 2))

    # Platt scaling
    lr = LogisticRegression(solver="lbfgs")
    lr.fit(x, y)
    y_cal = lr.predict_proba(x)[:, 1].astype(np.float32)

    ece_cal = expected_calibration_error(y, y_cal, n_bins=n_bins)
    brier_cal = float(np.mean((y_cal - y) ** 2))

    frac_pos_raw, mean_pred_raw = calibration_curve(y, x.reshape(-1), n_bins=n_bins, strategy="uniform")
    frac_pos_cal, mean_pred_cal = calibration_curve(y, y_cal, n_bins=n_bins, strategy="uniform")

    out_base = csv_path.stem.replace("predictions_", "platt_")
    out_json = OUT_DIR / f"{out_base}.json"
    out_png = OUT_DIR / f"{out_base}.png"

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "csv": str(csv_path),
                "n": int(len(rows)),
                "bins": int(n_bins),
                "raw": {"ece": float(ece_raw), "brier": brier_raw},
                "platt": {
                    "ece": float(ece_cal),
                    "brier": brier_cal,
                    "coef": float(lr.coef_[0][0]),
                    "intercept": float(lr.intercept_[0]),
                },
                "reliability_raw": {
                    "mean_predicted": [float(v) for v in mean_pred_raw],
                    "fraction_positives": [float(v) for v in frac_pos_raw],
                },
                "reliability_platt": {
                    "mean_predicted": [float(v) for v in mean_pred_cal],
                    "fraction_positives": [float(v) for v in frac_pos_cal],
                },
            },
            f,
            indent=2,
        )

    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect")
    plt.plot(mean_pred_raw, frac_pos_raw, "o-", label=f"Raw (ECE={ece_raw:.3f})")
    plt.plot(mean_pred_cal, frac_pos_cal, "o-", label=f"Platt (ECE={ece_cal:.3f})")
    plt.xlabel("Mean predicted confidence")
    plt.ylabel("Empirical accuracy")
    plt.title(f"Platt scaling: {csv_path.name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

    print("Platt scaling on top1_prob")
    print("=" * 72)
    print(f"CSV: {csv_path}")
    print(f"n: {len(rows)}")
    print(f"Raw:   ECE={ece_raw:.3f}  Brier={brier_raw:.3f}")
    print(f"Platt: ECE={ece_cal:.3f}  Brier={brier_cal:.3f}  coef={lr.coef_[0][0]:.3f}  intercept={lr.intercept_[0]:.3f}")
    print(f"Saved: {out_json}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()

