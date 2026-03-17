"""\
Calibrate/diagnose confidence using an existing prediction CSV.

This is *domain-relevant* calibration (e.g. raw_audio WAV evaluation) because it
uses the pipeline's own top1_prob values and true labels inferred from paths.

Outputs:
- reliability diagram (predicted confidence vs empirical accuracy)
- ECE (expected calibration error)
- threshold sweep (coverage vs accuracy)

Usage:
  python scripts/calibrate_confidence_from_predictions.py
  python scripts/calibrate_confidence_from_predictions.py --csv outputs/predictions/predictions_3stage_context_thr0.20.csv
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


if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_ROOT / "outputs" / "predictions" / "predictions_3stage_context_thr0.20.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AUDIO_DIR = PROJECT_ROOT / "data" / "raw_audio"

# Keep in sync with analyze_prediction_accuracy.py
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


@dataclass(frozen=True)
class Row:
    filepath: str
    detector_result: str
    top1_species: str
    top1_prob: float
    true_label: str


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
            fp = r.get("filepath", "")
            if not fp:
                continue
            try:
                p1 = float(r.get("top1_prob", "0") or 0.0)
            except ValueError:
                p1 = 0.0

            rows.append(
                Row(
                    filepath=fp,
                    detector_result=r.get("detector_result", ""),
                    top1_species=r.get("top1_species", ""),
                    top1_prob=p1,
                    true_label=get_mapped_label_from_filepath(fp),
                )
            )

    return rows


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    # Uniform bins in [0,1]
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if not np.any(mask):
            continue
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(y_prob[mask]))
        frac = float(np.sum(mask)) / float(n)
        ece += abs(acc - conf) * frac
    return float(ece)


def main() -> None:
    # Args: --csv <path> --bins <N>
    args = sys.argv[1:]
    csv_path = DEFAULT_CSV
    n_bins = 10

    i = 0
    while i < len(args):
        a = args[i]
        if a == "--csv":
            csv_path = Path(args[i + 1])
            i += 2
            continue
        if a == "--bins":
            n_bins = int(args[i + 1])
            i += 2
            continue
        raise SystemExit(f"ERROR: Unknown arg: {a}")

    rows = load_rows(csv_path)
    # Only evaluate items that passed detector
    rows = [r for r in rows if r.detector_result == "galago"]

    if not rows:
        raise SystemExit("ERROR: No detector-passed rows found (detector_result=='galago').")

    y_prob = np.asarray([r.top1_prob for r in rows], dtype=np.float32)
    y_true = np.asarray([1.0 if r.top1_species == r.true_label else 0.0 for r in rows], dtype=np.float32)

    # Reliability curve
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    ece = expected_calibration_error(y_true, y_prob, n_bins=n_bins)

    # Threshold sweep
    thresholds = np.arange(0.05, 1.001, 0.05)
    sweep = []
    for t in thresholds:
        mask = y_prob >= t
        covered = int(np.sum(mask))
        if covered == 0:
            sweep.append({"threshold": float(t), "coverage": 0.0, "accuracy_covered": 0.0})
            continue
        acc = float(np.mean(y_true[mask]))
        sweep.append(
            {
                "threshold": float(t),
                "coverage": float(covered / len(y_prob)),
                "accuracy_covered": acc,
            }
        )

    out_base = csv_path.stem.replace("predictions_", "calib_")
    out_json = OUT_DIR / f"{out_base}_reliability.json"
    out_png = OUT_DIR / f"{out_base}_reliability.png"

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "csv": str(csv_path),
                "n_rows": int(len(rows)),
                "n_bins": int(n_bins),
                "ece": float(ece),
                "reliability": {
                    "mean_predicted": [float(x) for x in mean_pred],
                    "fraction_positives": [float(x) for x in frac_pos],
                },
                "threshold_sweep": sweep,
            },
            f,
            indent=2,
        )

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.plot(mean_pred, frac_pos, "o-", label=f"Empirical (ECE={ece:.3f})")
    plt.xlabel("Mean predicted confidence (top1_prob)")
    plt.ylabel("Empirical accuracy (top1 correct)")
    plt.title(f"Reliability diagram: {csv_path.name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

    print("Confidence calibration (from predictions)")
    print("=" * 72)
    print(f"CSV: {csv_path}")
    print(f"Rows (detector passed): {len(rows)}")
    print(f"ECE (bins={n_bins}): {ece:.3f}")
    print(f"Saved: {out_json}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()

