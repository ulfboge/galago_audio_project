"""Compare 16-class vs 7-class models across confidence thresholds.

Uses existing prediction CSVs:
- outputs/predictions/predictions_all_species.csv (16 classes)
- outputs/predictions/predictions_top7_windowed_fixed.csv (7 classes)

For each model and threshold, reports:
- coverage (%% non-"uncertain")
- top-1 accuracy
- top-1 accuracy among covered files
- top-3 accuracy
"""
from __future__ import annotations

import csv
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CSV_16 = PROJECT_ROOT / "outputs" / "predictions" / "predictions_all_species.csv"
CSV_7 = PROJECT_ROOT / "outputs" / "predictions" / "predictions_top7_windowed_fixed.csv"

THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6]


@dataclass
class PredictionRow:
    true_label: str
    predicted_species: str
    predicted_prob: float
    top2_species: str
    top3_species: str


def load_predictions(path: Path) -> List[PredictionRow]:
    rows: List[PredictionRow] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                PredictionRow(
                    true_label=r["mapped_folder_label"],
                    predicted_species=r["predicted_species"],
                    predicted_prob=float(r["predicted_prob"]),
                    top2_species=r["top2_species"],
                    top3_species=r["top3_species"],
                )
            )
    return rows


def compute_metrics(rows: List[PredictionRow], threshold: float) -> Dict[str, float]:
    total = len(rows)
    if total == 0:
        return {
            "coverage": 0.0,
            "top1": 0.0,
            "top1_covered": 0.0,
            "top3": 0.0,
        }

    covered = 0
    correct_top1 = 0
    correct_top3 = 0

    for r in rows:
        # Re-interpret "uncertain" using the threshold
        if r.predicted_prob < threshold:
            # treat as uncertain
            pred = "uncertain"
        else:
            pred = r.predicted_species

        if pred != "uncertain":
            covered += 1

        # top-1 accuracy
        if pred == r.true_label:
            correct_top1 += 1

        # top-3 accuracy: check if true label in original top3 set
        if r.true_label in {r.predicted_species, r.top2_species, r.top3_species}:
            correct_top3 += 1

    coverage = covered / total * 100.0
    top1 = correct_top1 / total * 100.0
    top3 = correct_top3 / total * 100.0
    top1_covered = (correct_top1 / covered * 100.0) if covered > 0 else 0.0

    return {
        "coverage": coverage,
        "top1": top1,
        "top1_covered": top1_covered,
        "top3": top3,
    }


def print_table(name: str, rows: List[PredictionRow]) -> None:
    print(f"\n{name}:")
    print("-" * 60)
    print(f"{'thr':>4}  {'cov%':>7}  {'top1%':>7}  {'top1|cov%':>10}  {'top3%':>7}")
    print("-" * 60)
    for thr in THRESHOLDS:
        m = compute_metrics(rows, thr)
        print(
            f"{thr:>4.2f}  "
            f"{m['coverage']:7.1f}  "
            f"{m['top1']:7.1f}  "
            f"{m['top1_covered']:10.1f}  "
            f"{m['top3']:7.1f}"
        )


def main() -> None:
    print("Comparing 16-class vs 7-class models across thresholds")
    print("=" * 60)

    if not CSV_16.exists():
        print(f"ERROR: 16-class predictions not found at {CSV_16}")
        return
    if not CSV_7.exists():
        print(f"ERROR: 7-class predictions not found at {CSV_7}")
        return

    preds_16 = load_predictions(CSV_16)
    preds_7 = load_predictions(CSV_7)

    print(f"Loaded {len(preds_16)} rows for 16-class model")
    print(f"Loaded {len(preds_7)} rows for 7-class model")

    print_table("16-class model", preds_16)
    print_table("7-class model (top7_windowed_fixed)", preds_7)


if __name__ == "__main__":
    main()
