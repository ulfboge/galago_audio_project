"""\
Sweep classifier confidence thresholds for the 3-stage pipeline output.

This script reads:
- outputs/predictions/predictions_3stage_context.csv (must include top1_species/top1_prob)
- data/splits/raw_audio_holdout.json (optional, for holdout/train breakdown)

For each threshold, it reports:
- coverage (% files with top1_prob >= threshold)
- top-1 accuracy (with threshold gating; below threshold treated as 'uncertain')
- top-1 accuracy among covered files
- top-3 accuracy (ungated; true label in raw top-3)

Note: This does *not* re-run inference; it re-interprets existing probabilities.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRED_CSV = PROJECT_ROOT / "outputs" / "predictions" / "predictions_3stage_context.csv"
SPLIT_JSON = PROJECT_ROOT / "data" / "splits" / "raw_audio_holdout.json"

AUDIO_DIR = PROJECT_ROOT / "data" / "raw_audio"

THRESHOLDS = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6]

# Keep this in sync with scripts/analyze_prediction_accuracy.py
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
    """Extract mapped folder label from filepath."""
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
    true_label: str
    detector_result: str
    top1_species: str
    top1_prob: float
    top2_species: str
    top3_species: str


def load_rows(path: Path) -> list[Row]:
    if not path.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {path}")

    rows: list[Row] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"filepath", "detector_result", "top1_species", "top1_prob", "top2_species", "top3_species"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Predictions CSV missing columns: {sorted(missing)}")

        for r in reader:
            filepath = r["filepath"]
            true_label = get_mapped_label_from_filepath(filepath)

            try:
                top1_prob = float(r["top1_prob"])
            except ValueError:
                # If it can't be parsed, treat as 0.0 so it will be 'uncertain' at any reasonable threshold.
                top1_prob = 0.0

            rows.append(
                Row(
                    filepath=filepath,
                    true_label=true_label,
                    detector_result=r.get("detector_result", ""),
                    top1_species=r.get("top1_species", ""),
                    top1_prob=top1_prob,
                    top2_species=r.get("top2_species", ""),
                    top3_species=r.get("top3_species", ""),
                )
            )

    return rows


def load_split_files(path: Path) -> Optional[tuple[set[str], set[str]]]:
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    holdout = set(data.get("holdout_files", []))
    train = set(data.get("train_files", []))
    return holdout, train


def compute_metrics(rows: Iterable[Row], threshold: float) -> dict[str, float]:
    rows_list = list(rows)
    total = len(rows_list)
    if total == 0:
        return {"coverage": 0.0, "top1": 0.0, "top1_covered": 0.0, "top3": 0.0}

    covered = 0
    correct_top1 = 0
    correct_top3 = 0

    for r in rows_list:
        # detector gate (3-stage): only evaluate those that passed detector
        if r.detector_result != "galago":
            continue

        is_covered = r.top1_prob >= threshold
        if is_covered:
            covered += 1
            pred = r.top1_species
        else:
            pred = "uncertain"

        if pred == r.true_label:
            correct_top1 += 1

        if r.true_label in {r.top1_species, r.top2_species, r.top3_species}:
            correct_top3 += 1

    # NOTE: total already includes only rows_list; detector filtering is not expected for this dataset.
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


def print_table(name: str, rows: list[Row]) -> None:
    print(f"\n{name}:")
    print("-" * 72)
    print(f"{'thr':>5}  {'cov%':>7}  {'top1%':>7}  {'top1|cov%':>10}  {'top3%':>7}")
    print("-" * 72)
    for thr in THRESHOLDS:
        m = compute_metrics(rows, thr)
        print(
            f"{thr:>5.2f}  "
            f"{m['coverage']:7.1f}  "
            f"{m['top1']:7.1f}  "
            f"{m['top1_covered']:10.1f}  "
            f"{m['top3']:7.1f}"
        )


def main() -> None:
    print("3-stage threshold sweep (accuracy vs coverage)")
    print("=" * 72)

    rows = load_rows(PRED_CSV)
    print(f"Loaded {len(rows)} prediction rows")

    split = load_split_files(SPLIT_JSON)

    if split is None:
        print_table("All rows", rows)
        return

    holdout_files, train_files = split

    rows_holdout = [r for r in rows if r.filepath in holdout_files]
    rows_train = [r for r in rows if r.filepath in train_files]
    rows_other = [r for r in rows if (r.filepath not in holdout_files and r.filepath not in train_files)]

    if rows_other:
        print(f"WARNING: {len(rows_other)} rows not found in split JSON")

    print_table("Combined (holdout + train)", rows_holdout + rows_train)
    print_table("Holdout", rows_holdout)
    print_table("Train", rows_train)


if __name__ == "__main__":
    main()
