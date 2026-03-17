"""\
Summarize a gridsearch of pooling/RMS settings across confidence thresholds.

This script expects prediction CSVs produced by:
  scripts/predict_3stage_with_context.py

It reads all CSVs under:
  outputs/predictions/gridsearch_3stage_params/*.csv

For each CSV and each threshold, it computes:
- coverage (% with top1_prob >= threshold)
- top-1 accuracy (below threshold treated as 'uncertain')
- top-1 accuracy among covered
- top-3 accuracy (ungated; true label in raw top-3)

Also reports holdout/train split breakdown when split JSON is available.

Outputs:
- outputs/evaluation/gridsearch_threshold_sweep_summary.csv
- outputs/evaluation/gridsearch_threshold_sweep_summary.json
- outputs/evaluation/gridsearch_threshold_sweep_persistent_confusions.csv
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRID_DIR = PROJECT_ROOT / "outputs" / "predictions" / "gridsearch_3stage_params"
SPLIT_JSON = PROJECT_ROOT / "data" / "splits" / "raw_audio_holdout.json"
OUT_DIR = PROJECT_ROOT / "outputs" / "evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_SUMMARY_CSV = OUT_DIR / "gridsearch_threshold_sweep_summary.csv"
OUT_SUMMARY_JSON = OUT_DIR / "gridsearch_threshold_sweep_summary.json"
OUT_PERSISTENT_CSV = OUT_DIR / "gridsearch_threshold_sweep_persistent_confusions.csv"

AUDIO_DIR = PROJECT_ROOT / "data" / "raw_audio"

# Include high thresholds to see "high-confidence mistakes"
THRESHOLDS = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]

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
    true_label: str
    detector_result: str
    top1_species: str
    top1_prob: float
    top2_species: str
    top3_species: str


def load_split_sets(path: Path) -> Optional[tuple[set[str], set[str]]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return set(data.get("holdout_files", [])), set(data.get("train_files", []))


def load_rows(csv_path: Path) -> list[Row]:
    rows: list[Row] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"filepath", "detector_result", "top1_species", "top1_prob", "top2_species", "top3_species"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path.name} missing columns: {sorted(missing)}")
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
                    true_label=get_mapped_label_from_filepath(fp),
                    detector_result=r.get("detector_result", ""),
                    top1_species=r.get("top1_species", ""),
                    top1_prob=p1,
                    top2_species=r.get("top2_species", ""),
                    top3_species=r.get("top3_species", ""),
                )
            )
    return rows


def compute_metrics(rows: Iterable[Row], threshold: float) -> dict[str, float]:
    rs = [r for r in rows if r.detector_result == "galago"]
    total = len(rs)
    if total == 0:
        return {"n": 0, "coverage": 0.0, "top1": 0.0, "top1_covered": 0.0, "top3": 0.0}

    covered = 0
    correct_top1 = 0
    correct_top3 = 0

    for r in rs:
        is_cov = r.top1_prob >= threshold
        if is_cov:
            covered += 1
            pred = r.top1_species
        else:
            pred = "uncertain"

        if pred == r.true_label:
            correct_top1 += 1

        if r.true_label in {r.top1_species, r.top2_species, r.top3_species}:
            correct_top3 += 1

    return {
        "n": float(total),
        "coverage": covered / total * 100.0,
        "top1": correct_top1 / total * 100.0,
        "top1_covered": (correct_top1 / covered * 100.0) if covered else 0.0,
        "top3": correct_top3 / total * 100.0,
    }


def confusion_counts(rows: Iterable[Row], threshold: float) -> dict[tuple[str, str], list[str]]:
    """Confusions among *covered* predictions at a given threshold."""
    rs = [r for r in rows if r.detector_result == "galago" and r.top1_prob >= threshold]
    out: dict[tuple[str, str], list[str]] = defaultdict(list)
    for r in rs:
        if r.top1_species != r.true_label:
            out[(r.true_label, r.top1_species)].append(Path(r.filepath).name)
    return out


def main() -> None:
    if not GRID_DIR.exists():
        raise SystemExit(f"ERROR: grid dir not found: {GRID_DIR}")
    # Restrict to a single inference run per pooling/RMS setting.
    # Different classifier thresholds in the filename don't change the underlying top1_prob/top3;
    # they only change 'species_result' gating, which we ignore here.
    csvs = sorted(GRID_DIR.glob("*thr0p20_*.csv"))
    if not csvs:
        raise SystemExit(f"ERROR: no CSVs found under: {GRID_DIR}")

    split = load_split_sets(SPLIT_JSON)
    holdout_files: set[str] = set()
    train_files: set[str] = set()
    if split is not None:
        holdout_files, train_files = split

    summaries: list[dict[str, object]] = []

    # Persistent confusions across runs for each threshold (how many CSVs show this confusion)
    persistent: dict[tuple[float, str, str], dict[str, object]] = {}

    for csv_path in csvs:
        rows = load_rows(csv_path)

        rows_holdout = [r for r in rows if r.filepath in holdout_files] if holdout_files else []
        rows_train = [r for r in rows if r.filepath in train_files] if train_files else []

        for thr in THRESHOLDS:
            m_all = compute_metrics(rows, thr)
            m_h = compute_metrics(rows_holdout, thr) if rows_holdout else {"n": 0, "coverage": 0.0, "top1": 0.0, "top1_covered": 0.0, "top3": 0.0}
            m_t = compute_metrics(rows_train, thr) if rows_train else {"n": 0, "coverage": 0.0, "top1": 0.0, "top1_covered": 0.0, "top3": 0.0}

            summaries.append(
                {
                    "run_tag": csv_path.stem.replace("predictions_3stage_context_", ""),
                    "pred_csv": str(csv_path),
                    "threshold": thr,
                    "n": int(m_all["n"]),
                    "coverage_pct": float(m_all["coverage"]),
                    "top1_pct": float(m_all["top1"]),
                    "top1_covered_pct": float(m_all["top1_covered"]),
                    "top3_pct": float(m_all["top3"]),
                    "holdout_n": int(m_h["n"]),
                    "holdout_coverage_pct": float(m_h["coverage"]),
                    "holdout_top1_pct": float(m_h["top1"]),
                    "holdout_top3_pct": float(m_h["top3"]),
                    "train_n": int(m_t["n"]),
                    "train_coverage_pct": float(m_t["coverage"]),
                    "train_top1_pct": float(m_t["top1"]),
                    "train_top3_pct": float(m_t["top3"]),
                }
            )

            conf = confusion_counts(rows, thr)
            for (true_label, pred_label), files in conf.items():
                k = (thr, true_label, pred_label)
                e = persistent.setdefault(
                    k,
                    {
                        "threshold": thr,
                        "true_label": true_label,
                        "pred_label": pred_label,
                        "runs_with_confusion": 0,
                        "total_count": 0,
                        "example_files": [],
                    },
                )
                e["runs_with_confusion"] = int(e["runs_with_confusion"]) + 1
                e["total_count"] = int(e["total_count"]) + len(files)
                ex = list(e["example_files"])
                ex.extend(files[:5])
                # keep unique + limit
                seen = set()
                uniq = []
                for fn in ex:
                    if fn in seen:
                        continue
                    seen.add(fn)
                    uniq.append(fn)
                e["example_files"] = uniq[:15]

    # Sort summaries: by threshold then by top1 desc, then coverage desc
    summaries.sort(key=lambda r: (float(r["threshold"]), float(r["top1_pct"]), float(r["coverage_pct"])), reverse=True)

    with OUT_SUMMARY_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()) if summaries else [])
        w.writeheader()
        w.writerows(summaries)

    with OUT_SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump({"thresholds": THRESHOLDS, "n_runs": len(csvs), "summaries": summaries}, f, indent=2)

    # Persistent confusions CSV: sort by threshold then runs_with_confusion desc then total_count desc
    persistent_rows = list(persistent.values())
    persistent_rows.sort(key=lambda e: (float(e["threshold"]), int(e["runs_with_confusion"]), int(e["total_count"])), reverse=True)

    with OUT_PERSISTENT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["threshold", "true_label", "pred_label", "runs_with_confusion", "total_count", "example_files"],
        )
        w.writeheader()
        for e in persistent_rows[:200]:
            w.writerow(
                {
                    "threshold": e["threshold"],
                    "true_label": e["true_label"],
                    "pred_label": e["pred_label"],
                    "runs_with_confusion": e["runs_with_confusion"],
                    "total_count": e["total_count"],
                    "example_files": "; ".join(e.get("example_files") or []),
                }
            )

    print("Wrote:")
    print(f"- {OUT_SUMMARY_CSV}")
    print(f"- {OUT_SUMMARY_JSON}")
    print(f"- {OUT_PERSISTENT_CSV}")


if __name__ == "__main__":
    main()

