"""\
Grid search for the 3-stage pipeline parameters.

We vary:
- classifier threshold (affects 'uncertain' gating)
- RMS gate relative threshold (window selection)
- top-K pooling windows (window pooling)

For each combination we:
- run scripts/predict_3stage_with_context.py on the 69-WAV evaluation filelist
- write a versioned prediction CSV
- compute metrics (coverage/top1/top3 + holdout/train breakdown)
- aggregate "persistent confusions" across runs

Outputs:
- outputs/evaluation/gridsearch_3stage_params_summary.csv
- outputs/evaluation/gridsearch_3stage_params_summary.json
- outputs/evaluation/gridsearch_3stage_params_persistent_confusions.csv
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICT_SCRIPT = PROJECT_ROOT / "scripts" / "predict_3stage_with_context.py"

FILELIST_ALL = PROJECT_ROOT / "data" / "splits" / "raw_audio_all_filelist.txt"
SPLIT_JSON = PROJECT_ROOT / "data" / "splits" / "raw_audio_holdout.json"

OUT_PRED_DIR = PROJECT_ROOT / "outputs" / "predictions" / "gridsearch_3stage_params"
OUT_EVAL_DIR = PROJECT_ROOT / "outputs" / "evaluation"


# Keep in sync with analyze_prediction_accuracy.py / sweep_3stage_thresholds.py
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

AUDIO_DIR = PROJECT_ROOT / "data" / "raw_audio"


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
class PredRow:
    filepath: str
    true_label: str
    detector_result: str
    species_result: str
    top1_species: str
    top1_prob: float
    top2_species: str
    top3_species: str


def load_preds(csv_path: Path) -> list[PredRow]:
    rows: list[PredRow] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "filepath",
            "detector_result",
            "species_result",
            "top1_species",
            "top1_prob",
            "top2_species",
            "top3_species",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path.name} missing columns: {sorted(missing)}")

        for r in reader:
            fp = r["filepath"]
            try:
                p1 = float(r.get("top1_prob", "0") or 0.0)
            except ValueError:
                p1 = 0.0
            rows.append(
                PredRow(
                    filepath=fp,
                    true_label=get_mapped_label_from_filepath(fp),
                    detector_result=r.get("detector_result", ""),
                    species_result=r.get("species_result", ""),
                    top1_species=r.get("top1_species", ""),
                    top1_prob=p1,
                    top2_species=r.get("top2_species", ""),
                    top3_species=r.get("top3_species", ""),
                )
            )
    return rows


def load_split_sets() -> tuple[set[str], set[str]]:
    if not SPLIT_JSON.exists():
        return set(), set()
    with SPLIT_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return set(data.get("holdout_files", [])), set(data.get("train_files", []))


def is_confident_species(species_result: str) -> bool:
    return species_result not in {"uncertain", "error", "not_classified", "N/A", ""}


def compute_metrics(rows: Iterable[PredRow]) -> dict[str, float]:
    rows_list = [r for r in rows if r.detector_result == "galago"]
    total = len(rows_list)
    if total == 0:
        return {
            "n": 0,
            "coverage": 0.0,
            "top1": 0.0,
            "top1_covered": 0.0,
            "top3": 0.0,
            "uncertain_rate": 0.0,
        }

    confident = 0
    correct_top1 = 0
    correct_top3 = 0
    uncertain = 0

    for r in rows_list:
        if r.species_result == "uncertain":
            uncertain += 1
        if is_confident_species(r.species_result):
            confident += 1
            if r.species_result == r.true_label:
                correct_top1 += 1
        else:
            # Not confident -> can't be "correct top-1" by definition in this pipeline.
            pass

        if r.true_label in {r.top1_species, r.top2_species, r.top3_species}:
            correct_top3 += 1

    coverage = confident / total * 100.0
    top1 = correct_top1 / total * 100.0
    top1_covered = (correct_top1 / confident * 100.0) if confident else 0.0
    top3 = correct_top3 / total * 100.0
    uncertain_rate = uncertain / total * 100.0

    return {
        "n": total,
        "coverage": coverage,
        "top1": top1,
        "top1_covered": top1_covered,
        "top3": top3,
        "uncertain_rate": uncertain_rate,
    }


def confusion_counts(rows: Iterable[PredRow]) -> dict[tuple[str, str], list[tuple[str, float]]]:
    """Return mapping (true,pred) -> list of (filepath, top1_prob) examples (confident wrong only)."""
    m: dict[tuple[str, str], list[tuple[str, float]]] = {}
    for r in rows:
        if r.detector_result != "galago":
            continue
        if not is_confident_species(r.species_result):
            continue
        if r.species_result == r.true_label:
            continue
        key = (r.true_label, r.species_result)
        m.setdefault(key, []).append((r.filepath, r.top1_prob))
    # sort examples by probability desc for stable reporting
    for key in list(m.keys()):
        m[key].sort(key=lambda x: x[1], reverse=True)
    return m


def run_one(
    classifier_threshold: float,
    pool_topk: int,
    rms_gate_rel: float,
    *,
    detector_threshold: float = 0.30,
    rms_gate_abs: float = 1e-4,
    skip_existing: bool = True,
) -> Path:
    OUT_PRED_DIR.mkdir(parents=True, exist_ok=True)

    tag = f"thr{classifier_threshold:.2f}_topk{pool_topk}_rms{rms_gate_rel:.2f}".replace(".", "p")
    out_csv = OUT_PRED_DIR / f"predictions_3stage_context_{tag}.csv"

    if skip_existing and out_csv.exists() and out_csv.stat().st_size > 0:
        print(f"Skipping existing: {out_csv.name}")
        return out_csv

    cmd = [
        sys.executable,
        str(PREDICT_SCRIPT),
        "--filelist",
        str(FILELIST_ALL),
        "--out-csv",
        str(out_csv),
        "--detector-threshold",
        f"{detector_threshold}",
        "--classifier-threshold",
        f"{classifier_threshold}",
        "--pool-topk",
        str(pool_topk),
        "--rms-gate-rel",
        f"{rms_gate_rel}",
        "--rms-gate-abs",
        f"{rms_gate_abs}",
    ]

    print(f"\nRunning: {out_csv.name}")
    subprocess.run(cmd, check=True)
    return out_csv


def parse_args() -> dict[str, object]:
    args = sys.argv[1:]
    out: dict[str, object] = {
        "max_runs": None,
        "start_index": 0,
        "summarize_only": False,
        "skip_existing": True,
    }

    i = 0
    while i < len(args):
        a = args[i]
        if a == "--max-runs":
            out["max_runs"] = int(args[i + 1])
            i += 2
            continue
        if a == "--start-index":
            out["start_index"] = int(args[i + 1])
            i += 2
            continue
        if a == "--summarize-only":
            out["summarize_only"] = True
            i += 1
            continue
        if a == "--force":
            out["skip_existing"] = False
            i += 1
            continue
        raise SystemExit(f"ERROR: Unknown arg: {a}")

    return out


def main() -> None:
    print("Grid search: 3-stage params (threshold × topK × RMS gate)")
    print("=" * 72)

    opts = parse_args()
    max_runs = opts["max_runs"]
    start_index = int(opts["start_index"])
    summarize_only = bool(opts["summarize_only"])
    skip_existing = bool(opts["skip_existing"])

    if not FILELIST_ALL.exists():
        raise SystemExit(f"ERROR: Missing filelist: {FILELIST_ALL}")
    if not PREDICT_SCRIPT.exists():
        raise SystemExit(f"ERROR: Missing predict script: {PREDICT_SCRIPT}")

    # Parameter grid (from handover)
    CLASSIFIER_THRESHOLDS = [0.20, 0.30, 0.40]
    POOL_TOPK = [1, 3, 5]
    RMS_GATE_REL = [0.10, 0.20, 0.30]

    holdout_files, train_files = load_split_sets()

    OUT_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_csv = OUT_EVAL_DIR / "gridsearch_3stage_params_summary.csv"
    summary_json = OUT_EVAL_DIR / "gridsearch_3stage_params_summary.json"
    persistent_csv = OUT_EVAL_DIR / "gridsearch_3stage_params_persistent_confusions.csv"

    summaries: list[dict[str, object]] = []
    persistent_runs: dict[tuple[str, str], dict[str, object]] = {}

    combos: list[tuple[float, int, float]] = []
    for thr in CLASSIFIER_THRESHOLDS:
        for topk in POOL_TOPK:
            for rms in RMS_GATE_REL:
                combos.append((thr, topk, rms))

    if start_index < 0 or start_index >= len(combos):
        raise SystemExit(f"ERROR: --start-index must be in [0, {len(combos)-1}]")

    combos = combos[start_index:]
    if max_runs is not None:
        combos = combos[: int(max_runs)]

    total_runs = len(combos)
    run_idx = 0

    for thr, topk, rms in combos:
        run_idx += 1
        print(f"\n[{run_idx}/{total_runs}] thr={thr:.2f} topK={topk} rms_rel={rms:.2f}")

        out_csv = run_one(thr, topk, rms, skip_existing=skip_existing) if not summarize_only else (
            OUT_PRED_DIR / f"predictions_3stage_context_{f'thr{thr:.2f}_topk{topk}_rms{rms:.2f}'.replace('.', 'p')}.csv"
        )
        if not out_csv.exists() or out_csv.stat().st_size == 0:
            print(f"Missing (skipping metrics): {out_csv.name}")
            continue

        preds = load_preds(out_csv)

        # Metrics: overall + split breakdown
        m_all = compute_metrics(preds)
        m_holdout = compute_metrics([r for r in preds if r.filepath in holdout_files]) if holdout_files else compute_metrics([])
        m_train = compute_metrics([r for r in preds if r.filepath in train_files]) if train_files else compute_metrics([])

        row: dict[str, object] = {
            "run_tag": out_csv.stem.replace("predictions_3stage_context_", ""),
            "pred_csv": str(out_csv),
            "classifier_threshold": thr,
            "pool_topk_windows": topk,
            "rms_gate_rel": rms,
            # overall
            "n": int(m_all["n"]),
            "coverage_pct": float(m_all["coverage"]),
            "uncertain_pct": float(m_all["uncertain_rate"]),
            "top1_pct": float(m_all["top1"]),
            "top1_covered_pct": float(m_all["top1_covered"]),
            "top3_pct": float(m_all["top3"]),
            # holdout
            "holdout_n": int(m_holdout["n"]),
            "holdout_coverage_pct": float(m_holdout["coverage"]),
            "holdout_top1_pct": float(m_holdout["top1"]),
            "holdout_top3_pct": float(m_holdout["top3"]),
            # train
            "train_n": int(m_train["n"]),
            "train_coverage_pct": float(m_train["coverage"]),
            "train_top1_pct": float(m_train["top1"]),
            "train_top3_pct": float(m_train["top3"]),
        }
        summaries.append(row)

        # Confusions: count in how many runs each confusion appears (confident wrong)
        conf = confusion_counts(preds)
        for (true_label, pred_label), examples in conf.items():
            k = (true_label, pred_label)
            entry = persistent_runs.setdefault(
                k,
                {
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "runs_with_confusion": 0,
                    "total_count": 0,
                    "best_examples": [],  # list[(filepath, p)]
                },
            )
            entry["runs_with_confusion"] = int(entry["runs_with_confusion"]) + 1
            entry["total_count"] = int(entry["total_count"]) + len(examples)
            # keep top examples (highest p) across all runs
            best = list(entry["best_examples"])
            best.extend(examples[:5])
            best.sort(key=lambda x: x[1], reverse=True)
            entry["best_examples"] = best[:10]

    # Sort summaries by top1 (then top1_covered, then top3)
    summaries.sort(
        key=lambda r: (
            float(r["top1_pct"]),
            float(r["top1_covered_pct"]),
            float(r["top3_pct"]),
        ),
        reverse=True,
    )

    # Write summary CSV
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(summaries[0].keys()) if summaries else []
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summaries)

    # Write summary JSON (includes timestamp)
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": stamp,
                "n_runs": total_runs,
                "summary_csv": str(summary_csv),
                "summaries": summaries,
            },
            f,
            indent=2,
        )

    # Persistent confusions: sort by runs_with_confusion then total_count
    persistent = list(persistent_runs.values())
    persistent.sort(
        key=lambda e: (int(e["runs_with_confusion"]), int(e["total_count"])),
        reverse=True,
    )

    with persistent_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "true_label",
                "pred_label",
                "runs_with_confusion",
                "total_count",
                "top_examples",
            ],
        )
        w.writeheader()
        for e in persistent[:50]:
            top_examples = "; ".join(
                f"{Path(fp).name}@{p:.3f}" for fp, p in (e.get("best_examples") or [])
            )
            w.writerow(
                {
                    "true_label": e["true_label"],
                    "pred_label": e["pred_label"],
                    "runs_with_confusion": e["runs_with_confusion"],
                    "total_count": e["total_count"],
                    "top_examples": top_examples,
                }
            )

    print("\nDone.")
    print(f"- Summary CSV: {summary_csv}")
    print(f"- Summary JSON: {summary_json}")
    print(f"- Persistent confusions CSV: {persistent_csv}")


if __name__ == "__main__":
    main()

