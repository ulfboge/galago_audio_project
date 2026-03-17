"""\
Compare confusion patterns across multiple 3-stage prediction CSVs.

This is intended for comparing different *classifier thresholds* where only
`species_result` gating changes while `top1_species/top1_prob` stay the same.

Default inputs:
- outputs/predictions/predictions_3stage_context_thr0.20.csv
- outputs/predictions/predictions_3stage_context_thr0.25.csv
- outputs/predictions/predictions_3stage_context_thr0.30.csv

Outputs:
- Coverage/uncertain rates per threshold
- Top confusions at the lowest threshold with counts at higher thresholds
- Confusions that persist at the highest threshold (high-confidence mistakes)

Usage:
  python scripts/compare_confusions_by_threshold.py
  python scripts/compare_confusions_by_threshold.py --csv a.csv --csv b.csv --csv c.csv
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = PROJECT_ROOT / "data" / "raw_audio"

DEFAULT_CSVS = [
    PROJECT_ROOT / "outputs" / "predictions" / "predictions_3stage_context_thr0.20.csv",
    PROJECT_ROOT / "outputs" / "predictions" / "predictions_3stage_context_thr0.25.csv",
    PROJECT_ROOT / "outputs" / "predictions" / "predictions_3stage_context_thr0.30.csv",
]

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
class Pred:
    filepath: str
    true_label: str
    detector_result: str
    species_result: str
    top1_species: str
    top1_prob: float


def parse_args() -> list[Path]:
    args = sys.argv[1:]
    csvs: list[Path] = []

    i = 0
    while i < len(args):
        a = args[i]
        if a == "--csv":
            if i + 1 >= len(args):
                raise SystemExit("ERROR: --csv requires a path")
            csvs.append(Path(args[i + 1]))
            i += 2
            continue
        raise SystemExit(f"ERROR: Unknown arg: {a}")

    return csvs or DEFAULT_CSVS


def load_csv(path: Path) -> dict[str, Pred]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    by_fp: dict[str, Pred] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            fp = r.get("filepath", "")
            if not fp:
                continue
            true_label = get_mapped_label_from_filepath(fp)
            detector_result = r.get("detector_result", "")
            species_result = r.get("species_result", "")
            top1_species = r.get("top1_species", "")

            try:
                top1_prob = float(r.get("top1_prob", "0") or 0)
            except ValueError:
                top1_prob = 0.0

            by_fp[fp] = Pred(
                filepath=fp,
                true_label=true_label,
                detector_result=detector_result,
                species_result=species_result,
                top1_species=top1_species,
                top1_prob=top1_prob,
            )

    return by_fp


def summarize(preds: dict[str, Pred]) -> dict[str, int]:
    total = len(preds)
    galago = sum(1 for p in preds.values() if p.detector_result == "galago")
    uncertain = sum(
        1
        for p in preds.values()
        if p.detector_result == "galago" and p.species_result == "uncertain"
    )
    confident = sum(
        1
        for p in preds.values()
        if p.detector_result == "galago" and p.species_result not in {"uncertain", "error", "not_classified", "N/A", ""}
    )
    correct = sum(
        1
        for p in preds.values()
        if p.detector_result == "galago" and p.species_result == p.true_label
    )

    return {
        "total": total,
        "galago": galago,
        "confident": confident,
        "uncertain": uncertain,
        "correct": correct,
    }


def confusion_counts(preds: dict[str, Pred]) -> dict[tuple[str, str], int]:
    conf: dict[tuple[str, str], int] = defaultdict(int)
    for p in preds.values():
        if p.detector_result != "galago":
            continue
        if p.species_result in {"uncertain", "error", "not_classified", "N/A", ""}:
            continue
        if p.species_result != p.true_label:
            conf[(p.true_label, p.species_result)] += 1
    return conf


def confusion_examples(preds: dict[str, Pred], true_label: str, pred_label: str) -> list[Pred]:
    examples: list[Pred] = []
    for p in preds.values():
        if p.detector_result != "galago":
            continue
        if p.species_result != pred_label:
            continue
        if p.true_label != true_label:
            continue
        # only include confident predictions (not gated)
        if p.species_result in {"uncertain", "error", "not_classified", "N/A", ""}:
            continue
        examples.append(p)
    examples.sort(key=lambda x: x.top1_prob, reverse=True)
    return examples


def format_pct(num: int, den: int) -> str:
    if den <= 0:
        return "0.0%"
    return f"{(num/den*100):.1f}%"


def main() -> None:
    csv_paths = parse_args()

    loaded: list[tuple[str, dict[str, Pred]]] = []
    for p in csv_paths:
        loaded.append((p.name, load_csv(p)))

    # Build intersection of filepaths present in all csvs
    common = set(loaded[0][1].keys())
    for _, d in loaded[1:]:
        common &= set(d.keys())

    if not common:
        raise SystemExit("ERROR: No overlapping filepaths across provided CSVs")

    # Restrict to common to avoid accidental mismatches
    loaded_common: list[tuple[str, dict[str, Pred]]] = []
    for name, d in loaded:
        loaded_common.append((name, {k: d[k] for k in common}))

    print("Confusion comparison across thresholds")
    print("=" * 72)

    # Coverage summary
    print("\nCoverage summary:")
    for name, d in loaded_common:
        s = summarize(d)
        print(
            f"- {name}: "
            f"confident {s['confident']}/{s['galago']} ({format_pct(s['confident'], s['galago'])}), "
            f"uncertain {s['uncertain']}/{s['galago']} ({format_pct(s['uncertain'], s['galago'])}), "
            f"top1 {s['correct']}/{s['galago']} ({format_pct(s['correct'], s['galago'])})"
        )

    # Confusions per csv
    confs = [(name, confusion_counts(d)) for name, d in loaded_common]

    # Pick a baseline = first CSV
    base_name, base_conf = confs[0]

    # Top confusions at baseline
    base_sorted = sorted(base_conf.items(), key=lambda kv: kv[1], reverse=True)

    print("\nTop confusions at baseline (showing counts across all CSVs):")
    print("-" * 72)
    header = "true -> pred".ljust(62)
    for name, _ in confs:
        header += f" {name.split('_thr')[-1].replace('.csv',''):>8}"
    print(header)
    print("-" * 72)

    for (true_label, pred_label), base_count in base_sorted[:15]:
        row = f"{true_label} -> {pred_label}".ljust(62)
        for _, c in confs:
            row += f" {c.get((true_label, pred_label), 0):8d}"
        print(row)

    # Confusions that persist at the highest threshold
    high_name, high_conf = confs[-1]
    high_sorted = sorted(high_conf.items(), key=lambda kv: kv[1], reverse=True)

    print(f"\nConfusions that persist at highest threshold ({high_name}):")
    print("-" * 72)
    header = "true -> pred".ljust(62)
    for name, _ in confs:
        header += f" {name.split('_thr')[-1].replace('.csv',''):>8}"
    print(header)
    print("-" * 72)

    for (true_label, pred_label), _ in high_sorted[:10]:
        row = f"{true_label} -> {pred_label}".ljust(62)
        for _, c in confs:
            row += f" {c.get((true_label, pred_label), 0):8d}"
        print(row)

    # Show the exact WAVs behind the top persistent confusions (using baseline CSV's view)
    base_preds = loaded_common[0][1]
    print(f"\nExamples for top persistent confusions (from baseline CSV: {loaded_common[0][0]}):")
    print("-" * 72)
    for (true_label, pred_label), count in high_sorted[:5]:
        print(f"\n{true_label} -> {pred_label} ({count} files at highest threshold)")
        ex = confusion_examples(base_preds, true_label, pred_label)[:10]
        for p in ex:
            print(f"  - {Path(p.filepath).name} (p={p.top1_prob:.3f})")


if __name__ == "__main__":
    main()
