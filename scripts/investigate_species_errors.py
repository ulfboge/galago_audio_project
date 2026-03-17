"""\
Inspect per-species errors for a 3-stage prediction CSV.

Focus: species with 0% (or low) top-1 accuracy by showing what they get
confused with and whether the true label appears in the raw top-3.

Usage:
  python scripts/investigate_species_errors.py --species Otolemur_crassicaudatus
  python scripts/investigate_species_errors.py --csv outputs/predictions/predictions_3stage_context_thr0.20.csv --species Paragalago_rondoensis
"""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = PROJECT_ROOT / "data" / "raw_audio"
DEFAULT_CSV = PROJECT_ROOT / "outputs" / "predictions" / "predictions_3stage_context_thr0.20.csv"

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
    true_label: str
    detector_result: str
    species_result: str
    top1_species: str
    top1_prob: float
    top2_species: str
    top2_prob: float
    top3_species: str
    top3_prob: float


def _f(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def load_rows(path: Path) -> list[Row]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    rows: list[Row] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "filepath",
            "detector_result",
            "species_result",
            "top1_species",
            "top1_prob",
            "top2_species",
            "top2_prob",
            "top3_species",
            "top3_prob",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing columns: {sorted(missing)}")

        for r in reader:
            fp = r["filepath"]
            rows.append(
                Row(
                    filepath=fp,
                    true_label=get_mapped_label_from_filepath(fp),
                    detector_result=r.get("detector_result", ""),
                    species_result=r.get("species_result", ""),
                    top1_species=r.get("top1_species", ""),
                    top1_prob=_f(r.get("top1_prob", "")),
                    top2_species=r.get("top2_species", ""),
                    top2_prob=_f(r.get("top2_prob", "")),
                    top3_species=r.get("top3_species", ""),
                    top3_prob=_f(r.get("top3_prob", "")),
                )
            )
    return rows


def parse_args() -> tuple[Path, str, int]:
    args = sys.argv[1:]
    csv_path = DEFAULT_CSV
    species = ""
    limit = 25

    i = 0
    while i < len(args):
        a = args[i]
        if a == "--csv":
            csv_path = Path(args[i + 1])
            i += 2
            continue
        if a == "--species":
            species = args[i + 1]
            i += 2
            continue
        if a == "--limit":
            limit = int(args[i + 1])
            i += 2
            continue
        raise SystemExit(f"ERROR: Unknown arg: {a}")

    if not species:
        raise SystemExit("ERROR: --species is required")

    return csv_path, species, limit


def main() -> None:
    csv_path, target_species, limit = parse_args()
    rows = load_rows(csv_path)

    # Focus only on rows where true_label == target_species
    rows_t = [r for r in rows if r.true_label == target_species]

    if not rows_t:
        print(f"No rows found for species: {target_species}")
        return

    passed = [r for r in rows_t if r.detector_result == "galago"]
    filtered = [r for r in rows_t if r.detector_result != "galago"]

    correct_top1 = [r for r in passed if r.species_result == r.true_label]
    uncertain = [r for r in passed if r.species_result == "uncertain"]
    confident_wrong = [
        r
        for r in passed
        if r.species_result not in {"uncertain", "error", "not_classified", "N/A", ""} and r.species_result != r.true_label
    ]
    top3_hits = [
        r for r in passed if r.true_label in {r.top1_species, r.top2_species, r.top3_species}
    ]

    print(f"Investigating: {target_species}")
    print("=" * 72)
    print(f"CSV: {csv_path}")
    print(f"Total files (true label): {len(rows_t)}")
    print(f"  Passed detector: {len(passed)}")
    print(f"  Filtered by detector: {len(filtered)}")
    print("")
    print("Accuracy (within detector-passed):")
    print(f"  Top-1 (thresholded species_result): {len(correct_top1)}/{len(passed)} ({(len(correct_top1)/len(passed)*100):.1f}%)")
    print(f"  Uncertain: {len(uncertain)}/{len(passed)} ({(len(uncertain)/len(passed)*100):.1f}%)")
    print(f"  Top-3 hit (raw top1/top2/top3): {len(top3_hits)}/{len(passed)} ({(len(top3_hits)/len(passed)*100):.1f}%)")

    # What does it get predicted as?
    c_species_result = Counter(r.species_result for r in passed)
    c_top1 = Counter(r.top1_species for r in passed)

    print("\nMost common thresholded outputs (species_result):")
    for lab, cnt in c_species_result.most_common(10):
        print(f"  - {lab:30s} {cnt:3d}")

    print("\nMost common raw top-1 labels (top1_species):")
    for lab, cnt in c_top1.most_common(10):
        print(f"  - {lab:30s} {cnt:3d}")

    # Confusions (confident wrong only)
    conf = Counter(r.species_result for r in confident_wrong)
    if conf:
        print("\nTop confusions (confident wrong only):")
        for lab, cnt in conf.most_common(10):
            print(f"  - {target_species} -> {lab}: {cnt}")
    else:
        print("\nTop confusions: none (no confident wrong predictions)")

    # Show examples: highest-confidence wrong top1_prob (raw)
    examples = sorted(passed, key=lambda r: r.top1_prob, reverse=True)
    print(f"\nTop {min(limit, len(examples))} examples (sorted by top1_prob):")
    print("-" * 72)
    for r in examples[:limit]:
        name = Path(r.filepath).name
        in_top3 = r.true_label in {r.top1_species, r.top2_species, r.top3_species}
        mark = "TOP3" if in_top3 else "MISS"
        print(
            f"{name:50s}  "
            f"species_result={r.species_result:18s}  "
            f"top1={r.top1_species:22s}@{r.top1_prob:5.3f}  "
            f"top2={r.top2_species:22s}@{r.top2_prob:5.3f}  "
            f"top3={r.top3_species:22s}@{r.top3_prob:5.3f}  "
            f"{mark}"
        )


if __name__ == "__main__":
    main()

