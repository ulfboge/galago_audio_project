"""
Create a manual-review shortlist for high-impact species-confusion clips.

Prioritizes files where the model is either:
1) wrong within/near the acoustic cluster (dwarf galagos, rondoensis, Otolemur spp.), or
2) uncertain on those species.

Usage:
  python scripts/make_cluster_review_shortlist.py ^
    --csv outputs/predictions/predictions_3stage_malawi_balanced.csv ^
    --csv outputs/predictions/predictions_3stage_tanzania_tuned_strong_context.csv ^
    --out-csv outputs/evaluation/cluster_review_shortlist.csv ^
    --top-n 60

  Emit ingest-ready stubs (same columns as data/relabels/relabels_template.csv):
    --emit-relabel-stubs data/relabels/relabels_from_cluster_shortlist.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent))
from relabel_filename_hints import species_hint_from_wav_path


if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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

# Species in the dwarf-galago / Paragalago acoustic confusion cluster.
CLUSTER = {
    "Galagoides_demidovii",
    "Galagoides_sp_nov",
    "Otolemur_crassicaudatus",
    "Otolemur_garnettii",
    "Paragalago_granti",
    "Paragalago_orinus",
    "Paragalago_rondoensis",
    "Paragalago_zanzibaricus",
}

# Dominant pairwise confusions in TZ / Kenya Paragalago cluster.
RONDO_DEMI_PAIR = frozenset({"Paragalago_rondoensis", "Galagoides_demidovii"})
GRANTI_RONDO_PAIR = frozenset(
    {"Paragalago_granti", "Paragalago_rondoensis", "Paragalago_zanzibaricus"}
)


def parse_args(argv: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "csvs": [],
        "out_csv": str(PROJECT_ROOT / "outputs" / "evaluation" / "cluster_review_shortlist.csv"),
        "top_n": 60,
        "emit_relabel_stubs": None,
    }
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--csv":
            out["csvs"].append(argv[i + 1])
            i += 2
            continue
        if a == "--out-csv":
            out["out_csv"] = argv[i + 1]
            i += 2
            continue
        if a == "--top-n":
            out["top_n"] = int(argv[i + 1])
            i += 2
            continue
        if a == "--emit-relabel-stubs":
            out["emit_relabel_stubs"] = argv[i + 1]
            i += 2
            continue
        raise SystemExit(f"ERROR: Unknown arg: {a}")

    if not out["csvs"]:
        raise SystemExit("ERROR: Provide at least one --csv")
    return out


def true_label_from_filepath(filepath: str) -> str:
    p = Path(filepath)
    try:
        rel = p.resolve().relative_to(AUDIO_DIR.resolve())
        src_folder = rel.parts[0] if len(rel.parts) > 1 else p.parent.name
        return LABEL_MAP.get(src_folder, src_folder)
    except Exception:
        return LABEL_MAP.get(p.parent.name, p.parent.name)


def safe_float(v: str, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def review_priority(row: Dict[str, str], true_label: str) -> float:
    """
    Larger score = higher priority for human review.
    """
    pred = row.get("top1_species", "N/A")
    p1 = safe_float(row.get("top1_prob", "0"), 0.0)
    species_result = row.get("species_result", "")

    true_in_cluster = true_label in CLUSTER
    pred_in_cluster = pred in CLUSTER
    wrong = pred != true_label
    uncertain = species_result == "uncertain"

    score = 0.0

    # Main target: wrong decisions in/near the confusion cluster
    if wrong and (true_in_cluster or pred_in_cluster):
        score += 100.0
        score += p1 * 50.0  # high-confidence errors are especially useful

    # Uncertain on cluster species are also useful labeling targets
    if uncertain and true_in_cluster:
        score += 70.0
        score += (1.0 - p1) * 20.0

    # Extra bump for the hardest Tanzania issue
    if true_label == "Paragalago_rondoensis" and pred == "Galagoides_sp_nov":
        score += 30.0

    # Dominant pairwise confusion after TZ sp.nov handling: rondoensis <-> demidovii
    if true_label in RONDO_DEMI_PAIR and pred in RONDO_DEMI_PAIR and true_label != pred:
        score += 40.0

    # granti ↔ rondoensis / zanzibaricus (incremental calls etc.)
    if true_label in GRANTI_RONDO_PAIR and pred in GRANTI_RONDO_PAIR and true_label != pred:
        score += 45.0

    return score


def write_relabel_stub_csv(path: Path, shortlist: List[Dict[str, Any]]) -> None:
    """ingest_relabels.py format; species starts as folder label — curator edits if wrong."""
    relabel_fields = ["wav_path", "start_sec", "end_sec", "species", "population", "notes"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=relabel_fields)
        w.writeheader()
        for row in shortlist:
            fp = row["filepath"]
            try:
                wav_path = str(Path(fp).resolve())
            except Exception:
                wav_path = fp
            true_label = row["true_label"]
            sp_hint, hint_note = species_hint_from_wav_path(Path(wav_path), true_label)
            top1 = row.get("top1_species", "")
            sr = row.get("species_result", "")
            p1 = row.get("top1_prob", "")
            notes = (
                f"stub from cluster shortlist; folder_label={true_label}; "
                f"output={sr}; top1={top1} p1={p1}; EDIT species + add start/end if needed"
            )
            if hint_note:
                notes = f"{notes}; {hint_note}"
            w.writerow(
                {
                    "wav_path": wav_path,
                    "start_sec": "",
                    "end_sec": "",
                    "species": sp_hint,
                    "population": "",
                    "notes": notes[:500],
                }
            )


def main() -> int:
    opts = parse_args(sys.argv[1:])
    candidates: List[Dict[str, Any]] = []

    for csv_path_s in opts["csvs"]:
        csv_path = Path(csv_path_s)
        if not csv_path.exists():
            print(f"WARNING: Missing CSV, skipping: {csv_path}")
            continue

        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                fp = r.get("filepath", "")
                if not fp:
                    continue
                true_label = true_label_from_filepath(fp)
                score = review_priority(r, true_label)
                if score <= 0:
                    continue

                candidates.append(
                    {
                        "priority_score": round(score, 3),
                        "source_csv": str(csv_path),
                        "filepath": fp,
                        "true_label": true_label,
                        "species_result": r.get("species_result", ""),
                        "top1_species": r.get("top1_species", ""),
                        "top1_prob": r.get("top1_prob", ""),
                        "top2_species": r.get("top2_species", ""),
                        "top2_prob": r.get("top2_prob", ""),
                        "top3_species": r.get("top3_species", ""),
                        "top3_prob": r.get("top3_prob", ""),
                        "location_status": r.get("location_status", ""),
                        "lat": r.get("lat", ""),
                        "lon": r.get("lon", ""),
                    }
                )

    # One row per WAV: keep highest priority across overlapping CSVs
    best_by_fp: Dict[str, Dict[str, Any]] = {}
    for row in candidates:
        fp = row["filepath"]
        if fp not in best_by_fp or row["priority_score"] > best_by_fp[fp]["priority_score"]:
            best_by_fp[fp] = row

    rows_out = sorted(best_by_fp.values(), key=lambda x: x["priority_score"], reverse=True)
    rows_out = rows_out[: int(opts["top_n"])]

    out_csv = Path(str(opts["out_csv"]))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows_out:
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            writer.writeheader()
            writer.writerows(rows_out)
    else:
        out_csv.write_text("priority_score,source_csv,filepath,true_label\n", encoding="utf-8")

    print(f"Wrote {len(rows_out)} shortlist rows to: {out_csv} (deduped by filepath)")

    stub_path = opts.get("emit_relabel_stubs")
    if stub_path and rows_out:
        p = Path(stub_path)
        write_relabel_stub_csv(p, rows_out)
        print(f"Wrote {len(rows_out)} ingest stubs to: {p}")
    elif stub_path and not rows_out:
        print("NOTE: --emit-relabel-stubs given but shortlist empty; not writing stubs.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

