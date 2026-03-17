"""
Build a mapping from model class labels -> IUCN per-species GeoJSON file.

Why:
- Model labels are underscore-separated (e.g. "Paragalago_rondoensis")
- IUCN range data uses space-separated binomials in attributes (e.g. "Paragalago rondoensis")
- Some labels are synonyms / legacy spellings (e.g. Galagoides_demidovii vs IUCN "Galagoides demidoff")

This script reads:
- models/all_species/class_names_*.json (list of model labels)
- data/iucn/ranges_geojson/by_species/_index.csv (created by the zip splitter)

And writes:
- data/iucn_geojson_index_by_label.json (label -> geojson relpath + metadata)
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_OVERRIDES: Dict[str, Dict[str, str]] = {
    # Model label -> {iucn_sci_name, geojson_file}
    "Galagoides_demidovii": {
        "iucn_sci_name": "Galagoides demidoff",
        "geojson_file": "Galagoides_demidoff.geojson",
    },
    # Model currently uses a placeholder label; in this batch, IUCN provides the named species.
    "Galagoides_sp_nov": {
        "iucn_sci_name": "Galagoides kumbirensis",
        "geojson_file": "Galagoides_kumbirensis.geojson",
    },
}


@dataclass(frozen=True)
class IucnIndexRow:
    sci_name: str
    geojson_file: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--class-names-json",
        type=str,
        default=str(PROJECT_ROOT / "models" / "all_species" / "class_names_19.json"),
        help="Path to model class names JSON (list of labels).",
    )
    p.add_argument(
        "--iucn-index-csv",
        type=str,
        default=str(
            PROJECT_ROOT
            / "data"
            / "iucn"
            / "ranges_geojson"
            / "by_species"
            / "_index.csv"
        ),
        help="Path to _index.csv created by the IUCN zip splitter.",
    )
    p.add_argument(
        "--by-species-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "iucn" / "ranges_geojson" / "by_species"),
        help="Directory containing per-species GeoJSONs.",
    )
    p.add_argument(
        "--out-json",
        type=str,
        default=str(PROJECT_ROOT / "data" / "iucn_geojson_index_by_label.json"),
        help="Output JSON to write (recommended: tracked path outside data/iucn/).",
    )
    p.add_argument(
        "--overrides-json",
        type=str,
        default="",
        help="Optional JSON file with mapping overrides (merged over built-ins).",
    )
    p.add_argument(
        "--write-alias-files",
        action="store_true",
        help="If set, create alias .geojson copies for override targets (useful for direct label->file lookup).",
    )
    return p.parse_args()


def load_class_names(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise ValueError(f"class_names_json must be a JSON list[str]; got: {type(data)}")
    return data


def load_iucn_index_csv(path: Path) -> List[IucnIndexRow]:
    rows: List[IucnIndexRow] = []
    # NOTE: PowerShell/Excel may write UTF-8 with BOM. Use utf-8-sig to strip BOM.
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        def _clean_fieldname(s: str) -> str:
            s = s.strip()
            # Some CSV writers quote header names; strip surrounding quotes if present.
            if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
                s = s[1:-1]
            return s.strip()

        cleaned = [_clean_fieldname(fn) for fn in fieldnames]
        # Remap reader fieldnames in-place so row dicts use cleaned keys.
        reader.fieldnames = cleaned

        if "sci_name" not in cleaned or "file" not in cleaned:
            raise ValueError(f"Expected columns sci_name,file in {path}. Got: {fieldnames}")
        for r in reader:
            sci_name = (r.get("sci_name") or "").strip()
            geojson_file = (r.get("file") or "").strip()
            if not sci_name or not geojson_file:
                continue
            rows.append(IucnIndexRow(sci_name=sci_name, geojson_file=geojson_file))
    return rows


def normalize_space_name(s: str) -> str:
    return " ".join(s.strip().split()).lower()


def normalize_label_name(label: str) -> str:
    # "Paragalago_rondoensis" -> "paragalago rondoensis"
    return normalize_space_name(label.replace("_", " "))


def build_mapping(
    class_labels: List[str],
    iucn_rows: List[IucnIndexRow],
    by_species_dir: Path,
    overrides: Dict[str, Dict[str, str]],
    write_alias_files: bool,
) -> Tuple[Dict[str, Any], List[str]]:
    iucn_by_sciname_norm: Dict[str, IucnIndexRow] = {
        normalize_space_name(r.sci_name): r for r in iucn_rows
    }
    existing_files = {p.name for p in by_species_dir.glob("*.geojson")}

    out: Dict[str, Any] = {}
    missing: List[str] = []

    for label in class_labels:
        label_file = f"{label}.geojson"
        chosen: Optional[IucnIndexRow] = None
        note: str = ""

        # 1) Direct filename match (best case).
        if label_file in existing_files:
            # Look up sci_name (if possible) for metadata; if not, leave blank.
            sci_guess = normalize_label_name(label)
            chosen = iucn_by_sciname_norm.get(sci_guess)
            if chosen is None:
                chosen = IucnIndexRow(sci_name=label.replace("_", " "), geojson_file=label_file)
            note = "direct_filename_match"
        else:
            # 2) Match by sci_name (label -> spaced binomial).
            sci_guess = normalize_label_name(label)
            chosen = iucn_by_sciname_norm.get(sci_guess)
            if chosen is not None:
                note = "matched_by_sci_name"
            else:
                # 3) Overrides (synonyms / placeholder labels).
                ov = overrides.get(label)
                if ov:
                    ov_file = ov.get("geojson_file", "").strip()
                    ov_sci = ov.get("iucn_sci_name", "").strip()
                    if ov_file:
                        chosen = IucnIndexRow(sci_name=ov_sci or label.replace("_", " "), geojson_file=ov_file)
                        note = "override"

                        if write_alias_files:
                            # Create a copy so consumers can load "{label}.geojson" directly.
                            src = by_species_dir / ov_file
                            dst = by_species_dir / label_file
                            if src.exists() and not dst.exists():
                                dst.write_bytes(src.read_bytes())
                    else:
                        chosen = None

        if chosen is None:
            missing.append(label)
            out[label] = {
                "iucn_sci_name": None,
                "geojson_file": None,
                "geojson_relpath": None,
                "note": "missing",
            }
            continue

        relpath = str(Path("data") / "iucn" / "ranges_geojson" / "by_species" / chosen.geojson_file)
        out[label] = {
            "iucn_sci_name": chosen.sci_name,
            "geojson_file": chosen.geojson_file,
            "geojson_relpath": relpath.replace("\\", "/"),
            "note": note,
        }

    return out, missing


def main() -> int:
    args = parse_args()

    class_names_path = Path(args.class_names_json)
    iucn_index_csv = Path(args.iucn_index_csv)
    by_species_dir = Path(args.by_species_dir)
    out_json = Path(args.out_json)

    overrides = dict(DEFAULT_OVERRIDES)
    if args.overrides_json:
        p = Path(args.overrides_json)
        with p.open("r", encoding="utf-8") as f:
            user_overrides = json.load(f)
        if not isinstance(user_overrides, dict):
            raise ValueError("--overrides-json must be a JSON object")
        for k, v in user_overrides.items():
            if isinstance(k, str) and isinstance(v, dict):
                overrides[k] = {**overrides.get(k, {}), **v}

    if not class_names_path.exists():
        raise FileNotFoundError(f"Missing class names: {class_names_path}")
    if not iucn_index_csv.exists():
        raise FileNotFoundError(f"Missing IUCN index CSV: {iucn_index_csv}")
    if not by_species_dir.exists():
        raise FileNotFoundError(f"Missing by-species directory: {by_species_dir}")

    class_labels = load_class_names(class_names_path)
    iucn_rows = load_iucn_index_csv(iucn_index_csv)

    mapping, missing = build_mapping(
        class_labels=class_labels,
        iucn_rows=iucn_rows,
        by_species_dir=by_species_dir,
        overrides=overrides,
        write_alias_files=bool(args.write_alias_files),
    )

    payload = {
        "metadata": {
            "class_names_json": str(class_names_path).replace("\\", "/"),
            "iucn_index_csv": str(iucn_index_csv).replace("\\", "/"),
            "by_species_dir": str(by_species_dir).replace("\\", "/"),
            "missing_count": len(missing),
            "missing_labels": missing,
            "overrides": overrides,
        },
        "label_to_geojson": mapping,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote: {out_json}")
    if missing:
        print("Missing labels:")
        for m in missing:
            print(f"  - {m}")
    else:
        print("All labels mapped successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

