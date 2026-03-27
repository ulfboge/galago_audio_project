"""Append-only JSONL log for Gradio demo runs (filename + predictions + acoustic ranking)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_LOG_PATH = Path(__file__).resolve().parent / "logs" / "predictions.jsonl"


def _parse_acoustic_top10_string(raw: object) -> list[dict]:
    """
    Parse pipeline `acoustic_top10` string: 'Species_a 0.123 · Species_b 0.098 · ...'
    Returns [{"rank": 1, "species": "...", "prob": "0.123"}, ...]
    """
    if raw is None:
        return []
    s = str(raw).strip()
    if not s or s == "N/A":
        return []
    chunks = [c.strip() for c in s.split("·") if c.strip()]
    out: list[dict] = []
    for i, c in enumerate(chunks, 1):
        parts = c.rsplit(None, 1)
        if len(parts) == 2:
            sp, pr = parts[0], parts[1]
            out.append({"rank": i, "species": sp, "prob": pr})
        else:
            out.append({"rank": i, "species": c, "prob": None})
    return out


def append_prediction_log(
    row: dict,
    *,
    wav_name: str,
    profile: str,
    no_location: bool,
    lat: float | None,
    lon: float | None,
    infer_filename: bool,
    site_preset_label: str = "",
    log_path: Path | None = None,
) -> None:
    path = log_path or DEFAULT_LOG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    acoustic_parsed = _parse_acoustic_top10_string(row.get("acoustic_top10"))
    ac1 = acoustic_parsed[0] if len(acoustic_parsed) > 0 else {}
    ac2 = acoustic_parsed[1] if len(acoustic_parsed) > 1 else {}
    ac3 = acoustic_parsed[2] if len(acoustic_parsed) > 2 else {}

    entry = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "wav_filename": wav_name,
        "profile": profile,
        "no_location": no_location,
        "lat": lat,
        "lon": lon,
        "infer_filename": infer_filename,
        "site_preset_label": site_preset_label or None,
        "species_result": row.get("species_result"),
        "species_prob": row.get("species_prob"),
        "top1_species": row.get("top1_species"),
        "top1_prob": row.get("top1_prob"),
        "top2_species": row.get("top2_species"),
        "top2_prob": row.get("top2_prob"),
        "top3_species": row.get("top3_species"),
        "top3_prob": row.get("top3_prob"),
        "detector_result": row.get("detector_result"),
        "detector_prob": row.get("detector_prob"),
        "postprocess_mode": row.get("postprocess_mode"),
        "location_used": row.get("location_used"),
        "location_status": row.get("location_status"),
        # CNN acoustic ranking (före plats-omrankning); jämför Rondo vs Pugu m.m.
        "acoustic_top10_raw": row.get("acoustic_top10")
        if row.get("acoustic_top10") not in (None, "", "N/A")
        else None,
        "acoustic_top10_parsed": acoustic_parsed if acoustic_parsed else None,
        "acoustic_1_species": ac1.get("species"),
        "acoustic_1_prob": ac1.get("prob"),
        "acoustic_2_species": ac2.get("species"),
        "acoustic_2_prob": ac2.get("prob"),
        "acoustic_3_species": ac3.get("species"),
        "acoustic_3_prob": ac3.get("prob"),
    }

    # Var ligger Paragalago_rondoensis i akustisk ranking (om den finns bland top-10)
    rondo_rank = next(
        (x["rank"] for x in acoustic_parsed if x.get("species") == "Paragalago_rondoensis"),
        None,
    )
    entry["acoustic_rank_Paragalago_rondoensis"] = rondo_rank

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
