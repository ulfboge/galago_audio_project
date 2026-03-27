"""Append-only JSONL: observer says if demo prediction matched their expectation."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent / "logs" / "observer_feedback.jsonl"


def append_observer_feedback(
    snapshot: dict | None,
    agrees: str,
    believed_species: str,
    notes: str,
    *,
    out_path: Path | None = None,
    wav_tmp_path: "Path | None" = None,
    save_wav: bool = False,
) -> str:
    """
    Write one line to observer_feedback.jsonl. Returns short status for UI.
    `snapshot` comes from the last successful predict() in the demo (see upload_predict_gradio).

    Optional:
        wav_tmp_path: Gradio's temporary WAV file path (for opt-in archiving).
        save_wav:     True if the user ticked the consent checkbox.
                      When True, a copy is saved to DEMO_WAV_ARCHIVE_DIR and a
                      webhook notification is sent to DEMO_FEEDBACK_WEBHOOK_URL
                      (if that env-var is set).
    """
    if not snapshot or not snapshot.get("wav_filename"):
        return "Ingen körning att koppla till - kör **Kör förutsägelse** först."

    path = out_path or DEFAULT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    notes_s = (notes or "").strip() or None
    empty_marker = "(tomt / valde jag inte)"
    believed = (believed_species or "").strip()
    if not believed or believed == empty_marker:
        believed_clean = None
    else:
        believed_clean = believed

    if agrees == "Ja":
        agrees_flag: bool | None = True
    elif agrees == "Nej":
        agrees_flag = False
    else:
        agrees_flag = None

    entry = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "wav_filename": snapshot.get("wav_filename"),
        "model_species_result": snapshot.get("species_result"),
        "model_species_prob": snapshot.get("species_prob"),
        "model_top1_species": snapshot.get("top1_species"),
        "profile": snapshot.get("profile"),
        "observer_agrees_with_model": agrees_flag,
        "observer_agrees_label": agrees,
        "observer_believed_species": believed_clean,
        "observer_notes": notes_s,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # --- Webhook + opt-in WAV-arkivering ---
    status_parts = [f"Sparat i {path.name}"]
    try:
        from demo_webhook import notify_feedback  # pylint: disable=import-outside-toplevel
        sent, archived = notify_feedback(entry, wav_tmp_path, save_wav)
        if save_wav and archived:
            status_parts.append(f"WAV arkiverad ({archived.name})")
        elif save_wav and not archived:
            status_parts.append("WAV-arkivering misslyckades (se logg)")
        if sent:
            status_parts.append("Webhook skickad OK")
    except ImportError:
        pass  # demo_webhook.py inte tillgänglig — fortsätt tyst

    return " | ".join(status_parts)
