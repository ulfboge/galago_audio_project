"""
demo_webhook.py — Arkivera opt-in WAV och skicka HTTP-notis till utvecklaren.

Miljövariabler (sätt i .env eller i shell innan du startar demot):
  DEMO_FEEDBACK_WEBHOOK_URL   URL att POST:a JSON till när feedback sparas.
                               Fungerar med Slack incoming webhook, Discord webhook,
                               eller en egen endpoint. Lämna tom för att stänga av.
  DEMO_WAV_ARCHIVE_DIR        Katalog för WAV-kopior (opt-in).
                               Standard: demo/logs/wav_archive/

Inget av detta aktiveras automatiskt — webhook skickas bara om URL:en är satt,
och WAV sparas bara om användaren kryssar i rutan i demot.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

_DEFAULT_ARCHIVE = Path(__file__).resolve().parent / "logs" / "wav_archive"


def _archive_dir() -> Path:
    env = os.environ.get("DEMO_WAV_ARCHIVE_DIR", "").strip()
    return Path(env) if env else _DEFAULT_ARCHIVE


def archive_wav(tmp_path: Path, ts_utc: str) -> Path | None:
    """
    Kopiera tmp_path till arkivkatalogen med ett unikt namn.
    Filnamnet: <ts>_<hash>_<original_stem>.wav
    Returnerar dest-path vid lyckat, None annars.
    """
    try:
        dest_dir = _archive_dir()
        dest_dir.mkdir(parents=True, exist_ok=True)
        # Unikt suffix: de 8 första hex-tecknen av SHA-1 på originalnamnet
        h = hashlib.sha1(tmp_path.name.encode("utf-8")).hexdigest()[:8]
        # Tidstämpel på formen 20260327T142301Z (säker för filnamn)
        safe_ts = ts_utc[:19].replace(":", "").replace("-", "").replace("T", "T")
        stem = tmp_path.stem[:30]  # trunkera långa namn
        dest = dest_dir / f"{safe_ts}_{h}_{stem}.wav"
        shutil.copy2(tmp_path, dest)
        return dest
    except Exception as exc:
        print(f"[demo_webhook] WAV-arkivering misslyckades: {exc}", flush=True)
        return None


def _outgoing_json_for_service(url: str, payload: dict) -> dict:
    """
    Slack/Discord incoming webhooks kräver särskild form; annars skickas payload oförändrad.
    """
    u = url.lower()
    if "hooks.slack.com" in u:
        agree = payload.get("observer_agrees_label", "?")
        believed = payload.get("observer_believed_species") or "—"
        notes = payload.get("observer_notes") or "—"
        archived = payload.get("wav_archived") or "—"
        lines = [
            "*Galago demo — observer-feedback*",
            f"• Fil: `{payload.get('wav_filename', '?')}`",
            f"• Modell (output): `{payload.get('model_species_result', '?')}`",
            f"• Top-1: `{payload.get('model_top1_species', '?')}`",
            f"• Profil: `{payload.get('profile', '?')}`",
            f"• Observatör håller med: `{agree}`",
            f"• Observatör trodde art: `{believed}`",
            f"• Anteckning: {notes}",
            f"• WAV arkiverad: `{archived}`",
        ]
        return {"text": "\n".join(lines)}
    if "discord.com/api/webhooks" in u or "discordapp.com/api/webhooks" in u:
        text = (
            "**Galago demo — feedback**\n"
            f"Fil: `{payload.get('wav_filename')}`\n"
            f"Modell: `{payload.get('model_species_result')}` | "
            f"trodde: `{payload.get('observer_believed_species')}` | "
            f"håller med: `{payload.get('observer_agrees_label')}`\n"
            f"Arkiv: `{payload.get('wav_archived')}`"
        )
        return {"content": text[:2000]}
    return payload


def send_webhook(payload: dict) -> bool:
    """
    POST payload som JSON till DEMO_FEEDBACK_WEBHOOK_URL.
    Returnerar True om servern svarade < 400, annars False.
    Tyst vid timeout / nätverksfel (loggas till stdout).
    """
    url = os.environ.get("DEMO_FEEDBACK_WEBHOOK_URL", "").strip()
    if not url:
        return False
    try:
        body_obj = _outgoing_json_for_service(url, payload)
        data = json.dumps(body_obj, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            ok = resp.status < 400
            if not ok:
                print(f"[demo_webhook] Webhook HTTP {resp.status}", flush=True)
            return ok
    except Exception as exc:
        print(f"[demo_webhook] Webhook misslyckades: {exc}", flush=True)
        return False


def notify_feedback(
    entry: dict,
    wav_tmp_path: "Path | None",
    save_wav: bool,
) -> "tuple[bool, Path | None]":
    """
    Arkivera WAV (om save_wav=True och tmp_path finns) och skicka webhook.

    Args:
        entry:        Den sparade feedback-raden (från demo_observer_feedback).
        wav_tmp_path: Gradios temporära WAV-fil (None om okänd).
        save_wav:     True om användaren kryssat i opt-in-rutan.

    Returns:
        (webhook_sent: bool, archived_path: Path | None)
    """
    archived: "Path | None" = None
    ts = entry.get("ts_utc") or datetime.now(timezone.utc).isoformat()

    if save_wav and wav_tmp_path is not None and wav_tmp_path.exists():
        archived = archive_wav(wav_tmp_path, ts)

    # Bygg payload: samma fält som entry + arkivinfo
    payload = dict(entry)
    payload["wav_archived"] = archived.name if archived else None
    payload["source"] = "galago-demo"

    sent = send_webhook(payload)
    return sent, archived
