"""
Local demo: upload a WAV → run 3-stage galago pipeline → show best guess + top-3.

Models load once and stay in memory (see cached_predictor.py).

Requires: pip install -r requirements-demo.txt  (or: pip install gradio)
Run from repo root:
  python demo/upload_predict_gradio.py

Port: defaults to 7860, or set GRADIO_SERVER_PORT. If 7860 is busy, tries 7861–7869.

The map runs in an iframe at `/galago-map` (FastAPI route + vendored Leaflet).
Clicks update lat/long fields via same-origin parent DOM — or use **Förvald plats** / manual entry.

Predictions append to demo/logs/predictions.jsonl when logging is on (see demo_prediction_log.py):
operational top-3, full acoustic_top10_parsed, acoustic_1..3_*, acoustic_rank_Paragalago_rondoensis.

Observer feedback (optional): after a run, answer whether the model matched your expectation; saved to
demo/logs/observer_feedback.jsonl (see demo_observer_feedback.py).

UI: **Nästa fil** clears upload + result (keeps plats/profil/sessionslista). **Töm sessionslista** clears
the in-memory table of filename ↔ guess for this browser tab.
"""
from __future__ import annotations

import json
import os
import re
import time
import traceback
from collections import deque
from pathlib import Path


def _load_dotenv_if_present() -> None:
    """Läs repots .env om python-dotenv finns (slipper manuell export i shell)."""
    root = Path(__file__).resolve().parents[1]
    p = root / ".env"
    if not p.is_file():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(p)
    except ImportError:
        pass


_load_dotenv_if_present()

# Compatibility shim: gradio 4.44.0 imports HfFolder which was removed in
# huggingface_hub 0.26+. HF Spaces pre-installs huggingface_hub>=0.30, so we
# stub it back before gradio is imported inside main().
try:
    from huggingface_hub import HfFolder as _HfFolder  # noqa: F401
except ImportError:
    import huggingface_hub as _hfhub

    class _HfFolderStub:
        @staticmethod
        def get_token() -> "str | None":
            import os
            return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    _hfhub.HfFolder = _HfFolderStub  # type: ignore[attr-defined]
    del _hfhub

try:
    from ensure_hf_hub_models import ensure_hf_hub_models

    ensure_hf_hub_models()
except ImportError:
    pass

from cached_predictor import get_cached_predictor
from demo_observer_feedback import append_observer_feedback
from demo_prediction_log import append_prediction_log

# Gradio 4.44.x has a bug where gradio_client passes additionalProperties=True
# (a bool) into _json_schema_to_python_type, which then does `"const" in True`
# and raises TypeError. Patch it to return "Any" for non-dict schemas.
try:
    import gradio_client.utils as _gc_utils
    _orig_j2p = _gc_utils._json_schema_to_python_type

    def _patched_j2p(schema, defs=None):
        if not isinstance(schema, dict):
            return "Any"
        return _orig_j2p(schema, defs)

    _gc_utils._json_schema_to_python_type = _patched_j2p
except Exception:
    pass

# ── Säkerhetsgränser ────────────────────────────────────────────────────────
# Max filstorlek för uppladdad WAV (bytes). Standard 50 MB.
MAX_WAV_BYTES: int = int(os.environ.get("DEMO_MAX_WAV_MB", "50")) * 1024 * 1024

# Rate limiting: max antal körningar per tidsfönster (per server-instans).
_RATE_WINDOW_SEC: int = int(os.environ.get("DEMO_RATE_WINDOW_SEC", "60"))
_RATE_MAX_CALLS: int = int(os.environ.get("DEMO_RATE_MAX_CALLS", "30"))
_rate_timestamps: deque = deque()  # globalt per process


def _check_rate_limit() -> str | None:
    """
    Enkel token-bucket per server-process.
    Returnerar ett felmeddelande om gränsen är nådd, annars None.
    """
    now = time.monotonic()
    cutoff = now - _RATE_WINDOW_SEC
    while _rate_timestamps and _rate_timestamps[0] < cutoff:
        _rate_timestamps.popleft()
    if len(_rate_timestamps) >= _RATE_MAX_CALLS:
        return (
            f"**Hastighetsgräns:** max {_RATE_MAX_CALLS} körningar per "
            f"{_RATE_WINDOW_SEC} sekunder nådd. Försök igen om en stund."
        )
    _rate_timestamps.append(now)
    return None

SITE_PRESETS_PATH = Path(__file__).resolve().parent / "site_presets.json"

REPO_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES_19_JSON = REPO_ROOT / "models" / "all_species" / "class_names_19.json"

PROFILES = [
    "balanced",
    "conservative",
    "max_coverage",
    "kenya_balanced_auto",
    "tanzania_balanced_auto",
    "malawi_balanced_auto",
]


def _load_observer_species_choices() -> list[str]:
    placeholder = "(tomt / valde jag inte)"
    if not CLASS_NAMES_19_JSON.exists():
        return [placeholder]
    try:
        data = json.loads(CLASS_NAMES_19_JSON.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [placeholder] + [str(x) for x in data]
    except (OSError, json.JSONDecodeError):
        pass
    return [placeholder]


def _load_site_presets() -> list[dict]:
    if not SITE_PRESETS_PATH.exists():
        return [{"id": "", "label": "(No presets file)", "lat": None, "lon": None}]
    data = json.loads(SITE_PRESETS_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []
    return data


def _gradio_file_url(path: Path) -> str:
    """Gradio /file= route for assets under allowed_paths (absolute path)."""
    return "/file=" + str(path.resolve()).replace("\\", "/")


def _map_iframe_document_html() -> str:
    """Standalone map page served at /galago-map (vendored Leaflet, same-origin parent DOM)."""
    return """<!DOCTYPE html>
<html lang="sv">
<head>
  <meta charset="utf-8" />
  <title>Galago platskarta</title>
  <link rel="stylesheet" href="/galago-static/leaflet/leaflet.css" />
  <script src="/galago-static/leaflet/leaflet.js"></script>
  <style>
    html, body { margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; }
    #map { width: 100%; height: 100%; min-height: 400px; }
  </style>
</head>
<body>
  <div id="map"></div>
  <script>
  (function () {
    function parentEl(id) {
      try { return window.parent.document.getElementById(id); } catch (e) { return null; }
    }
    function setField(elemId, value) {
      var root = parentEl(elemId);
      if (!root) return;
      var inp = root.querySelector("textarea, input");
      if (!inp) return;
      inp.value = value;
      inp.dispatchEvent(new Event("input", { bubbles: true }));
      inp.dispatchEvent(new Event("change", { bubbles: true }));
    }
    function wireCopyButton() {
      var copyBtn = parentEl("galago-copy-coords");
      if (!copyBtn || copyBtn.__galagoWired) return;
      copyBtn.__galagoWired = true;
      copyBtn.addEventListener("click", function () {
        var c = window.parent.__galagoLastCoords;
        if (!c) return;
        var line = c.lat + "\\t" + c.lon;
        var copyMsg = parentEl("galago-copy-msg");
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(line).then(function () {
            if (copyMsg) copyMsg.textContent = "Kopierat!";
          }).catch(function () {
            if (copyMsg) copyMsg.textContent = "Kunde inte kopiera — markera siffrorna manuellt.";
          });
        } else if (copyMsg) {
          copyMsg.textContent = "Ingen clipboard-API — kopiera från rutan ovan.";
        }
      });
    }
    function pushCoords(la, lo) {
      setField("galago_paste_coords", la + ", " + lo);
      setField("galago_lat", la);
      setField("galago_lon", lo);
      try { window.parent.__galagoLastCoords = { lat: la, lon: lo }; } catch (e) {}
      var out = parentEl("galago-coord-out");
      if (out) {
        out.innerHTML =
          "<strong>Latitude:</strong> " + la + "<br><strong>Longitude:</strong> " + lo +
          "<br><small>Fälten nedan uppdaterades automatiskt.</small>";
      }
      var copyBtn = parentEl("galago-copy-coords");
      if (copyBtn) copyBtn.disabled = false;
    }
    function setReady() {
      var out = parentEl("galago-coord-out");
      if (out) out.textContent = "Klicka på kartan för lat/lon (WGS84, decimalgrader).";
      wireCopyButton();
    }

    var map = L.map("map", { scrollWheelZoom: true }).setView([-5, 25], 4);
    L.tileLayer("https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png", {
      maxZoom: 20,
      subdomains: "abcd",
      attribution: "&copy; OpenStreetMap &copy; CARTO",
    }).addTo(map);

    var marker = null;
    map.on("click", function (e) {
      var la = e.latlng.lat.toFixed(6);
      var lo = e.latlng.lng.toFixed(6);
      if (marker) map.removeLayer(marker);
      marker = L.marker(e.latlng).addTo(map);
      pushCoords(la, lo);
    });

    function reflow() {
      try { map.invalidateSize({ pan: false }); } catch (e) {}
    }
    reflow();
    requestAnimationFrame(reflow);
    setTimeout(reflow, 150);
    setTimeout(reflow, 600);
    setTimeout(function () { reflow(); setReady(); }, 1200);
    window.addEventListener("resize", reflow);
  })();
  </script>
</body>
</html>"""


def _register_map_routes(app, demo_dir: Path) -> None:
    """Serve map HTML + vendored Leaflet on the outer FastAPI app (survives Gradio launch)."""
    leaflet_dir = demo_dir / "vendor" / "leaflet"
    if not (leaflet_dir / "leaflet.js").is_file():
        return

    from fastapi.responses import HTMLResponse
    from starlette.staticfiles import StaticFiles

    if not getattr(app.state, "galago_map_routes", False):
        app.mount(
            "/galago-static/leaflet",
            StaticFiles(directory=str(leaflet_dir)),
            name="galago_leaflet_vendor",
        )

        @app.get("/galago-map", include_in_schema=False)
        def _galago_map_page():
            return HTMLResponse(_map_iframe_document_html())

        app.state.galago_map_routes = True


def _map_shell_html() -> str:
    """Iframe to /galago-map — isolated layout; iframe JS updates parent lat/lon fields."""
    return (
        "<p><strong>Karta</strong> — klicka för lat/lon (WGS84, decimalgrader). "
        "Klick uppdaterar fälten <em>Latitude</em>, <em>Longitude</em> och "
        "<em>Klistra koordinater</em> nedan. "
        "<a href=\"/galago-map\" target=\"_blank\" rel=\"noopener\">Öppna karta i ny flik</a> om rutan är tom.</p>"
        '<iframe id="galago-map-iframe" title="Välj plats på kartan" src="/galago-map" '
        'style="width:100%;height:420px;border:1px solid #ccc;border-radius:8px;display:block;background:#e8e8e8">'
        "</iframe>"
        '<div id="galago-coord-out" style="margin-top:10px;padding:10px;background:#f4f4f4;border-radius:6px;font-family:ui-monospace,monospace">'
        "Laddar karta…"
        "</div>"
        "<p style=\"margin-top:8px\">"
        '<button type="button" id="galago-copy-coords" disabled>Kopiera lat och lon</button>'
        '<span id="galago-copy-msg" style="margin-left:8px;font-size:0.9rem;color:#063"></span>'
        "</p>"
    )


def row_to_markdown(r: dict) -> str:
    """Huvudresultat + visad top-3. Akustisk top-10 visas i en separat ruta under (tydligare i Gradio)."""
    lines = [
        "## Result (best operational guess)",
        "",
        f"- **Output species:** `{r.get('species_result', '')}`",
        f"- **Detector:** `{r.get('detector_result', '')}` (P galago ≈ {r.get('detector_prob', '')})",
        f"- **Location context:** `{r.get('location_used', '')}` — {r.get('location_status', '')}",
        f"- **Post-process:** `{r.get('postprocess_action', '')}` (`{r.get('postprocess_mode', '')}`)",
        "",
        "### Visad top-3 (CNN:s tre bästa — ev. **ny ordning** efter plats)",
        f"1. `{r.get('top1_species', '')}` — {r.get('top1_prob', '')}",
        f"2. `{r.get('top2_species', '')}` — {r.get('top2_prob', '')}",
        f"3. `{r.get('top3_species', '')}` — {r.get('top3_prob', '')}",
        "",
        "_Plats kan bara permutera dessa tre — en art som inte är CNN-top-3 kan inte flyttas in._",
        "",
        "**Akustisk top-10** (var t.ex. rondoensis ligger i modellen) finns i **rutan under**.",
        "",
        "_Akustisk likhet — inte taxonomiskt bevis._",
    ]
    return "\n".join(lines)


def acoustic_top10_markdown(r: dict | None) -> str:
    """Egen panel: numrerad lista så den syns tydligt i Gradio (inte en enda lång rad)."""
    if r is None:
        return "### Akustisk top-10 (CNN, utan plats)\n\n_Ingen körning ännu._"
    s = r.get("acoustic_top10", "N/A")
    if not s or str(s).strip() == "N/A":
        return "### Akustisk top-10 (CNN, utan plats)\n\n_Ingen data (t.ex. detectorfel eller ingen klassificering)._"
    chunks = [c.strip() for c in str(s).split("·") if c.strip()]
    lines = [
        "### Akustisk top-10 (poolad CNN, **utan** plats-omrankning)",
        "",
        "Sorterad från högst till lägst sannolikhet. Här syns om t.ex. `Paragalago_rondoensis` ligger utanför top-3.",
        "",
    ]
    for i, c in enumerate(chunks, 1):
        parts = c.rsplit(None, 1)
        if len(parts) == 2:
            sp, pr = parts
            lines.append(f"{i}. `{sp}` — {pr}")
        else:
            lines.append(f"{i}. {c}")
    lines.append("")
    lines.append("_Samma värden som i pipelinen `acoustic_top10`; bara formatterade som lista._")
    return "\n".join(lines)


def _coords_or_none(lat: float | None, lon: float | None) -> tuple[float | None, float | None]:
    """Gradio Number often sends 0 instead of empty; (0,0) is not a real field default — treat as unset."""
    if lat is None and lon is None:
        return None, None
    if lat is not None and lon is not None and lat == 0 and lon == 0:
        return None, None
    return lat, lon


def parse_lat_lon_paste(text: str | None) -> tuple[float | None, float | None, str]:
    """
    Parse one line: lat then lon. Accepts comma, tab, or whitespace between values.
    """
    if text is None:
        return None, None, "Tom rad."
    s = str(text).strip()
    if not s:
        return None, None, "Tom rad — klistra två tal (latitude först, sedan longitude)."

    s = s.replace(";", ",")
    if "\t" in s:
        parts = [p.strip() for p in s.split("\t") if p.strip()]
    elif "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
    else:
        parts = [p.strip() for p in re.split(r"\s+", s) if p.strip()]

    if len(parts) < 2:
        return None, None, "Behöver två tal: lat, lon (t.ex. -6.8, 39.28 eller tab från kartan)."

    try:
        la, lo = float(parts[0]), float(parts[1])
    except ValueError:
        return None, None, "Kunde inte läsa tal — använd decimalpunkt (.) och lat först."

    if not (-90 <= la <= 90):
        return None, None, f"Latitude {la} ska ligga i [-90, 90]."
    if not (-180 <= lo <= 180):
        return None, None, f"Longitude {lo} ska ligga i [-180, 180]."

    return la, lo, f"OK — satt lat={la}, lon={lo}."


def apply_site_preset(preset_label: str, presets: list[dict]):
    for p in presets:
        if p.get("label") == preset_label:
            la, lo = p.get("lat"), p.get("lon")
            if la is None or lo is None:
                return None, None
            return float(la), float(lo)
    return None, None


def _wav_path_from_gradio(audio_path: object) -> Path | None:
    """Normalize Gradio Audio output; must use type=\"filepath\" on the component."""
    if audio_path is None:
        return None
    if isinstance(audio_path, (tuple, list)):
        return None
    if isinstance(audio_path, Path):
        return audio_path if audio_path.exists() else None
    if isinstance(audio_path, str) and audio_path.strip():
        p = Path(audio_path)
        return p if p.exists() else None
    return None


SESSION_KEYS = ("Fil", "Trolig art (output)", "Top-1 klass", "P (output)", "Profil")


def _session_row(wav_name: str, row: dict, profile: str) -> dict:
    return {
        "Fil": wav_name,
        "Trolig art (output)": str(row.get("species_result", "")),
        "Top-1 klass": str(row.get("top1_species", "")),
        "P (output)": str(row.get("species_prob", "")),
        "Profil": profile,
    }


def _session_to_table_rows(history: list[dict]) -> list[list]:
    return [[h.get(k, "") for k in SESSION_KEYS] for h in history]


_ACOUSTIC_EMPTY = "### Akustisk top-10 (CNN, utan plats)\n\n_Ingen körning — ladda upp en `.wav` och klicka **Kör förutsägelse**._"
_ACOUSTIC_SKIP = "### Akustisk top-10 (CNN, utan plats)\n\n_Klassificeraren kördes inte (ogiltig fil eller fel före modellen)._"
_ACOUSTIC_ERR = "### Akustisk top-10 (CNN, utan plats)\n\n_Kunde inte beräknas — se **Debug** ovan._"


def run_predict(
    audio_path: object,
    profile: str,
    infer_filename: bool,
    no_location: bool,
    preset_label: str,
    lat: float | None,
    lon: float | None,
    log_this: bool,
    tanzania_no_spnov_remap: bool,
    presets: list[dict],
):
    """
    Returns (markdown, debug_log, session_entry_or_none, acoustic_top10_markdown, feedback_snapshot_or_none).
    feedback_snapshot binds the next observer feedback row to this predict().
    """
    wav = _wav_path_from_gradio(audio_path)
    if audio_path is not None and wav is None:
        return (
            "**Fel:** Kunde inte läsa uppladdad fil. Kontrollera att **Recording** använder en `.wav`-fil "
            "och att komponenten är satt till filväg (filepath).",
            "Gradio gav inte en giltig sökväg (eller returnerade numpy-data i stället för fil).",
            None,
            _ACOUSTIC_SKIP,
            None,
        )
    if wav is None:
        return "Ladda upp en **.wav**-fil.", "", None, _ACOUSTIC_EMPTY, None

    # macOS / ZIP: AppleDouble sidecars look like "._MyRecording.wav" — not valid audio; librosa fails.
    if wav.name.startswith("._"):
        return (
            "**Fel:** Filnamnet börjar med `._` — det är nästan alltid en **macOS-metadatafil** (AppleDouble), "
            "inte själva WAV-inspelningen. Ladda upp filen med samma namn **utan** `._` i början "
            "(eller packa upp / kopiera om från källan så att du får den riktiga `.wav`-filen).",
            "AppleDouble sidecar (._*); detector/load fails. Use the real audio file.",
            None,
            _ACOUSTIC_SKIP,
            None,
        )

    # ── Filändelse: måste vara .wav ─────────────────────────────────────────
    if wav.suffix.lower() != ".wav":
        return (
            f"**Fel:** Endast `.wav`-filer stöds (fick `{wav.suffix}`). "
            "Konvertera filen till WAV innan uppladdning.",
            f"Otillåten filändelse: {wav.suffix}",
            None,
            _ACOUSTIC_SKIP,
            None,
        )

    # ── Filstorlek ──────────────────────────────────────────────────────────
    wav_size = wav.stat().st_size
    if wav_size > MAX_WAV_BYTES:
        mb = wav_size / (1024 * 1024)
        limit_mb = MAX_WAV_BYTES / (1024 * 1024)
        return (
            f"**Fel:** Filen är för stor ({mb:.1f} MB). Max tillåten storlek är {limit_mb:.0f} MB.",
            f"Fil för stor: {mb:.1f} MB > {limit_mb:.0f} MB",
            None,
            _ACOUSTIC_SKIP,
            None,
        )

    # ── Rate limiting ───────────────────────────────────────────────────────
    rate_err = _check_rate_limit()
    if rate_err:
        return rate_err, "Rate limit nådd.", None, _ACOUSTIC_SKIP, None

    if no_location:
        lat_effective: float | None = None
        lon_effective: float | None = None
    else:
        lat_effective, lon_effective = _coords_or_none(lat, lon)

    post_override = None
    if tanzania_no_spnov_remap and profile == "tanzania_balanced_auto":
        post_override = "none"

    try:
        pred = get_cached_predictor()
        pred.set_deployment_profile(profile)
        row = pred.predict(
            wav.resolve(),
            infer_location_from_filename=infer_filename,
            lat=lat_effective,
            lon=lon_effective,
            postprocess_mode_override=post_override,
        )
        if log_this:
            append_prediction_log(
                row,
                wav_name=wav.name,
                profile=profile,
                no_location=no_location,
                lat=lat_effective,
                lon=lon_effective,
                infer_filename=infer_filename,
                site_preset_label=preset_label,
            )
        log = (
            f"Profile={profile} in-process (models cached). postprocess_mode={row.get('postprocess_mode')}\n"
            f"wav={wav.name}\n"
            f"location={'skipped' if no_location else f'lat={lat_effective}, lon={lon_effective}'}\n"
        )
        if log_this:
            log += "Appended to demo/logs/predictions.jsonl\n"
        entry = _session_row(wav.name, row, profile)
        feedback_snapshot = {
            "wav_filename": wav.name,
            "wav_tmp_path": str(wav.resolve()),
            "species_result": row.get("species_result"),
            "species_prob": row.get("species_prob"),
            "top1_species": row.get("top1_species"),
            "profile": profile,
        }
        return (
            row_to_markdown(row),
            log,
            entry,
            acoustic_top10_markdown(row),
            feedback_snapshot,
        )
    except Exception as e:
        tb = traceback.format_exc()
        return f"**Error:** `{e}`\n\n```\n{tb}\n```", tb, None, _ACOUSTIC_ERR, None


def _patch_gradio_app_for_map(demo_dir: Path) -> None:
    """Register /galago-map after Gradio builds its FastAPI app (launch() recreates demo.app)."""
    from gradio.routes import App

    if getattr(App, "_galago_map_patch_applied", False):
        return

    _orig = App.create_app

    def _create_app_with_map(*args, **kwargs):
        app = _orig(*args, **kwargs)
        _register_map_routes(app, demo_dir)
        return app

    App.create_app = _create_app_with_map
    App._galago_map_patch_applied = True


def main() -> None:
    import gradio as gr

    presets = _load_site_presets()
    preset_labels = [p["label"] for p in presets if "label" in p]
    if not preset_labels:
        preset_labels = ["(No presets)"]

    species_observer_choices = _load_observer_species_choices()

    demo_dir = Path(__file__).resolve().parent
    _patch_gradio_app_for_map(demo_dir)

    with gr.Blocks(title="Galago call demo") as demo:
        gr.Markdown(
            "# Galago — akustisk demo\n\n"
            "Ladda en **galago-.wav** (mono/stereo spelar mindre roll; modellen fönstrar signalen). "
            "Första körningen laddar modeller — därefter snabbare.\n\n"
            "**Profiler:** *balanced* för blandad användning. *kenya_* / *tanzania_* / *malawi_* när du vill "
            "matcha trösklar och (för TZ/Malawi) särskilda efterbehandlingsregler."
        )
        with gr.Accordion("ℹ️ Dataskydd och loggning", open=False):
            gr.Markdown(
                "**Vad loggas?**\n\n"
                "När du kör en förutsägelse sparas modellens output (art, sannolikhet, profil) "
                "lokalt på servern i `demo/logs/predictions.jsonl` — *om* loggning är påslagen i "
                "Avancerat-sektionen. Ingen information skickas utanför servern vid körning.\n\n"
                "**Frivillig feedback**\n\n"
                "Om du fyller i *Din bedömning* och klickar **Spara min bedömning** sparas ditt svar "
                "(art, om du håller med, eventuell anteckning) i `demo/logs/observer_feedback.jsonl`. "
                "Denna information kan användas för att förbättra modellen.\n\n"
                "**Frivillig WAV-delning**\n\n"
                "Kryssar du i *Jag samtycker till att en kopia av inspelningen sparas* sparas en kopia "
                "av din WAV-fil på servern i `demo/logs/wav_archive/`. Filen används enbart för att "
                "granska och förbättra modellen. Utan kryss sparas *ingen* ljudfil — bara metadata.\n\n"
                "**Åtkomst**\n\n"
                "Loggfiler och WAV-kopior är bara tillgängliga för den som driver servern. "
                "Inga data delas med tredje part. Kontakta serveransvarig för att begära radering."
            )
        audio = gr.Audio(
            label="Inspelning (.wav)",
            type="filepath",
            sources=["upload"],
        )
        profile = gr.Dropdown(
            choices=PROFILES,
            value="balanced",
            label="Profil (deployment)",
        )

        with gr.Accordion("Plats och karta", open=True):
            gr.HTML(_map_shell_html())
            no_location = gr.Checkbox(
                value=False,
                label="Ingen plats — skicka inte lat/long till kontext (bara akustik + profiler)",
            )
            site_dd = gr.Dropdown(
                choices=preset_labels,
                value=preset_labels[0],
                label="Förvald plats (fyller lat/long)",
            )
            with gr.Row():
                lat = gr.Number(
                    label="Latitude (decimal, WGS84)",
                    value=None,
                    info="Inte DMS. Ex: -6.8",
                    elem_id="galago_lat",
                )
                lon = gr.Number(
                    label="Longitude (decimal, WGS84)",
                    value=None,
                    info="Ex: 39.28",
                    elem_id="galago_lon",
                )
            with gr.Row():
                paste_coords = gr.Textbox(
                    label="Klistra koordinater (lat, sedan lon)",
                    placeholder="-6.8, 39.28  eller  tab från kartans Kopiera-knapp",
                    lines=1,
                    scale=4,
                    elem_id="galago_paste_coords",
                )
                apply_paste_btn = gr.Button("Tillämpa", scale=1)
            coord_paste_status = gr.Textbox(
                label="",
                value="",
                lines=1,
                max_lines=1,
                interactive=False,
                show_label=False,
            )

        with gr.Accordion("Avancerat", open=False):
            gr.Markdown(
                "_Filnamn med **Pugu** / **Rondo** tolkas som Tanzania-kontext när rutan nedan är på._"
            )
            infer_fn = gr.Checkbox(
                value=True,
                label="Härled Tanzania från filnamn (Pugu / Rondo)",
            )
            tz_raw = gr.Checkbox(
                value=False,
                label="Tanzania-profil: stäng av sp.nov → rondoensis (vis rå klassificering)",
            )
            log_this = gr.Checkbox(
                value=True,
                label="Logga till demo/logs/predictions.jsonl (inkl. akustisk top-10 / plats 1–3 + rondoensis-rank)",
            )

        session_hist = gr.State([])
        last_run_ctx = gr.State(None)

        with gr.Row():
            btn = gr.Button("Kör förutsägelse", variant="primary")
            next_btn = gr.Button("Nästa fil (rensa uppladdning + resultat)")
            clear_session_btn = gr.Button("Töm sessionslista")

        out_md = gr.Markdown()
        acoustic_md = gr.Markdown(value=_ACOUSTIC_EMPTY)
        log = gr.TextArea(label="Debug / traceback", lines=10)
        gr.Markdown(
            "**Sessionslista** — en rad per lyckad körning (filnamn, modellens output-art, top-1, sannolikhet, profil). "
            "Listan lever bara i den här webbläsarfliken tills du tömmer den eller startar om servern."
        )
        session_df = gr.Dataframe(
            headers=list(SESSION_KEYS),
            value=[],
            label="Sammanställning (denna körning)",
            interactive=False,
            wrap=True,
        )

        with gr.Accordion("Din bedömning (valfritt)", open=False):
            gr.Markdown(
                "Efter **Kör förutsägelse**: svara om modellens **output-art** stämmer med vad **du** trodde "
                "(fältet påverkar inte modellen — det sparas för senare analys)."
            )
            observer_agrees = gr.Radio(
                choices=["Ja", "Nej", "Vill inte svara"],
                value="Vill inte svara",
                label="Stämmer modellens förslag (output species) med vad du trodde?",
            )
            observer_believed = gr.Dropdown(
                choices=species_observer_choices,
                value=species_observer_choices[0],
                label="Vilken art trodde du? (valfritt)",
                allow_custom_value=False,
            )
            observer_notes = gr.Textbox(
                label="Anteckning (valfritt)",
                placeholder="t.ex. Pugu vs Rondo, call-typ, osäkerhet …",
                lines=2,
            )
            save_wav_cb = gr.Checkbox(
                value=False,
                label=(
                    "Jag samtycker till att en kopia av den uppladdade inspelningen sparas "
                    "för granskning och modellförbättring"
                ),
                info=(
                    "Utan kryss sparas bara metadata (art, bedömning, tidstämpel). "
                    "Kryssa i för att även spara en kopia av WAV-filen lokalt på servern."
                ),
            )
            save_feedback_btn = gr.Button("Spara min bedömning")
            feedback_status = gr.Textbox(
                label="",
                value="",
                lines=1,
                interactive=False,
                show_label=False,
            )

        def _on_preset(lbl: str):
            la, lo = apply_site_preset(lbl, presets)
            if la is None:
                return gr.update(), gr.update()
            return gr.update(value=la), gr.update(value=lo)

        site_dd.change(_on_preset, inputs=[site_dd], outputs=[lat, lon])

        def _apply_paste_line(line: str):
            la, lo, msg = parse_lat_lon_paste(line)
            if la is None:
                return gr.update(), gr.update(), msg
            return gr.update(value=la), gr.update(value=lo), msg

        apply_paste_btn.click(
            _apply_paste_line,
            inputs=[paste_coords],
            outputs=[lat, lon, coord_paste_status],
        )

        def _predict_with_session(
            audio_p,
            prof,
            inf_fn,
            no_loc,
            site,
            la,
            lo,
            log_t,
            tz_r,
            hist,
        ):
            md, lg, entry, ac, snap = run_predict(
                audio_p,
                prof,
                inf_fn,
                no_loc,
                site,
                la,
                lo,
                log_t,
                tz_r,
                presets,
            )
            new_hist = list(hist) if isinstance(hist, list) else []
            if entry is not None:
                new_hist.append(entry)
            return (
                md,
                lg,
                new_hist,
                gr.update(value=_session_to_table_rows(new_hist)),
                ac,
                snap,
                "",
            )

        btn.click(
            _predict_with_session,
            inputs=[
                audio,
                profile,
                infer_fn,
                no_location,
                site_dd,
                lat,
                lon,
                log_this,
                tz_raw,
                session_hist,
            ],
            outputs=[out_md, log, session_hist, session_df, acoustic_md, last_run_ctx, feedback_status],
        )

        def _save_observer_feedback(ctx, agrees: str, believed: str, notes: str, sv: bool):
            wav_tmp = None
            if ctx and ctx.get("wav_tmp_path"):
                p = Path(ctx["wav_tmp_path"])
                wav_tmp = p if p.exists() else None
            msg = append_observer_feedback(
                ctx, agrees, believed, notes,
                wav_tmp_path=wav_tmp,
                save_wav=bool(sv),
            )
            return msg

        save_feedback_btn.click(
            _save_observer_feedback,
            inputs=[last_run_ctx, observer_agrees, observer_believed, observer_notes, save_wav_cb],
            outputs=[feedback_status],
        )

        def _next_file_reset():
            return (
                gr.update(value=None),
                "",
                _ACOUSTIC_EMPTY,
                "",
                gr.update(value=""),
                gr.update(value=""),
                None,
                "",
                gr.update(value="Vill inte svara"),
                gr.update(value=species_observer_choices[0]),
                gr.update(value=""),
                gr.update(value=False),
            )

        next_btn.click(
            _next_file_reset,
            outputs=[
                audio,
                out_md,
                acoustic_md,
                log,
                paste_coords,
                coord_paste_status,
                last_run_ctx,
                feedback_status,
                observer_agrees,
                observer_believed,
                observer_notes,
                save_wav_cb,
            ],
        )

        def _clear_session():
            return [], gr.update(value=[])

        clear_session_btn.click(_clear_session, outputs=[session_hist, session_df])

    base = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    for port in range(base, base + 10):
        try:
            demo.launch(
                server_name="0.0.0.0",
                server_port=port,
                allowed_paths=[str(demo_dir)],
                show_api=False,
            )
            break
        except OSError as e:
            err = str(e).lower()
            if "empty port" in err or "cannot find" in err:
                print(f"Port {port} busy, trying next...", flush=True)
                continue
            raise
    else:
        raise RuntimeError(f"No free Gradio port in range {base}–{base + 9}")


if __name__ == "__main__":
    main()
