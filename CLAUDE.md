# CLAUDE.md

Den här filen ger Claude (och andra AI-assistenter) kontext för att arbeta i **galago_audio_project**.

## Vad projektet är

En CNN-baserad pipeline för **akustisk artidentifiering av galagos** (familjen Galagidae) från WAV-inspelningar. Ljud omvandlas till 128×128 mel-spektrogram och klassificeras i tre steg:

**detektor → klassificerare → kontext-omrankning**

Målet är en webbapp där användare laddar upp fältinspelningar och får artförslag med osäkerhetshantering — plus en väg att samla in feedback och förbättra modellen över tid.

| Del | Plats |
|-----|--------|
| Kod (GitHub) | [github.com/ulfboge/galago_audio_project](https://github.com/ulfboge/galago_audio_project) |
| Live-demo (HF Space) | [huggingface.co/spaces/ulfboge/galago-call-demo](https://huggingface.co/spaces/ulfboge/galago-call-demo) |
| Modellvikter (HF Hub) | [huggingface.co/ulfboge/galago-demo-vikter](https://huggingface.co/ulfboge/galago-demo-vikter) |

Repot ligger lokalt i `c:\Users\galag\GitHub\galago_audio_project` (flyttat från `jobb-ansökningar/research/` i juni 2026).

---

## Utvecklingsfas (juni 2026)

**Fas:** Live-demo på HF Space **fungerar** (inferens, platsväljare, profiler). Modellförbättring via kuraterad relabel + omträning — **pågår**; nya vikter **inte** uppladdade till Hub än.

| Område | Status |
|--------|--------|
| HF Space build / inferens | ✅ Starlette-pin, `species_ranges.json`, modeller från Hub |
| Karta (lat/lon) i Gradio | ✅ Iframe `/galago-map` + patch av `App.create_app` (commit `524d854`) |
| Observer-feedback / Slack på Space | ⚠️ Ej verifierat i denna sprint |
| Granti incremental-call | ⚠️ Hub-vikter: rondoensis; lokal omträning: zanzibaricus — inte granti |
| Hub-vikter vs lokal omträning | ⚠️ Space kör **gamla** `improved_best` från Hub; lokal omträning juni 2026 ej pushad |
| Regional eval efter juni-omträning | ❌ Ej körd (`run_eval_regions.py --tag after_retrain_2026-06`) |
| HF-token rotation | ❌ Öppen (exponerad mars 2026) |

**Senaste handover:** `docs/session_2026-06-24_space_fix.md` · **Operativ lista:** `docs/next_steps_2026-06-24.md`

---

## Setup

```bash
pip install -r requirements.txt          # full (TF, librosa, gradio, etc.)
pip install -r requirements-demo.txt     # lättare: bara demo-UI
```

Alla skript löser sökvägar relativt repots rot via `Path(__file__).resolve().parents[N]` — kör alltid från rotmappen.

Modellfiler finns **inte i git**. Lokalt: `demo/ensure_hf_hub_models.py` laddar ner vid behov. På HF Space: sätt `GALAGO_HF_MODEL_REPO=ulfboge/galago-demo-vikter`.

**Deploy till Space:** `git push origin main` **och** `git push space main` (två remotes).

---

## Vanliga kommandon

**Inferens på en WAV:**
```bash
python scripts/predict_3stage_with_context.py --wav <path/to/file.wav>
python scripts/predict_3stage_with_context.py --wav <file.wav> --profile balanced --lat -1.3 --lon 36.8
```

**Gradio-demo (lokal app):**
```bash
python demo/upload_predict_gradio.py   # http://127.0.0.1:7860
```

**Träna klassificeraren:**
```bash
python scripts/train_cnn_all_species_improved.py
python scripts/train_cnn_all_species_improved.py --epochs 50
```

**Utvärdera per region:**
```bash
python scripts/run_eval_regions.py
python scripts/run_eval_regions.py --tag after_retrain_2026-06
```

**Granti-relabel (exempel):**
```bash
python scripts/ingest_relabels.py --csv data/relabels/relabels_granti_incremental_2026-06.csv \
  --out-filelist data/splits/mels_relabels_additions.txt \
  --append-filelist data/splits/mels_cluster_rondoensis_train.txt
```

**Granska demo-feedback:**
```bash
python scripts/review_feedback.py
python scripts/export_feedback_to_relabels.py --dry-run
```

---

## Arkitektur: 3-stegs pipeline

`scripts/predict_3stage_with_context.py` är huvudingången för inferens.

1. **Detektor** (`models/detector/galago_detector_best.keras`) — binär: galagoläte eller inte. Tröskel ~0.3.
2. **Klassificerare** (`models/all_species/galago_cnn_all_19classes_improved_best.keras` standard) — CNN på mel-bilder, 19 arter. Fönster 2.5 s / hop 1.25 s, top-K pooling. Operativ tröskel 0.35; rekommenderad högkonfidens-tröskel 0.6.
3. **Kontext-omrankning** (`scripts/context_reranker.py`) — Bayesiansk omrankning med plats (IUCN-polygoner), månad och tid. Läser `data/species_ranges.json` och `data/iucn/`.

Demot cachar modeller via `demo/cached_predictor.py`.

### Mel-parametrar (måste vara konsekventa)

```
SR=22050, N_MELS=128, N_FFT=2048, HOP_LENGTH=512, FMIN=200, FMAX=10000, TARGET_FRAMES=128
```

**Kritisk fix (mars 2026):** spektrogram måste vändas vertikalt och normaliseras till `[0,1]` — detta var orsaken till stora regionala noggrannhetsfall innan fixen.

### Deployment-profiler

`configs/deployment_profiles.json` — namngivna trösklar: `max_coverage`, `balanced`, `conservative`, `kenya_balanced`, `tanzania_balanced`, `malawi_balanced`. Skicka `--profile <namn>` till inferensskriptet.

---

## Data och modeller

```
data/
  melspectrograms/          # tränings-PNG:er, en undermapp per art
  raw_audio/                # fält-WAV:er
  species_ranges.json       # platspriorer (incheckad i git för Space)
  iucn/                     # IUCN GeoJSON (polygonpriorer)
  splits/                   # train/holdout-fillistor
  relabels/                 # manuella om-etiketteringar (vissa CSV incheckade)
models/
  detector/
  all_species/              # 16/19-klass
demo/
  upload_predict_gradio.py  # Gradio-app (HF Space entry)
  galago_map_embed.html     # fristående karta (referens; Space använder /galago-map)
  vendor/leaflet/           # vendored Leaflet för kart-routen
  logs/                     # predictions.jsonl, observer_feedback.jsonl
```

Aktiva skript (ignorera legacy `top6`, `top7`, `6class`, `7class`):
- `predict_3stage_with_context.py` — inferens
- `train_cnn_all_species_improved.py` — träning
- `ingest_raw_audio_to_training_mels.py` — fältljud → mels
- `ingest_relabels.py` — kuraterade CSV → mels + splits
- `make_cluster_review_shortlist.py` — confusion-shortlist (inkl. granti↔rondoensis)
- `context_reranker.py`, `evaluate_confusion.py`, `run_eval_regions.py`

---

## Demo & feedback

Gradio-appen har:
- uppladdning av `.wav` med storleksgräns och rate limiting
- **karta:** iframe `src="/galago-map"`; Leaflet vendored; iframe uppdaterar lat/lon i parent-DOM
- **Förvald plats** dropdown (fallback utan karta)
- observer-feedback (stämmer / trodd art / anteckning)
- opt-in checkbox för WAV-kopia till `demo/logs/wav_archive/`
- webhook till Slack via `DEMO_FEEDBACK_WEBHOOK_URL` i `.env`
- export till relabel-CSV via `scripts/export_feedback_to_relabels.py`

**Kart-implementation (viktigt för felsökning):** Gradio 4.44 strippar `<script>` i `gr.HTML`. Routes för `/galago-map` registreras via patch av `gradio.routes.App.create_app` i `demo/upload_predict_gradio.py` — **inte** före `demo.launch()` (routes försvinner då). Se `docs/session_2026-06-24_space_fix.md`.

Se även `docs/handover_2026-03-27.md` och `docs/demo_live_feedback_roadmap.md`.

---

## Modellstatus — granti incremental (juni 2026)

Testfil: `data/raw_audio/Paragalago_granti/Incremental call  G granti.wav` (Kenya ~ -1.3, 36.8).

| Modell | Incremental | Buzz G granti | Buzz-Screech+ Grunt |
|--------|-------------|---------------|---------------------|
| Hub / före omträning | Paragalago_rondoensis p≈1.0 | Paragalago_granti ✓ | Paragalago_granti ✓ |
| Lokal `improved_best` efter omträning (26 ep, val 95.4%) | Paragalago_zanzibaricus 0.89 | **Galago_moholi** 1.0 (regression) | Paragalago_granti 0.96 ✓ |

Relabel ingest: **1 ny mel** för incremental (center 2.5 s, utan `start_sec`/`end_sec`). **Ladda inte upp** juni-vikter till Hub förrän incremental + Buzz verifierats — snippet-relabel rekommenderas.

---

## Nästa steg (prioriterad checklista)

### 1. Modell + Hub (hög ROI)

- [ ] Snippet-relabel för incremental call (`start_sec`/`end_sec` i `data/relabels/relabels_granti_incremental_2026-06.csv`)
- [ ] `ingest_relabels.py` → omträna → testa alla tre granti-WAV:er lokalt
- [ ] `python scripts/run_eval_regions.py --tag after_retrain_2026-06`
- [ ] Vid acceptabelt resultat: ladda upp `galago_cnn_all_19classes_improved_best.keras` till `ulfboge/galago-demo-vikter`
- [ ] Övriga relabel-rader i `data/relabels/` (~175 kandidater) — se `docs/confusion_relabel_workflow.md`

### 2. Live-demo

- [x] HF Space startar och inferens fungerar
- [x] Karta: klick → lat/lon (commit `524d854`)
- [ ] Smoke-test observer-feedback + Slack webhook på Space
- [ ] Rotera exponerad HF-token (`docs/next_steps_2026-06-24.md` §1)

### 3. Feedback → träning (när data finns)

- [ ] `python scripts/review_feedback.py --only-pending`
- [ ] `python scripts/export_feedback_to_relabels.py` → ingest → retrain

### 4. Längre sikt

- [ ] SQLite/Postgres för feedbackstatus när volymen växer
- [ ] GitHub Pages-landningssida → HF Space
- [ ] Malawi ft4-vikter i Hub om inte redan där

---

## Viktiga dokument

| Dokument | Innehåll |
|----------|----------|
| `README.md` | Projektöversikt, arter, prestanda |
| `docs/session_2026-06-24_space_fix.md` | Juni 2026: Space, karta, granti-relabel, omträning |
| `docs/next_steps_2026-06-24.md` | Operativ checklista juni 2026 |
| `docs/immediate_next_steps.md` | Relabel → ingest → retrain → eval |
| `docs/confusion_relabel_workflow.md` | Shortlist → kurering → ingest |
| `docs/handover_2026-03-27.md` | Demo-feedback, webhook, HF-förberedelse |
| `docs/space_deploy_status_2026-03-27.md` | HF Space/Model-repo-status |
| `docs/session_eval_2026-03_retrain_v2.md` | Referens-siffror regional eval |
| `docs/evaluation_caveats.md` | Bias i eval (mappnamn vs kuraterad sanning) |
| `demo/README_spaces.md` | Guide för Hugging Face Spaces |

---

## Gör / gör inte

**Gör:**
- Kör skript från repots rot
- Håll mel-parametrar konsekventa mellan träning och inferens
- Tolka lågkonfidens-prediktioner som osäkra, inte som saknad art
- Använd `--profile` och `--lat`/`--lon` när plats är känd
- Testa lokalt **innan** Hub-uppladdning av nya vikter

**Gör inte:**
- Committa `.env`, modellvikter (`.keras`) eller `demo/logs/wav_archive/`
- Träna om automatiskt på råa uppladdningar utan kuratering
- Ändra mel-parametrar utan omträning
- Anta att klassificeraren identifierar individer eller ljudtyper — bara akustisk artlikhet
- Registrera FastAPI-routes på `demo.app` före `launch()` — de försvinner när Gradio skapar appen
