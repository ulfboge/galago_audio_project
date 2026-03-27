---
title: Galago Call Demo
emoji: 🦔
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "4.44.0"
app_file: demo/upload_predict_gradio.py
pinned: false
license: mit
---

# Galago — akustisk demo (Hugging Face Space)

Ladda upp en `.wav`-inspelning och kör 3-stegs pipeline (detektor → CNN → platskontext).

## Vikter finns inte i GitHub-klonen

`models/` är gitignorerad. På Space måste vikterna finnas **antingen** genom nedladdning från Hub **eller** genom att du committar dem i ett separat Space-repo med LFS.

**Rekommenderat:** skapa ett **Model repo** (eller Dataset) på Hugging Face, ladda upp filer med **samma relativa sökvägar** som i projektet:

| Sökväg i Hub-repot | Syfte |
|---------------------|--------|
| `models/detector/galago_detector_best.keras` | Detektor |
| `models/all_species/galago_cnn_all_19classes_improved_best.keras` | Standardklassificerare (balanced m.fl.) |
| `models/all_species/class_names_19.json` | Etikettlista i demo-UI |
| `models/all_species/galago_cnn_malawi_spnov_ft4_best.keras` | *Valfritt* — krävs för Malawi-profiler |
| `data/species_ranges.json` | *Valfritt* — string-baserade plats-priors (annars neutral prior) |

Exempel (kör från **projektroten** där `models/...` finns lokalt):

```python
from huggingface_hub import HfApi

repo_id = "ditt-namn/galago-demo-vikter"
paths = [
    "models/detector/galago_detector_best.keras",
    "models/all_species/galago_cnn_all_19classes_improved_best.keras",
    "models/all_species/class_names_19.json",
]
api = HfApi()
for p in paths:
    api.upload_file(
        path_or_fileobj=p,
        path_in_repo=p,
        repo_id=repo_id,
        repo_type="model",
    )
```

(Alternativ: Hub-webben eller `git lfs` i Model-repot med samma katalogstruktur.)

## Secrets / variabler på Space

I **Settings → Variables and secrets** (Space):

| Namn | Typ | Beskrivning |
|------|-----|-------------|
| `GALAGO_HF_MODEL_REPO` | Secret eller variable | Model-repo-id, t.ex. `ditt-namn/galago-demo-vikter` |
| `GALAGO_HF_MODEL_REVISION` | Variable | Valfritt branch/tag (standard: Hub default) |
| `DEMO_FEEDBACK_WEBHOOK_URL` | Secret | Slack/Discord-webhook för feedback-notiser |
| `HF_TOKEN` | Secret | Bara om vikt-repot är **privat** (läs-token) |

Vid start anropar demot `ensure_hf_hub_models` och hämtar filer som saknas under repots rot.

## Monorepo vs. dedikerat Space-repo

- **Samma GitHub-repo som Space:** HF läser **`README.md` i repots rot** för SDK-metadata (YAML-frontmatter). Detta fil (`demo/README_spaces.md`) är en **mall**: kopiera innehållet till rot-`README.md` *eller* använd en gren / separat klone där rot-`README.md` = denna mall (GitHub-sidan kan då visa dubbla `---`-block om du inte trimmar — många använder en **egen Space-klon** med kort README).
- **Rot-beroenden:** Space kör `pip install -r requirements.txt` från **roten** (TensorFlow CPU, librosa, gradio, `huggingface_hub`, …). `packages.txt` installerar `libsndfile1` för ljud på Linux.

## Lokal körning

```bash
pip install -r requirements-demo.txt   # lättvikt, om modeller redan finns lokalt
# eller
pip install -r requirements.txt
python demo/upload_predict_gradio.py
```

## Checklista innan första build

1. `configs/deployment_profiles.json` är incheckad i Git (Space-klonen behöver den).
2. `GALAGO_HF_MODEL_REPO` pekar på repo där **minst** de tre obligatoriska filerna finns.
3. Webhook: sätt `DEMO_FEEDBACK_WEBHOOK_URL` om du vill ha Slack/Discord vid sparad feedback.
