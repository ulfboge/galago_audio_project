# Sessionlogg — HF Space + granti-relabel (2026-06-24 / 25)

Handover efter Space-fix, kartfix, demo-test och omträning.

**Status 2026-06-25:** Space live med fungerande karta. Lokal omträning klar (val 95.4%); **nya vikter ej på Hub**. Incremental granti fortfarande fel (Hub: rondoensis; lokal: zanzibaricus).

---

## HF Space — deploy & fixar

### Problem (mars–juni 2026)

1. **`TypeError: unhashable type: 'dict'`** — Gradio 4.44 + Starlette 1.0 → pin `starlette>=0.37.2,<1.0` (`3ef054e`)
2. **Saknad `species_ranges.json`** — incheckad i git
3. **Karta** — flera misslyckade försök (data-URL iframe, inline Leaflet, `/file=`, `mount_gradio_app`+uvicorn)

### Kartfix — slutgiltig lösning (`524d854`)

- Iframe `src="/galago-map"` med **egen FastAPI-route** + vendored Leaflet (`demo/vendor/leaflet/`)
- Route registreras i **patch av `gradio.routes.App.create_app`** (routes före `launch()` försvinner — `launch()` byter ut `demo.app`)
- Iframe-JS uppdaterar parent-DOM (`galago_lat`, `galago_lon`, `galago_paste_coords`) — samma origin
- Fallback: länk «Öppna karta i ny flik»; **Förvald plats** dropdown utan karta
- Karttiles från Carto CDN (inuti iframe)

| Commit | Försök |
|--------|--------|
| `368e577` | Inline Leaflet (CDN) — klick/strular på HF |
| `492eb14` | Vendored Leaflet + `/file=` + `galago_map.js` |
| `2ddf827` | `Blocks(head=…)` + `Blocks(js=…)` |
| `941bd35` | `invalidateSize` / ResizeObserver |
| `a919adb` | Iframe + `galago_map_embed.html` + postMessage |
| `67ed5d1` | FastAPI `/galago-map` — 404 (routes före launch) |
| `8c9c6b0` | `mount_gradio_app` — Space hängde / ofullständig start |
| `4dc9c92` / `2106cb4` / `524d854` | `demo.launch()` + `create_app`-patch |

**Deploy:** `git push origin main` + `git push space main`

---

## Demo smoke-test — granti-filer

Plats (metadata): lat **-1.3**, lon **36.8** (Kenya).

### Före juni-omträning (Hub-vikter)

| Fil | Resultat |
|-----|----------|
| Buzz G granti | Paragalago_granti ✓ |
| Buzz-Screech+ Grunt | Paragalago_granti ✓ |
| Incremental call | Paragalago_rondoensis p≈1.0 ✗ |

### Efter lokal omträning (`improved_best`, 26 ep, val acc 95.4%)

| Fil | Resultat |
|-----|----------|
| Buzz G granti | **Galago_moholi** 1.0 ✗ (regression) |
| Buzz-Screech+ Grunt | Paragalago_granti 0.96 ✓ |
| Incremental call | **Paragalago_zanzibaricus** 0.89 ✗ (granti 0.07, rondoensis 0.04) |

**Space** kör fortfarande **gamla Hub-vikter** → incremental visar rondoensis tills nya vikter laddas upp (och då troligen zanzibaricus, inte granti).

Tolka inte p=1.0 som definitiv taxon — särskilt för incremental call-typ.

---

## Modellförbättring — granti ↔ rondoensis

### Predict + shortlist

```text
data/splits/granti_review_filelist.txt (3 WAV)
→ outputs/predictions/predictions_granti_review_2026-06.csv
```

`make_cluster_review_shortlist.py` utökad: **granti / zanzibaricus / orinus** i CLUSTER + boost granti↔rondoensis.

### Relabel + ingest

- CSV: `data/relabels/relabels_granti_incremental_2026-06.csv` (`ingest=yes`, **inga** `start_sec`/`end_sec`)
- `ingest_relabels.py`: **Added 1** (Incremental), **Skipped 2** (Buzz-mels fanns)
- Append: `data/splits/mels_cluster_rondoensis_train.txt`

### Omträning (klar 2026-06-25)

```bash
python scripts/train_cnn_all_species_improved.py --epochs 50
```

- **Val accuracy:** 95.4% (19 klasser)
- **Epochs:** 26 (early stopping)
- **Outputs:** `galago_cnn_all_19classes_improved_best.keras`, `*_final.keras`, `class_names_19.json`
- **Träningsskript:** `matplotlib.use("Agg")` tillagt (krasch vid plottning på Windows efter save)

**Ej körd än:** `run_eval_regions.py --tag after_retrain_2026-06`

---

## Öppna punkter

1. **Snippet-relabel** för incremental (`start_sec`/`end_sec`) — se `docs/confusion_relabel_workflow.md`
2. **Verifiera Buzz** efter nästa omträning (moholi-regression i juni-körningen)
3. **Ladda upp vikter** till `ulfboge/galago-demo-vikter` endast efter lokal acceptans
4. **Rotera HF-token** (exponerad mars 2026)
5. **Observer-feedback / Slack** på Space — ej testat
6. **~175 relabel-kandidater** i `data/relabels/` — bredare Tanzania/confusion-arbete

---

## Snabbreferens

| Vad | Var |
|-----|-----|
| Live-demo | https://huggingface.co/spaces/ulfboge/galago-call-demo |
| Hub-vikter | https://huggingface.co/ulfboge/galago-demo-vikter |
| Relabel-workflow | `docs/confusion_relabel_workflow.md` |
| Grant relabel CSV | `data/relabels/relabels_granti_incremental_2026-06.csv` |
| Kartkod | `demo/upload_predict_gradio.py` (`_patch_gradio_app_for_map`, `/galago-map`) |

**Deploy-rutin:** `git push origin main` + `git push space main`
