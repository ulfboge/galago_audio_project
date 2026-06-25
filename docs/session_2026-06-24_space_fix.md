# Sessionlogg — HF Space + granti-relabel (2026-06-24 / 25)

Handover efter Space-fix, demo-test och modellförbättringsstart.

---

## HF Space — deploy & fixar

### Problem (mars–juni 2026)

1. **`TypeError: unhashable type: 'dict'`** — Gradio 4.44 + Starlette 1.0 (fix: pin Starlette &lt;1.0, commit `3ef054e`)
2. **Saknad `species_ranges.json`** — incheckad i git
3. **Karta utan klick** — `data:`-URL-iframe blockerade Leaflet (commit `368e577`, otillräcklig på Gradio 4.44)
4. **Ingen karta alls** — Gradio 4.44 **strippar inline `<script>`** i `gr.HTML`; CDN-länkar i samma block räcker inte

### Kartfix (commit följer denna session)

- Leaflet **vendored** under `demo/vendor/leaflet/`
- **v3:** `gr.HTML` har bara div-markup; Leaflet laddas via `gr.Blocks(head=…)` + `gr.Blocks(js=…)` (Gradio 4.44 strippar `<script>` i HTML-komponenten)
- Klick uppdaterar `galago_lat`, `galago_lon`, `galago_paste_coords` automatiskt
- Karttiles fortfarande från Carto CDN (OSM 403 i embed)

**Deploy:** `git push origin main` + `git push space main`

---

## Demo smoke-test — granti-fil

Fil: `data/raw_audio/Paragalago_granti/Incremental call  G granti.wav`  
Inspelningsplats (metadata): lat **-1.3**, lon **36.8** (Kenya).

| Profil | Plats | Resultat |
|--------|-------|----------|
| `balanced` | ingen | `Paragalago_rondoensis` p=1.0 |
| `kenya_balanced_auto` | ingen / med coords | samma — **modellfel**, inte Space-bugg |

- **Buzz** / **Buzz-Screech+ Grunt** (samma mapp): korrekt `Paragalago_granti` (p≈1.0 / 0.96)
- **Incremental call**: akustisk confusion med rondoensis-klustret; plats-omrankning kan **inte** lyfta in granti (rank 6, utanför CNN top-3)

Tolka inte p=1.0 som “definitivt rondoensis” för denna call type.

---

## Modellförbättring — granti ↔ rondoensis (körd 2026-06-25)

### Predict (Kenya-profil)

```text
data/splits/granti_review_filelist.txt  (3 WAV)
→ outputs/predictions/predictions_granti_review_2026-06.csv
```

| Fil | Prediktion |
|-----|------------|
| Buzz G granti | Paragalago_granti ✓ |
| Buzz-Screech+ Grunt | Paragalago_granti ✓ |
| Incremental call | Paragalago_rondoensis ✗ |

### Shortlist

```bash
python scripts/make_cluster_review_shortlist.py \
  --csv outputs/predictions/predictions_granti_review_2026-06.csv \
  --out-csv outputs/evaluation/cluster_review_granti_2026-06.csv \
  --emit-relabel-stubs data/relabels/relabels_from_cluster_shortlist_granti_2026-06.csv
```

- `make_cluster_review_shortlist.py` utökad: **granti / zanzibaricus / orinus** i CLUSTER + boost för granti↔rondoensis-par

### Relabel + ingest

- Kuraterad CSV: `data/relabels/relabels_granti_incremental_2026-06.csv` (`ingest=yes` på 3 rader)
- `ingest_relabels.py` resultat:
  - **Added: 1** (Incremental call — ny mel)
  - **Skipped: 2** (Buzz-filer — PNG fanns redan under `Paragalago_granti/relabel__*`)
  - Append till `data/splits/mels_cluster_rondoensis_train.txt`

### Omträning (pågår lokalt)

```bash
python scripts/train_cnn_all_species_improved.py --epochs 50
```

Logg: `outputs/train_retrain_2026-06.log`  
Vikter: `models/all_species/galago_cnn_all_19classes_improved_best.keras` (när klar)

### Efter träning (nästa session)

```bash
python scripts/predict_3stage_with_context.py \
  --wav "data/raw_audio/Paragalago_granti/Incremental call  G granti.wav" \
  --profile kenya_balanced_auto --lat -1.3 --lon 36.8

python scripts/run_eval_regions.py --tag after_retrain_2026-06
```

Jämför mot `docs/session_eval_2026-03_retrain_v2.md`. Vid lyckad incremental-fix: ladda upp nya vikter till `ulfboge/galago-demo-vikter`.

---

## Öppna punkter

1. **Rotera HF-token** (exponerad mars 2026) — `docs/next_steps_2026-06-24.md` §1
2. **Verifiera karta** på Space efter senaste push (ska synas + klicka)
3. **Incremental granti** — om fortfarande fel efter omträning: snippet-windows (`start_sec`/`end_sec`) i relabel, inte bara center 2.5 s
4. **Observer-feedback / Slack** på Space — ej testat

---

## Snabbreferens

| Vad | Var |
|-----|-----|
| Live-demo | https://huggingface.co/spaces/ulfboge/galago-call-demo |
| Relabel-workflow | `docs/confusion_relabel_workflow.md` |
| Senaste Space-fix | `3ef054e` (Starlette), `368e577` (karta v1), + kartfix v2 denna session |
| Grant relabel CSV | `data/relabels/relabels_granti_incremental_2026-06.csv` |

**Deploy-rutin:** `git push origin main` + `git push space main`
