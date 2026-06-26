# Sessionlogg — granti buzz-ft, tail-pool, Kenya geo-guard (2026-06-26)

Handover efter modellförbättring, Hub-deploy och geografisk inferensfix för Kenya.

---

## Bakgrund

Tre Kenya-taggade granti-klipp gav fel art i demo/Hub:

| Fil | Problem (före) |
|-----|----------------|
| `Buzz  G granti.wav` | orinus (p≈1.0) |
| `Incremental call  G granti.wav` | rondoensis / otolemur (pooling) |
| `Buzz-Screech+ Grunt G granti.wav` | oftast OK |

Rotorsaker:

1. **Inferens:** top-3 confidence-pooling väljer höga tidiga fönster (blandade långa klipp).
2. **Träning:** buzz-likhet granti/orinus; `Paragalago_orinus: 1.5` i `SPECIES_OVERSAMPLE_BOOST`.
3. **Geografi:** `Paragalago_granti` finns **inte** i Kenya enligt `data/species_ranges.json` — mappnamn/filnamn ≠ fältart vid Kenya-koordinater.

---

## Vad som gjordes

### 1. Relabel + ingest (tidigare i sprinten)

- `data/relabels/relabels_granti_incremental_2026-06.csv` — Buzz + incremental-snippets
- Relabel-mels under `data/melspectrograms/Paragalago_granti/` (bl.a. `relabel__Buzz__G_granti__full.png`)

### 2. Buzz-cluster finetune

**Nya skript:**

- `scripts/make_granti_buzz_cluster_filelist.py` — buzz-PNG:er i granti/orinus-klustret, **12× boost** för Kenya `Buzz  G granti`
- `scripts/finetune_granti_buzz_cluster.py` — finetune från `improved_best`

**Output:** `models/all_species/galago_cnn_granti_buzz_ft_best.keras`  
**Backup:** `models/all_species/galago_cnn_all_19classes_improved_best_pre_buzz.keras`

**Smoke-test (buzz-ft, akustik):**

| Fil | Resultat |
|-----|----------|
| Buzz G granti | granti (1.0) ✓ |
| Incremental call | granti (0.79–0.97) ✓ |
| Buzz-Screech+ Grunt | granti (1.0) ✓ |

### 3. Inferens: tail-pool vid konflikt

**Fil:** `scripts/predict_3stage_with_context.py`  
**Commit:** `2e75268`

När poolade top-K-fönster **inte enas** och filen är **> 5 s** → om-poola från **svans** (≥ max(5 s, halva filen)).

Fixar incremental där sant ljud ligger sent (t.ex. granti vid 7.5 s men oto/rondoensis tidigt).

### 4. Regional eval (`after_granti_buzz_ft`)

```bash
python scripts/run_eval_regions.py --tag after_granti_buzz_ft \
  --classifier-model models/all_species/galago_cnn_granti_buzz_ft_best.keras
```

| Region | Top-1 | Kommentar |
|--------|-------|-----------|
| Kenya (34) | 94.1% | granti 3/3 ✓; **Knock→granti** ✗ |
| Tanzania (10) | 90.0% | |
| Malawi (24) | 95.8% | |

Jämförelse mars-baseline (`improved_best`): Kenya **97.1%** → buzz-ft **−3 pp** men granti-kluster fixat.

**CSVs:** `outputs/predictions/predictions_3stage_eval_{kenya,tanzania,malawi}_after_granti_buzz_ft.csv`

### 5. Knock-regression (`O_garnettii_Knock.wav`)

Blandad fil: tid ≈ granti, sent ≈ garnettii.

| Modell | Knock |
|--------|-------|
| `improved_best` | garnettii (0.96) ✓ |
| `granti_buzz_ft` | granti (0.97) ✗ |

Buzz-finetune förstärker granti i tidiga fönster; tail-pool räcker inte. Garnettii-anchor-finetune testades — fixade inte Knock, sänkte Malawi.

**Åtgärd framåt:** relabel tidiga knock-segment eller separat strategi — inte blockerande för buzz-deploy.

### 6. Hub-deploy (buzz-ft + tail-pool-kod)

| Steg | Status |
|------|--------|
| Tail-pool push GitHub + Space | `2e75268` |
| buzz-ft uppladdad som `galago_cnn_all_19classes_improved_best.keras` | `ulfboge/galago-demo-vikter` |

Space hämtar vikter vid start via `GALAGO_HF_MODEL_REPO`. Vid cache-problem: **Factory reboot** i Space-inställningar.

### 7. Kenya geo-guard (geografisk inferens)

**Problem:** Buzz-ft + Kenya-plats gav **granti** med `[Unlikely here]` — biogeografiskt fel (granti: Malawi/Mozambique/Tanzania/Zimbabwe, inte Kenya).

**Lösning:**

- `postprocess_mode: kenya_geo_guard` på `kenya_balanced` / `kenya_balanced_auto`
- `context_alpha: 1.0` på Kenya-profiler
- Om top-1 är out-of-range i Kenya (granti, orinus, rondoensis, zanzibaricus) → försök bästa **in-range** kandidat från akustisk top-10; annars **`uncertain`**
- Demo visar fortfarande **Top-1 klass** (akustik) separat från **Trolig art (output)**

**Commit:** `26752aa`  
**Filer:** `predict_3stage_with_context.py`, `configs/deployment_profiles.json`, `docs/deployment_geographic_assumptions.md`

**Verifierat lokalt (buzz-ft, Kenya −1.3, 36.8, `kenya_balanced_auto`):**

| Fil | Trolig art | Top-1 |
|-----|------------|-------|
| Buzz G granti | **uncertain** | granti |
| Incremental call | **uncertain** | granti |

### 8. Demo: varför `uncertain` trots hög Top-1?

**Trolig art** = efter profilregler. **Top-1 klass** = ren akustik.

`tanzania_balanced_auto` har `consensus_min_count: 2` — samma art i ≥2 poolade fönster. `Incremental call` är blandad → ofta bara 1/2 granti → **uncertain** trots hög Top-1 (t.ex. p≈0.97).

För Tanzania med granti som output: `tanzania_balanced` eller `--consensus-min-count 0`.

---

## Commits (denna sprint-del)

| Commit | Innehåll |
|--------|----------|
| `2e75268` | Tail-pool fallback vid oeniga fönster |
| `26752aa` | Kenya geo-guard + profil + `deployment_geographic_assumptions.md` |

**Deploy:** `git push origin main` + `git push space main` (båda körda för `26752aa`).

---

## Viktiga sökvägar

| Vad | Plats |
|-----|--------|
| Buzz-ft vikter (lokal) | `models/all_species/galago_cnn_granti_buzz_ft_best.keras` |
| Pre-buzz backup | `models/all_species/galago_cnn_all_19classes_improved_best_pre_buzz.keras` |
| Hub (Space default) | `ulfboge/galago-demo-vikter` → `galago_cnn_all_19classes_improved_best.keras` (= buzz-ft) |
| Kenya-profil | `kenya_balanced_auto` i demo |
| Geo-antaganden | `docs/deployment_geographic_assumptions.md` |
| Eval-caveats | `docs/evaluation_caveats.md` |

---

## Kända tradeoffs / öppet

1. **Kenya granti-mapp** — folder-accuracy ≠ geografisk sanning; tre filer i `Paragalago_granti/` med Kenya-koordinat i `recording_locations.json`.
2. **Knock→granti** med buzz-ft på Kenya-eval.
3. **Orinus→zanzibaricus** (`G_orinus_Rapid_Yaps.wav`) — kvar från mars.
4. **Hub buzz-ft** — bra för akustisk granti-match, Kenya-demo kräver **`kenya_balanced_auto`** + plats för geo-guard.
5. **Ocommittat lokalt** — finetune-skript, relabel-CSV, övriga docs/skript (ej i `26752aa`).

---

## Kommandon (upprepa)

```bash
# Buzz finetune
python scripts/make_granti_buzz_cluster_filelist.py
python scripts/finetune_granti_buzz_cluster.py

# Inferens Kenya
python scripts/predict_3stage_with_context.py \
  --wav "data/raw_audio/Paragalago_granti/Buzz  G granti.wav" \
  --profile kenya_balanced_auto --lat -1.3 --lon 36.8

# Regional eval buzz-ft
python scripts/run_eval_regions.py --tag after_granti_buzz_ft \
  --classifier-model models/all_species/galago_cnn_granti_buzz_ft_best.keras

# Deploy Space
git push origin main
git push space main
```

---

## Nästa steg (förslag)

1. Smoke-testa [galago-call-demo](https://huggingface.co/spaces/ulfboge/galago-call-demo) efter rebuild: Kenya-plats + `kenya_balanced_auto` + granti-klipp → **uncertain**.
2. Kurera **korrekt Kenya-art** för de tre granti-mappfilerna (cocos? garnettii?) → relabel + omträning.
3. Åtgärda **Knock-regression** (relabel tidiga segment eller återställ pre-buzz för vissa profiler).
4. Överväg `Paragalago_orinus: 1.5` bort från `SPECIES_OVERSAMPLE_BOOST` inför full omträning.
