# Geographic assumptions for deployment profiles

These are **operational priors** for the classifier + post-processing. They are not a substitute for local verification.

## Tanzania (Pugu, Rondo, Pande hills recordings)

**Assumption:** *Galagoides sp.nov* is **not** part of the expected species list for these Tanzania sites in this project. A model prediction of `Galagoides_sp_nov` on Tanzania-tagged audio is treated as **acoustic confusion** with taxa in the *Paragalago rondoensis* / dwarf-galago cluster (see deployment profiles).

**Profiles:**

| Profile | Behaviour when model emits `Galagoides_sp_nov` |
|---------|-----------------------------------------------|
| `tanzania_balanced_auto` | Same remap as aggressive (default Tanzania ops profile in `deployment_profiles.json`). To disable: put **`--postprocess-mode none` after `--profile tanzania_balanced_auto`**. |
| `tanzania_site_rules` | Never emit confident sp.nov: remap to `Paragalago_rondoensis` only when polygon prior says sp.nov unlikely **and** top-2 is rondoensis; otherwise **`uncertain`**. |
| `tanzania_site_rules_aggressive` | Remap sp.nov → **`Paragalago_rondoensis`** for Tanzania site tokens (filename / location). |

Malawi sp.nov populations use **`malawi_*`** profiles — **do not** use Tanzania sp.nov-remap logic there.

## Malawi

Recordings under `G.sp.nov.1` / `G.sp.nov.3` are mapped to **`Galagoides_sp_nov`** for training and evaluation. Use **`malawi_balanced`** / **`malawi_balanced_auto`** and a correct **`recording_locations.json`** entry for Malawi lat/lon.

## Kenya

Kenya subset files use **`kenya_balanced_auto`** (or `kenya_balanced`). Sp.nov is not assumed absent; do not apply Tanzania sp.nov remapping.

**Assumption:** `Paragalago_granti`, `Paragalago_orinus`, `Paragalago_rondoensis`, and `Paragalago_zanzibaricus` are **not** expected in Kenya (see `data/species_ranges.json`). Kenya-plausible dwarf galago: **`Paragalago_cocos`**; common larger galago: **`Otolemur_garnettii`**.

**Profiles `kenya_balanced` / `kenya_balanced_auto`** set `postprocess_mode: kenya_geo_guard` and `context_alpha: 1.0`. When lat/lon (or location string Kenya) indicates Kenya and the acoustic top-1 is out of range, the pipeline tries the best **in-range** species from the acoustic top-10; otherwise **`uncertain`**. Acoustic top-1 is still shown in demo as *Top-1 klass*.

Disable for classifier-only eval: `--postprocess-mode none` after `--profile kenya_balanced_auto`.

**Data caveat:** Some WAVs under `data/raw_audio/Paragalago_granti/` are tagged Kenya in `recording_locations.json` but use the taxon name *granti* in the filename — folder/filename accuracy ≠ geographic truth; see `docs/evaluation_caveats.md`.

## Batch regional evaluation

To run Kenya, Tanzania (`tanzania_balanced_auto`, with TZ sp.nov remap), and Malawi in one pass and print accuracy summaries:

```powershell
python scripts/run_eval_regions.py
```

See `docs/immediate_next_steps.md` §3 for options (`--tag`, `--predict-only`, `--tanzania-postprocess none`).

**Evaluation:** reported folder-based accuracy can mis-rank files that are mis-filed or snippet-mismatched — see `docs/evaluation_caveats.md`. Example A/B run (default TZ vs classifier-only): `docs/session_eval_2026-03_retrain_v2.md`.
