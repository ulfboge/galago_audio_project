# Next steps — 2026-06-24 / 25

State after re-activating the project in June 2026 (~3 months after March session).

**Where we are:** HF Space is **live** (inferens + karta). Lokal omträning juni 2026 klar men **inte** på Hub. Incremental granti fortfarande fel; snippet-relabel är nästa modellsteg.

---

## 1. Rotate the exposed HF token (security — still open)

A Hugging Face token was exposed in chat during the March 2026 deploy debugging.

- Revoke at https://huggingface.co/settings/tokens
- New token with write access to `ulfboge/galago-demo-vikter` and `ulfboge/galago-call-demo`
- `hf auth login --token <new_token> --add-to-git-credential --force`
- Update `.env` if stored there

---

## 2. HF Space — status

| Item | Status |
|------|--------|
| Build / Starlette crash | ✅ `3ef054e` |
| `species_ranges.json` on Space | ✅ |
| Inferens + modell-download | ✅ |
| Karta lat/lon | ✅ `524d854` (`/galago-map` + `create_app`-patch) |
| Upload WAV → prediction (UI) | ✅ (rapporterat fungerande) |
| Observer-feedback + Slack | ⚠️ Not verified this sprint |
| Deploy | `git push origin main` + `git push space main` |

Details: `docs/session_2026-06-24_space_fix.md`

---

## 3. Granti incremental — modell (highest ROI now)

**Problem:** `Incremental call  G granti.wav` — Hub: `Paragalago_rondoensis`; lokal omträning: `Paragalago_zanzibaricus` (inte granti). Buzz-call **regressed** to `Galago_moholi` after June retrain.

**Done:**
- Relabel CSV + ingest (1 new mel, no snippets)
- Full retrain: val acc **95.4%**, 26 epochs
- `make_cluster_review_shortlist.py` extended for granti cluster

**Next:**

```bash
# 1) Edit snippet windows in relabel CSV (critical for incremental)
#    data/relabels/relabels_granti_incremental_2026-06.csv

python scripts/ingest_relabels.py \
  --csv data/relabels/relabels_granti_incremental_2026-06.csv \
  --out-filelist data/splits/mels_relabels_additions.txt \
  --append-filelist data/splits/mels_cluster_rondoensis_train.txt

python scripts/train_cnn_all_species_improved.py --epochs 50

# 2) Smoke-test all three granti WAVs before Hub upload
python scripts/predict_3stage_with_context.py \
  --wav "data/raw_audio/Paragalago_granti/Incremental call  G granti.wav" \
  --profile balanced --lat -1.3 --lon 36.8

python scripts/run_eval_regions.py --tag after_retrain_2026-06
```

**Do not upload** `improved_best.keras` to Hub until incremental + Buzz are acceptable locally.

Reference eval (March 2026, `docs/session_eval_2026-03_retrain_v2.md`):

| Region | Top-1 | Top-3 |
|--------|-------|-------|
| Malawi (24) | 100.0% | 100.0% |
| Kenya (34) | 97.1% | 97.1% |
| Tanzania (11) | 81.8% | 100.0% |

---

## 4. Broader relabel backlog

~175 candidate rows in `data/relabels/` — Tanzania / rondoensis confusion still weak spot. See `docs/confusion_relabel_workflow.md` and `docs/immediate_next_steps.md`.

---

## 5. User feedback

```bash
python scripts/review_feedback.py --only-pending
python scripts/export_feedback_to_relabels.py --dry-run
```

Demo log still mostly March test data unless Space accumulated real usage.

---

## 6. Hub model repo

Space pulls `galago_cnn_all_19classes_improved_best.keras` from `ulfboge/galago-demo-vikter` — **still pre-June retrain** until you upload.

Optional: confirm Malawi ft4 weights in Hub (`galago_cnn_malawi_spnov_ft4_best.keras`).

---

## Background

- **Mel inference fix (2026-03-25):** vertical flip + `[0,1]` scale — do not revert.
- **Active classifier (production default):** `galago_cnn_all_19classes_improved_best.keras`
- **Map routes:** must be registered inside patched `App.create_app`, not before `demo.launch()`
- **Trainer:** uses `matplotlib` Agg backend (headless-safe)
